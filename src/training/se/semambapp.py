"""SEMamba++ generator and discriminator updates.

The generator is the independent network in ``src.se``. The loss weights match
the published training config: adversarial 1, feature matching 1 (the matching
term itself is scaled by 2), mel 0.1, magnitude 0.9, phase 0.3, complex 0.1,
consistency 0.1. The waveform L1 term is not in that sum. The discriminators
are an independent multi-resolution stack and a four-band stack. The reference
repository has no license file, so its discriminator source is not copied.

Update order is discriminator, then generator. ExponentialLR steps once per
finished epoch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import backward
from src.training.errors import TrainingError
from src.training.se.discriminators import MultiResolutionDiscriminator, SubbandDiscriminator
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.spectral import (
    compressed_complex_losses,
    feature_matching,
    least_squares_discriminator,
    least_squares_generator,
    log_mel_l1,
)
from src.training.se.stft import mag_pha_istft, mag_pha_stft

_MAG = 0.9
_PHASE = 0.3
_COMPLEX = 0.1
_CONSISTENCY = 0.1
_MEL = 0.1
_ADV = 1.0
_FM = 1.0


class SEMambaTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        if config["loss"] != "official":
            raise TrainingError("SEMamba++ training uses loss official")
        architecture = dict(model.get("architecture") or {})
        stft = model.get("stft") or {}
        self.n_fft = int(architecture.get("n_fft", stft.get("n_fft", 400)))
        self.hop = int(stft.get("hop", 100 if self.n_fft == 400 else max(self.n_fft // 4, 1)))
        self.win = int(stft.get("win", self.n_fft))
        self.compress = 0.3
        self.sample_rate = int(model["sampling_rate"])
        windows = stft.get("disc_windows") or (256, 512, 1024)
        self.disc_windows = tuple(int(item) for item in windows)
        self.grad_clip = config.get("grad_clip")
        with seed_model_init(config["seed"]):
            self.generator = build_model("semambapp", architecture or None, variant=model.get("variant")).to(device)
            self.discriminator_mrd = MultiResolutionDiscriminator(self.disc_windows).to(device)
            self.discriminator_band = SubbandDiscriminator(self.n_fft).to(device)
        discriminator_parameters = list(self.discriminator_mrd.parameters()) + list(self.discriminator_band.parameters())
        self.optim_g = build_optimizer(self.generator.parameters(), config)
        self.optim_d = build_optimizer(discriminator_parameters, config)
        self.sched_g = build_scheduler(self.optim_g, config)
        self.sched_d = build_scheduler(self.optim_d, config)

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)
        self.discriminator_mrd = wrap_module(self.discriminator_mrd, process)
        self.discriminator_band = wrap_module(self.discriminator_band, process)

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator_mrd.train()
        self.discriminator_band.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        unwrap(self.discriminator_mrd).eval()
        unwrap(self.discriminator_band).eval()

    def learning_rates(self) -> dict[str, float]:
        return {
            "generator": float(self.optim_g.param_groups[0]["lr"]),
            "discriminator": float(self.optim_d.param_groups[0]["lr"]),
        }

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        step_scheduler(self.sched_g, validation)
        step_scheduler(self.sched_d, validation)

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def export_for_inference(self) -> dict[str, Any]:
        return {"generator": self.generator_state_dict()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": "official",
            "generator": self.generator_state_dict(),
            "discriminator_mrd": unwrap(self.discriminator_mrd).state_dict(),
            "discriminator_band": unwrap(self.discriminator_band).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "sched_g": None if self.sched_g is None else self.sched_g.state_dict(),
            "sched_d": None if self.sched_d is None else self.sched_d.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "official":
            raise TrainingError("checkpoint loss does not match SEMamba++")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.discriminator_mrd.load_state_dict(state["discriminator_mrd"], strict=True)
        self.discriminator_band.load_state_dict(state["discriminator_band"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        self.optim_d.load_state_dict(state["optim_d"])
        if self.sched_g is not None and state.get("sched_g") is not None:
            self.sched_g.load_state_dict(state["sched_g"])
            self.sched_d.load_state_dict(state["sched_d"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        discriminators = [self.discriminator_mrd, self.discriminator_band]
        self.optim_d.zero_grad(set_to_none=True)
        disc_total = 0.0
        for index, batch in enumerate(batches):
            clean, noisy = _waves(batch, self.device)
            with torch.no_grad():
                parts = self._generate(clean, noisy, unwrap(self.generator))
            real_scores, _ = _scores(unwrap(self.discriminator_mrd), unwrap(self.discriminator_band), clean)
            fake_scores, _ = _scores(unwrap(self.discriminator_mrd), unwrap(self.discriminator_band), parts["audio"].detach())
            loss = sum(
                least_squares_discriminator(real, fake) for real, fake in zip(real_scores, fake_scores, strict=True)
            )
            _finite(loss, "semamba discriminator")
            backward(loss / accumulation, discriminators, last=index == len(batches) - 1)
            disc_total += float(loss.detach())
        clip_gradients(unwrap(self.discriminator_mrd), self.grad_clip)
        clip_gradients(unwrap(self.discriminator_band), self.grad_clip)
        self.optim_d.step()

        frozen = [parameter for module in discriminators for parameter in unwrap(module).parameters()]
        for parameter in frozen:
            parameter.requires_grad_(False)
        self.optim_g.zero_grad(set_to_none=True)
        gen_total = 0.0
        try:
            for index, batch in enumerate(batches):
                clean, noisy = _waves(batch, self.device)
                parts = self._generate(clean, noisy, self.generator)
                _, real_features = _scores(unwrap(self.discriminator_mrd), unwrap(self.discriminator_band), clean)
                fake_scores, fake_features = _scores(
                    unwrap(self.discriminator_mrd), unwrap(self.discriminator_band), parts["audio"]
                )
                adversarial = sum(least_squares_generator(score) for score in fake_scores)
                matching = feature_matching(real_features, fake_features, scale=2.0)
                spectral = compressed_complex_losses(
                    parts["clean_mag"],
                    parts["mag"],
                    parts["clean_phase"],
                    parts["phase"],
                    parts["clean_complex"],
                    parts["complex"],
                    parts["recomputed"],
                )
                mel = log_mel_l1(clean, parts["audio"], self.sample_rate)
                loss = (
                    adversarial * _ADV
                    + matching * _FM
                    + mel * _MEL
                    + spectral["magnitude"] * _MAG
                    + spectral["phase"] * _PHASE
                    + spectral["complex"] * _COMPLEX
                    + spectral["consistency"] * _CONSISTENCY
                )
                _finite(loss, "semamba generator")
                backward(loss / accumulation, [self.generator], last=index == len(batches) - 1)
                gen_total += float(loss.detach())
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        count = len(batches)
        return {"generator": gen_total / count, "discriminator": disc_total / count}

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean, noisy = _waves(batch, self.device)
        with torch.no_grad():
            parts = self._generate(clean, noisy, unwrap(self.generator))
            value = float(F.l1_loss(clean, parts["audio"]).detach())
        metric = _metric(value)
        return {
            "losses": {
                "generator": metric,
                "time": metric,
                "discriminator": _metric(status="na", reason="validation reports the waveform L1, not the GAN"),
            },
            "pesq": {"status": "na", "value": None, "reason": "not computed", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "not computed", "count": 0},
        }

    def _generate(self, clean: torch.Tensor, noisy: torch.Tensor, generator: torch.nn.Module) -> dict[str, torch.Tensor]:
        clean_mag, clean_phase, clean_complex = mag_pha_stft(clean, self.n_fft, self.hop, self.win, self.compress)
        noisy_mag, noisy_phase, _ = mag_pha_stft(noisy, self.n_fft, self.hop, self.win, self.compress)
        magnitude, phase = generator(noisy_mag, noisy_phase)
        if magnitude.shape != noisy_mag.shape:
            magnitude = _match_bins(magnitude, noisy_mag.shape[-2])
            phase = _match_bins(phase, noisy_phase.shape[-2])
        complex_spectrum = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
        audio = mag_pha_istft(magnitude, phase, self.n_fft, self.hop, self.win, self.compress)
        if audio.shape[-1] != clean.shape[-1]:
            audio = _fit_time(audio, clean.shape[-1])
        recomputed_mag, _, recomputed = mag_pha_stft(audio, self.n_fft, self.hop, self.win, self.compress)
        del recomputed_mag
        return {
            "clean_mag": clean_mag,
            "clean_phase": clean_phase,
            "clean_complex": clean_complex,
            "mag": magnitude,
            "phase": phase,
            "complex": complex_spectrum,
            "recomputed": recomputed,
            "audio": audio,
        }


def _scores(mrd, band, waveform):
    mrd_scores, mrd_features = mrd(waveform)
    band_scores, band_features = band(waveform)
    return mrd_scores + band_scores, mrd_features + band_features


def _match_bins(value: torch.Tensor, bins: int) -> torch.Tensor:
    current = value.shape[-2]
    if current == bins:
        return value
    if current > bins:
        return value[..., :bins, :]
    return F.pad(value, (0, 0, 0, bins - current))


def _fit_time(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.shape[-1] > length:
        return waveform[..., :length]
    return F.pad(waveform, (0, length - waveform.shape[-1]))


def _waves(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    clean = batch["clean"].to(device)
    noisy = batch["noisy"].to(device)
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise TrainingError(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    return clean, noisy


def _finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
