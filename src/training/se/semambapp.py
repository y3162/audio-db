"""SEMamba++ generator and discriminator updates.

The generator is the independent network in ``src.se``. The loss weights match
the published training config: adversarial 1, feature matching 1 (the matching
term itself is the sum of per-map L1 means scaled by 2), mel 0.1, magnitude
0.9, phase 0.3, complex 0.1, consistency 0.1. The waveform L1 term is commented
out of that sum. Discriminators are the published multi-scale sub-band CQT
stack and the three-resolution magnitude stack, implemented in
``semamba_disc`` without importing the reference tree.

Update order is discriminator, then generator. One micro-batch reuses the
generator forward. More than one micro-batch forwards the generator again
after the discriminator step, because keeping every graph does not fit and
the generator has no dropout. Each mean loss is divided by the accumulation
count. ExponentialLR steps once per finished epoch. The reference selective
scan is recomputed while training so a 3-second full-width step fits in 16 GB.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import accumulation_scope
from src.training.errors import TrainingError
from src.training.se.semamba_disc import MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.spectral import (
    compressed_complex_losses,
    feature_matching,
    least_squares_discriminator,
    least_squares_generator,
    summed_slaney_log_mel,
)
from src.training.se.stft import semamba_mag_phase_istft, semamba_mag_phase_stft

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
        self.grad_clip = config.get("grad_clip")
        with seed_model_init(config["seed"]):
            self.generator = build_model("semambapp", architecture or None, variant=model.get("variant")).to(device)
            self.discriminator_mrd = MultiResolutionDiscriminator().to(device)
            self.discriminator_band = MultiScaleSubbandCQTDiscriminator(self.sample_rate).to(device)
        discriminator_parameters = list(self.discriminator_mrd.parameters()) + list(self.discriminator_band.parameters())
        self.optim_g = build_optimizer(self.generator.parameters(), config)
        self.optim_d = build_optimizer(discriminator_parameters, config)
        self.sched_g = build_scheduler(self.optim_g, config)
        self.sched_d = build_scheduler(self.optim_d, config)
        self.capture_update = False
        self.capture_gradients = False
        self.last_capture = None

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
        # One micro-batch keeps its generator graph and reuses that waveform.
        # Several micro-batches would otherwise retain every graph until the
        # generator backward. The generator has no dropout, so each later
        # micro-batch is forwarded again after the discriminator step and the
        # previous graph is released. A single batch still reuses one forward.
        reuse_graph = len(batches) == 1
        cached: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]] = []
        disc_total = 0.0
        for index, batch in enumerate(batches):
            last = index == len(batches) - 1
            clean, noisy = _waves(batch, self.device)
            if reuse_graph:
                with accumulation_scope([self.generator], last=True):
                    parts = self._generate(clean, noisy, self.generator)
                audio = parts["audio"]
                cached.append((clean, noisy, parts))
            else:
                with torch.no_grad():
                    preview = self._generate(clean, noisy, unwrap(self.generator))
                audio = preview["audio"]
                cached.append((clean, noisy, None))
            with accumulation_scope(discriminators, last=last):
                real_scores, _ = _scores(self.discriminator_mrd, self.discriminator_band, clean)
                fake_scores, _ = _scores(
                    self.discriminator_mrd,
                    self.discriminator_band,
                    audio.detach(),
                )
                loss = sum(
                    least_squares_discriminator(real, fake) for real, fake in zip(real_scores, fake_scores, strict=True)
                )
                _finite(loss, "semamba discriminator")
                # Each term is a mean. Dividing by the micro-batch count makes the
                # summed gradient the mean over the global batch, which is what one
                # official batch computes. Leaving the mean unscaled would add the
                # micro-batch gradients and take a larger AdamW step.
                (loss / accumulation).backward()
            disc_total += float(loss.detach())
        clip_gradients(unwrap(self.discriminator_mrd), self.grad_clip)
        clip_gradients(unwrap(self.discriminator_band), self.grad_clip)
        disc_norm, disc_finite = _grad_norm(unwrap(self.discriminator_mrd), unwrap(self.discriminator_band))
        if self.capture_update and self.capture_gradients:
            disc_grads = _clone_grads(unwrap(self.discriminator_mrd), unwrap(self.discriminator_band))
        else:
            disc_grads = None
        self.optim_d.step()

        frozen = [parameter for module in discriminators for parameter in unwrap(module).parameters()]
        for parameter in frozen:
            parameter.requires_grad_(False)
        self.optim_g.zero_grad(set_to_none=True)
        gen_total = 0.0
        components = {name: 0.0 for name in ("adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")}
        produced: list[torch.Tensor] = []
        try:
            for index, (clean, noisy, parts) in enumerate(cached):
                if parts is None:
                    with accumulation_scope([self.generator], last=index == len(cached) - 1):
                        parts = self._generate(clean, noisy, self.generator)
                        gen_total += self._generator_backward(clean, parts, accumulation, components)
                else:
                    gen_total += self._generator_backward(clean, parts, accumulation, components)
                produced.append(parts["audio"].detach())
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        clip_gradients(unwrap(self.generator), self.grad_clip)
        gen_norm, gen_finite = _grad_norm(unwrap(self.generator))
        if self.capture_update:
            self.last_capture = {
                "audio": [audio.to(device="cpu", dtype=torch.float32).clone() for audio in produced],
                "generator_grad": _clone_grads(unwrap(self.generator)) if self.capture_gradients else None,
                "discriminator_grad": disc_grads,
                "generator_grad_finite": gen_finite,
                "discriminator_grad_finite": disc_finite,
            }
        self.optim_g.step()
        count = len(batches)
        metrics = {
            "generator": gen_total / count,
            "discriminator": disc_total / count,
            "generator_grad_norm": gen_norm,
            "discriminator_grad_norm": disc_norm,
        }
        metrics.update({name: value / count for name, value in components.items()})
        return metrics

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean, noisy = _waves(batch, self.device)
        with torch.no_grad():
            parts = self._generate(clean, noisy, unwrap(self.generator))
            value = float(F.l1_loss(clean, parts["audio"]).detach())
        from src.evaluation.metrics.sisdr import score_si_sdr
        from src.evaluation.metrics.stoi import score_stoi
        from src.evaluation.metrics.pesq import score_pesq

        reference, estimate = clean[0], parts["audio"][0]
        metric = _metric(value)
        return {
            "lengths": {"input": int(noisy[0].numel()), "output": int(estimate.numel()),
                        "reference": int(reference.numel())},
            "sample_rate": self.sample_rate,
            "si_sdr": score_si_sdr(reference, estimate),
            "losses": {
                "generator": metric,
                "time": metric,
                "discriminator": _metric(status="na", reason="validation reports the waveform L1, not the GAN"),
            },
            "pesq": score_pesq(reference, estimate, self.sample_rate),
            "stoi": score_stoi(reference, estimate, self.sample_rate),
        }

    def _generator_backward(
        self,
        clean: torch.Tensor,
        parts: dict[str, torch.Tensor],
        accumulation: int,
        components: dict[str, float],
    ) -> float:
        _, real_features = _scores(unwrap(self.discriminator_mrd), unwrap(self.discriminator_band), clean)
        fake_scores, fake_features = _scores(
            unwrap(self.discriminator_mrd), unwrap(self.discriminator_band), parts["audio"]
        )
        adversarial = sum(least_squares_generator(score) for score in fake_scores)
        matching = feature_matching(real_features, fake_features, scale=2.0, reduction="sum")
        spectral = compressed_complex_losses(
            parts["clean_mag"],
            parts["mag"],
            parts["clean_phase"],
            parts["phase"],
            parts["clean_complex"],
            parts["complex"],
            parts["recomputed"],
            phase_mode="matrix",
        )
        mel = summed_slaney_log_mel(clean, parts["audio"], self.sample_rate)
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
        (loss / accumulation).backward()
        components["adversarial"] += float(adversarial.detach())
        components["matching"] += float(matching.detach())
        components["mel"] += float(mel.detach())
        for name in ("magnitude", "phase", "complex", "consistency"):
            components[name] += float(spectral[name].detach())
        return float(loss.detach())

    def _generate(self, clean: torch.Tensor, noisy: torch.Tensor, generator: torch.nn.Module) -> dict[str, torch.Tensor]:
        clean_mag, clean_phase, clean_complex = semamba_mag_phase_stft(
            clean, self.n_fft, self.hop, self.win, self.compress, addeps=False
        )
        noisy_mag, noisy_phase, _ = semamba_mag_phase_stft(
            noisy, self.n_fft, self.hop, self.win, self.compress, addeps=False
        )
        magnitude, phase = generator(noisy_mag, noisy_phase)
        if magnitude.shape != noisy_mag.shape:
            magnitude = _match_bins(magnitude, noisy_mag.shape[-2])
            phase = _match_bins(phase, noisy_phase.shape[-2])
        complex_spectrum = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
        audio = semamba_mag_phase_istft(
            magnitude, phase, self.n_fft, self.hop, self.win, self.compress, length=clean.shape[-1]
        )
        if audio.shape[-1] != clean.shape[-1]:
            audio = _fit_time(audio, clean.shape[-1])
        recomputed_mag, _, recomputed = semamba_mag_phase_stft(
            audio, self.n_fft, self.hop, self.win, self.compress, addeps=True
        )
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


def _grad_norm(*modules: torch.nn.Module) -> tuple[float, bool]:
    total = 0.0
    finite = True
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                finite = False
            total += float(grad.square().sum())
    return total**0.5, finite


def _clone_grads(*modules: torch.nn.Module) -> dict[str, torch.Tensor]:
    cloned = {}
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            prefix = module.__class__.__name__
            cloned[f"{prefix}.{name}"] = parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
    return cloned


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
