"""SEMamba++ rank-0 validation. The shadow is a weight copy and is not saved.

The comparison task keeps its own ``train_update`` and ``validation_report``.
This module scores the copied generator with waveform L1, PESQ, STOI, and
SI-SDR. It does not call the comparison task and it does not step a scheduler.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.calc.distributed import unwrap
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.metrics import _aggregate_validation
from src.experiments.calc.rng import seed_model_init
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.models.semambapp.model import (
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from src.experiments.runtime import ProcessRuntime
from src.models import build_model
from src.experiments.calc.stft import semamba_mag_phase_istft, semamba_mag_phase_stft

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


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}


class ValidationShadow:
    """Generator and discriminators built in the same order as the comparison task.

    Validation copies the live weights over these modules and then scores the
    generator. The discriminators are loaded so the shadow stays the same
    object the previous task constructed. The report does not use them.
    """

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.device = device
        model = config["model"]
        if config["loss"] != "official":
            raise TrainingError("SEMamba++ training uses loss official")
        architecture = dict(model.get("architecture") or {})
        mrd_resolutions = architecture.pop("mrd_resolutions", None)
        mrd_channels = architecture.pop("mrd_channels", None)
        cqt_hops = architecture.pop("cqt_hops", None)
        cqt_octaves = architecture.pop("cqt_octaves", None)
        cqt_bins = architecture.pop("cqt_bins_per_octave", None)
        cqt_channels = architecture.pop("cqt_channels", None)
        stft = model.get("stft") or {}
        self.n_fft = int(architecture.get("n_fft", stft.get("n_fft", 400)))
        self.hop = int(stft.get("hop", 100 if self.n_fft == 400 else max(self.n_fft // 4, 1)))
        self.win = int(stft.get("win", self.n_fft))
        self.compress = 0.3
        self.sample_rate = int(model["sampling_rate"])
        with seed_model_init(config["seed"]):
            self.generator = build_model("semambapp", architecture or None, variant=model.get("variant")).to(device)
            if mrd_resolutions is None and mrd_channels is None:
                self.discriminator_mrd = MultiResolutionDiscriminator().to(device)
            else:
                self.discriminator_mrd = MultiResolutionDiscriminator(
                    mrd_resolutions,
                    32 if mrd_channels is None else int(mrd_channels),
                ).to(device)
            if cqt_hops is None and cqt_octaves is None and cqt_bins is None and cqt_channels is None:
                self.discriminator_band = MultiScaleSubbandCQTDiscriminator(self.sample_rate).to(device)
            else:
                self.discriminator_band = MultiScaleSubbandCQTDiscriminator(
                    self.sample_rate,
                    hops=cqt_hops,
                    octaves=cqt_octaves,
                    bins=cqt_bins,
                    channels=128 if cqt_channels is None else int(cqt_channels),
                ).to(device)

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        unwrap(self.discriminator_mrd).eval()
        unwrap(self.discriminator_band).eval()

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


def validation_model(config: dict[str, Any], device: torch.device) -> ValidationShadow:
    """Build the shadow modules used to score validation on a weight copy."""

    return ValidationShadow(config, device)

# --- Rank 0 copies weights into the shadow and restores the cursor and RNG. ---

class SEMambaValidator:
    """Copy live weights into one shadow task and return the shared summary."""

    def __init__(self, *, generator: Any, mrd: Any, cqt: Any, optim_g: torch.optim.Optimizer, scheduler_g: Any, grad_clip: float | None) -> None:
        self.generator = generator
        self.mrd = mrd
        self.cqt = cqt
        self.optim_g = optim_g
        self.scheduler_g = scheduler_g
        self.grad_clip = grad_clip
        self._shadow: Any = None

    def invalidate(self) -> None:
        self._shadow = None

    def evaluate(self, runtime: ProcessRuntime, data: Any, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        cursor = (int(data.epoch), int(data.update_index))
        rng = runtime.capture_rng()
        error: str | None = None
        summary: dict[str, Any] | None = None
        try:
            if runtime.rank == 0:
                try:
                    summary = self._rank0(runtime, data, trainer_epoch, component_id)
                except (TrainingError, RuntimeFailure, RuntimeError, ValueError, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
        finally:
            runtime.restore_rng(rng)
        errors = runtime.gather_objects(error)
        failure = next((item for item in errors if item), None)
        if failure:
            raise RuntimeFailure(f"validation failed: {failure}")
        gathered = runtime.gather_objects(summary)
        shared = gathered[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure(f"component {component_id}: rank 0 did not return a validation summary")
        moved = (int(data.epoch), int(data.update_index))
        if moved != cursor:
            raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return shared

    def _rank0(self, runtime: ProcessRuntime, data: Any, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        task = self._shadow_task(runtime)
        reports: list[dict[str, Any]] = []
        for batch in data.validation_epoch(trainer_epoch):
            if not isinstance(batch, Batch):
                raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
            batch = runtime.move(batch)
            report = task.validation_report({"clean": batch.targets, "noisy": batch.inputs})
            reports.append(report)
        if not reports:
            raise RuntimeFailure(f"component {component_id}: rank 0 validation produced no batches")
        summary = _aggregate_validation(reports)
        summary["cases"] = [
            {
                "lengths": report["lengths"],
                "losses": report["losses"],
                "pesq": report["pesq"],
                "sample_rate": report["sample_rate"],
                "si_sdr": report["si_sdr"],
                "stoi": report["stoi"],
            }
            for report in reports
        ]
        return summary

    def _shadow_task(self, runtime: ProcessRuntime) -> Any:
        if self._shadow is None:
            architecture = dict(self.generator.unwrapped().architecture)
            hop = int(architecture.pop("hop"))
            win = int(architecture.pop("win"))
            sampling_rate = int(architecture.pop("sampling_rate"))
            group = self.optim_g.param_groups[0]
            betas = group["betas"]
            self._shadow = validation_model(
                {
                    "grad_clip": self.grad_clip,
                    "loss": "official",
                    "model": {
                        "architecture": architecture,
                        "name": "semambapp",
                        "sampling_rate": sampling_rate,
                        "stft": {"hop": hop, "win": win},
                        "variant": "semambapp",
                    },
                    "optimizer": {
                        "betas": [float(betas[0]), float(betas[1])],
                        "lr": float(group["lr"]),
                        "name": "adamw",
                        "weight_decay": float(group["weight_decay"]),
                    },
                    "scheduler": {"gamma": float(self.scheduler_g.gamma), "name": "exponential"},
                    "seed": int(self.generator.unwrapped().seed),
                },
                runtime.device,
            )
        task = self._shadow
        task.generator.load_state_dict(self.generator.unwrapped().generator.state_dict(), strict=True)
        task.discriminator_mrd.load_state_dict(self.mrd.unwrapped().discriminator.state_dict(), strict=True)
        task.discriminator_band.load_state_dict(self.cqt.unwrapped().discriminator.state_dict(), strict=True)
        task.eval_mode()
        return task
