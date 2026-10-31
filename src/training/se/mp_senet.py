"""MP-SENet generator and metric-discriminator updates.

The generator module is the inference implementation from ``src.se``. The
discriminator and the losses stay in this package. ``enhance()`` is not called
during training. ``loss: reconstruction`` is a time L1 objective for foundation
tests. It is not the official MP-SENet objective.
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
from src.training.se.discriminator import MetricDiscriminator
from src.training.se.losses import discriminator_loss, generator_losses, metric_targets
from src.training.se.stft import mag_pha_istft, mag_pha_stft


class MPSENetTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.loss_name = config["loss"]
        model = config["model"]
        self.spec = model
        if self.loss_name not in ("official", "reconstruction"):
            raise TrainingError(f"unknown loss {self.loss_name!r}")
        if int(model["dense_channel"]) % 4 != 0:
            raise TrainingError("model.dense_channel must be divisible by the 4 attention heads")
        with seed_model_init(config["seed"]):
            self.generator = build_model("mp_senet", model).to(device)
            self.discriminator = MetricDiscriminator().to(device) if self.loss_name == "official" else None
        optimizer = config["optimizer"]
        betas = (float(optimizer["betas"][0]), float(optimizer["betas"][1]))
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=float(optimizer["lr"]),
            betas=betas,
            weight_decay=float(optimizer["weight_decay"]),
        )
        self.sched_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=float(config["scheduler"]["gamma"])
        )
        self.optim_d = None
        self.sched_d = None
        if self.discriminator is not None:
            self.optim_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=float(optimizer["lr"]),
                betas=betas,
                weight_decay=float(optimizer["weight_decay"]),
            )
            self.sched_d = torch.optim.lr_scheduler.ExponentialLR(
                self.optim_d, gamma=float(config["scheduler"]["gamma"])
            )

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)
        if self.discriminator is not None:
            self.discriminator = wrap_module(self.discriminator, process)

    def train_mode(self) -> None:
        self.generator.train()
        if self.discriminator is not None:
            self.discriminator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()
        if self.discriminator is not None:
            unwrap(self.discriminator).eval()

    def learning_rates(self) -> dict[str, float]:
        rates = {"generator": float(self.optim_g.param_groups[0]["lr"])}
        if self.optim_d is not None:
            rates["discriminator"] = float(self.optim_d.param_groups[0]["lr"])
        return rates

    def step_schedulers(self) -> None:
        self.sched_g.step()
        if self.sched_d is not None:
            self.sched_d.step()

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss_name,
            "generator": unwrap(self.generator).state_dict(),
            "discriminator": None
            if self.discriminator is None
            else unwrap(self.discriminator).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": None if self.optim_d is None else self.optim_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": None if self.sched_d is None else self.sched_d.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["loss"] != self.loss_name:
            raise TrainingError("checkpoint loss does not match the task")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        self.sched_g.load_state_dict(state["sched_g"])
        if self.discriminator is None:
            if state["discriminator"] is not None:
                raise TrainingError("checkpoint has a discriminator but this run does not")
            return
        self.discriminator.load_state_dict(state["discriminator"], strict=True)
        self.optim_d.load_state_dict(state["optim_d"])
        self.sched_d.load_state_dict(state["sched_d"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        if self.loss_name == "reconstruction":
            return self._reconstruction(batches, accumulation)
        return self._official(batches, accumulation)

    def validation_metrics(self, batch: dict[str, Any]) -> dict[str, float]:
        clean, noisy = self._waveforms(batch)
        with torch.no_grad():
            parts = self._generate(clean, noisy, generator=unwrap(self.generator))
        metrics = {
            "time": float(F.l1_loss(clean, parts["audio"]).detach()),
            "magnitude": float(F.mse_loss(parts["clean_mag"], parts["mag"]).detach()),
        }
        if self.loss_name == "official":
            targets = metric_targets(clean, parts["audio"], int(self.spec["sampling_rate"]))
            if targets is not None:
                metrics["pesq_normalized"] = float(targets.mean())
        return metrics

    def _reconstruction(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(batches):
            clean, noisy = self._waveforms(batch)
            parts = self._generate(clean, noisy, generator=self.generator)
            loss = F.l1_loss(clean, parts["audio"]) / accumulation
            backward(loss, [self.generator], last=index == len(batches) - 1)
            total += float(loss.detach()) * accumulation
        self.optim_g.step()
        return {"time": total / len(batches)}

    def _official(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        if self.discriminator is None or self.optim_d is None:
            raise TrainingError("official loss requires a discriminator")
        raw_discriminator = unwrap(self.discriminator)
        self.optim_d.zero_grad(set_to_none=True)
        discriminator_total = 0.0
        for index, batch in enumerate(batches):
            clean, noisy = self._waveforms(batch)
            with torch.no_grad():
                parts = self._generate(clean, noisy, generator=unwrap(self.generator))
            ones = torch.ones(clean.shape[0], device=self.device)
            _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
            real = self.discriminator(parts["clean_mag"], parts["clean_mag"])
            fake = self.discriminator(parts["clean_mag"], parts["mag_hat"])
            targets = metric_targets(clean, parts["audio"], int(self.spec["sampling_rate"]))
            loss = discriminator_loss(real, fake, ones, targets) / accumulation
            backward(loss, [self.discriminator], last=index == len(batches) - 1)
            discriminator_total += float(loss.detach()) * accumulation
        self.optim_d.step()

        frozen = [parameter for parameter in raw_discriminator.parameters() if parameter.requires_grad]
        for parameter in frozen:
            parameter.requires_grad_(False)
        self.optim_g.zero_grad(set_to_none=True)
        generator_total = 0.0
        try:
            for index, batch in enumerate(batches):
                clean, noisy = self._waveforms(batch)
                parts = self._generate(clean, noisy, generator=self.generator)
                ones = torch.ones(clean.shape[0], device=self.device)
                # The unwrapped discriminator avoids a second DDP all-reduce.
                # Its parameters are frozen, so the metric gradient reaches the generator only.
                metric = raw_discriminator(parts["clean_mag"], parts["mag_hat"])
                losses = generator_losses(
                    parts["clean_mag"],
                    parts["mag"],
                    parts["clean_phase"],
                    parts["phase"],
                    parts["clean_complex"],
                    parts["complex"],
                    parts["recomputed_complex"],
                    clean,
                    parts["audio"],
                    metric,
                    ones,
                )
                loss = losses["total"] / accumulation
                backward(loss, [self.generator], last=index == len(batches) - 1)
                generator_total += float(loss.detach()) * accumulation
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        self.optim_g.step()
        return {
            "generator": generator_total / len(batches),
            "discriminator": discriminator_total / len(batches),
        }

    def _waveforms(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise TrainingError(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        return clean, noisy

    def _generate(self, clean: torch.Tensor, noisy: torch.Tensor, *, generator: torch.nn.Module) -> dict[str, torch.Tensor]:
        spec = self.spec
        clean_mag, clean_phase, clean_complex = mag_pha_stft(
            clean, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
        )
        noisy_mag, noisy_phase, _noisy_complex = mag_pha_stft(
            noisy, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
        )
        magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
        audio = mag_pha_istft(
            magnitude, phase, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
        )
        if audio.shape[-1] != clean.shape[-1]:
            raise TrainingError(
                f"iSTFT length {audio.shape[-1]} does not match the waveform length {clean.shape[-1]}"
            )
        magnitude_hat, phase_hat, recomputed = mag_pha_stft(
            audio, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
        )
        return {
            "clean_mag": clean_mag,
            "clean_phase": clean_phase,
            "clean_complex": clean_complex,
            "mag": magnitude,
            "phase": phase,
            "complex": complex_spectrum,
            "audio": audio,
            "mag_hat": magnitude_hat,
            "phase_hat": phase_hat,
            "recomputed_complex": recomputed,
        }


def _conv_out(size: int) -> int:
    return (size + 2 - 4) // 2 + 1


def _require_discriminator_resolution(n_freq: int, n_time: int) -> None:
    """InstanceNorm rejects a 1x1 map while the discriminator is training.

    Four stride-2 convolutions shrink the spectrogram before that norm. The
    official 400-point STFT and 2-second segments stay large enough. A smaller
    setup is rejected instead of switching the norm to evaluation statistics.
    """
    freq, time = int(n_freq), int(n_time)
    for _ in range(4):
        freq, time = _conv_out(freq), _conv_out(time)
        if freq < 1 or time < 1 or freq * time <= 1:
            raise TrainingError(
                "official loss needs a larger spectrogram so the metric discriminator "
                "InstanceNorm sees more than one spatial location. Increase segment_size or n_fft."
            )
