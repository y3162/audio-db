"""Independent MP-SENet learning task for numeric comparison.

This is not the production updater. ``train_update`` keeps its own
discriminator-then-generator loop, including the manual gradient average that
skips missing gradients. Production validation calls ``evaluator.report_batch``
directly and does not construct this task. ``validation_report`` delegates to
that same report so the batch formula is not copied.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.experiments.calc.distributed import ProcessInfo, unwrap, wrap_module
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.scope import accumulation_scope
from src.experiments.calc.errors import TrainingError
from src.experiments.official.mp_senet.model import MetricDiscriminator
from src.experiments.official.mp_senet.dataset import (
    official_center_istft_length,
    require_discriminator_resolution as _require_discriminator_resolution,
)
from src.experiments.calc.losses import discriminator_loss, generator_losses, metric_targets
from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft


class MPSENetTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.loss_name = config["loss"]
        model = config["model"]
        self.spec = model
        if self.loss_name not in ("official", "reconstruction"):
            raise TrainingError(f"unknown loss {self.loss_name!r}")
        heads = int(model.get("n_heads", 4))
        if heads < 1 or int(model["dense_channel"]) % heads != 0:
            raise TrainingError(f"model.dense_channel must be divisible by the {heads} attention heads")
        with seed_model_init(config["seed"]):
            self.generator = build_model("mp_senet", model).to(device)
            if self.loss_name != "official":
                self.discriminator = None
            elif "disc_dim" in model:
                self.discriminator = MetricDiscriminator(dim=int(model["disc_dim"])).to(device)
            else:
                self.discriminator = MetricDiscriminator().to(device)
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

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        del validation
        self.sched_g.step()
        if self.sched_d is not None:
            self.sched_d.step()

    def export_for_inference(self) -> dict[str, Any]:
        return {"generator": self.generator_state_dict()}

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

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Official or research report. The formula lives in ``evaluator``.

        This task does not call the production updater. The report function
        does not build an optimizer.
        """

        from src.experiments.official.mp_senet.evaluator import report_batch

        discriminator = None if self.discriminator is None else unwrap(self.discriminator)
        return report_batch(
            generator=unwrap(self.generator),
            discriminator=discriminator,
            spec=self.spec,
            device=self.device,
            batch=batch,
            official=self.loss_name == "official",
        )

    def _reconstruction(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(batches):
            with accumulation_scope([self.generator], last=index == len(batches) - 1):
                clean, noisy = self._waveforms(batch)
                parts = self._generate(clean, noisy, generator=self.generator)
                loss = F.l1_loss(clean, parts["audio"]) / accumulation
                loss.backward()
            total += float(loss.detach()) * accumulation
        self.optim_g.step()
        mean = total / len(batches)
        return {"generator": mean, "time": mean}

    def _official(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        if self.discriminator is None or self.optim_d is None:
            raise TrainingError("official loss requires a discriminator")
        raw_discriminator = unwrap(self.discriminator)
        # One generator graph per micro-batch, matching official train.py.
        # The discriminator sees a detached magnitude. A second forward would
        # be a different CUDA evaluation and would feed the generator loss a
        # waveform the discriminator did not score.
        self.optim_d.zero_grad(set_to_none=True)
        self.optim_g.zero_grad(set_to_none=True)
        discriminator_total = 0.0
        stored: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        for index, batch in enumerate(batches):
            clean, noisy = self._waveforms(batch)
            last = index == len(batches) - 1
            # The generator graphs are retained until after the discriminator step.
            # Reduce explicitly after all delayed backward calls: multiple DDP
            # forwards before backward cannot safely select the last reduction.
            with accumulation_scope([self.generator], last=False):
                parts = self._generate(clean, noisy, generator=self.generator)
            stored.append((clean, parts))
            ones = torch.ones(clean.shape[0], device=self.device)
            _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
            with accumulation_scope([self.discriminator], last=last):
                real = self.discriminator(parts["clean_mag"], parts["clean_mag"])
                fake = self.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
                targets = metric_targets(clean, parts["audio"], int(self.spec["sampling_rate"]))
                loss = discriminator_loss(real, fake, ones, targets) / accumulation
                loss.backward()
            discriminator_total += float(loss.detach()) * accumulation
        self.optim_d.step()

        frozen = [parameter for parameter in raw_discriminator.parameters() if parameter.requires_grad]
        for parameter in frozen:
            parameter.requires_grad_(False)
        component_totals = {
            name: 0.0 for name in ("total", "magnitude", "phase", "complex", "consistency", "metric", "time")
        }
        try:
            for clean, parts in stored:
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
                loss.backward()
                for name, value in losses.items():
                    component_totals[name] += float(value.detach())
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        if isinstance(self.generator, DistributedDataParallel):
            import torch.distributed as dist
            for parameter in unwrap(self.generator).parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(dist.get_world_size())
        self.optim_g.step()
        count = len(batches)
        return {
            "generator": component_totals["total"] / count,
            "discriminator": discriminator_total / count,
            "magnitude": component_totals["magnitude"] / count,
            "phase": component_totals["phase"] / count,
            "complex": component_totals["complex"] / count,
            "consistency": component_totals["consistency"] / count,
            "metric": component_totals["metric"] / count,
            "time": component_totals["time"] / count,
        }

    def _waveforms(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise TrainingError(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        return clean, noisy

    def _generate(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        *,
        generator: torch.nn.Module,
        allow_official_istft_length: bool = False,
    ) -> dict[str, torch.Tensor]:
        spec = self.spec
        hop = int(spec["hop_size"])
        clean_mag, clean_phase, clean_complex = mag_pha_stft(
            clean, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
        )
        noisy_mag, noisy_phase, _noisy_complex = mag_pha_stft(
            noisy, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
        )
        magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
        audio = mag_pha_istft(magnitude, phase, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"])
        waveform_length = int(clean.shape[-1])
        audio_length = int(audio.shape[-1])
        if audio_length != waveform_length:
            official_length = official_center_istft_length(waveform_length, hop)
            if not (allow_official_istft_length and audio_length == official_length):
                raise TrainingError(
                    f"iSTFT length {audio_length} does not match the waveform length {waveform_length}"
                )
        try:
            magnitude_hat, phase_hat, recomputed = mag_pha_stft(
                audio, spec["n_fft"], hop, spec["win_size"], spec["compress_factor"]
            )
        except (RuntimeError, TrainingError):
            if audio_length == waveform_length or not allow_official_istft_length:
                raise
            magnitude_hat = None
        else:
            magnitude_hat = (magnitude_hat, phase_hat, recomputed)
        result = {
            "clean_mag": clean_mag,
            "clean_phase": clean_phase,
            "clean_complex": clean_complex,
            "mag": magnitude,
            "phase": phase,
            "complex": complex_spectrum,
            "audio": audio,
        }
        if magnitude_hat is not None:
            hat, phase_hat, recomputed = magnitude_hat
            result["mag_hat"] = hat
            result["phase_hat"] = phase_hat
            result["recomputed_complex"] = recomputed
        return result
