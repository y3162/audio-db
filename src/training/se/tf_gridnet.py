"""TF-GridNet training with the ESPnet multi-resolution L1 objective.

``urgent24`` is TFGridNetV3 on a complex STFT (n_fft 256 and hop 128 at 8 kHz,
scaled with the sample rate) and variance normalization. ``ms-snsd`` is the
waveform TFGridNet with its own Hann STFT. Both minimize ``mr_l1_tfd``.
SI-SNR is monitored at weight 0 in the official recipes and is not added to
the training loss. ``loss: waveform_l1`` stays on ``WaveformTask``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import backward
from src.training.errors import TrainingError
from src.training.se.optim import build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.spectral import multi_resolution_l1, si_sdr_loss


_URGENT_WINDOWS = (256, 512, 768, 1024)


class TFGridNetTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        self.variant = model.get("variant") or "urgent24"
        if self.variant not in ("urgent24", "ms-snsd"):
            raise TrainingError(f"unsupported TF-GridNet variant {self.variant!r}")
        if config["loss"] != "official":
            raise TrainingError("TFGridNetTask implements loss official (mr_l1_tfd)")
        self.normalize_variance = self.variant == "urgent24"
        self.grad_clip = config.get("grad_clip", 1.0 if self.variant == "urgent24" else None)
        with seed_model_init(config["seed"]):
            self.generator = build_model(
                "tf_gridnet",
                model.get("architecture") or None,
                variant=self.variant,
            ).to(device)
        self.optim_g = build_optimizer(self.generator.parameters(), config)
        self.sched_g = build_scheduler(self.optim_g, config)

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)

    def train_mode(self) -> None:
        self.generator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()

    def learning_rates(self) -> dict[str, float]:
        return {"generator": float(self.optim_g.param_groups[0]["lr"])}

    def step_schedulers(self, validation: dict[str, Any] | None = None) -> None:
        step_scheduler(self.sched_g, validation)

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        return {"separator." + name: value for name, value in self.generator_state_dict().items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": "official",
            "variant": self.variant,
            "generator": self.generator_state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "sched_g": None if self.sched_g is None else self.sched_g.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "official" or state.get("variant") != self.variant:
            raise TrainingError("checkpoint does not match this TF-GridNet task")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        if self.sched_g is not None and state.get("sched_g") is not None:
            self.sched_g.load_state_dict(state["sched_g"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        monitor = 0.0
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            estimate = self._enhance(self.generator, noisy)
            loss = multi_resolution_l1(
                clean,
                estimate,
                _URGENT_WINDOWS,
                normalize_variance=self.normalize_variance,
            )
            _raise_if_nonfinite(loss, "mr_l1_tfd")
            backward(loss / accumulation, [self.generator], last=index == len(batches) - 1)
            total += float(loss.detach())
            with torch.no_grad():
                monitor += float(si_sdr_loss(estimate, clean).detach())
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        count = len(batches)
        return {"generator": total / count, "mr_l1_tfd": total / count, "si_sdr": monitor / count}

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        with torch.no_grad():
            estimate = self._enhance(unwrap(self.generator), noisy)
            value = float(
                multi_resolution_l1(
                    clean, estimate, _URGENT_WINDOWS, normalize_variance=self.normalize_variance
                ).detach()
            )
        metric = _metric(value)
        return {
            "losses": {
                "generator": metric,
                "mr_l1_tfd": metric,
                "discriminator": _metric(status="na", reason="mr_l1_tfd has no discriminator"),
            },
            "pesq": {"status": "na", "value": None, "reason": "not computed for mr_l1_tfd", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "not computed for mr_l1_tfd", "count": 0},
        }

    def _enhance(self, module: torch.nn.Module, noisy: torch.Tensor) -> torch.Tensor:
        if self.variant == "ms-snsd":
            return module(noisy)
        return _urgent_waveform(module, noisy, int(self.config["model"]["sampling_rate"]))


def _urgent_waveform(module: torch.nn.Module, noisy: torch.Tensor, sample_rate: int) -> torch.Tensor:
    n_fft = 256 * sample_rate // 8000
    hop = 128 * sample_rate // 8000
    if n_fft < 2 or noisy.shape[-1] <= n_fft // 2:
        raise TrainingError(f"urgent24 STFT needs more than {n_fft // 2} samples at {sample_rate} Hz")
    scale = noisy.std(dim=-1, keepdim=True).clamp_min(1e-8)
    window = torch.hann_window(n_fft, device=noisy.device, dtype=noisy.dtype)
    spectrum = torch.stft(noisy / scale, n_fft, hop, n_fft, window, return_complex=True).transpose(1, 2)
    enhanced = module(spectrum).transpose(1, 2)
    return torch.istft(enhanced, n_fft, hop, n_fft, window, length=noisy.shape[-1]) * scale


def _raise_if_nonfinite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and (value is None or value != value or value in (float("inf"), float("-inf"))):
        return {"status": "na", "value": None, "reason": "non-finite loss"}
    return {"status": status, "value": value, "reason": reason}
