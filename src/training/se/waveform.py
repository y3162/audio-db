"""Waveform L1 training for SE models whose forward is already in ``src.se``.

This is not an official training reproduction. The loss is time-domain L1,
there is no discriminator, and the official multi-stage or GAN objectives are
not implemented. ``enhance()`` is not called.
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

_NAMES = {"tf_gridnet"}


class WaveformTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        if model["name"] not in _NAMES:
            raise TrainingError(f"waveform L1 training is not implemented for {model['name']}")
        if config["loss"] != "waveform_l1":
            raise TrainingError("this task only implements loss waveform_l1")
        with seed_model_init(config["seed"]):
            self.generator = build_model(
                model["name"],
                model.get("architecture") or None,
                variant=model.get("variant"),
            ).to(device)
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

    def distribute(self, process: ProcessInfo) -> None:
        self.generator = wrap_module(self.generator, process)

    def train_mode(self) -> None:
        self.generator.train()

    def eval_mode(self) -> None:
        unwrap(self.generator).eval()

    def learning_rates(self) -> dict[str, float]:
        return {"generator": float(self.optim_g.param_groups[0]["lr"])}

    def step_schedulers(self) -> None:
        self.sched_g.step()

    def generator_state_dict(self) -> dict[str, torch.Tensor]:
        return unwrap(self.generator).state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss": "waveform_l1",
            "generator": unwrap(self.generator).state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "sched_g": self.sched_g.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("loss") != "waveform_l1":
            raise TrainingError("checkpoint loss does not match waveform_l1")
        if isinstance(self.generator, DistributedDataParallel):
            raise TrainingError("load a checkpoint before wrapping DistributedDataParallel")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.optim_g.load_state_dict(state["optim_g"])
        self.sched_g.load_state_dict(state["sched_g"])

    def train_update(self, batches: list[dict[str, Any]], accumulation: int) -> dict[str, float]:
        self.train_mode()
        self.optim_g.zero_grad(set_to_none=True)
        total = 0.0
        for index, batch in enumerate(batches):
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            estimate = _predict(self.config["model"], self.generator, noisy)
            if estimate.shape != clean.shape:
                raise TrainingError(
                    f"estimate shape {tuple(estimate.shape)} does not match the teacher {tuple(clean.shape)}"
                )
            loss = F.l1_loss(clean, estimate) / accumulation
            backward(loss, [self.generator], last=index == len(batches) - 1)
            total += float(loss.detach()) * accumulation
        self.optim_g.step()
        return {"time": total / len(batches), "generator": total / len(batches)}

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        with torch.no_grad():
            estimate = _predict(self.config["model"], unwrap(self.generator), noisy)
        value = float(F.l1_loss(clean, estimate).detach())
        metric = {"status": "ok", "value": value, "reason": None}
        missing = {"status": "na", "value": None, "reason": "waveform_l1 has no discriminator or reference metric in this task"}
        return {
            "losses": {"generator": metric, "time": metric, "discriminator": missing},
            "pesq": {"status": "na", "value": None, "reason": "not computed for waveform_l1", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "not computed for waveform_l1", "count": 0},
        }


def _predict(model: dict[str, Any], module: torch.nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    name = model["name"]
    if name == "tf_gridnet":
        variant = model.get("variant") or "urgent24"
        if variant == "ms-snsd":
            return module(noisy)
        if variant in (None, "urgent24"):
            return _gridnet_v3_waveform(module, noisy, model.get("stft") or {})
        raise TrainingError(f"unsupported TF-GridNet variant {variant!r}")
    raise TrainingError(f"no waveform prediction for {name}")


def _gridnet_v3_waveform(module: torch.nn.Module, noisy: torch.Tensor, architecture: dict[str, Any]) -> torch.Tensor:
    n_fft = int(architecture.get("n_fft", 256))
    hop = int(architecture.get("hop", n_fft // 2))
    window = torch.hann_window(n_fft, device=noisy.device, dtype=noisy.dtype)
    spectrum = torch.stft(noisy, n_fft, hop, n_fft, window, return_complex=True).transpose(1, 2)
    enhanced = module(spectrum).transpose(1, 2)
    return torch.istft(enhanced, n_fft, hop, n_fft, window, length=noisy.shape[-1])

