"""TF-Locoformer training for the DNS-Interspeech 2020 recipe.

The network normalizes mixture variance before the STFT. The objective is
ESPnet ``mr_l1_tfd`` with windows 256/512/768/1024, time-domain weight 0.5,
and variance normalization left off inside the loss. SI-SNR is monitored at
weight 0 and is not added to the training loss. The optimizer is AdamW.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from src.se import build_model
from src.training.core.distributed import ProcessInfo, unwrap, wrap_module
from src.training.core.rng import seed_model_init
from src.training.core.trainer import accumulation_scope
from src.training.errors import TrainingError
from src.training.se.initialization import apply_espnet_initialization
from src.training.se.optim import WarmupReduceLROnPlateau, build_optimizer, build_scheduler, clip_gradients, step_scheduler
from src.training.se.spectral import multi_resolution_l1, si_sdr_loss


_WINDOWS = (256, 512, 768, 1024)


class TFLocoformerTask:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        model = config["model"]
        self.variant = model.get("variant") or "dns"
        if self.variant != "dns":
            raise TrainingError(f"unsupported TF-Locoformer variant {self.variant!r}")
        if config["loss"] != "official":
            raise TrainingError("TFLocoformerTask implements loss official (mr_l1_tfd)")
        self.grad_clip = config.get("grad_clip", 5.0)
        architecture = dict(model.get("architecture") or {})
        architecture.update(model.get("stft") or {})
        with seed_model_init(config["seed"]):
            self.generator = build_model("tf_locoformer", architecture, variant=self.variant)
            if model.get("init"):
                apply_espnet_initialization(self.generator, model["init"])
        self.generator = self.generator.to(device)
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
        return self.generator_state_dict()

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
            raise TrainingError("checkpoint does not match this TF-Locoformer task")
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
            with accumulation_scope([self.generator], last=index == len(batches) - 1):
                clean = batch["clean"].to(self.device)
                noisy = batch["noisy"].to(self.device)
                try:
                    estimate = self.generator(noisy)
                except ValueError as exc:
                    raise TrainingError(str(exc)) from exc
                loss = multi_resolution_l1(clean, estimate, _WINDOWS, time_domain_weight=0.5, normalize_variance=False)
                _raise_if_nonfinite(loss, "mr_l1_tfd")
                (loss / accumulation).backward()
            total += float(loss.detach())
            with torch.no_grad():
                monitor += float(si_sdr_loss(estimate, clean).detach())
        clip_gradients(unwrap(self.generator), self.grad_clip)
        self.optim_g.step()
        if isinstance(self.sched_g, WarmupReduceLROnPlateau):
            self.sched_g.step_batch()
        count = len(batches)
        return {"generator": total / count, "mr_l1_tfd": total / count, "si_sdr": monitor / count}

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        clean = batch["clean"].to(self.device)
        noisy = batch["noisy"].to(self.device)
        with torch.no_grad():
            try:
                estimate = unwrap(self.generator)(noisy)
            except ValueError as exc:
                raise TrainingError(str(exc)) from exc
            value = float(
                multi_resolution_l1(
                    clean, estimate, _WINDOWS, time_domain_weight=0.5, normalize_variance=False
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
            "si_snr": _positive_si_sdr(clean, estimate),
        }


def _raise_if_nonfinite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss).all():
        raise TrainingError(f"non-finite {name} loss")


def _metric(value: float | None = None, *, status: str = "ok", reason: str | None = None) -> dict[str, Any]:
    if status == "ok" and value is not None and value == value and value not in (float("inf"), float("-inf")):
        return {"status": "ok", "value": value, "reason": None}
    return {"status": "na", "value": None, "reason": reason or "non-finite loss"}


def _positive_si_sdr(clean: torch.Tensor, estimate: torch.Tensor) -> dict[str, Any]:
    """SI-SDR in dB. The shared chunk ranker maximizes this value."""
    value = -float(si_sdr_loss(estimate, clean).detach())
    return _metric(value)
