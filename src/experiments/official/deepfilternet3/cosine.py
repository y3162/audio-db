"""Per-step cosine table. The trainer applies it before the optimizer step."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.deepfilternet3.reference import deepfilter_cosine_schedules


class CosineTable:
    """Own the rebuilt learning-rate and weight-decay tables for one epoch length."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        schedule_epochs: int,
        lr_min: float,
        lr_warmup: float,
        warmup_epochs: int,
        weight_decay_end: float,
        steps_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self.schedule_epochs = schedule_epochs
        self.lr_min = lr_min
        self.lr_warmup = lr_warmup
        self.warmup_epochs = warmup_epochs
        self.weight_decay_end = weight_decay_end
        self.steps = steps_per_epoch
        self.rates, self.decays = self.tables(steps_per_epoch)

    def tables(self, steps_per_epoch: int) -> tuple[list[float], list[float]]:
        config = {
            "optimizer": {"lr": 0.001, "weight_decay": 1e-12},
            "scheduler": {
                "epochs": self.schedule_epochs,
                "lr_min": self.lr_min,
                "lr_warmup": self.lr_warmup,
                "name": "deepfilter_cosine",
                "steps_per_epoch": steps_per_epoch,
                "warmup_epochs": self.warmup_epochs,
                "weight_decay_end": self.weight_decay_end,
            },
        }
        try:
            return deepfilter_cosine_schedules(config)
        except TrainingError as exc:
            raise BuildError(str(exc)) from exc

    def retarget(self, steps_per_epoch: int) -> None:
        if steps_per_epoch == self.steps and self.rates:
            return
        self.rates, self.decays = self.tables(steps_per_epoch)
        self.steps = steps_per_epoch

    def install(self, steps_per_epoch: int, rates: list[float], decays: list[float]) -> None:
        self.steps = steps_per_epoch
        self.rates = rates
        self.decays = decays

    def apply(self, index: int) -> tuple[float, float]:
        """Copy one table row onto the optimizer before it steps."""

        if index < 0 or index >= len(self.rates):
            raise RuntimeFailure(
                f"deepfilter cosine step {index} is outside the schedule of {len(self.rates)}"
            )
        learning_rate = self.rates[index]
        decay = self.decays[index]
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
            if float(group["weight_decay"]) > 0.0:
                group["weight_decay"] = decay
        return learning_rate, decay


def open_epoch(
    data: Any,
    cosine: CosineTable,
    progress: Any,
    *,
    epoch: int,
    index: int,
    accumulation: int,
    component_id: str,
) -> tuple[Any, int]:
    """Open the epoch view and rebuild the cosine table for its optimizer-step count."""

    from src.experiments.trainers import _close_epoch_view

    view = data.train_epoch(epoch, start_update=index)
    try:
        if view.updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if view.updates % accumulation != 0:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} has {view.updates} micro-batches, "
                f"which is not divisible by accumulation {accumulation}"
            )
        steps = view.updates // accumulation
        cosine.retarget(steps)
        progress.schedule_steps = cosine.steps
        if index > view.updates or index % accumulation != 0:
            raise RuntimeFailure(f"component {component_id}: update {index} is not an optimizer boundary")
        if index == view.updates:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
            )
    except Exception:
        _close_epoch_view(view)
        raise
    return view, steps
