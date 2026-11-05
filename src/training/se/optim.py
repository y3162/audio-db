"""Optimizers and epoch schedulers shared by speech-enhancement tasks.

The trainer calls ``step_scheduler`` once per finished epoch. A plateau
scheduler also needs the validation loss from that epoch. Gradient
accumulation does not change how often an epoch scheduler steps.
``WarmupReduceLROnPlateau.step_batch`` is the exception: the TF-Locoformer
task calls it after every optimizer update, and ``step_scheduler`` only
applies the plateau half at epoch end.
"""

from __future__ import annotations

from typing import Any

import torch

from src.training.errors import TrainingError


def build_optimizer(parameters, config: dict[str, Any], *, lr: float | None = None) -> torch.optim.Optimizer:
    optimizer = config["optimizer"]
    kwargs = {
        "lr": float(optimizer["lr"] if lr is None else lr),
        "betas": (float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        "weight_decay": float(optimizer["weight_decay"]),
    }
    name = optimizer["name"]
    if name == "adamw":
        return torch.optim.AdamW(parameters, amsgrad=bool(optimizer.get("amsgrad", False)), **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, amsgrad=bool(optimizer.get("amsgrad", False)), **kwargs)
    raise TrainingError(f"unsupported optimizer {name}")


class WarmupReduceLROnPlateau:
    """ESPnet ``WarmupReduceLROnPlateau`` for one optimizer.

    Each optimizer update calls :meth:`step_batch`. While
    ``step_num <= warmup_steps`` the learning rate is
    ``initial_lr * step_num / warmup_steps``. After warmup, a finished
    validation epoch calls :meth:`step_validation` and reduces the rate by
    ``factor`` when the metric has failed to improve for more than
    ``patience`` epochs. Batch steps after warmup do not change the rate.
    The first optimizer update still uses the constructor learning rate,
    because ESPnet steps this scheduler after ``optimizer.step``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 3,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if factor >= 1.0:
            raise TrainingError("scheduler.factor must be < 1")
        if mode not in ("min", "max") or threshold_mode not in ("rel", "abs"):
            raise TrainingError("unsupported warmup plateau mode")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.step_num = 0
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.cooldown = int(cooldown)
        self.cooldown_counter = 0
        self.mode = mode
        self.eps = float(eps)
        self.last_epoch = 0
        self.num_bad_epochs = 0
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [float(group["initial_lr"]) for group in optimizer.param_groups]
        self.min_lrs = [float(min_lr)] * len(optimizer.param_groups)
        self.mode_worse = float("inf") if mode == "min" else float("-inf")
        self.best = self.mode_worse

    def step_batch(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base * (self.step_num / self.warmup_steps)

    def step_validation(self, metric: float) -> None:
        if self.step_num <= self.warmup_steps:
            return
        current = float(metric)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def state_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.mode_worse = float("inf") if self.mode == "min" else float("-inf")

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min" and self.threshold_mode == "rel":
            return current < best * (1.0 - self.threshold)
        if self.mode == "min":
            return current < best - self.threshold
        if self.threshold_mode == "rel":
            return current > best * (1.0 + self.threshold)
        return current > best + self.threshold

    def _reduce_lr(self) -> None:
        for group, floor in zip(self.optimizer.param_groups, self.min_lrs):
            old = float(group["lr"])
            new = max(old * self.factor, floor)
            if old - new > self.eps:
                group["lr"] = new


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    schedule = config["scheduler"]
    name = schedule["name"]
    if name == "none":
        return None
    if name == "warmup_reduce_on_plateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            warmup_steps=int(schedule["warmup_steps"]),
            mode=schedule["mode"],
            factor=float(schedule["factor"]),
            patience=int(schedule["patience"]),
            threshold=float(schedule["threshold"]),
            threshold_mode=schedule["threshold_mode"],
            cooldown=int(schedule["cooldown"]),
            min_lr=float(schedule["min_lr"]),
            eps=float(schedule["eps"]),
        )
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(schedule["gamma"]))
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(schedule["step_size"]), gamma=float(schedule["gamma"])
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(schedule["factor"]),
            patience=int(schedule["patience"]),
            threshold=1e-4,
            threshold_mode="rel",
        )
    raise TrainingError(f"unsupported scheduler {name}")


def step_scheduler(scheduler, validation: dict[str, Any] | None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, WarmupReduceLROnPlateau):
        metric = _validation_loss(validation)
        if metric is None:
            return
        scheduler.step_validation(metric)
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        metric = _validation_loss(validation)
        if metric is None:
            return
        scheduler.step(metric)
        return
    scheduler.step()


def clip_gradients(module: torch.nn.Module, limit: float | None) -> None:
    if limit is None:
        return
    torch.nn.utils.clip_grad_norm_(module.parameters(), float(limit))


def _validation_loss(validation: dict[str, Any] | None) -> float | None:
    if not validation:
        return None
    losses = validation.get("losses") or {}
    generator = losses.get("generator") or {}
    value = generator.get("value")
    if generator.get("status") != "ok" or value is None:
        return None
    return float(value)
