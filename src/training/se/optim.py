"""Optimizers and epoch schedulers shared by speech-enhancement tasks.

The trainer calls ``step_scheduler`` once per finished epoch. A plateau
scheduler also needs the validation loss from that epoch. Gradient
accumulation does not change how often an epoch scheduler steps.
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


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    schedule = config["scheduler"]
    name = schedule["name"]
    if name == "none":
        return None
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
        )
    raise TrainingError(f"unsupported scheduler {name}")


def step_scheduler(scheduler, validation: dict[str, Any] | None) -> None:
    if scheduler is None:
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
