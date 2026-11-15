"""Constructor checks for the TF-Locoformer trainer."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.optim import WarmupReduceLROnPlateau
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.runtime import PreparedModel, require_prepared
from src.experiments.training.objectives import coerce_spectral_objective


def require_locoformer_setup(
    model: PreparedModel,
    objective: torch.nn.Module,
    data: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupReduceLROnPlateau,
    validation_profile: str,
) -> tuple[torch.nn.Module, str]:
    """Refuse a loop that cannot run the DNS warmup schedule. Return the objective and optimizer id."""

    require_prepared(model)
    objective = coerce_spectral_objective(objective, label="DNS")
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    if not isinstance(scheduler, WarmupReduceLROnPlateau):
        raise BuildError("scheduler must be WarmupReduceLROnPlateau")
    if scheduler.mode != "min":
        raise BuildError("DNS scheduler mode must be min")
    if not isinstance(validation_profile, str) or not validation_profile:
        raise BuildError("validation_profile must be a non-empty string")
    optimizer_id = getattr(optimizer, "_experiment_component_id", None)
    if not isinstance(optimizer_id, str):
        raise BuildError("optimizer has no component id")
    return objective, optimizer_id


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def optional_patience(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError("early_stopping_patience must be an integer >= 0")
    return value


def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)


def require_group_size(data: Any, accumulation: int, component_id: str) -> None:
    group_size = getattr(data, "group_size", None)
    if group_size is None:
        return
    if group_size != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} does not match accumulation {accumulation}"
        )
