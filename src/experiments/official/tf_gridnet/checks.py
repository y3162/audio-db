"""Constructor checks for the TF-GridNet trainer.

These look at the objects the YAML named. They do not train, validate, or save.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.runtime import PreparedModel, require_prepared
from src.experiments.training.objectives import coerce_spectral_objective


def require_gridnet_setup(
    model: PreparedModel,
    objective: torch.nn.Module,
    data: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
) -> tuple[torch.nn.Module, str]:
    """Refuse a loop that cannot run the MS-SNSD boundary. Return the objective and optimizer id."""

    require_prepared(model)
    objective = coerce_spectral_objective(objective, label="MS-SNSD")
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        raise BuildError("scheduler must be ReduceLROnPlateau")
    if scheduler.mode != "min":
        raise BuildError("MS-SNSD scheduler mode must be min")
    optimizer_id = getattr(optimizer, "_experiment_component_id", None)
    if not isinstance(optimizer_id, str):
        raise BuildError("optimizer has no component id")
    return objective, optimizer_id


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
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
