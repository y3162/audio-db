"""DeepFilterNet3 build checks. The AdamW base values stay fixed."""

from __future__ import annotations

import torch

from src.experiments.calc.batch_schedule import scheduled_batch_size
from src.experiments.errors import BuildError, RuntimeFailure


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be an integer >= 1")
    return value


def non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value


def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)


def require_adamw(optimizer: torch.optim.Optimizer) -> str:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError("DeepFilterNet3 optimizer must be AdamW")
    group = optimizer.param_groups[0]
    if float(group["lr"]) != 0.001:
        raise BuildError("DeepFilterNet3 learning rate must be 0.001")
    betas = tuple(group["betas"])
    if betas != (0.9, 0.999):
        raise BuildError("DeepFilterNet3 betas must be (0.9, 0.999)")
    if float(group["weight_decay"]) != 1e-12:
        raise BuildError("DeepFilterNet3 weight decay must be 1e-12")
    if group.get("amsgrad") is not True:
        raise BuildError("DeepFilterNet3 AdamW requires amsgrad")
    component_id = getattr(optimizer, "_experiment_component_id", None)
    if not isinstance(component_id, str):
        raise BuildError("optimizer has no component id")
    return component_id


def parse_schedule(raw: list[list[int]] | None) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise BuildError("batch_schedule must be a non-empty list of [epoch, batch_size]")
    schedule = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise BuildError("batch_schedule entries must be [epoch, batch_size]")
        schedule.append(
            (non_negative_int("batch_schedule epoch", item[0]), positive_int("batch_schedule batch_size", item[1]))
        )
    if schedule[0][0] != 0:
        raise BuildError("batch_schedule must start at epoch 0")
    return schedule


def apply_batch_schedule(data: object, epoch: int, schedule: list[tuple[int, int]] | None, limit: int) -> None:
    if schedule is None:
        return
    size = scheduled_batch_size(epoch, limit, schedule)
    current = getattr(getattr(data, "batcher", None), "micro_batch", None)
    if current == size:
        return
    rebind = getattr(data, "rebind_micro_batch", None)
    if not callable(rebind):
        raise RuntimeFailure("data cannot rebind a DeepFilterNet3 micro-batch")
    rebind(size)


def require_group_size(data: object, accumulation: int, component_id: str) -> None:
    group_size = getattr(data, "group_size", None)
    if group_size is not None and group_size != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} "
            f"does not match accumulation {accumulation}"
        )
