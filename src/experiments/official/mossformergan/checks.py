"""MossFormerGAN build checks. The two AdamW learning rates stay distinct."""

from __future__ import annotations

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mossformergan.reference import OFFICIAL_ADAMW_WEIGHT_DECAY

OFFICIAL_FAITHFUL = "official_faithful"
LEAKAGE_SAFE = "leakage_safe"
CLAIMS = frozenset({OFFICIAL_FAITHFUL, LEAKAGE_SAFE, "research"})
OFFICIAL_LEARNING_RATE = 0.0005
OFFICIAL_DISCRIMINATOR_LEARNING_RATE = 0.001
OFFICIAL_BETAS = (0.9, 0.999)
OFFICIAL_CLIP = 10.0
COUNTER_KEYS = (
    "micro_batches_processed",
    "micro_batches_succeeded",
    "micro_batches_failed",
    "optimizer_updates",
    "official_step",
)


def empty_counters() -> dict[str, int]:
    return {key: 0 for key in COUNTER_KEYS}


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)


def require_adamw(optimizer: torch.optim.Optimizer, name: str, *, lr: float) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError(f"{name} must be AdamW")
    if len(optimizer.param_groups) != 1:
        raise BuildError(f"{name} must have one parameter group")
    group = optimizer.param_groups[0]
    if float(group["lr"]) != float(lr):
        raise BuildError(f"{name} learning rate must be {lr}")
    betas = tuple(float(item) for item in group["betas"])
    if betas != OFFICIAL_BETAS:
        raise BuildError(f"{name} betas must be {OFFICIAL_BETAS}")
    if float(group["weight_decay"]) != float(OFFICIAL_ADAMW_WEIGHT_DECAY):
        raise BuildError(f"{name} weight decay must be the PyTorch AdamW default {OFFICIAL_ADAMW_WEIGHT_DECAY}")
    if float(group["eps"]) != 1e-8 or bool(group["amsgrad"]):
        raise BuildError(f"{name} must keep AdamW eps 1e-8 and amsgrad false")


def require_disjoint(generator: torch.optim.Optimizer, discriminator: torch.optim.Optimizer) -> None:
    left = {id(parameter) for group in generator.param_groups for parameter in group["params"]}
    right = {id(parameter) for group in discriminator.param_groups for parameter in group["params"]}
    if not left or not right:
        raise BuildError("optimizer has no parameters")
    if left & right:
        raise BuildError("generator and discriminator share parameters")


def refuse_amp(runtime: object) -> None:
    if getattr(runtime, "precision", "fp32") == "amp_fp16":
        raise RuntimeFailure(
            "MossFormerGAN refuses amp_fp16. Each successful micro-batch clips, and the "
            "generator step and the discriminator step are not one scaler decision."
        )


def require_group_size(data: object, accumulation: int, component_id: str) -> None:
    group_size = getattr(data, "group_size", None)
    if group_size is None:
        return
    if int(group_size) != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} "
            f"does not match accumulation {accumulation}"
        )
