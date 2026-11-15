"""SEMamba++ build checks. The official AdamW and ExponentialLR values stay fixed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.experiments.errors import BuildError
from src.experiments.official.semambapp.models import (
    SEMambaPPCQTDiscriminator,
    SEMambaPPGenerator,
    SEMambaPPMRD,
)
from src.experiments.runtime import require_prepared

OFFICIAL_LR = 0.0002
OFFICIAL_BETAS = (0.8, 0.99)
OFFICIAL_WEIGHT_DECAY = 0.01
OFFICIAL_GAMMA = 0.99
SUBSTITUTE_CLAIM = "protocol_conformant_substitute_materials"


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be a non-negative integer")
    return value


def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{name} must be a positive number")
    number = float(value)
    if number <= 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise BuildError(f"{name} must be a positive finite number")
    return number


def require_adamw(optimizer: torch.optim.Optimizer, name: str) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError(f"{name} must be AdamW")
    if not optimizer.param_groups:
        raise BuildError(f"{name} has no parameters")
    for group in optimizer.param_groups:
        if float(group["lr"]) != OFFICIAL_LR:
            raise BuildError(f"{name} learning rate must be {OFFICIAL_LR}")
        betas = tuple(float(value) for value in group["betas"])
        if betas != OFFICIAL_BETAS:
            raise BuildError(f"{name} betas must be {OFFICIAL_BETAS}")
        if float(group["weight_decay"]) != OFFICIAL_WEIGHT_DECAY:
            raise BuildError(f"{name} weight_decay must be {OFFICIAL_WEIGHT_DECAY}")
        if float(group["eps"]) != 1e-8:
            raise BuildError(f"{name} eps must be 1e-8")
        if bool(group.get("amsgrad", False)):
            raise BuildError(f"{name} amsgrad must stay off")


def require_scheduler(scheduler: torch.optim.lr_scheduler.ExponentialLR, name: str) -> None:
    if not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
        raise BuildError(f"{name} must be ExponentialLR")
    if float(scheduler.gamma) != OFFICIAL_GAMMA:
        raise BuildError(f"{name} gamma must be {OFFICIAL_GAMMA}")


def require_disjoint(optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer) -> None:
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("SEMamba++ optimizers must each own parameters")


@dataclass(frozen=True)
class SEMambaSetup:
    optim_g_id: str
    optim_d_id: str
    claim_label: str
    max_logical_steps: int
    accumulation: int
    grad_clip: float | None
    validation_interval: int
    best_checkpoint_start_epoch: int
    max_epochs: int | None
    stop_after_logical_steps: int | None


def require_semamba_setup(
    generator: Any,
    mrd: Any,
    cqt: Any,
    data: Any,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
    scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
    *,
    generator_objective: torch.nn.Module,
    discriminator_objective: torch.nn.Module,
    max_logical_steps: int,
    accumulation: int,
    grad_clip: float | None,
    validation_interval: int,
    best_checkpoint_start_epoch: int,
    max_epochs: int | None,
    stop_after_logical_steps: int | None,
    claim_label: str,
) -> SEMambaSetup:
    require_prepared(generator, SEMambaPPGenerator, "generator must be a prepared SEMambaPPGenerator")
    require_prepared(mrd, SEMambaPPMRD, "mrd must be a prepared SEMambaPPMRD")
    require_prepared(cqt, SEMambaPPCQTDiscriminator, "cqt must be a prepared SEMambaPPCQTDiscriminator")
    if generator.unwrapped().architecture != mrd.unwrapped().architecture:
        raise BuildError("MRD initialization architecture does not match the generator")
    if generator.unwrapped().architecture != cqt.unwrapped().architecture:
        raise BuildError("CQT initialization architecture does not match the generator")
    if generator.unwrapped().seed != mrd.unwrapped().seed or generator.unwrapped().seed != cqt.unwrapped().seed:
        raise BuildError("generator, MRD, and CQT seeds differ")
    require_adamw(optim_g, "optim_g")
    require_adamw(optim_d, "optim_d")
    require_scheduler(scheduler_g, "scheduler_g")
    require_scheduler(scheduler_d, "scheduler_d")
    if not isinstance(generator_objective, torch.nn.Module) or not isinstance(discriminator_objective, torch.nn.Module):
        raise BuildError("SEMamba++ objectives must be loss modules")
    require_disjoint(optim_g, optim_d)
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    optim_g_id = getattr(optim_g, "_experiment_component_id", None)
    optim_d_id = getattr(optim_d, "_experiment_component_id", None)
    if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
        raise BuildError("optimizers have no component id")
    if not isinstance(claim_label, str) or not claim_label:
        raise BuildError("claim_label must be a non-empty string")
    if claim_label == "official_reproduction":
        raise BuildError("semambapp.gsr.official is blocked")
    if stop_after_logical_steps is not None:
        stop_after_logical_steps = positive_int("stop_after_logical_steps", stop_after_logical_steps)
    return SEMambaSetup(
        optim_g_id=optim_g_id,
        optim_d_id=optim_d_id,
        claim_label=claim_label,
        max_logical_steps=positive_int("max_logical_steps", max_logical_steps),
        accumulation=positive_int("accumulation", accumulation),
        grad_clip=None if grad_clip is None else positive_float("grad_clip", grad_clip),
        validation_interval=non_negative_int("validation_interval", validation_interval),
        best_checkpoint_start_epoch=non_negative_int(
            "best_checkpoint_start_epoch", best_checkpoint_start_epoch
        ),
        max_epochs=None if max_epochs is None else positive_int("max_epochs", max_epochs),
        stop_after_logical_steps=stop_after_logical_steps,
    )


def require_group_size(data: object, accumulation: int, component_id: str) -> None:
    from src.experiments.errors import RuntimeFailure

    group_size = getattr(data, "group_size", None)
    if group_size is None:
        return
    if int(group_size) != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} "
            f"does not match accumulation {accumulation}"
        )
