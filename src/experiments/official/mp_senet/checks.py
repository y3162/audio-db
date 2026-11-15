"""Constructor checks for the MP-SENet GAN trainer.

These look at the objects the YAML named. They do not train, validate, or
save, and they do not keep a copy of the trainer.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.errors import BuildError
from src.experiments.official.mp_senet.models import (
    MPSENetGANGenerator,
    MPSENetMetricDiscriminator,
    require_official_architecture,
)
from src.experiments.runtime import PreparedModel, require_prepared
from src.experiments.training.official_objectives import require_mp_senet

_CLAIMS = frozenset({"official_faithful", "leakage_safe", "research"})


def require_mp_gan_setup(
    generator: PreparedModel,
    discriminator: PreparedModel,
    data: Any,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
    scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
    signal_objective: torch.nn.Module,
    metric_objective: torch.nn.Module,
    discriminator_objective: torch.nn.Module,
    experiment_claim: str,
) -> tuple[str, str]:
    """Refuse a pair that cannot run the official GAN loop. Return optimizer ids."""

    require_prepared(generator, MPSENetGANGenerator, "generator must be a prepared MPSENetGANGenerator")
    require_prepared(
        discriminator,
        MPSENetMetricDiscriminator,
        "discriminator must be a prepared MPSENetMetricDiscriminator",
    )
    if generator.unwrapped().architecture != discriminator.unwrapped().architecture:
        raise BuildError("discriminator initialization architecture does not match the generator")
    if generator.unwrapped().seed != discriminator.unwrapped().seed:
        raise BuildError("generator and discriminator seeds differ")
    if not isinstance(optim_g, torch.optim.AdamW) or not isinstance(optim_d, torch.optim.AdamW):
        raise BuildError("MP-SENet GAN optimizers must be AdamW")
    if not isinstance(scheduler_g, torch.optim.lr_scheduler.ExponentialLR):
        raise BuildError("scheduler_g must be ExponentialLR")
    if not isinstance(scheduler_d, torch.optim.lr_scheduler.ExponentialLR):
        raise BuildError("scheduler_d must be ExponentialLR")
    _require_disjoint(optim_g, optim_d)
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    if experiment_claim not in _CLAIMS:
        raise BuildError("experiment_claim must be official_faithful, leakage_safe, or research")
    require_official_architecture(generator.unwrapped().architecture, experiment_claim)
    require_mp_senet(signal_objective, metric_objective, discriminator_objective, experiment_claim)
    optim_g_id = getattr(optim_g, "_experiment_component_id", None)
    optim_d_id = getattr(optim_d, "_experiment_component_id", None)
    if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
        raise BuildError("optimizers have no component id")
    return optim_g_id, optim_d_id


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value


def _require_disjoint(optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer) -> None:
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("MP-SENet GAN optimizers must each own parameters")
