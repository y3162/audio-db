"""Loss terms and weighted objectives constructed from experiment YAML.

A term reads a named context and returns one scalar. An objective adds those
scalars in the injected term order. It does not call backward or an optimizer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.errors import BuildError, RuntimeFailure


class ObjectiveOutput:
    """Total loss and the unweighted term values in addition order.

    ``records`` is ``(name, unweighted value, weight)``. ``total`` is the
    left fold of the weighted terms. A weight of ``1.0`` does not multiply,
    so a single official term keeps the term's own graph.
    """

    def __init__(
        self,
        total: torch.Tensor,
        records: tuple[tuple[str, torch.Tensor, float], ...],
    ) -> None:
        self.total = total
        self.records = records

    @property
    def terms(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value, _weight in self.records}

    @property
    def weights(self) -> dict[str, float]:
        return {name: weight for name, _value, weight in self.records}

    def unweighted(self, name: str) -> torch.Tensor:
        for item, value, _weight in self.records:
            if item == name:
                return value
        raise KeyError(name)


def apply_weight(value: torch.Tensor, weight: float) -> torch.Tensor:
    """Scale a term. ``1.0`` returns the same tensor."""

    if weight == 1.0:
        return value
    return value * weight


def require_tensor(context: Mapping[str, Any], key: str, term: str) -> torch.Tensor:
    if key not in context:
        raise ValueError(f"loss term {term} requires context[{key!r}]")
    value = context[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"loss term {term} context[{key!r}] must be a tensor")
    return value


def require_sequence(context: Mapping[str, Any], key: str, term: str) -> Sequence[Any]:
    if key not in context:
        raise ValueError(f"loss term {term} requires context[{key!r}]")
    value = context[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"loss term {term} context[{key!r}] must be a non-empty sequence")
    return value


class WeightedObjective(torch.nn.Module):
    """Left-fold ``weight * term(context)`` in the injected term order.

    ``weights`` must name each term once. ``0.0`` is a real coefficient: the
    term still runs, and its gradient is zero. Negative finite weights are
    allowed. ``check_finite`` raises when the total is not finite. Official
    GAN steps leave it false so a non-finite micro-batch can still be skipped
    by the trainer.
    """

    def __init__(
        self,
        terms: Sequence[torch.nn.Module],
        weights: Mapping[str, float],
        *,
        check_finite: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence) or not terms:
            raise BuildError("objective terms must be a non-empty sequence")
        if not isinstance(weights, Mapping):
            raise BuildError("objective weights must be a mapping")
        if not isinstance(check_finite, bool):
            raise BuildError("check_finite must be a boolean")
        order: list[str] = []
        seen: set[str] = set()
        for term in terms:
            name = getattr(term, "name", None)
            if not isinstance(name, str) or not name:
                raise BuildError("each loss term needs a non-empty name")
            if name in seen:
                raise BuildError(f"duplicate loss term name {name}")
            seen.add(name)
            order.append(name)
        cleaned: dict[str, float] = {}
        for name, value in weights.items():
            if not isinstance(name, str) or not name:
                raise BuildError("objective weight names must be strings")
            cleaned[name] = _finite_weight(value, name)
        missing = [name for name in order if name not in cleaned]
        extra = [name for name in cleaned if name not in seen]
        if missing or extra:
            raise BuildError(
                "objective weights must match the term names; "
                f"missing {missing}, extra {extra}"
            )
        self.term_modules = torch.nn.ModuleList(list(terms))
        self.order = tuple(order)
        self.weights = {name: cleaned[name] for name in order}
        self.check_finite = check_finite

    def term(self, name: str) -> torch.nn.Module:
        for module in self.term_modules:
            if module.name == name:
                return module
        raise KeyError(name)

    def forward(self, context: Mapping[str, Any]) -> ObjectiveOutput:
        if not isinstance(context, Mapping):
            raise ValueError("loss context must be a mapping")
        total: torch.Tensor | None = None
        records: list[tuple[str, torch.Tensor, float]] = []
        for module in self.term_modules:
            value = module(context)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ValueError(f"loss term {module.name} must return a scalar tensor")
            weight = self.weights[module.name]
            records.append((module.name, value, weight))
            scaled = apply_weight(value, weight)
            total = scaled if total is None else total + scaled
        if total is None:
            raise ValueError("objective has no terms")
        if self.check_finite and not bool(torch.isfinite(total.detach()).all()):
            raise TrainingError("objective loss is not finite")
        return ObjectiveOutput(total, tuple(records))


def term_floats(result: ObjectiveOutput) -> dict[str, float]:
    """Weighted term values as Python floats, in addition order."""

    return {
        name: float(apply_weight(value, weight).detach())
        for name, value, weight in result.records
    }


def call_objective(module: torch.nn.Module, context: Mapping[str, Any]) -> ObjectiveOutput:
    result = module(context)
    if not isinstance(result, ObjectiveOutput):
        raise RuntimeFailure("loss objective must return ObjectiveOutput")
    return result


def coerce_spectral_objective(objective: torch.nn.Module, *, label: str) -> WeightedObjective:
    """Accept the historical single module or an explicit weighted objective.

    Direct callers still pass ``MultiResolutionL1Loss``. Experiment YAML passes
    ``WeightedObjective``. A weight of 1 keeps that module's graph.
    """

    from src.experiments.training.losses import MultiResolutionL1Loss

    if isinstance(objective, MultiResolutionL1Loss):
        if objective.normalize_variance:
            raise BuildError(f"{label} loss cannot normalize variance")
        objective = WeightedObjective([objective], {objective.name: 1.0})
    if not isinstance(objective, WeightedObjective):
        raise BuildError(f"{label} objective must be WeightedObjective")
    for module in objective.term_modules:
        if isinstance(module, MultiResolutionL1Loss) and module.normalize_variance:
            raise BuildError(f"{label} loss cannot normalize variance")
    return objective


def _finite_weight(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"weight {name} must be a finite number")
    weight = float(value)
    if not math.isfinite(weight):
        raise BuildError(f"weight {name} must be finite")
    return weight
