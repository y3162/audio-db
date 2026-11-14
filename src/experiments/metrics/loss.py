"""Weighted mean of a batch loss. The value is the loss itself, so smaller is better."""

from __future__ import annotations

from typing import Mapping

from src.experiments.errors import BuildError
from src.experiments.metrics.base import MetricResult, finite_number, reduced_mean, require_name, stats_number


class LossMetric:
    """Sample-weighted mean of batch-mean losses.

    ``weight`` is the number of utterances in the batch. A non-finite loss is a
    failure and is left out of the sum. It is not replaced with zero.
    """

    direction = "min"
    unit = "loss"

    def __init__(self, name: str = "loss", require_complete: bool = True) -> None:
        try:
            self.name = require_name(name)
        except ValueError as exc:
            raise BuildError(str(exc)) from exc
        if not isinstance(require_complete, bool):
            raise BuildError("require_complete must be a bool")
        self.require_complete = require_complete
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._weight = 0.0
        self._failures = 0.0
        self._excluded = 0.0

    def update(self, value: object, weight: object) -> None:
        count = finite_number(weight)
        if count is None or count <= 0:
            raise BuildError("loss weight must be a positive finite number")
        number = finite_number(value)
        if number is None:
            self._failures += count
            return
        self._sum += number * count
        self._weight += count

    def local_stats(self) -> dict[str, float]:
        return {"sum": self._sum, "weight": self._weight, "failures": self._failures, "excluded": self._excluded}

    def result_from(self, stats: Mapping[str, float]) -> MetricResult:
        return reduced_mean(
            name=self.name,
            direction=self.direction,
            unit=self.unit,
            total=stats_number(stats, "sum"),
            count=stats_number(stats, "weight"),
            failures=stats_number(stats, "failures"),
            excluded=stats_number(stats, "excluded"),
            require_complete=self.require_complete,
        )
