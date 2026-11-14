"""When a research evaluator is allowed to run."""

from __future__ import annotations

from typing import Mapping

from src.experiments.errors import BuildError

UNITS = frozenset({"epoch", "logical_step", "boundary", "end"})


class EvaluationSchedule:
    """One YAML ``every`` block.

    ``epoch`` and ``logical_step`` fire on that stride. Epoch 0 is included.
    ``boundary`` fires on the official validation boundary. ``end`` fires when
    training completes. A completed token is not run again after resume.
    """

    def __init__(self, every: Mapping[str, object] | None) -> None:
        if every is None:
            every = {"unit": "boundary", "interval": 1}
        if not isinstance(every, Mapping):
            raise BuildError("every must be a mapping")
        unit = every.get("unit", "boundary")
        interval = every.get("interval", 1)
        if unit not in UNITS:
            raise BuildError("every.unit must be epoch, logical_step, boundary, or end")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise BuildError("every.interval must be a positive integer")
        if unit in {"boundary", "end"} and interval != 1:
            raise BuildError(f"every.interval must be 1 when unit is {unit}")
        unknown = sorted(set(every) - {"unit", "interval"})
        if unknown:
            raise BuildError(f"unknown every field {unknown[0]}")
        self.unit = str(unit)
        self.interval = int(interval)

    def due(self, *, logical_step: int, epoch: int | None, kind: str) -> bool:
        if self.unit == "boundary":
            return kind == "boundary"
        if self.unit == "end":
            return kind == "end"
        if self.unit == "epoch":
            return kind == "boundary" and epoch is not None and int(epoch) % self.interval == 0
        if self.unit == "logical_step":
            return kind in {"step", "boundary"} and logical_step > 0 and logical_step % self.interval == 0
        return False

    def token(self, *, logical_step: int, epoch: int | None, source: str) -> tuple[str, str, int]:
        if self.unit == "epoch":
            if epoch is None:
                raise BuildError("epoch schedule requires an epoch")
            return ("epoch", source, int(epoch))
        return (self.unit, source, int(logical_step))
