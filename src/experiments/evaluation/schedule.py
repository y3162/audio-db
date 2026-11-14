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
        self.progress_key = "epoch" if self.unit == "epoch" else "logical_step" if self.unit == "logical_step" else None

    def due(
        self,
        *,
        logical_step: int | None = None,
        epoch: int | None = None,
        kind: str,
        progress: Mapping[str, int] | None = None,
    ) -> bool:
        values = _progress_values(logical_step, epoch, progress)
        if self.unit == "boundary":
            return kind == "boundary"
        if self.unit == "end":
            return kind == "end"
        if self.unit == "epoch":
            if "epoch" not in values:
                raise BuildError("epoch schedule requires progress epoch")
            return kind == "boundary" and int(values["epoch"]) % self.interval == 0
        if self.unit == "logical_step":
            if "logical_step" not in values:
                raise BuildError("logical_step schedule requires progress logical_step")
            step = int(values["logical_step"])
            return kind in {"step", "boundary"} and step > 0 and step % self.interval == 0
        return False

    def token(
        self,
        *,
        logical_step: int | None = None,
        epoch: int | None = None,
        source: str,
        progress: Mapping[str, int] | None = None,
    ) -> tuple[str, str, int]:
        values = _progress_values(logical_step, epoch, progress)
        if self.unit == "epoch":
            if "epoch" not in values:
                raise BuildError("epoch schedule requires progress epoch")
            return ("epoch", source, int(values["epoch"]))
        if self.unit == "logical_step" and "logical_step" not in values:
            raise BuildError("logical_step schedule requires progress logical_step")
        if "logical_step" in values:
            return (self.unit, source, int(values["logical_step"]))
        if "epoch" in values:
            return (self.unit, source, int(values["epoch"]))
        if self.unit == "end":
            return ("end", source, 1)
        if "event" in values:
            return (self.unit, source, int(values["event"]))
        raise BuildError(f"{self.unit} schedule needs progress logical_step, epoch, or event")


def _progress_values(
    logical_step: int | None,
    epoch: int | None,
    progress: Mapping[str, int] | None,
) -> dict[str, int]:
    values: dict[str, int] = {}
    if progress is not None:
        for key, value in progress.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise BuildError(f"progress {key} must be an integer")
            values[str(key)] = int(value)
    if logical_step is not None:
        if isinstance(logical_step, bool) or not isinstance(logical_step, int):
            raise BuildError("logical_step schedule requires progress logical_step")
        values.setdefault("logical_step", int(logical_step))
    if epoch is not None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise BuildError("epoch schedule requires progress epoch")
        values.setdefault("epoch", int(epoch))
    return values
