"""Shared result of one research metric.

A metric accumulates utterance statistics and reduces them. It does not train,
stop, or choose a checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MetricResult:
    """One reduced metric. ``value`` is None unless ``status`` is ``ok``."""

    name: str
    status: str
    value: float | None
    count: int
    weight: float
    failures: int
    excluded: int
    direction: str
    unit: str
    reason: str | None = None
    partial_value: float | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "count": self.count,
            "direction": self.direction,
            "excluded": self.excluded,
            "failures": self.failures,
            "metric": self.name,
            "partial_value": self.partial_value,
            "reason": self.reason,
            "status": self.status,
            "unit": self.unit,
            "value": self.value,
            "weight": self.weight,
        }


def finite_number(value: object) -> float | None:
    """Return a finite float, or None for booleans, non-numbers, and non-finite values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def require_name(name: str) -> str:
    if not isinstance(name, str) or not name or "." in name or any(char.isspace() for char in name):
        raise ValueError("metric name must be a non-empty string without a dot or spaces")
    return name


def empty_result(name: str, *, direction: str, unit: str, reason: str) -> MetricResult:
    return MetricResult(
        name=name,
        status="incomplete",
        value=None,
        count=0,
        weight=0.0,
        failures=0,
        excluded=0,
        direction=direction,
        unit=unit,
        reason=reason,
        partial_value=None,
    )


def reduced_mean(
    *,
    name: str,
    direction: str,
    unit: str,
    total: float,
    count: float,
    failures: float,
    excluded: float,
    require_complete: bool,
) -> MetricResult:
    """Mean of successful utterances. Failures are never filled with zero."""

    ok = int(count)
    partial = None if ok <= 0 else float(total) / float(count)
    if partial is not None and not math.isfinite(partial):
        partial = None
    failed = int(failures) > 0
    dropped = int(excluded) > 0
    if ok <= 0 or (require_complete and (failed or dropped)):
        reason = "no_successful_utterances" if ok <= 0 else "incomplete_utterances"
        return MetricResult(
            name=name,
            status="incomplete",
            value=None,
            count=ok,
            weight=float(count),
            failures=int(failures),
            excluded=int(excluded),
            direction=direction,
            unit=unit,
            reason=reason,
            partial_value=partial,
        )
    if partial is None or not math.isfinite(partial):
        return empty_result(name, direction=direction, unit=unit, reason="non_finite_aggregate")
    return MetricResult(
        name=name,
        status="ok",
        value=float(partial),
        count=ok,
        weight=float(count),
        failures=int(failures),
        excluded=int(excluded),
        direction=direction,
        unit=unit,
        reason=None,
        partial_value=float(partial),
    )


def stats_number(stats: Mapping[str, float], key: str) -> float:
    value = stats.get(key, 0.0)
    number = finite_number(value)
    if number is None:
        raise ValueError(f"reduced statistic {key} is not finite")
    return number
