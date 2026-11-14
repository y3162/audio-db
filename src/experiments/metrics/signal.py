"""Utterance means of SI-SDR, PESQ, STOI, and ESTOI.

The numbers come from ``src.evaluation.metrics``. SI-SDR here is the positive
decibel score from ``score_si_sdr``, not the negative training loss.
"""

from __future__ import annotations

from typing import Mapping

import math

from src.experiments.errors import BuildError
from src.experiments.metrics.base import MetricResult, reduced_mean, require_name, stats_number


class _UtteranceMean:
    direction = "max"
    unit = ""

    def __init__(self, name: str, *, require_complete: bool = True) -> None:
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
        self._ok = 0.0
        self._failures = 0.0
        self._excluded = 0.0

    def update_status(self, status: object, value: object) -> None:
        if status == "ok":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                self._failures += 1.0
                return
            number = float(value)
            self._sum += number
            self._ok += 1.0
            return
        if status in {"na", "disabled"}:
            self._excluded += 1.0
            return
        self._failures += 1.0

    def local_stats(self) -> dict[str, float]:
        return {"sum": self._sum, "ok": self._ok, "failures": self._failures, "excluded": self._excluded}

    def result_from(self, stats: Mapping[str, float]) -> MetricResult:
        return reduced_mean(
            name=self.name,
            direction=self.direction,
            unit=self.unit,
            total=stats_number(stats, "sum"),
            count=stats_number(stats, "ok"),
            failures=stats_number(stats, "failures"),
            excluded=stats_number(stats, "excluded"),
            require_complete=self.require_complete,
        )


class SISDRMetric(_UtteranceMean):
    """Mean utterance SI-SDR in dB. Higher is better. Perfect reconstruction is not a finite score."""

    unit = "dB"

    def __init__(self, name: str = "si_sdr", require_complete: bool = True) -> None:
        super().__init__(name, require_complete=require_complete)

    def score(self, reference: torch.Tensor, estimate: torch.Tensor) -> None:
        from src.evaluation.metrics.sisdr import score_si_sdr

        report = score_si_sdr(reference, estimate)
        self.update_status(report.get("status"), report.get("value"))


class PESQMetric(_UtteranceMean):
    """Mean of successful PESQ scores. Unavailable, na, and errors stay out of the value."""

    unit = "mos"

    def __init__(self, name: str = "pesq", require_complete: bool = True) -> None:
        super().__init__(name, require_complete=require_complete)

    def score(self, reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> None:
        from src.evaluation.metrics.pesq import score_pesq

        report = score_pesq(reference, estimate, sample_rate)
        self.update_status(report.get("status"), report.get("value"))


class STOIMetric(_UtteranceMean):
    """Mean of successful STOI scores."""

    unit = "correlation"

    def __init__(self, name: str = "stoi", require_complete: bool = True) -> None:
        super().__init__(name, require_complete=require_complete)

    def score(self, reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> None:
        from src.evaluation.metrics.stoi import score_stoi

        report = score_stoi(reference, estimate, sample_rate, extended=False)
        self.update_status(report.get("status"), report.get("value"))


class ESTOIMetric(_UtteranceMean):
    """Mean of successful ESTOI scores."""

    unit = "correlation"

    def __init__(self, name: str = "estoi", require_complete: bool = True) -> None:
        super().__init__(name, require_complete=require_complete)

    def score(self, reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> None:
        from src.evaluation.metrics.stoi import score_stoi

        report = score_stoi(reference, estimate, sample_rate, extended=True)
        self.update_status(report.get("status"), report.get("value"))
