"""Corpus WER. The value is total edit count over total reference words."""

from __future__ import annotations

from typing import Mapping

from src.experiments.errors import BuildError
from src.experiments.metrics.base import MetricResult, finite_number, require_name


class WerMetric:
    """Micro-averaged word error rate. Lower is better.

    An utterance with no reference text, a failed hypothesis, or a non-finite
    count is a failure. Empty reference tokenizations are excluded. Neither
    case becomes ``0.0``. The score is adoptable only when every required
    utterance succeeded and the reference word count is positive.
    """

    direction = "min"
    unit = "ratio"

    def __init__(self, name: str = "wer", normalization: str = "wer-norm-v1", recognizer: object | None = None) -> None:
        try:
            self.name = require_name(name)
        except ValueError as exc:
            raise BuildError(str(exc)) from exc
        if recognizer is not None and not callable(getattr(recognizer, "transcribe", None)):
            raise BuildError("WER recognizer must provide transcribe(audio, sample_rate)")
        self.recognizer = recognizer
        if not isinstance(normalization, str) or not normalization:
            raise BuildError("normalization must be a non-empty string")
        from src.evaluation.metrics.wer import normalize_text

        try:
            normalize_text("ok", normalization)
        except Exception as exc:
            raise BuildError(f"unknown WER normalization {normalization}: {exc}") from exc
        self.normalization = normalization
        self.reset()

    def reset(self) -> None:
        self._counts = {"S": 0.0, "D": 0.0, "I": 0.0, "N": 0.0, "failures": 0.0, "excluded": 0.0, "ok": 0.0}

    def update(self, reference: object, hypothesis: object) -> None:
        from src.evaluation.metrics.wer import score_pair

        if not isinstance(reference, str) or reference.strip() == "":
            self._counts["failures"] += 1.0
            return
        if not isinstance(hypothesis, str):
            self._counts["failures"] += 1.0
            return
        try:
            pair = score_pair(reference, hypothesis, self.normalization)
            counts = pair["normalized"]
        except Exception:
            self._counts["failures"] += 1.0
            return
        if not isinstance(counts, dict):
            self._counts["failures"] += 1.0
            return
        words = finite_number(counts.get("N"))
        if words is None:
            self._counts["failures"] += 1.0
            return
        if words == 0:
            self._counts["excluded"] += 1.0
            return
        for key in ("S", "D", "I", "N"):
            number = finite_number(counts.get(key))
            if number is None:
                self._counts["failures"] += 1.0
                return
        for key in ("S", "D", "I", "N"):
            self._counts[key] += float(counts[key])
        self._counts["ok"] += 1.0

    def local_stats(self) -> dict[str, float]:
        return dict(self._counts)

    def result_from(self, stats: Mapping[str, float]) -> MetricResult:
        failures = int(stats.get("failures", 0.0))
        excluded = int(stats.get("excluded", 0.0))
        ok = int(stats.get("ok", 0.0))
        substitutions = float(stats.get("S", 0.0))
        deletions = float(stats.get("D", 0.0))
        insertions = float(stats.get("I", 0.0))
        words = float(stats.get("N", 0.0))
        partial = None
        if words > 0:
            partial = (substitutions + deletions + insertions) / words
        if failures or excluded or ok <= 0 or words <= 0 or partial is None:
            reason = "incomplete_utterances" if failures or excluded else "no_reference_words"
            return MetricResult(
                name=self.name,
                status="incomplete",
                value=None,
                count=ok,
                weight=words,
                failures=failures,
                excluded=excluded,
                direction=self.direction,
                unit=self.unit,
                reason=reason,
                partial_value=partial,
            )
        return MetricResult(
            name=self.name,
            status="ok",
            value=float(partial),
            count=ok,
            weight=words,
            failures=0,
            excluded=0,
            direction=self.direction,
            unit=self.unit,
            reason=None,
            partial_value=float(partial),
        )
