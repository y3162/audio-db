"""Score the conditions produced for one utterance."""

from __future__ import annotations

from typing import Any

import torch

from src.evaluation.config import MetricsConfig
from src.evaluation.metrics.pesq import score_pesq
from src.evaluation.metrics.sisdr import score_si_sdr
from src.evaluation.metrics.stoi import score_stoi
from src.evaluation.metrics.wer import score_pair


def score_utterance(
    metrics: MetricsConfig,
    *,
    clean: torch.Tensor | None,
    estimates: dict[str, torch.Tensor],
    sample_rate: int,
    reference: str | None,
    hypotheses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "wer": _wer(metrics, reference, hypotheses),
        "pesq": _reference_metric("pesq", metrics.pesq, clean, estimates, sample_rate, score_pesq),
        "stoi": _reference_metric(
            "stoi",
            metrics.stoi,
            clean,
            estimates,
            sample_rate,
            lambda ref, est, rate: score_stoi(ref, est, rate, extended=False),
        ),
        "estoi": _reference_metric(
            "estoi",
            metrics.estoi,
            clean,
            estimates,
            sample_rate,
            lambda ref, est, rate: score_stoi(ref, est, rate, extended=True),
        ),
        "si_sdr": _si_sdr(metrics.si_sdr, clean, estimates),
    }


def not_computed(reason: str = "not_computed") -> dict[str, Any]:
    return {
        "wer": {"status": "na", "reason": reason, "values": {}},
        "pesq": {"status": "na", "reason": reason, "values": {}},
        "stoi": {"status": "na", "reason": reason, "values": {}},
        "estoi": {"status": "na", "reason": reason, "values": {}},
        "si_sdr": {"status": "na", "reason": reason, "values": {}},
    }


def _wer(metrics: MetricsConfig, reference: str | None, hypotheses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not metrics.wer:
        return {"status": "disabled", "normalization": metrics.normalization, "values": {}}
    if reference is None:
        return {
            "status": "na",
            "reason": "no_reference_transcript",
            "normalization": metrics.normalization,
            "values": {},
        }
    values = {}
    for condition, hypothesis in hypotheses.items():
        if hypothesis.get("text") is None:
            values[condition] = {
                "status": "na",
                "reason": hypothesis.get("error") or "not_transcribed",
            }
            continue
        values[condition] = score_pair(reference, hypothesis["text"], metrics.normalization)
    return {"status": "ok", "normalization": metrics.normalization, "values": values}


def _reference_metric(name, enabled, clean, estimates, sample_rate, function) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled", "values": {}}
    if clean is None:
        return {
            "status": "na",
            "reason": "no_clean_reference",
            "values": {condition: {"status": "na", "reason": "no_clean_reference", "value": None} for condition in estimates},
        }
    values = {}
    for condition, estimate in estimates.items():
        values[condition] = function(clean, estimate, sample_rate)
    statuses = {item["status"] for item in values.values()}
    if statuses == {"ok"}:
        status = "ok"
    elif "unavailable" in statuses:
        status = "unavailable"
    elif "error" in statuses:
        status = "error"
    else:
        status = "ok"
    reason = None
    for item in values.values():
        if item.get("reason"):
            reason = item["reason"]
            break
    return {"status": status, "reason": reason, "metric": name, "values": values}


def _si_sdr(enabled: bool, clean: torch.Tensor | None, estimates: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled", "values": {}}
    if clean is None:
        return {
            "status": "na",
            "reason": "no_clean_reference",
            "values": {condition: {"status": "na", "reason": "no_clean_reference", "value": None} for condition in estimates},
        }
    values = {condition: score_si_sdr(clean, estimate) for condition, estimate in estimates.items()}
    return {"status": "ok", "values": values}
