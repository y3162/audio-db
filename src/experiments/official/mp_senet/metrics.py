"""MP-SENet validation aggregation.

This is the mean of the rank-0 batch reports. It is not
``calc.metrics._aggregate_validation``: that helper has no ``audio_lengths``
or ``pesq_policy``, and it reads ``status`` with a required key. An
unavailable PESQ stays unavailable and is not replaced with zero.
"""

from __future__ import annotations

from typing import Any


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of batch-level official reports. Unavailable stays unavailable."""

    names: list[str] = []
    for report in reports:
        for name in report["losses"]:
            if name not in names:
                names.append(name)
    losses = {
        name: combine_metric([report["losses"][name] for report in reports if name in report["losses"]])
        for name in names
    }
    pesq = combine_metric([report["pesq"] for report in reports])
    policy = next((report["pesq"].get("pesq_policy") for report in reports if isinstance(report.get("pesq"), dict)), None)
    if policy is not None:
        pesq["pesq_policy"] = policy
    lengths: list[dict[str, int]] = []
    for report in reports:
        rows = report.get("audio_lengths")
        if isinstance(rows, list):
            lengths.extend(rows)
    return {
        "audio_lengths": lengths,
        "losses": losses,
        "pesq": pesq,
        "stoi": combine_metric([report["stoi"] for report in reports]),
    }


def combine_metric(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean the finite ok values. One unavailable item makes the result unavailable."""

    if any(item.get("status") == "unavailable" for item in items):
        reason = next(item.get("reason") for item in items if item.get("status") == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in items if item.get("status") == "ok" and item.get("value") is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item.get("reason") for item in items if item.get("reason")), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}
