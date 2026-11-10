"""Validation metric aggregation shared with the reference reports."""

from __future__ import annotations

from typing import Any


def _combine_metric(items: list[dict[str, Any]]) -> dict[str, Any]:
    if any(item["status"] == "unavailable" for item in items):
        reason = next(item["reason"] for item in items if item["status"] == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in items if item["status"] == "ok" and item["value"] is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item.get("reason") for item in items if item.get("reason")), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}


def _aggregate_validation(reports: list[dict[str, Any]]) -> dict[str, Any]:
    names: list[str] = []
    for report in reports:
        for name in report["losses"]:
            if name not in names:
                names.append(name)
    losses = {
        name: _combine_metric([report["losses"][name] for report in reports if name in report["losses"]])
        for name in names
    }
    result = {
        "losses": losses,
        "pesq": _combine_metric([report["pesq"] for report in reports]),
        "stoi": _combine_metric([report["stoi"] for report in reports]),
    }
    if reports and all("si_sdr" in report for report in reports):
        result["si_sdr"] = _combine_metric([report["si_sdr"] for report in reports])
    if reports and all("si_snr" in report for report in reports):
        result["si_snr"] = _combine_metric([report["si_snr"] for report in reports])
    return result
