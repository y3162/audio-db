"""MP-SENet official PESQ selection.

The trainer calls this after validation and before the epoch schedulers.
Writing ``best_generator.pt`` stays in ``artifacts.py``. A missing or
non-finite PESQ does not become zero.
"""

from __future__ import annotations

import math
from typing import Any


def consider_best_pesq(
    epoch: int,
    summary: dict[str, Any],
    *,
    start_epoch: int,
    best_pesq: float | None,
    logical_step: int,
) -> dict[str, Any]:
    """Keep the earlier checkpoint unless PESQ is strictly higher after the start epoch.

    The returned record does not change ``best_pesq``. The trainer stores
    ``pesq``, ``epoch``, and ``logical_step`` when ``selected`` is true.
    """

    pesq = summary.get("pesq")
    if not isinstance(pesq, dict):
        pesq = {"status": "na", "value": None, "reason": "missing pesq"}
    status = pesq.get("status")
    value = pesq.get("value")
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "logical_step": int(logical_step),
        "pesq": None if value is None else float(value),
        "pesq_status": status,
        "reason": None,
        "selected": False,
    }
    if epoch < start_epoch:
        record["reason"] = "before_start_epoch"
        return record
    if status != "ok" or value is None:
        record["reason"] = "unavailable" if status == "unavailable" else "not_ok"
        return record
    score = float(value)
    if not math.isfinite(score):
        record["reason"] = "non_finite"
        return record
    if best_pesq is not None and score <= float(best_pesq):
        record["reason"] = "not_higher"
        return record
    record["selected"] = True
    record["reason"] = "improved"
    return record
