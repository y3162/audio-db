"""Official validation, PESQ selection, and both epoch-end schedulers.

The trainer decides that a boundary is pending and saves that fact first.
This function then runs the official steps and returns the scalar metrics.
It does not own the progress counters, does not receive the trainer, and
does not save. The caller records the closed boundary, scores research, and
offers the boundary checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.experiments.selectors import scalar_metrics


def apply_official_boundary(
    boundary: dict[str, Any],
    *,
    validate: Callable[[], dict[str, Any]],
    consider_best: Callable[[int, dict[str, Any]], dict[str, Any]],
    record_history: Callable[[dict[str, Any]], None],
    write_best: Callable[[dict[str, Any]], None],
    scheduler_g: Any,
    scheduler_d: Any,
    optim_g: Any,
    optim_d: Any,
    data: Any,
    mark_phase: Callable[[str], None],
) -> tuple[dict[str, float], bool]:
    """Validate, select a strict PESQ improvement, then step both schedulers.

    A validation interval does not step the schedulers. When the same update
    validates and ends the epoch, selection finishes before either
    ``ExponentialLR`` moves. ``mark_phase("pending_scheduler")`` runs only in
    that case, before the scheduler steps. The returned bool is that step.
    """

    metrics: dict[str, float] = {}
    if boundary["validate"] and not boundary["validation_done"]:
        summary = validate()
        metrics = scalar_metrics(summary)
        decision = consider_best(int(boundary["trainer_epoch"]), summary)
        record_history(decision)
        write_best(decision)
        boundary["validation_done"] = True
        boundary["selection_done"] = True
        boundary["pesq"] = decision["pesq"]
        boundary["pesq_status"] = decision["pesq_status"]
        boundary["selected"] = decision["selected"]
        if boundary["scheduler"]:
            mark_phase("pending_scheduler")
    stepped = False
    if boundary["scheduler"] and not boundary["scheduler_done"]:
        scheduler_g.step()
        scheduler_d.step()
        stepped = True
        boundary["scheduler_done"] = True
        boundary["lr_g"] = float(optim_g.param_groups[0]["lr"])
        boundary["lr_d"] = float(optim_d.param_groups[0]["lr"])
    if boundary["epoch_done"]:
        data.epoch = int(boundary["trainer_epoch"]) + 1
        data.update_index = 0
    boundary["closed"] = True
    return metrics, stepped
