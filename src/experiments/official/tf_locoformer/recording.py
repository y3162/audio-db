"""Deferred step log for TF-Locoformer.

``lr`` is the rate used for the step. ``scheduled_lr`` is the rate after
``step_batch``. ``step_num`` is read by the trainer at drain time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.experiments.execution.journal import StepJournal


class DeferredStepLog:
    """At most one unflushed step row."""

    def __init__(self) -> None:
        self._item: dict[str, Any] | None = None

    def queue(self, record: dict[str, Any]) -> None:
        self._item = record

    def drain(
        self,
        journal: StepJournal,
        lifecycle: Any,
        optimizer_updates: Mapping[str, int],
        step_num: int,
    ) -> None:
        item = self._item
        if item is None:
            return
        self._item = None
        journal.write_metrics(
            {
                "applied": item["applied"],
                "epoch": item["epoch"],
                "logical_step": item["logical_step"],
                "loss": item["loss"],
                "lr": item["used_lr"],
                "optimizer_updates": dict(optimizer_updates),
                "sample_id": item["sample_id"],
                "scheduled_lr": item["scheduled_lr"],
                "si_sdr": item["si_sdr"],
                "step_num": int(step_num),
                "terms": item["terms"],
            }
        )
        journal.write_sample(
            {
                "epoch": item["epoch"],
                "logical_step": item["logical_step"],
                "loss": item["loss"],
                "lr": item["used_lr"],
                "sample_id": item["sample_id"],
                "scheduled_lr": item["scheduled_lr"],
                "si_sdr": item["si_sdr"],
            }
        )
        if not item["offer_step"]:
            return
        step_metrics = item["step_metrics"]
        logical_step = item["logical_step"]
        if step_metrics:
            lifecycle.save_if_needed("step", logical_step=logical_step, metrics=step_metrics)
        else:
            lifecycle.save_if_needed("step", logical_step=logical_step)
