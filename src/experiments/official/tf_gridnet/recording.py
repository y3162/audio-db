"""Deferred step log for TF-GridNet.

The trainer queues one finished step and drains it before the next forward,
or immediately when the epoch is over. The held record is not checkpoint
state. ``optimizer_updates`` is read at drain time so the row matches the
counters after ``record_step``. A step save is offered only when the trainer
marked the row as a continuing step.
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

    def drain(self, journal: StepJournal, lifecycle: Any, optimizer_updates: Mapping[str, int]) -> None:
        """Write the held row and offer its step save when the trainer asked."""

        item = self._item
        if item is None:
            return
        self._item = None
        loss_value = item["loss"]
        applied = item["applied"]
        sample_id = item["sample_id"]
        learning_rate = item["learning_rate"]
        epoch = item["epoch"]
        logical_step = item["logical_step"]
        offer_step = item["offer_step"]
        terms = item["terms"]
        step_metrics = item["step_metrics"]
        journal.write_metrics(
            {
                "applied": applied,
                "epoch": epoch,
                "logical_step": logical_step,
                "loss": loss_value,
                "lr": learning_rate,
                "optimizer_updates": dict(optimizer_updates),
                "sample_id": sample_id,
                "terms": terms,
            }
        )
        journal.write_sample(
            {
                "epoch": epoch,
                "logical_step": logical_step,
                "loss": loss_value,
                "lr": learning_rate,
                "sample_id": sample_id,
            }
        )
        if not offer_step:
            return
        if step_metrics:
            lifecycle.save_if_needed("step", logical_step=logical_step, metrics=step_metrics)
        else:
            lifecycle.save_if_needed("step", logical_step=logical_step)
