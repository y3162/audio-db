"""Identical trainer steps.

These functions replace copies whose meaning, errors, and order already
matched. They do not choose a model's update, validation, or save point.

``open_grouped_window`` is the TF-GridNet and TF-Locoformer accumulation
window: one list per micro-batch, and a window that does not fit is refused.
``open_flat_window_at`` is the MP-SENet and SE-Mamba++ window: the epoch length
must be divisible by the accumulation, and the micro-batches are flattened.
``open_flat_window`` passes a trainer's data and accumulation into it.
DeepFilterNet3 retargets its cosine table while opening a window.
MossFormerGAN reads one micro-batch in ``fit``. Those two stay in their
trainers.

``close_training_run`` is the research ``end`` score, the final offer, and
``lifecycle.finish``. ``finish_training_run`` passes a trainer's attributes
into it for TF-GridNet, TF-Locoformer, DeepFilterNet3, and SE-Mamba++.
MP-SENet calls ``close_training_run`` with its progress counters. Passing
``prepare`` asks the final offer to broadcast rank buffers before it writes.
MossFormerGAN records ``halt_reason`` instead of ``stop_reason``, so it does
not call this.

``stop_after_step`` is the rank-agreed interrupt, including a logical-step
cap. ``stop_requested`` reads those two values from a trainer. Each trainer
keeps ``_stop_requested`` so a test can replace it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.results import RunResult
from src.experiments.stopping import control_requests_stop
from src.experiments.trainers import _close_epoch_view


def stop_after_step(
    logical_step: int,
    stop_after_logical_steps: int | None,
    runtime: object | None = None,
) -> bool:
    """Return whether every rank should stop for an interrupt or a step cap."""

    from src.experiments.signals import ranks_should_stop

    capped = stop_after_logical_steps is not None and logical_step >= stop_after_logical_steps
    return ranks_should_stop(runtime, extra=capped)


def stop_requested(trainer: Any, runtime: object | None = None) -> bool:
    """Read the step cap from a trainer and agree the interrupt."""

    return stop_after_step(trainer.logical_step, trainer.stop_after_logical_steps, runtime)


def close_training_run(
    reason: str,
    *,
    research: Any,
    runtime: Any,
    lifecycle: Any,
    data: Any,
    logical_step: int,
    optimizer_updates: Mapping[str, int],
    training_control: Any,
    set_stop_reason: Callable[[str], None],
    prepare: Any = None,
) -> RunResult:
    """Score a completed run, offer the final checkpoint, and finish.

    The saved trainer state still has the previous stop reason: ``set_stop_reason``
    runs after the final offer. Early stopping renames only the returned result.
    ``prepare=None`` does not broadcast buffers.
    """

    if reason == "completed":
        epoch = getattr(data, "epoch", None)
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            epoch = None
        extra = research.score(
            runtime,
            output_directory=lifecycle.output_directory,
            config_identity=lifecycle.config_identity,
            kind="end",
            epoch=epoch,
            logical_step=logical_step,
            stopping=training_control,
        )
        if extra:
            lifecycle.save_if_needed(
                "final",
                logical_step=logical_step,
                metrics=extra,
                prepare=prepare,
            )
        else:
            lifecycle.save_if_needed(
                "final",
                logical_step=logical_step,
                prepare=prepare,
            )
    set_stop_reason(reason)
    if reason == "completed" and control_requests_stop(training_control):
        reason = "early_stopping"
    return lifecycle.finish(
        RunResult(
            stop_reason=reason,
            logical_step=logical_step,
            optimizer_updates=dict(optimizer_updates),
            output_dir=lifecycle.output_directory,
        )
    )


def finish_training_run(trainer: Any, reason: str, *, prepare: Any = None) -> RunResult:
    """Finish a trainer that still stores its counters as attributes."""

    def set_stop_reason(value: str) -> None:
        trainer.stop_reason = value

    return close_training_run(
        reason,
        research=trainer.research,
        runtime=trainer.runtime,
        lifecycle=trainer.lifecycle,
        data=trainer.data,
        logical_step=trainer.logical_step,
        optimizer_updates=trainer.optimizer_updates,
        training_control=getattr(trainer, "training_control", None),
        set_stop_reason=set_stop_reason,
        prepare=prepare,
    )


def open_grouped_window(
    trainer: Any,
    component_id: str,
) -> tuple[int, int, Any, list[list[Batch]]]:
    """Take the next accumulation window without flattening its micro-batches."""

    epoch = int(trainer.data.epoch)
    index = int(trainer.data.update_index)
    view = trainer.data.train_epoch(epoch, start_update=index)
    try:
        if view.updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if index > view.updates:
            raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
        if index == view.updates:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
            )
        if index + trainer.accumulation > view.updates:
            raise RuntimeFailure(
                f"component {component_id}: accumulation {trainer.accumulation} "
                f"does not fit at update {index}"
            )
        planned = [view.batch(index + offset) for offset in range(trainer.accumulation)]
    except Exception:
        _close_epoch_view(view)
        raise
    return epoch, index, view, planned


def open_flat_window_at(
    data: Any,
    accumulation: int,
    component_id: str,
) -> tuple[int, int, Any, list[Batch]]:
    """Take the next accumulation window as one flat micro-batch list.

    The epoch length must be divisible by the accumulation. A short tail is
    not padded here.
    """

    epoch = int(data.epoch)
    index = int(data.update_index)
    view = data.train_epoch(epoch, start_update=index)
    try:
        if view.updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if view.updates % accumulation != 0:
            raise RuntimeFailure(
                f"component {component_id}: epoch micro-batches {view.updates} "
                f"are not divisible by accumulation {accumulation}"
            )
        if index > view.updates:
            raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
        if index == view.updates:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
            )
        if index + accumulation > view.updates:
            raise RuntimeFailure(
                f"component {component_id}: accumulation {accumulation} does not fit at update {index}"
            )
        batches: list[Batch] = []
        for offset in range(accumulation):
            batches.extend(view.batch(index + offset))
    except Exception:
        _close_epoch_view(view)
        raise
    return epoch, index, view, batches


def open_flat_window(trainer: Any, component_id: str) -> tuple[int, int, Any, list[Batch]]:
    """Open the flat window stored on a trainer."""

    return open_flat_window_at(trainer.data, trainer.accumulation, component_id)
