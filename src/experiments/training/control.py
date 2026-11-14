"""Identical trainer steps.

These functions replace copies whose meaning, errors, and order already
matched. They do not choose a model's update, validation, or save point.

``open_grouped_window`` is the TF-GridNet and TF-Locoformer accumulation
window: one list per micro-batch, and a window that does not fit is refused.
``open_flat_window`` is the MP-SENet and SE-Mamba++ window: the epoch length
must be divisible by the accumulation, and the micro-batches are flattened.
DeepFilterNet3 retargets its cosine table while opening a window.
MossFormerGAN reads one micro-batch in ``fit``. Those two stay in their
trainers.

``finish_training_run`` is the research ``end`` score, the final offer, and
``lifecycle.finish`` used by TF-GridNet, TF-Locoformer, MP-SENet,
DeepFilterNet3, and SE-Mamba++. Passing ``prepare`` asks the final offer to
broadcast rank buffers before it writes. MossFormerGAN records ``halt_reason``
instead of ``stop_reason``, so it does not call this.

``stop_requested`` is the six official trainers' rank-agreed interrupt,
including a logical-step cap. Each trainer keeps the method so a test can
replace it.
"""

from __future__ import annotations

from typing import Any

from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.results import RunResult
from src.experiments.stopping import should_stop
from src.experiments.trainers import _close_epoch_view


def stop_requested(trainer: Any, runtime: object | None = None) -> bool:
    """Return whether every rank should stop for an interrupt or a step cap."""

    from src.experiments.signals import ranks_should_stop

    capped = trainer.stop_after_logical_steps is not None and trainer.logical_step >= trainer.stop_after_logical_steps
    return ranks_should_stop(runtime, extra=capped)


def finish_training_run(trainer: Any, reason: str, *, prepare: Any = None) -> RunResult:
    """Score a completed run, offer the final checkpoint, and finish.

    ``stop_reason`` keeps the reason from the caller. ``should_stop`` can
    still rename the returned result to ``early_stopping``. ``prepare=None``
    does not broadcast buffers.
    """

    if reason == "completed":
        epoch = getattr(trainer.data, "epoch", None)
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            epoch = None
        extra = trainer.research.score(
            trainer.runtime,
            output_directory=trainer.lifecycle.output_directory,
            config_identity=trainer.lifecycle.config_identity,
            kind="end",
            epoch=epoch,
            logical_step=trainer.logical_step,
            stopping=getattr(trainer, "training_control", None),
        )
        if extra:
            trainer.lifecycle.save_if_needed(
                "final",
                logical_step=trainer.logical_step,
                metrics=extra,
                prepare=prepare,
            )
        else:
            trainer.lifecycle.save_if_needed(
                "final",
                logical_step=trainer.logical_step,
                prepare=prepare,
            )
    trainer.stop_reason = reason
    if reason == "completed" and should_stop(trainer):
        reason = "early_stopping"
    return trainer.lifecycle.finish(
        RunResult(
            stop_reason=reason,
            logical_step=trainer.logical_step,
            optimizer_updates=dict(trainer.optimizer_updates),
            output_dir=trainer.lifecycle.output_directory,
        )
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


def open_flat_window(trainer: Any, component_id: str) -> tuple[int, int, Any, list[Batch]]:
    """Take the next accumulation window as one flat micro-batch list.

    The epoch length must be divisible by the accumulation. A short tail is
    not padded here.
    """

    epoch = int(trainer.data.epoch)
    index = int(trainer.data.update_index)
    view = trainer.data.train_epoch(epoch, start_update=index)
    try:
        if view.updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if view.updates % trainer.accumulation != 0:
            raise RuntimeFailure(
                f"component {component_id}: epoch micro-batches {view.updates} "
                f"are not divisible by accumulation {trainer.accumulation}"
            )
        if index > view.updates:
            raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
        if index == view.updates:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
            )
        if index + trainer.accumulation > view.updates:
            raise RuntimeFailure(
                f"component {component_id}: accumulation {trainer.accumulation} does not fit at update {index}"
            )
        batches: list[Batch] = []
        for offset in range(trainer.accumulation):
            batches.extend(view.batch(index + offset))
    except Exception:
        _close_epoch_view(view)
        raise
    return epoch, index, view, batches
