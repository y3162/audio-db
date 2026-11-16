"""Identical trainer steps.

These functions replace copies whose meaning, errors, and order already
matched. They do not choose a model's update, validation, or save point.

``open_grouped_window`` is the TF-GridNet and TF-Locoformer accumulation
window: one list per micro-batch. A short tail is that many micro-batches,
not an error and not padding. ``open_flat_window_at`` is the MP-SENet and
SE-Mamba++ window. A full window still has ``accumulation`` micro-batches.
A short tail is shorter, and the micro-batches are flattened.
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


def snapshot_fields(obj: object) -> dict[str, Any]:
    """Copy the plain counters an update mutates. Modules and optimizers stay put."""

    saved: dict[str, Any] = {}
    for name, value in vars(obj).items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            saved[name] = value
        elif isinstance(value, list):
            saved[name] = list(value)
        elif isinstance(value, dict):
            saved[name] = dict(value)
    return saved


def restore_fields(obj: object, saved: dict[str, Any]) -> None:
    """Put ``snapshot_fields`` back, including lists that were appended in place."""

    for name, value in saved.items():
        current = getattr(obj, name)
        if isinstance(value, list) and isinstance(current, list):
            current.clear()
            current.extend(value)
        elif isinstance(value, dict) and isinstance(current, dict):
            current.clear()
            current.update(value)
        else:
            setattr(obj, name, value)


def finite_gradient_report(
    modules: list[Any],
    *,
    loss: float,
    sample_id: str,
    component_id: str,
) -> dict[str, Any]:
    """Report gradient connectivity after one probe backward.

    A model that sets ``find_unused_parameters`` may leave some parameters
    without a gradient. Every other trainable parameter needs one. No
    connected gradient is a failure either way.
    """

    missing = 0
    present = 0
    allow_missing = False
    for module in modules:
        policy = getattr(module, "policy", None)
        if getattr(policy, "find_unused_parameters", False) is True:
            allow_missing = True
        parameters = getattr(module, "parameters", ())
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing += 1
            else:
                present += 1
    if present < 1 or (missing and not allow_missing):
        raise RuntimeFailure(
            f"component {component_id}: preflight left {missing} trainable parameters without gradients"
        )
    return {
        "finite": True,
        "loss": float(loss),
        "parameters_missing_grad": missing,
        "parameters_with_grad": present,
        "sample_id": sample_id,
        "status": "ok",
    }


def window_divisor(accumulation: int, window: int) -> int:
    """Denominator for one optimizer window.

    A full window returns ``accumulation`` itself. A short tail returns how
    many micro-batches it actually holds. The two integers differ only then,
    so a full window still divides by the same value as before.
    """

    if window == accumulation:
        return accumulation
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window > accumulation:
        raise RuntimeFailure(f"window {window} is outside 1..{accumulation}")
    return window


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
        span = min(trainer.accumulation, view.updates - index)
        planned = [view.batch(index + offset) for offset in range(span)]
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

    A short tail returns fewer micro-batches. It is not padded and it is not
    dropped. A full window still returns ``accumulation`` of them.
    """

    epoch = int(data.epoch)
    index = int(data.update_index)
    view = data.train_epoch(epoch, start_update=index)
    try:
        if view.updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if index > view.updates:
            raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
        if index == view.updates:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
            )
        span = min(accumulation, view.updates - index)
        batches: list[Batch] = []
        for offset in range(span):
            batches.extend(view.batch(index + offset))
    except Exception:
        _close_epoch_view(view)
        raise
    return epoch, index, view, batches


def open_flat_window(trainer: Any, component_id: str) -> tuple[int, int, Any, list[Batch]]:
    """Open the flat window stored on a trainer."""

    return open_flat_window_at(trainer.data, trainer.accumulation, component_id)
