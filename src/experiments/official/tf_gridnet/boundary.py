"""Epoch-end order for TF-GridNet.

Validation, ``ReduceLROnPlateau``, n-best selection, and the early-stop flag
run in that order. The trainer has already saved ``boundary_pending``. This
module does not score research and does not offer the closed boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.official.tf_gridnet.artifacts import write_nbest
from src.experiments.official.tf_gridnet.selection import early_stop, rank_validation_loss
from src.experiments.official.tf_gridnet.state import PENDING_VALIDATION, GridNetProgress
from src.experiments.official.tf_gridnet.validation import GridNetValidator
from src.experiments.runtime import ProcessRuntime
from src.experiments.trainers import _rank


@dataclass(frozen=True)
class BoundaryOutcome:
    """Validation loss after the boundary has closed and the cursor has moved."""

    trainer_epoch: int
    loss: float


class GridNetBoundary:
    """One pending epoch: validation, scheduler, n-best, then early stop."""

    def __init__(
        self,
        validator: GridNetValidator,
        scheduler: Any,
        optimizer: Any,
        model: Any,
        *,
        keep_nbest: int,
        early_stopping_patience: int | None,
    ) -> None:
        self.validator = validator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.keep_nbest = keep_nbest
        self.early_stopping_patience = early_stopping_patience

    def run(
        self,
        progress: GridNetProgress,
        runtime: ProcessRuntime,
        data: Any,
        output_directory: Path,
        component_id: str,
        on_phase: Callable[[str], None],
    ) -> BoundaryOutcome:
        """Close one pending epoch. ``on_phase`` sees each intermediate phase."""

        if progress.phase != PENDING_VALIDATION or progress.finished_epoch is None or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        trainer_epoch = int(progress.finished_epoch)
        espnet_epoch = trainer_epoch + 1
        summary = self.validator.evaluate(data, runtime, trainer_epoch, component_id)
        progress.note_validation(float(summary["loss"]), int(summary["weight"]))
        progress.phase = "pending_scheduler"
        on_phase(progress.phase)
        self.scheduler.step(summary["loss"])
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        progress.note_scheduler(learning_rate)
        progress.phase = "pending_selection"
        on_phase(progress.phase)
        _select(
            progress,
            runtime,
            output_directory,
            self.model,
            espnet_epoch,
            float(summary["loss"]),
            self.keep_nbest,
        )
        boundary = progress.boundary
        if boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        boundary["selection_done"] = True
        progress.close_boundary()
        progress.roll_epoch(data, trainer_epoch)
        on_phase(progress.phase)
        if self.early_stopping_patience is not None and early_stop(
            progress.history, espnet_epoch, self.early_stopping_patience
        ):
            progress.early_stopped = True
        return BoundaryOutcome(trainer_epoch=trainer_epoch, loss=float(summary["loss"]))


def _select(
    progress: GridNetProgress,
    runtime: ProcessRuntime,
    directory: Path,
    model: Any,
    espnet_epoch: int,
    loss: float,
    keep_nbest: int,
) -> None:
    """Rank valid.loss and write the official n-best files."""

    progress.append_history({"epoch": int(espnet_epoch), "loss": float(loss)})
    ranked = rank_validation_loss(progress.history, keep_nbest)
    error: str | None = None
    if _rank(runtime) == 0:
        try:
            write_nbest(
                directory,
                model.unwrapped().export_for_inference(),
                espnet_epoch,
                ranked,
                keep_nbest,
                progress.logical_step,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = runtime.gather_objects(error)
    failure = next((item for item in errors if item), None)
    if failure:
        raise RuntimeFailure(f"n-best selection failed: {failure}")
    runtime.barrier()
