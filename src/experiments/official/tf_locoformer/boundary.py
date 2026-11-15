"""Epoch-end order for TF-Locoformer.

Validation runs, then the caller steps ``WarmupReduceLROnPlateau`` with
``step_validation``, then selection. This module does not call ``step_batch``
and does not call ``scheduler.step``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.official.tf_gridnet.selection import early_stop
from src.experiments.official.tf_locoformer.state import PENDING_VALIDATION, LocoformerProgress
from src.experiments.official.tf_locoformer.validation import LocoformerValidator
from src.experiments.runtime import ProcessRuntime


@dataclass(frozen=True)
class BoundaryOutcome:
    trainer_epoch: int
    loss: float
    si_snr: float


class LocoformerBoundary:
    """One pending epoch. The trainer supplies the plateau step and ``_select``."""

    def __init__(self, validator: LocoformerValidator, *, early_stopping_patience: int | None) -> None:
        self.validator = validator
        self.early_stopping_patience = early_stopping_patience

    def run(
        self,
        progress: LocoformerProgress,
        runtime: ProcessRuntime,
        data: Any,
        output_directory: Path,
        component_id: str,
        on_phase: Callable[[str], None],
        step_validation: Callable[[float], float],
        select: Callable[..., None],
    ) -> BoundaryOutcome:
        if progress.phase != PENDING_VALIDATION or progress.finished_epoch is None or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        trainer_epoch = int(progress.finished_epoch)
        espnet_epoch = trainer_epoch + 1
        summary = self.validator.evaluate(data, runtime, trainer_epoch, component_id)
        loss = float(summary["loss"])
        si_snr = float(summary["si_snr"])
        progress.note_validation(loss, si_snr, int(summary["weight"]))
        progress.phase = "pending_scheduler"
        on_phase(progress.phase)
        learning_rate = step_validation(loss)
        progress.note_scheduler(learning_rate)
        progress.phase = "pending_selection"
        on_phase(progress.phase)
        select(runtime, output_directory, espnet_epoch, loss, si_snr)
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
        return BoundaryOutcome(trainer_epoch=trainer_epoch, loss=loss, si_snr=si_snr)
