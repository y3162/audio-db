"""Official validation, PESQ selection, and both epoch-end schedulers.

The trainer opens a pending boundary and saves that fact first. ``run`` then
performs the official steps and returns what the trainer still has to record.
It does not save, score research, or move the data cursor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.artifacts import publish_best
from src.experiments.official.mp_senet.selection import apply_pesq_selection
from src.experiments.official.mp_senet.state import PENDING_BOUNDARY, MPTrainingProgress
from src.experiments.selectors import scalar_metrics


@dataclass(frozen=True)
class BoundaryResult:
    """What remains after validation and the epoch-end schedulers.

    ``metrics`` are the finite official scalars. ``roll_epoch`` tells the
    progress owner to advance the data cursor. ``validate`` is the flag that
    was stored with the pending boundary, including when validation was
    already done and this call did not score again.
    """

    metrics: dict[str, float]
    roll_epoch: bool
    validate: bool
    trainer_epoch: int


class OfficialBoundary:
    """One pending boundary: validation, best PESQ, then both schedulers."""

    def __init__(
        self,
        validator: Any,
        scheduler_g: Any,
        scheduler_d: Any,
        optim_g: Any,
        optim_d: Any,
        generator: Any,
        experiment_claim: str,
    ) -> None:
        self.validator = validator
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.generator = generator
        self.experiment_claim = experiment_claim

    def run(
        self,
        progress: MPTrainingProgress,
        runtime: Any,
        output_directory: Any,
        component_id: str,
        *,
        start_epoch: int,
        on_phase: Callable[[str], None],
    ) -> BoundaryResult:
        """Validate, select, then step both schedulers when the epoch is over.

        A validation interval does not step the schedulers. When the same
        update validates and ends the epoch, selection finishes before either
        ``ExponentialLR`` moves. ``on_phase("pending_scheduler")`` runs only
        in that case, before the scheduler steps.
        """

        if progress.phase != PENDING_BOUNDARY or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending GAN boundary")
        boundary = progress.boundary
        metrics: dict[str, float] = {}
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self.validator.evaluate(runtime, int(boundary["trainer_epoch"]), component_id)
            metrics = scalar_metrics(summary)
            decision = apply_pesq_selection(
                progress,
                int(boundary["trainer_epoch"]),
                summary,
                start_epoch=start_epoch,
            )
            progress.append_history(decision)
            publish_best(
                runtime,
                output_directory,
                decision,
                self.generator.unwrapped().export_for_inference(),
                self.experiment_claim,
            )
            progress.mark_validation(decision)
            if boundary["scheduler"]:
                progress.mark_phase("pending_scheduler")
                on_phase("pending_scheduler")
        if boundary["scheduler"] and not boundary["scheduler_done"]:
            self.scheduler_g.step()
            self.scheduler_d.step()
            progress.mark_scheduler_done(
                float(self.optim_g.param_groups[0]["lr"]),
                float(self.optim_d.param_groups[0]["lr"]),
            )
        result = BoundaryResult(
            metrics=metrics,
            roll_epoch=bool(boundary["epoch_done"]),
            validate=bool(boundary["validate"]),
            trainer_epoch=int(boundary["trainer_epoch"]),
        )
        progress.close_boundary()
        on_phase(progress.phase)
        return result
