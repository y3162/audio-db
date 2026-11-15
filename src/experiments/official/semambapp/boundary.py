"""Finish one SEMamba++ boundary. Validation does not select a best generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.official.semambapp.state import PENDING_BOUNDARY, SEMambaProgress
from src.experiments.official.semambapp.validation import SEMambaValidator
from src.experiments.runtime import ProcessRuntime
from src.experiments.selectors import scalar_metrics
from src.experiments.stopping import observe_validation


@dataclass(frozen=True)
class BoundaryOutcome:
    trainer_epoch: int
    metrics: dict[str, float]
    validated: bool


class SEMambaBoundary:
    """Validate when the boundary asks, then step both ExponentialLR schedulers together."""

    def __init__(self, validator: SEMambaValidator, *, optim_g: Any, optim_d: Any, scheduler_g: Any, scheduler_d: Any) -> None:
        self.validator = validator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d

    def run(
        self,
        progress: SEMambaProgress,
        runtime: ProcessRuntime,
        data: Any,
        component_id: str,
        on_phase: Any,
    ) -> BoundaryOutcome:
        if progress.phase != PENDING_BOUNDARY or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending GAN boundary")
        boundary = progress.boundary
        trainer_epoch = int(boundary["trainer_epoch"])
        metrics: dict[str, float] = {}
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self.validator.evaluate(runtime, data, trainer_epoch, component_id)
            metrics = scalar_metrics(summary)
            progress.history.append(
                {
                    "epoch": trainer_epoch,
                    "logical_step": int(progress.logical_step),
                    "reason": "semambapp_does_not_select_best_generator",
                    "selected": False,
                    "summary": summary,
                }
            )
            boundary["validation_done"] = True
            boundary["selection_done"] = True
            if boundary["scheduler"]:
                progress.phase = "pending_scheduler"
                on_phase(progress.phase)
        if boundary["scheduler"] and not boundary["scheduler_done"]:
            self.scheduler_g.step()
            self.scheduler_d.step()
            progress.scheduler_steps += 1
            boundary["scheduler_done"] = True
            boundary["lr_g"] = float(self.optim_g.param_groups[0]["lr"])
            boundary["lr_d"] = float(self.optim_d.param_groups[0]["lr"])
        if boundary["epoch_done"]:
            data.epoch = trainer_epoch + 1
            data.update_index = 0
        boundary["closed"] = True
        progress.close_boundary(boundary)
        on_phase(progress.phase)
        return BoundaryOutcome(trainer_epoch, metrics, bool(boundary["validate"]))


def published_metrics(
    outcome: BoundaryOutcome,
    runtime: ProcessRuntime,
    *,
    research: Any,
    lifecycle: Any,
    logical_step: int,
    training_control: Any,
) -> dict[str, float]:
    """Attach the research score after validation and both schedulers have finished."""

    if outcome.validated:
        observe_validation(
            training_control,
            outcome.metrics,
            step=logical_step,
            scoped=research.attached,
        )
    extra = research.score(
        runtime,
        output_directory=lifecycle.output_directory,
        config_identity=lifecycle.config_identity,
        kind="boundary",
        epoch=outcome.trainer_epoch,
        logical_step=logical_step,
        official=outcome.metrics if outcome.validated else {},
        stopping=training_control,
    )
    return {**outcome.metrics, **extra} if extra else outcome.metrics
