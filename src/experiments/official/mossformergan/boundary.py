"""Close one MossFormerGAN epoch after validation and the plateau decision."""

from __future__ import annotations

from typing import Any, Callable

from src.experiments.errors import RuntimeFailure
from src.experiments.official.mossformergan.microbatch import MicroBatch
from src.experiments.official.mossformergan.plateau import apply_plateau
from src.experiments.official.mossformergan.state import PENDING_BOUNDARY, MossProgress
from src.experiments.official.mossformergan.validation import evaluate
from src.experiments.runtime import ProcessRuntime
from src.experiments.selectors import scalar_metrics
from src.experiments.stopping import observe_validation


def open_boundary(progress: MossProgress, *, epoch: int, validate: bool) -> None:
    if progress.phase != "in_epoch":
        raise RuntimeFailure("an epoch boundary is already pending")
    progress.boundary = {
        "closed": False,
        "plateau_done": False,
        "trainer_epoch": int(epoch),
        "validate": validate,
        "validation_done": False,
        "validation_summary": None,
    }
    progress.phase = PENDING_BOUNDARY


def finish_boundary(
    progress: MossProgress,
    runtime: ProcessRuntime,
    data: Any,
    micro: MicroBatch,
    *,
    component_id: str,
    max_logical_steps: int,
    n_fft: int,
    hop: int,
    win: int,
    sample_rate: int,
    offer: Callable[..., None],
    research: Any,
    lifecycle: Any,
    training_control: Any,
) -> None:
    if progress.phase != PENDING_BOUNDARY or progress.boundary is None:
        raise RuntimeFailure(f"component {component_id}: there is no pending MossFormerGAN boundary")
    boundary = progress.boundary
    if boundary["validate"] and not boundary["validation_done"]:
        summary = evaluate(
            runtime, data, micro.generator, micro.discriminator,
            trainer_epoch=int(boundary["trainer_epoch"]), n_fft=n_fft, hop=hop, win=win, sample_rate=sample_rate,
        )
        boundary["validation_summary"] = summary
        boundary["validation_done"] = True
        offer("boundary_pending")
    if boundary["validate"] and not boundary["plateau_done"]:
        summary = boundary.get("validation_summary")
        if not isinstance(summary, dict):
            raise RuntimeFailure(f"component {component_id}: validation summary is missing")
        boundary["plateau_action"] = apply_plateau(
            progress, summary, generator=micro.generator, discriminator=micro.discriminator,
            optim_g=micro.optim_g, optim_d=micro.optim_d,
        )
        boundary["plateau_done"] = True
    micro.discard_window()
    no_success = progress.counters["micro_batches_succeeded"] == progress.epoch_success_start
    no_step = progress.counters["optimizer_updates"] == progress.epoch_update_start
    if no_step:
        progress.epochs_without_step += 1
    else:
        progress.epochs_without_step = 0
    data.epoch = int(boundary["trainer_epoch"]) + 1
    data.update_index = 0
    boundary["closed"] = True
    metrics = scalar_metrics(boundary.get("validation_summary"))
    progress.completed_boundaries.append(dict(boundary))
    progress.boundary = None
    progress.phase = "in_epoch"
    if progress.stop_training:
        if progress.halt_reason == "running":
            progress.halt_reason = "plateau_stop"
    elif no_success:
        progress.halt_reason = "no_successful_micro_batch"
        progress.stop_training = True
    elif progress.epochs_without_step >= max_logical_steps:
        progress.halt_reason = "accumulation_window_never_closed"
        progress.stop_training = True
    if boundary["validate"]:
        observe_validation(training_control, metrics, step=progress.logical_step, scoped=research.attached)
        extra = research.score(
            runtime,
            output_directory=lifecycle.output_directory,
            config_identity=lifecycle.config_identity,
            kind="boundary",
            epoch=int(boundary["trainer_epoch"]),
            logical_step=progress.logical_step,
            official=metrics,
            stopping=training_control,
        )
        if extra:
            metrics = {**metrics, **extra}
    offer("boundary", metrics)
