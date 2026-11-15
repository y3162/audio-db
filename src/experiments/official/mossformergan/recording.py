"""Record a finished MossFormerGAN pair. A skipped micro-batch does not advance the logical step."""

from __future__ import annotations

from typing import Any

from src.experiments.execution.journal import StepJournal
from src.experiments.official.mossformergan.state import MossProgress
from src.experiments.results import RunResult
from src.experiments.runtime import ProcessRuntime
from src.experiments.stopping import control_requests_stop


def note_micro_batch(
    progress: MossProgress,
    runtime: ProcessRuntime,
    journal: StepJournal,
    stepped: dict[str, Any] | None,
    *,
    epoch: int,
    research: Any,
    lifecycle: Any,
    training_control: Any,
    offer: Any,
) -> None:
    if stepped is None:
        offer("partial")
        return
    progress.note_closed_pair()
    progress.step_log.append(stepped)
    journal.write_metrics(
        {
            "discriminator": stepped["discriminator"],
            "generator": stepped["generator"],
            "logical_step": stepped["logical_step"],
            "loss": stepped["loss"],
            "sample_id": stepped["sample_id"],
        }
    )
    journal.write_sample(
        {"logical_step": progress.logical_step, "loss": stepped["loss"], "sample_id": stepped["sample_id"]}
    )
    step_metrics = research.score(
        runtime,
        output_directory=lifecycle.output_directory,
        config_identity=lifecycle.config_identity,
        kind="step",
        epoch=epoch,
        logical_step=progress.logical_step,
        stopping=training_control,
    )
    if step_metrics:
        offer("step", step_metrics)
    else:
        offer("step")


def finish_run(
    reason: str,
    *,
    research: Any,
    runtime: Any,
    lifecycle: Any,
    data: Any,
    progress: MossProgress,
    training_control: Any,
    prepare: Any,
) -> RunResult:
    if reason == "completed":
        epoch = getattr(data, "epoch", None)
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            epoch = None
        extra = research.score(
            runtime, output_directory=lifecycle.output_directory, config_identity=lifecycle.config_identity,
            kind="end", epoch=epoch, logical_step=progress.logical_step, stopping=training_control,
        )
        if extra:
            lifecycle.save_if_needed("final", logical_step=progress.logical_step, metrics=extra, prepare=prepare)
        else:
            lifecycle.save_if_needed("final", logical_step=progress.logical_step, prepare=prepare)
    if progress.halt_reason in {"initialized", "running"}:
        progress.halt_reason = reason
    if reason == "completed" and control_requests_stop(training_control):
        reason = "early_stopping"
    return lifecycle.finish(
        RunResult(
            stop_reason=reason, logical_step=progress.logical_step,
            optimizer_updates=dict(progress.optimizer_updates), output_dir=lifecycle.output_directory,
        )
    )
