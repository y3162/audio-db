"""Close one DeepFilterNet3 epoch: equal-weight validation, strict best, then patience."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.errors import RuntimeFailure
from src.experiments.official.deepfilternet3.artifacts import write_best
from src.experiments.official.deepfilternet3.selection import note_early_stop
from src.experiments.official.deepfilternet3.state import PENDING_VALIDATION, DFProgress
from src.experiments.official.deepfilternet3.validation import evaluate
from src.experiments.runtime import ProcessRuntime
from src.experiments.trainers import _rank
from src.se.models.deepfilternet3.dsp import DFStream


def finish(
    progress: DFProgress,
    runtime: ProcessRuntime,
    data: Any,
    model: Any,
    stream: DFStream | None,
    objective: torch.nn.Module,
    *,
    patience: int,
    component_id: str,
    output_directory: Any,
) -> float:
    if progress.phase != PENDING_VALIDATION or progress.finished_epoch is None:
        raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
    trainer_epoch = int(progress.finished_epoch)
    loss = evaluate(runtime, data, model, stream, objective, trainer_epoch, component_id)
    progress.plateau_metric, progress.plateau_count, stop = note_early_stop(
        progress.plateau_metric, progress.plateau_count, loss, patience
    )
    progress.history.append({"epoch": trainer_epoch, "loss": loss})
    improved = progress.best_loss is None or loss < progress.best_loss
    if improved:
        progress.best_loss = loss
        progress.best_epoch = trainer_epoch
    _write_best(runtime, model, output_directory, progress.best_epoch, progress.best_loss, progress.logical_step, improved)
    progress.phase = "in_epoch"
    progress.finished_epoch = None
    progress.boundary = None
    data.epoch = trainer_epoch + 1
    data.update_index = 0
    if stop:
        progress.early_stopped = True
    return float(loss)


def _write_best(
    runtime: ProcessRuntime,
    model: Any,
    directory: Any,
    best_epoch: int | None,
    best_loss: float | None,
    logical_step: int,
    improved: bool,
) -> None:
    if not improved or _rank(runtime) != 0:
        runtime.barrier()
        return
    write_best(
        directory,
        {
            "epoch": best_epoch,
            "generator": model.unwrapped().export_for_inference(),
            "logical_step": logical_step,
            "loss": best_loss,
        },
    )
    runtime.barrier()
