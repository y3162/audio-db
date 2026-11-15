"""Write one DeepFilterNet3 step row from the progress that was just committed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.deepfilternet3.artifacts import write_rfft
from src.experiments.official.deepfilternet3.state import DFProgress
from src.experiments.runtime import ProcessRuntime
from src.experiments.trainers import _rank


def note_step(
    journal: StepJournal,
    progress: DFProgress,
    *,
    applied: bool,
    epoch: int,
    loss: float,
    learning_rate: float,
    sample_id: str,
    weight_decay: float,
    schedule_index: int,
) -> None:
    journal.write_metrics(
        {
            "applied": applied,
            "epoch": epoch,
            "logical_step": progress.logical_step,
            "loss": loss,
            "lr": learning_rate,
            "optimizer_updates": dict(progress.optimizer_updates),
            "sample_id": sample_id,
            "weight_decay": weight_decay,
        }
    )
    journal.write_sample(
        {
            "epoch": epoch,
            "logical_step": progress.logical_step,
            "loss": loss,
            "lr": learning_rate,
            "sample_id": sample_id,
            "schedule_index": schedule_index,
            "weight_decay": weight_decay,
        }
    )


def record_rfft(runtime: ProcessRuntime, output_directory: Any) -> None:
    """Write the kernel selected on this host. Rank 0 writes the file."""

    problem = ""
    if _rank(runtime) == 0:
        try:
            from src.se.models.deepfilternet3.dsp import rfft_runtime_record

            write_rfft(Path(output_directory), rfft_runtime_record())
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
    if runtime.world_size > 1:
        reported = runtime.gather_objects(problem)
        problem = next((item for item in reported if item), "")
    if problem:
        raise RuntimeFailure(f"DeepFilterNet3 realfft kernel: {problem}")
