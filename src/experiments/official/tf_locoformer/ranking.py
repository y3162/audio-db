"""n-best files for TF-Locoformer.

``valid.loss`` is minimized and ``valid.si_snr`` is maximized. The trainer's
``_select`` calls this so a test can replace the whole selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.errors import RuntimeFailure
from src.experiments.official.tf_gridnet.evaluator import rank_validation_loss
from src.experiments.official.tf_locoformer.artifacts import write_selection
from src.experiments.official.tf_locoformer.selection import rank_validation_si_snr
from src.experiments.official.tf_locoformer.state import LocoformerProgress
from src.experiments.runtime import ProcessRuntime
from src.experiments.trainers import _rank


def select_checkpoints(
    progress: LocoformerProgress,
    runtime: ProcessRuntime,
    directory: Path,
    model: Any,
    espnet_epoch: int,
    loss: float,
    si_snr: float,
    keep_nbest: int,
    validation_profile: str,
) -> None:
    """Rank both criteria and write the official selection files."""

    progress.append_history({"epoch": int(espnet_epoch), "loss": float(loss), "si_snr": float(si_snr)})
    by_loss = rank_validation_loss(progress.history, keep_nbest)
    by_si = rank_validation_si_snr(progress.history, keep_nbest)
    error: str | None = None
    if _rank(runtime) == 0:
        try:
            write_selection(
                directory,
                model.unwrapped().export_for_inference(),
                espnet_epoch,
                by_loss,
                by_si,
                keep_nbest,
                progress.logical_step,
                validation_profile,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = runtime.gather_objects(error)
    failure = next((item for item in errors if item), None)
    if failure:
        raise RuntimeFailure(f"n-best selection failed: {failure}")
    runtime.barrier()
