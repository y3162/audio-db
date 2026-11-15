"""DeepFilterNet3 validation, early stopping, and best-generator files.

Each validation utterance is scored alone and averaged with equal weight.
This module does not step the optimizer and does not offer a checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.deepfilternet3.dataset import clean_noisy
from src.experiments.official.deepfilternet3.loss import loss_tensor
from src.experiments.runtime import ProcessRuntime
from src.experiments.trainers import _rank
from src.se.models.deepfilternet3.dsp import DFStream

# --- Equal-weight validation. Rank 0 scores; every rank restores RNG and cursor. ---

def evaluate(
    runtime: ProcessRuntime,
    data: Any,
    model: Any,
    stream: DFStream | None,
    objective: torch.nn.Module,
    trainer_epoch: int,
    component_id: str,
) -> float:
    was_training = bool(model.training)
    cursor = (int(data.epoch), int(data.update_index))
    rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
    values: list[float] = []
    try:
        model.eval()
        if _rank(runtime) == 0:
            with torch.no_grad():
                for batch in data.validation_epoch(trainer_epoch):
                    if not isinstance(batch, Batch):
                        raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                    batch = runtime.move(batch)
                    clean, noisy = clean_noisy(batch)
                    for row in range(int(clean.shape[0])):
                        value = loss_tensor(
                            stream,
                            clean[row : row + 1],
                            noisy[row : row + 1],
                            model.unwrapped(),
                            objective,
                        )
                        if not bool(torch.isfinite(value).all()):
                            raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                        values.append(float(value.detach()))
        gathered = runtime.gather_objects(values)
    finally:
        restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
        if was_training:
            model.train()
        moved = (int(data.epoch), int(data.update_index))
        if moved != cursor:
            raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
    scores = next((item for item in gathered if item), None)
    if not scores:
        raise RuntimeFailure(f"component {component_id}: validation produced no utterances")
    return sum(scores) / len(scores)


# --- Patience on the equal-weight validation loss. ---

def note_early_stop(metric: float | None, count: int, loss: float, patience: int) -> tuple[float, int, bool]:
    """Reset the counter on a strict improvement. Stop at ``patience`` misses."""

    if patience < 1:
        raise RuntimeFailure("early_stopping_patience must be >= 1")
    if metric is None or loss < metric:
        return loss, 0, False
    count += 1
    return metric, count, count >= patience


# --- Best generator and the rfft record. Rank 0 writes. ---

def write_best(directory: Path, payload: dict[str, Any]) -> None:
    target = directory / "valid.loss.best.pt"
    temporary = directory / ".valid.loss.best.pt.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeFailure(f"best checkpoint failed: {exc}") from exc

def write_rfft(directory: Path, record: dict[str, Any]) -> None:
    path = Path(directory) / "rfft.json"
    temporary = path.with_name(".rfft.json.tmp")
    try:
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
