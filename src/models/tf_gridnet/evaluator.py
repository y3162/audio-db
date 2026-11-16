"""TF-GridNet validation, valid.loss ranking, and n-best inference files.

The injected objective supplies the validation scalar. This module does not
step the scheduler and does not offer a checkpoint. ``valid.si_snr`` is not a
selection key.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from src.experiments.calc.average import average_checkpoint_files, remove_symlink, replace_symlink
from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.training.objectives import call_objective

# --- Rank-local validation. The training cursor and RNG are restored. ---

class GridNetValidator:
    """Evaluate one epoch with the injected objective."""

    def __init__(self, model: PreparedModel, objective: torch.nn.Module) -> None:
        self.model = model
        self.objective = objective

    def evaluate(self, data: object, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, float | int]:
        """Return the weighted loss. The training cursor and RNG are restored."""

        module = self.model
        was_training = bool(module.training)
        cursor = (int(data.epoch), int(data.update_index))
        rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
        local_sum = 0.0
        local_count = 0
        try:
            module.eval()
            with torch.no_grad():
                for batch in data.validation_epoch(trainer_epoch):
                    if not isinstance(batch, Batch):
                        raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                    batch = runtime.move(batch)
                    prediction = module.inference(batch.inputs)
                    value = _objective_total(self.objective, prediction, batch.targets)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise RuntimeFailure(f"component {component_id}: validation loss is not a scalar")
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                    weight = int(batch.inputs.shape[0])
                    local_sum += float(value.detach()) * weight
                    local_count += weight
            gathered = runtime.gather_objects({"count": local_count, "sum": local_sum})
            total = 0.0
            count = 0
            for item in gathered:
                total += float(item["sum"])
                count += int(item["count"])
            if count < 1:
                raise RuntimeFailure(f"component {component_id}: validation produced no batches")
            loss = total / count
        finally:
            restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
            if was_training:
                module.train()
            moved = (int(data.epoch), int(data.update_index))
            if moved != cursor:
                raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return {"loss": loss, "weight": count}


def _objective_total(objective: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the total of the injected objective."""

    try:
        result = call_objective(objective, {"estimate": prediction, "target": target})
    except ValueError as exc:
        raise RuntimeFailure(str(exc)) from exc
    return result.total

# --- Weighted mean of validation parts. ---

def weighted_validation_loss(parts: list[tuple[float, int]]) -> tuple[float, int]:
    """Mean of batch-mean losses weighted by sample count.

    A rank mean of means is a different number when the ranks do not hold
    the same number of samples. The weight is the number of waveforms in
    the batch, which is the count ``_reduce_weighted_reports`` multiplies
    by ``len(case_id)``.
    """

    if not parts:
        raise RuntimeFailure("validation produced no batches")
    total = 0.0
    count = 0
    for loss, weight in parts:
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise RuntimeFailure("validation weight must be a positive integer")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise RuntimeFailure("validation loss must be a number")
        total += float(loss) * weight
        count += weight
    if count < 1:
        raise RuntimeFailure("validation produced no batches")
    return total / count, count

# --- n-best by valid.loss, and early stop. ---

def rank_validation_loss(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Lowest ``valid.loss`` first. Equal losses keep the earlier epoch."""

    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise RuntimeFailure("keep_nbest must be a positive integer")
    ordered = sorted(history, key=lambda row: (float(row["loss"]), int(row["epoch"])))
    return ordered[:keep]


def early_stop(history: list[dict[str, Any]], espnet_epoch: int, patience: int) -> bool:
    """Stop when the best epoch is more than ``patience`` epochs behind.

    The best row is the lowest loss, then the earlier epoch. This is the
    chunk early-stopping rule. It does not read ``valid.si_snr``.
    """

    if not history:
        return False
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise RuntimeFailure("early_stopping_patience must be an integer >= 0")
    best = min(history, key=lambda row: (float(row["loss"]), int(row["epoch"])))
    return int(espnet_epoch) - int(best["epoch"]) > patience

# --- Official n-best inference weights. These are not resume checkpoints. ---

_EPOCH_FILE = re.compile(r"^(\d+)epoch\.pt$")


def write_nbest(
    directory: Path,
    exported: dict[str, torch.Tensor],
    espnet_epoch: int,
    ranked: list[dict[str, Any]],
    keep: int,
    logical_step: int,
) -> None:
    """Write the epoch export and the valid.loss n-best files. Rank 0 only."""

    if not ranked:
        raise RuntimeFailure("n-best selection received an empty history")
    _save_state(directory / f"{espnet_epoch}epoch.pt", exported)
    best = ranked[0]
    if int(best["epoch"]) == int(espnet_epoch):
        _save_state(directory / "valid.loss.best.pt", exported)
        _write_json(
            directory / "valid.loss.best.pt.json",
            {
                "epoch": int(espnet_epoch),
                "global_step": int(logical_step),
                "metric": "loss",
                "value": float(best["loss"]),
            },
        )
    kept = {int(row["epoch"]) for row in ranked}
    for path in directory.glob("*epoch.pt"):
        match = _EPOCH_FILE.match(path.name)
        if match and int(match.group(1)) not in kept:
            path.unlink()
    best_name = f"{int(best['epoch'])}epoch.pt"
    if keep > 1 and len(ranked) >= keep:
        epochs = [int(row["epoch"]) for row in ranked]
        averaged = average_checkpoint_files([directory / f"{epoch}epoch.pt" for epoch in epochs])
        average_name = f"valid.loss.ave_{keep}best.pt"
        average_path = directory / average_name
        _save_state(average_path, averaged)
        _write_json(
            directory / f"{average_name}.json",
            {
                "average": "espnet_nbest_float_mean_int_sum",
                "criterion": "valid.loss.min",
                "epochs": epochs,
                "files": [f"{epoch}epoch.pt" for epoch in epochs],
                "global_step": int(logical_step),
                "metric": "loss",
                "values": [float(row["loss"]) for row in ranked],
            },
        )
        replace_symlink(directory / "valid.loss.ave.pt", average_name)
        # Left from the epochs before n-best existed. Its epoch file may be gone by now.
        remove_symlink(directory / "valid.loss.ave_1best.pt")
        return
    replace_symlink(directory / "valid.loss.ave_1best.pt", best_name)
    replace_symlink(directory / "valid.loss.ave.pt", "valid.loss.ave_1best.pt")


def _save_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
