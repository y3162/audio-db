"""TF-GridNet inference files. These are not resume checkpoints.

Rank 0 writes one ``{epoch}epoch.pt`` per finished epoch, ``valid.loss.best.pt``
when that epoch is the best, and the ESPnet n-best average links. The trainer
decides the ranking and calls :func:`write_nbest`. This module does not choose
a metric or a step.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from src.experiments.calc.average import average_checkpoint_files, remove_symlink, replace_symlink
from src.experiments.errors import RuntimeFailure

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
