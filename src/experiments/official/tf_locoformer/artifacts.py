"""TF-Locoformer inference files. These are not resume checkpoints.

Rank 0 writes one ``{epoch}epoch.pt`` per finished epoch, the loss n-best
average, and a separate ``valid.si_snr.best.pt``. The trainer ranks both
metrics and calls :func:`write_selection`. This module does not choose which
epoch is best. SI-SNR does not write an average.
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


def write_selection(
    directory: Path,
    exported: dict[str, torch.Tensor],
    espnet_epoch: int,
    by_loss: list[dict[str, Any]],
    by_si: list[dict[str, Any]],
    keep: int,
    logical_step: int,
    validation_profile: str,
) -> None:
    """Write the epoch export, loss n-best, and SI-SNR best. Rank 0 only.

    Loss uses the ESPnet n-best average. SI-SNR keeps its own best file and
    protects its n-best epoch files. It does not write an average. Equal
    scores already resolved to the earlier epoch in the ranked lists.
    """

    if not by_loss:
        raise RuntimeFailure("n-best selection received an empty loss history")
    directory.mkdir(parents=True, exist_ok=True)
    _save_state(directory / f"{espnet_epoch}epoch.pt", exported)
    loss_best = by_loss[0]
    if int(loss_best["epoch"]) == int(espnet_epoch):
        _save_state(directory / "valid.loss.best.pt", exported)
        _write_json(
            directory / "valid.loss.best.pt.json",
            {
                "criterion": "valid.loss.min",
                "epoch": int(espnet_epoch),
                "global_step": int(logical_step),
                "metric": "loss",
                "validation_profile": validation_profile,
                "value": float(loss_best["loss"]),
            },
        )
    if by_si and int(by_si[0]["epoch"]) == int(espnet_epoch):
        _save_state(directory / "valid.si_snr.best.pt", exported)
        _write_json(
            directory / "valid.si_snr.best.pt.json",
            {
                "criterion": "valid.si_snr.max",
                "epoch": int(espnet_epoch),
                "global_step": int(logical_step),
                "metric": "si_snr",
                "sign": "si_sdr_db",
                "validation_profile": validation_profile,
                "value": float(by_si[0]["si_snr"]),
            },
        )
    kept = {int(row["epoch"]) for row in by_loss + by_si}
    for path in directory.glob("*epoch.pt"):
        match = _EPOCH_FILE.match(path.name)
        if match and int(match.group(1)) not in kept:
            path.unlink()
    best_name = f"{int(loss_best['epoch'])}epoch.pt"
    if keep > 1 and len(by_loss) >= keep:
        epochs = [int(row["epoch"]) for row in by_loss]
        averaged = average_checkpoint_files([directory / f"{epoch}epoch.pt" for epoch in epochs])
        average_name = f"valid.loss.ave_{keep}best.pt"
        _save_state(directory / average_name, averaged)
        _write_json(
            directory / f"{average_name}.json",
            {
                "average": "espnet_nbest_float_mean_int_sum",
                "criterion": "valid.loss.min",
                "epochs": epochs,
                "files": [f"{epoch}epoch.pt" for epoch in epochs],
                "global_step": int(logical_step),
                "metric": "loss",
                "validation_profile": validation_profile,
                "values": [float(row["loss"]) for row in by_loss],
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
