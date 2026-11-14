"""TF-GridNet official ranking and early stop.

``valid.loss`` is the only ranking key. File names and the n-best average
are written by ``artifacts.py``. This module does not touch the disk.
"""

from __future__ import annotations

from typing import Any

from src.experiments.errors import RuntimeFailure


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
