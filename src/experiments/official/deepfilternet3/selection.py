"""DeepFilterNet3 official best-loss patience.

A strict improvement resets the miss count. The trainer writes
``valid.loss.best`` through ``artifacts.py`` when the loss improved.
"""

from __future__ import annotations

from src.experiments.errors import RuntimeFailure


def note_early_stop(metric: float | None, count: int, loss: float, patience: int) -> tuple[float, int, bool]:
    """Reset the counter on a strict improvement. Stop at ``patience`` misses."""

    if patience < 1:
        raise RuntimeFailure("early_stopping_patience must be >= 1")
    if metric is None or loss < metric:
        return loss, 0, False
    count += 1
    return metric, count, count >= patience
