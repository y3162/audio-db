"""Checkpoint fields for a data plan.

The stored position is the number of optimizer updates that have finished.
Prefetched audio that the optimizer has not consumed is not part of this state.
"""

from __future__ import annotations

from typing import Any

from src.experiments.data.errors import ReproductionError


def require_resume_match(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse a checkpoint whose recomputed plan does not match ``saved``."""
    if not isinstance(saved, dict):
        raise ReproductionError("checkpoint data_state must be an object")
    if saved != current:
        keys = sorted(set(saved) | set(current))
        changed = [key for key in keys if saved.get(key) != current.get(key)]
        raise ReproductionError(
            "checkpoint data_state does not match this data plan: " + ", ".join(changed)
        )
