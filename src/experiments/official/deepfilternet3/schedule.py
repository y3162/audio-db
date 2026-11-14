"""DeepFilterNet3 cosine-table index.

The trainer applies the cosine step before the optimizer update. This
function only names the table row for the current epoch position.
"""

from __future__ import annotations

from src.experiments.errors import RuntimeFailure


def schedule_index(epoch: int, index_in_epoch: int, steps_per_epoch: int) -> int:
    """DeepFilterNet's table index. Past epochs are not summed."""

    if epoch < 0 or index_in_epoch < 0 or steps_per_epoch < 1:
        raise RuntimeFailure("schedule position is invalid")
    if index_in_epoch >= steps_per_epoch:
        raise RuntimeFailure(
            f"index {index_in_epoch} is outside an epoch of {steps_per_epoch} updates"
        )
    return epoch * steps_per_epoch + index_in_epoch
