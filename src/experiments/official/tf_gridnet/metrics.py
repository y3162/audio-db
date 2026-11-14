"""TF-GridNet validation aggregation.

The trainer decides when validation runs. This module only reduces the
batch losses it is given.
"""

from __future__ import annotations

from src.experiments.errors import RuntimeFailure


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
