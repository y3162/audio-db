"""Scale one optimizer window by the samples that window actually holds.

A full window whose every rank has the same positive micro-batch size uses
``loss / accumulation``. That division is bitwise-identical to the previous
loop. Any short window, partial micro-batch, or uneven rank uses

    scale_i = n_i * world_size / n_global

so an all-reduced mean of the local means is the mean over real samples.
An empty rank contributes scale 0 and must still run the collective.
"""

from __future__ import annotations


def window_is_uniform(rank_counts: list[list[int]], accumulation: int) -> bool:
    """True when every rank has ``accumulation`` equal, positive micro-batches."""

    if accumulation < 1 or not rank_counts:
        return False
    width = rank_counts[0][0] if rank_counts[0] else 0
    if width < 1:
        return False
    for counts in rank_counts:
        if len(counts) != accumulation or any(count != width for count in counts):
            return False
    return True


def sample_scales(
    counts: list[int],
    *,
    world_size: int,
    n_global: int,
) -> list[float]:
    """Per-micro multipliers. Empty micros are 0. ``n_global`` 0 is all zeros."""

    if world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if n_global < 0:
        raise ValueError("n_global must be >= 0")
    if n_global == 0:
        return [0.0 for _ in counts]
    return [count * world_size / n_global for count in counts]
