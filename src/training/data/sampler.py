"""Epoch order as a pure function of the seed.

The permutation is rebuilt from ``(seed, epoch)``. Resuming an epoch does not
require the sampler generator state, only how many micro-batches were finished.
"""

from __future__ import annotations

import torch

from src.training.core.rng import mix_seed
from src.training.errors import TrainingError


def epoch_indices(size: int, seed: int, epoch: int, world_size: int, rank: int) -> list[int]:
    """Return this rank's sample indexes for one epoch, in shuffle order.

    The tail that does not fill every rank is omitted so each rank sees the
    same count. The omitted indexes change with the epoch because the shuffle
    does.
    """
    if size < 1:
        raise TrainingError("training dataset is empty")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise TrainingError(f"invalid rank {rank} for world_size {world_size}")
    if epoch < 0:
        raise TrainingError("epoch must be zero or positive")
    generator = torch.Generator()
    generator.manual_seed(mix_seed("order", seed, epoch))
    order = torch.randperm(size, generator=generator).tolist()
    usable = (size // world_size) * world_size
    if usable == 0:
        raise TrainingError(f"dataset size {size} is smaller than world_size {world_size}")
    return order[:usable][rank::world_size]


def crop_start(case_id: str, seed: int, epoch: int, length: int, segment: int) -> int:
    """Crop offset for one case. The same id, seed, and epoch always match.

    The offset does not depend on which other samples were drawn first.
    """
    if length < 1 or segment < 1:
        raise TrainingError("crop length and segment must be positive")
    if length <= segment:
        return 0
    return mix_seed("crop", seed, epoch, case_id) % (length - segment + 1)
