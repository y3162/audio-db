"""Research sampler ``weighted_categorical_v1``.

This is not an official ``rank_order`` policy. The numeric contract is fixed in
``docs/training/experiment-framework/DATA_MIGRATION_PLAN.md`` before this
module is used by ``OrderSampler``. Unweighted epochs do not call it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import gcd, isfinite
from functools import reduce
from typing import Any

from src.experiments.errors import DataError
from src.reproduction.pipeline.seeds import mix_seed

ALGORITHM = "weighted_categorical_v1"
RNG_NAME = "splitmix64-v1"
RANK_SPLIT = "stride_drop_tail_v1"
WEIGHT_SCALE = 1_000_000
_MASK = (1 << 64) - 1


def weighted_order_parameters(seed: int) -> dict[str, Any]:
    """Plan-hash parameters for one weighted epoch. Official params are not reused."""

    return {
        "algorithm": ALGORITHM,
        "draws": "training_count",
        "rank_split": RANK_SPLIT,
        "replacement": True,
        "rng": RNG_NAME,
        "training_seed": int(seed),
        "weight_scale": WEIGHT_SCALE,
    }


def normalize_weight_table(weights: object, training_ids: Sequence[str]) -> dict[str, float]:
    """Require one finite non-negative weight for every training sample id.

    Effective values use the same 6-decimal text as ``ids.digest``. Missing
    keys are not filled in. A total of zero is rejected.
    """

    if isinstance(weights, (str, bytes)) or not isinstance(weights, Mapping):
        raise DataError("sampling weights must be a mapping")
    expected = tuple(training_ids)
    if len(set(expected)) != len(expected):
        raise DataError("training sample ids are not unique")
    if len(weights) != len(expected) or set(weights) != set(expected):
        raise DataError("sampling weights must list every training sample once")
    normalized: dict[str, float] = {}
    total = 0
    for sample_id in expected:
        if not isinstance(sample_id, str):
            raise DataError("training sample ids must be strings")
        normalized[sample_id] = _effective_weight(weights[sample_id])
        total += _micro_weight(normalized[sample_id])
    if total <= 0:
        raise DataError("sampling weights must have a positive total")
    return normalized


def categorical_draws(weights: Sequence[float], seed: int, epoch: int, draws: int) -> list[int]:
    """With-replacement indexes. ``weights`` follows the canonical sample order."""

    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise DataError("weighted draw count must be a positive integer")
    micros = [_micro_weight(_effective_weight(value)) for value in weights]
    support = [index for index, weight in enumerate(micros) if weight > 0]
    masses = [micros[index] for index in support]
    if not masses:
        raise DataError("sampling weights must have a positive total")
    divisor = reduce(gcd, masses)
    masses = [weight // divisor for weight in masses]
    total = sum(masses)
    prefix: list[int] = []
    running = 0
    for weight in masses:
        running += weight
        prefix.append(running)
    bits = max(total.bit_length(), 1)
    width = ((bits + 63) // 64) * 64
    space = 1 << width
    span = (space // total) * total
    stream = mix_seed(ALGORITHM, int(seed), int(epoch))
    chosen: list[int] = []
    counter = 0
    words = width // 64
    while len(chosen) < draws:
        value = 0
        for _ in range(words):
            counter += 1
            value = (value << 64) | _splitmix64((stream + counter) & _MASK)
        if value < span:
            chosen.append(support[_find_prefix(prefix, value % total)])
    return chosen


def rank_sample_positions(
    weights: Sequence[float],
    seed: int,
    epoch: int,
    world_size: int,
    rank: int,
) -> list[int]:
    """Global draws of length ``len(weights)``, then ``stride_drop_tail_v1``."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise DataError("world_size must be a positive integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
        raise DataError("rank is outside this world")
    global_positions = categorical_draws(weights, seed, epoch, len(weights))
    usable = (len(global_positions) // world_size) * world_size
    return global_positions[:usable][rank::world_size]


def _effective_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError("sampling weight must be a finite non-negative number")
    number = float(value)
    if not isfinite(number):
        raise DataError("sampling weight must be a finite non-negative number")
    text = format(number, ".6f")
    if text.startswith("-"):
        raise DataError("sampling weight must be a finite non-negative number")
    return float(text)


def _micro_weight(value: float) -> int:
    text = format(value, ".6f")
    whole, frac = text.split(".")
    return int(whole) * WEIGHT_SCALE + int(frac)


def _splitmix64(value: int) -> int:
    mixed = (value + 0x9E3779B97F4A7C15) & _MASK
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _MASK
    return (mixed ^ (mixed >> 31)) & _MASK


def _find_prefix(prefix: Sequence[int], bucket: int) -> int:
    low = 0
    high = len(prefix) - 1
    while low < high:
        mid = (low + high) // 2
        if bucket < prefix[mid]:
            high = mid
        else:
            low = mid + 1
    return low
