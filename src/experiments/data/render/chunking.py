"""ESPnet chunk-iterator plans for one or more ranks.

The plan is a function of the utterance list, the lengths, the seed, the
ESPnet epoch, and the batch shape. It does not read waveforms. Each rank
builds its own ``numpy.random.RandomState(epoch + seed)`` after the utterance
shard is chosen, which is what separate ESPnet processes do.

ESPnet's base order is the shape-file order from ``UnsortedBatchSampler``.
This function keeps the caller's list order and does not sort it. The
MS-SNSD dataset passes utterance ids in ``speech_mix_shape`` order.

``global_batch`` must be divisible by ``world_size``. The pinned ESPnet
factory gives the remainder to the lowest ranks instead. The published
TF-GridNet (4/2) and TF-Locoformer (4/4) recipes divide evenly, and an uneven
per-rank batch would change the loss reduction, so this planner refuses it.

Training and validation both stop at the shortest rank (ESPnet
``iterator_stop``). The utterance window for the next epoch still advances by
``num_samples``; dropped chunk batches are not carried over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest


@dataclass(frozen=True)
class PlannedChunk:
    case_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class ChunkEpochPlan:
    """Kept batches after ``iterator_stop``. ``raw_batch_counts`` is before it."""

    espnet_epoch: int
    world_size: int
    seed: int
    updates: int
    per_rank_batch: int
    per_rank_cache: int
    rank_batches: tuple[tuple[tuple[PlannedChunk, ...], ...], ...]
    raw_rank_batches: tuple[tuple[tuple[PlannedChunk, ...], ...], ...]
    raw_batch_counts: tuple[int, ...]
    discarded_batches: tuple[int, ...]
    discarded_tail_chunks: tuple[int, ...]
    selected: tuple[tuple[str, ...], ...]
    plan_hash: str
    stats: dict[str, Any]


def select_source_ids(
    case_ids: Sequence[str],
    *,
    epoch: int,
    seed: int,
    num_samples: int,
    shuffle: bool = True,
) -> list[str]:
    """One rank's utterances for an ESPnet epoch.

    ``epoch`` is one-based. The input sampler is the ESPnet ``RawSampler``:
    one fixed unshuffled list, reshuffled once per cycle with
    ``RandomState(cycle + seed)``.
    """
    if epoch < 1 or num_samples < 1 or not case_ids:
        raise ValueError("epoch, num_samples and input must be positive")
    count = len(case_ids)
    cursor = num_samples * (epoch - 1)
    result: list[str] = []
    remaining = num_samples
    while remaining:
        cycle, offset = divmod(cursor, count)
        indices = list(range(count))
        if shuffle:
            np.random.RandomState(cycle + seed).shuffle(indices)
        take = min(count - offset, remaining)
        result.extend(case_ids[i] for i in indices[offset : offset + take])
        remaining -= take
        cursor += take
    return result


def iterate_chunks(
    samples: Sequence[tuple[str, int]],
    *,
    epoch: int,
    seed: int,
    chunk_length: int,
    batch_size: int,
    cache_size: int,
    shuffle: bool = True,
    shift_ratio: float = 0.5,
    pad_short: bool = False,
) -> tuple[list[tuple[PlannedChunk, ...]], int]:
    """Chunk one rank. The returned integer is the discarded cache tail."""
    if chunk_length < 2 or batch_size < 1 or cache_size < 1:
        raise ValueError("invalid chunk or batch size")
    shift = int(chunk_length * shift_ratio)
    if shift < 1:
        raise ValueError("chunk shift must be positive")
    rng = np.random.RandomState(epoch + seed)
    cache: list[PlannedChunk] = []
    limit = max(cache_size, batch_size)
    batches: list[tuple[PlannedChunk, ...]] = []

    def flush() -> None:
        nonlocal cache
        if shuffle:
            order = np.arange(len(cache))
            rng.shuffle(order)
            cache = [cache[i] for i in order]
        while len(cache) >= batch_size:
            batches.append(tuple(cache[:batch_size]))
            cache = cache[batch_size:]

    for case_id, length in samples:
        if length <= chunk_length:
            if pad_short:
                if length < 1:
                    raise ReproductionError("cannot pad an empty utterance")
                cache.append(PlannedChunk(case_id, 0, chunk_length))
                if len(cache) > limit:
                    flush()
            continue
        # Match ESPnet's random choice even for a single candidate.
        width = int(rng.choice([chunk_length], size=1).item())
        count = (length - width) // shift + 1
        offset = rng.randint(0, (length - width) % shift + 1) if shuffle else 0
        cache.extend(
            PlannedChunk(case_id, offset + i * shift, offset + i * shift + width)
            for i in range(count)
        )
        if len(cache) > limit:
            flush()
    flush()
    return batches, len(cache)


def plan_chunk_epoch(
    samples: Sequence[tuple[str, int]],
    *,
    espnet_epoch: int,
    seed: int,
    world_size: int,
    num_samples: int | None,
    chunk_length: int,
    global_batch: int,
    num_cache_chunks: int,
    shuffle: bool,
    shift_ratio: float = 0.5,
    pad_short: bool = False,
) -> ChunkEpochPlan:
    """Plan every rank, then keep ``min(rank lengths)`` batches."""
    if espnet_epoch < 1 or seed < 0:
        raise ReproductionError("espnet epoch and seed must be non-negative, and epoch starts at 1")
    if world_size < 1 or global_batch < 1 or num_cache_chunks < 1:
        raise ReproductionError("world_size, global_batch and num_cache_chunks must be positive")
    if global_batch % world_size != 0:
        raise ReproductionError(
            f"global_batch {global_batch} is not divisible by world_size {world_size}. "
            "ESPnet would give the remainder to the lowest ranks; this planner refuses that split."
        )
    if len(samples) < world_size:
        raise ReproductionError(
            f"{len(samples)} utterances cannot be sharded across world_size {world_size}"
        )
    per_rank_batch = global_batch // world_size
    # ChunkIterFactory stores max(num_cache_chunks // world, batch_size).
    per_rank_cache = max(num_cache_chunks // world_size, per_rank_batch)
    rank_kept: list[tuple[tuple[PlannedChunk, ...], ...]] = []
    raw_counts: list[int] = []
    tails: list[int] = []
    selected_rows: list[tuple[str, ...]] = []
    for rank in range(world_size):
        shard = list(samples[rank::world_size])
        if num_samples is None:
            chosen = list(shard)
            if shuffle:
                order = list(range(len(chosen)))
                np.random.RandomState(espnet_epoch + seed).shuffle(order)
                chosen = [chosen[i] for i in order]
        else:
            ids = select_source_ids(
                [case_id for case_id, _ in shard],
                epoch=espnet_epoch,
                seed=seed,
                num_samples=num_samples,
                shuffle=shuffle,
            )
            length_of = {case_id: length for case_id, length in shard}
            chosen = [(case_id, int(length_of[case_id])) for case_id in ids]
        batches, tail = iterate_chunks(
            chosen,
            epoch=espnet_epoch,
            seed=seed,
            chunk_length=chunk_length,
            batch_size=per_rank_batch,
            cache_size=per_rank_cache,
            shuffle=shuffle,
            shift_ratio=shift_ratio,
            pad_short=pad_short,
        )
        rank_kept.append(tuple(batches))
        raw_counts.append(len(batches))
        tails.append(tail)
        selected_rows.append(tuple(case_id for case_id, _ in chosen))
    updates = min(raw_counts) if raw_counts else 0
    kept = tuple(tuple(batches[:updates]) for batches in rank_kept)
    discarded = tuple(count - updates for count in raw_counts)
    plan_data = {
            "algorithm": "espnet_chunk_v1",
            "espnet_epoch": espnet_epoch,
            "seed": seed,
            "world_size": world_size,
            "num_samples": num_samples,
            "chunk_length": chunk_length,
            "shift": int(chunk_length * shift_ratio),
            "per_rank_batch": per_rank_batch,
            "per_rank_cache": per_rank_cache,
            "shuffle": shuffle,
            "iterator_stop": "min_rank_batches",
            "discarded_batches": list(discarded),
            "ranks": [
                [[{"case_id": chunk.case_id, "start": chunk.start, "stop": chunk.stop} for chunk in batch] for batch in batches]
                for batches in kept
            ],
        }
    if pad_short:
        plan_data["pad_short"] = True
    plan_hash = digest(plan_data)
    return ChunkEpochPlan(
        espnet_epoch=espnet_epoch,
        world_size=world_size,
        seed=seed,
        updates=updates,
        per_rank_batch=per_rank_batch,
        per_rank_cache=per_rank_cache,
        rank_batches=kept,
        raw_rank_batches=tuple(rank_kept),
        raw_batch_counts=tuple(raw_counts),
        discarded_batches=discarded,
        discarded_tail_chunks=tuple(tails),
        selected=tuple(selected_rows),
        plan_hash=plan_hash,
        stats={
            "order_policy": "espnet_chunk_v1",
            "base_order": "caller_list",
            "iterator_stop": "min_rank_batches",
            "rng": "RandomState(epoch + seed) independently on each rank",
            "model_seed": "ESPnet set_all_random_seed(seed + epoch) is the same on every rank",
            "cache_rule": "max(num_cache_chunks // world_size, per_rank_batch)",
            "batch_remainder": "refused",
            "utterance_cursor": None if num_samples is None else num_samples * espnet_epoch,
            "raw_batch_counts": list(raw_counts),
            "discarded_batches": list(discarded),
            "discarded_tail_chunks": list(tails),
        },
    )
