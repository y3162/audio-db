"""Order policies.

Each function returns this rank's dataset positions for one epoch. The same
arguments always return the same positions. Official random sequences that
depend on DataLoader workers are not reproduced; those differences are recorded
in ``POLICY_NOTES``.
"""

from __future__ import annotations

import math
import random

import torch

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import ORDER_POLICIES
from src.experiments.data.sampling.seeds import mix_seed

# What this implementation matches, and what it does not.
POLICY_NOTES: dict[str, dict[str, str]] = {
    "global_shuffle_stride_v1": {
        "matches": "algorithm",
        "detail": (
            "Same permutation as src.experiments.calc.sampler.epoch_indices: "
            "torch.randperm seeded by mix_seed('order', seed, epoch), then "
            "stride [rank::world]. A tail that does not fill every rank is "
            "given to the first ranks, one sample each, and is not copied."
        ),
    },
    "torch_distributed_sampler_v1": {
        "matches": "algorithm_when_divisible",
        "detail": (
            "torch.randperm(N) with generator seed 0+epoch, then [rank::world] "
            "over the complete stride groups. Leftover samples go to the first "
            "ranks once each. When N is divisible by world_size this is "
            "DistributedSampler(drop_last=True, seed 0). It does not pad. "
            "The training seed is not used."
        ),
    },
    "fixed_single_shuffle_v1": {
        "matches": "algorithm",
        "detail": (
            "One random.Random(1234).shuffle of the dataset positions, reused "
            "for every epoch. This is the MP-SENet 1-GPU Dataset shuffle. "
            "The sequence is CPython's Random.shuffle on this interpreter, "
            "not a promise about an older Python. The training seed is not used. "
            "world_size 1 keeps the full list. A larger world strides that shuffle "
            "and gives leftover samples to the first ranks once each."
        ),
    },
    "rank_mod_fixed_v1": {
        "matches": "algorithm",
        "detail": (
            "ClearerVoice DistributedSampler's complete stride groups: "
            "randperm(N//world) with seed 0+epoch, expand by world. Rank r "
            "owns a contiguous slice of those groups. Leftover indexes are "
            "appended once, to the first ranks, instead of repeating the front. "
            "When N is divisible by world_size the slices match the official "
            "sampler. Official non-DDP training uses DataLoader(shuffle=True); "
            "that sequence is not reproduced. The training seed is not used."
        ),
    },
    "per_rank_independent_v1": {
        "matches": "distribution",
        "detail": (
            "Each rank builds one permutation L_r with mix_seed('per_rank', "
            "training_seed, rank). Epoch order follows torch_distributed_sampler_v1 "
            "and indexes L_r, so ranks can overlap or omit samples. Official "
            "SEMamba++ shuffling is unseeded; the seed is audio-db determinism "
            "and is not an official random sequence. Official DistributedSampler "
            "uses drop_last=False and the DataLoader then uses drop_last=True. "
            "This policy truncates with drop_last=True before batching, so the "
            "official sampler's padded tail is not reproduced. world_size 1 still "
            "reshuffles every epoch; the official 1-GPU loader does not."
        ),
    },
    "dfn3_online_ceil_v1": {
        "matches": "distribution",
        "detail": (
            "DeepFilterNet3 is one process. audio-db shuffles sample indices with "
            "mix_seed('dfn3-order', protocol, snapshot, seed, split, epoch), then "
            "shards with [rank::world] and does not repeat samples to equalize "
            "rank lengths. The last batch is kept. "
            "world_size 1 does not pad, so the update count is ceil(N / batch), "
            "which is the official drop_last=false loader. Sample audio does not "
            "depend on rank or world size. This is not the official main-thread "
            "Rng that is seeded once with 42 and then consumed across epochs."
        ),
    },
    "espnet_chunk_v1": {
        "matches": "algorithm",
        "detail": (
            "Implemented by plan_chunk_epoch, not by a flat index list. "
            "The caller list is the base order (ESPnet's shape-file order is "
            "not available until generation). Shard with [rank::world], select "
            "num_samples utterances per rank with RandomState(cycle + seed), "
            "then chunk with a fresh RandomState(epoch + seed) on each rank. "
            "Cache is max(num_cache_chunks // world, per-rank batch). "
            "iterator_stop keeps min(rank batch counts). "
            "global_batch must divide by world_size. "
            "TF-GridNet at 0d136c5 and TF-Locoformer at 90eed8e use this "
            "iterator; the 90eed8e differences that remain at the DNS defaults "
            "do not change the split."
        ),
    },
}


def resolve_order_policy(order: str | list[str], world_size: int) -> str:
    """Pick the policy a protocol rule names for this world size.

    MP-SENet stores both the 1-GPU shuffle and the DistributedSampler. Any
    other list must contain exactly one policy.
    """
    if isinstance(order, str):
        names = [order]
    elif isinstance(order, list) and order and all(isinstance(item, str) for item in order):
        names = list(order)
    else:
        raise ReproductionError("rule order must be a policy name or a list of names")
    unknown = [name for name in names if name not in ORDER_POLICIES]
    if unknown:
        raise ReproductionError(f"unknown order policy: {', '.join(unknown)}")
    if (
        set(names) == {"fixed_single_shuffle_v1", "torch_distributed_sampler_v1"}
        and len(names) == 2
    ):
        if world_size < 1:
            raise ReproductionError("world_size must be positive")
        return "fixed_single_shuffle_v1" if world_size == 1 else "torch_distributed_sampler_v1"
    if len(names) == 1:
        return names[0]
    raise ReproductionError(f"cannot choose one order policy from {names}")


def rank_order(
    policy: str,
    size: int,
    *,
    seed: int,
    epoch: int,
    world_size: int,
    rank: int,
) -> list[int]:
    """Dataset positions for ``rank`` in ``epoch``.

    Positions refer to the case list before shuffling. Each position is
    returned on at most one rank.
    """
    _check_rank(size, world_size, rank, epoch)
    if policy == "global_shuffle_stride_v1":
        return _global_shuffle_stride(size, seed, epoch, world_size, rank)
    if policy == "torch_distributed_sampler_v1":
        return _torch_distributed(size, world_size, rank, epoch, sampler_seed=0)
    if policy == "fixed_single_shuffle_v1":
        return _fixed_single_shuffle(size, world_size, rank)
    if policy == "rank_mod_fixed_v1":
        return _rank_mod_fixed(size, world_size, rank, epoch, sampler_seed=0)
    if policy == "per_rank_independent_v1":
        return _per_rank_independent(size, seed, epoch, world_size, rank)
    if policy == "espnet_chunk_v1":
        raise ReproductionError(
            "espnet_chunk_v1 is planned by plan_chunk_epoch; a flat rank index list does not describe chunks"
        )
    raise ReproductionError(f"unknown order policy: {policy}")


def order_params(policy: str, seed: int) -> dict[str, int | bool | str]:
    """Parameters that define the permutation, for the checkpoint."""
    if policy == "global_shuffle_stride_v1":
        return {"mix": "order", "training_seed": seed, "reshuffle_each_epoch": True}
    if policy == "torch_distributed_sampler_v1":
        return {"sampler_seed": 0, "epoch_added": True, "drop_last": False, "pad": "none", "training_seed_used": False}
    if policy == "fixed_single_shuffle_v1":
        return {"shuffle_seed": 1234, "reshuffle_each_epoch": False, "training_seed_used": False}
    if policy == "rank_mod_fixed_v1":
        return {"sampler_seed": 0, "epoch_added": True, "pad": "none", "training_seed_used": False}
    if policy == "per_rank_independent_v1":
        return {"training_seed": seed, "sampler_seed": 0, "list": "mix_seed per_rank"}
    if policy == "espnet_chunk_v1":
        return {
            "rng": "numpy.RandomState(epoch + seed) per rank",
            "sampler": "RawSampler cycle + seed",
            "shard": "rank::world",
            "iterator_stop": "min_rank_batches",
            "training_seed": seed,
        }
    raise ReproductionError(f"unknown order policy: {policy}")


def _check_rank(size: int, world_size: int, rank: int, epoch: int) -> None:
    if size < 1:
        raise ReproductionError("training case list is empty")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ReproductionError(f"invalid rank {rank} for world_size {world_size}")
    if epoch < 0:
        raise ReproductionError("epoch must be zero or positive")


def _randperm(size: int, generator_seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(int(generator_seed))
    return torch.randperm(size, generator=generator).tolist()


def _cover_stride(order: list[int], world_size: int, rank: int) -> list[int]:
    """Stride a global order. Leftover items go once to the first ranks."""

    usable = (len(order) // world_size) * world_size
    base = order[:usable][rank::world_size] if usable else []
    tail = order[usable:]
    if rank < len(tail):
        return [*base, tail[rank]]
    return list(base)


def _global_shuffle_stride(size: int, seed: int, epoch: int, world_size: int, rank: int) -> list[int]:
    return _cover_stride(_randperm(size, mix_seed("order", seed, epoch)), world_size, rank)


def _torch_distributed(size: int, world_size: int, rank: int, epoch: int, *, sampler_seed: int) -> list[int]:
    return _cover_stride(_randperm(size, sampler_seed + epoch), world_size, rank)


def _fixed_single_shuffle(size: int, world_size: int, rank: int) -> list[int]:
    order = list(range(size))
    random.Random(1234).shuffle(order)
    return _cover_stride(order, world_size, rank)


def _rank_mod_fixed(size: int, world_size: int, rank: int, epoch: int, *, sampler_seed: int) -> list[int]:
    """Complete ClearerVoice stride groups, plus each leftover index once.

    When ``size`` is divisible by ``world_size`` the slice matches
    ``DistributedSampler`` with its default padding, because that padding
    adds nothing. A short dataset gives the first ranks one index and the
    others an empty list.
    """

    base = size // world_size
    chosen: list[int] = []
    if base >= 1:
        generator = torch.Generator()
        generator.manual_seed(sampler_seed + epoch)
        ind = torch.randperm(base, generator=generator) * world_size
        indices: list[int] = []
        for offset in range(world_size):
            indices.extend((ind + offset).tolist())
        start = rank * base
        chosen = indices[start : start + base]
    tail = size - base * world_size
    if rank < tail:
        chosen.append(base * world_size + rank)
    return chosen


def distributed_sampler_coverage(size: int, world_size: int) -> dict[str, int]:
    """Counts for ClearerVoice ``DistributedSampler`` with its default shuffle.

    ``num_samples`` is ``ceil(size / world_size)`` on every rank, so each rank
    has the same number of samples. The shuffle builds only complete stride
    groups, which drops ``size % world_size`` tail indexes. The sampler then
    repeats the front of that list until ``ceil(size / world_size) * world_size``.
    Those repeated indexes are scored again. DataLoader ``drop_last`` stays
    false, so a short last batch is kept and still counts as one batch.
    """
    if size < 1 or world_size < 1:
        raise ReproductionError("sampler coverage needs a positive dataset and world size")
    if size // world_size < 1:
        raise ReproductionError(
            f"rank_mod_fixed_v1 cannot assign {size} samples across world_size {world_size}"
        )
    num_samples = int(math.ceil(size / world_size))
    unique = (size // world_size) * world_size
    return {
        "num_samples_per_rank": num_samples,
        "dropped_tail": size - unique,
        "padded_duplicates": num_samples * world_size - unique,
        "scored_slots": num_samples * world_size,
        "unique_assigned": unique,
    }


def recipe_validation_positions(count: int, world_size: int, rank: int) -> tuple[list[int], str]:
    """Indexes one rank scores in recipe validation.

    When ``count >= world_size > 1``, this is ``rank_mod_fixed_v1``. When the
    validation set is shorter than the world, rank ``r`` receives index ``r``
    if ``r < count`` and otherwise an empty list. The same utterance is not
    scored on two ranks.
    """
    if count < 0 or world_size < 1 or rank < 0 or rank >= world_size:
        raise ReproductionError(
            f"invalid validation positions count={count} world_size={world_size} rank={rank}"
        )
    if world_size > 1 and count >= world_size:
        return (
            rank_order(
                "rank_mod_fixed_v1",
                count,
                seed=0,
                epoch=0,
                world_size=world_size,
                rank=rank,
            ),
            "rank_mod_fixed_v1",
        )
    if world_size > 1 and count < world_size:
        return ([rank] if rank < count else []), "cover_once"
    return list(range(count)), "single_process"


def sampler_batch_count(num_samples: int, batch_size: int) -> int:
    """Batches one rank yields when ``drop_last`` is false."""
    if num_samples < 1 or batch_size < 1:
        raise ReproductionError("batch count needs a positive sample count and batch size")
    return int(math.ceil(num_samples / batch_size))


def _per_rank_independent(size: int, seed: int, epoch: int, world_size: int, rank: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(mix_seed("per_rank", seed, rank))
    local = torch.randperm(size, generator=generator).tolist()
    positions = _torch_distributed(size, world_size, rank, epoch, sampler_seed=0)
    return [local[position] for position in positions]
