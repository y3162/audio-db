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

from src.reproduction.errors import ReproductionError
from src.reproduction.ids import ORDER_POLICIES
from src.reproduction.pipeline.seeds import mix_seed

# What this implementation matches, and what it does not.
POLICY_NOTES: dict[str, dict[str, str]] = {
    "global_shuffle_stride_v1": {
        "matches": "algorithm",
        "detail": (
            "Same permutation as src.training.data.sampler.epoch_indices: "
            "torch.randperm seeded by mix_seed('order', seed, epoch), then "
            "drop the tail and take [rank::world]."
        ),
    },
    "torch_distributed_sampler_v1": {
        "matches": "algorithm_when_divisible",
        "detail": (
            "torch.randperm(N) with generator seed 0+epoch, keep N//world*world "
            "items, then [rank::world]. This is torch.utils.data.DistributedSampler "
            "with drop_last=True and seed 0. MP-SENet constructs DistributedSampler "
            "without drop_last, so the default pads instead of truncating. The two "
            "agree whenever N is divisible by world_size (the VoiceBank 11,572 "
            "list is divisible by 2 and by 4). The training seed is not used."
        ),
    },
    "fixed_single_shuffle_v1": {
        "matches": "algorithm",
        "detail": (
            "One random.Random(1234).shuffle of the dataset positions, reused "
            "for every epoch. This is the MP-SENet 1-GPU Dataset shuffle. "
            "The sequence is CPython's Random.shuffle on this interpreter, "
            "not a promise about an older Python. The training seed is not used. "
            "world_size 1 keeps the full list; a larger world then applies the "
            "same tail drop and stride, which the official 1-GPU loader never does."
        ),
    },
    "rank_mod_fixed_v1": {
        "matches": "algorithm",
        "detail": (
            "ClearerVoice DistributedSampler: randperm(N//world) with seed "
            "0+epoch, expand by world, then pad by repeating the front up to "
            "ceil(N/world) per rank. Rank r therefore owns indexes congruent "
            "to r (mod world) from the first (N//world)*world items. Leftover "
            "indexes at the tail are dropped and earlier indexes are repeated. "
            "Official non-DDP training uses DataLoader(shuffle=True) instead; "
            "that RandomSampler sequence is not reproduced. Validation order "
            "uses epoch 0 of this policy. The training seed is not used."
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
            "pads the list so every rank has the same length when world_size > 1 "
            "(torch DistributedSampler drop_last=False). The last batch is kept. "
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

    Positions refer to the case list before shuffling. A position may repeat
    when the official sampler pads.
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
        return {"sampler_seed": 0, "epoch_added": True, "drop_last": True, "training_seed_used": False}
    if policy == "fixed_single_shuffle_v1":
        return {"shuffle_seed": 1234, "reshuffle_each_epoch": False, "training_seed_used": False}
    if policy == "rank_mod_fixed_v1":
        return {"sampler_seed": 0, "epoch_added": True, "pad": "repeat_front", "training_seed_used": False}
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


def _global_shuffle_stride(size: int, seed: int, epoch: int, world_size: int, rank: int) -> list[int]:
    order = _randperm(size, mix_seed("order", seed, epoch))
    usable = (size // world_size) * world_size
    if usable == 0:
        raise ReproductionError(f"dataset size {size} is smaller than world_size {world_size}")
    return order[:usable][rank::world_size]


def _torch_distributed(size: int, world_size: int, rank: int, epoch: int, *, sampler_seed: int) -> list[int]:
    usable = (size // world_size) * world_size
    if usable == 0:
        raise ReproductionError(f"dataset size {size} is smaller than world_size {world_size}")
    order = _randperm(size, sampler_seed + epoch)
    return order[:usable][rank::world_size]


def _fixed_single_shuffle(size: int, world_size: int, rank: int) -> list[int]:
    order = list(range(size))
    random.Random(1234).shuffle(order)
    usable = (size // world_size) * world_size
    if usable == 0:
        raise ReproductionError(f"dataset size {size} is smaller than world_size {world_size}")
    return order[:usable][rank::world_size]


def _rank_mod_fixed(size: int, world_size: int, rank: int, epoch: int, *, sampler_seed: int) -> list[int]:
    """ClearerVoice ``DistributedSampler.__iter__`` for one rank."""
    num_samples = int(math.ceil(size / world_size))
    total = num_samples * world_size
    base = size // world_size
    if base < 1:
        raise ReproductionError(
            f"rank_mod_fixed_v1 cannot assign {size} samples across world_size {world_size}"
        )
    generator = torch.Generator()
    generator.manual_seed(sampler_seed + epoch)
    ind = torch.randperm(base, generator=generator) * world_size
    indices: list[int] = []
    for offset in range(world_size):
        indices.extend((ind + offset).tolist())
    if len(indices) > total:
        raise ReproductionError("rank_mod_fixed_v1 produced too many indexes")
    indices += indices[: total - len(indices)]
    if len(indices) != total:
        raise ReproductionError("rank_mod_fixed_v1 padding did not fill the epoch")
    start = rank * num_samples
    chosen = indices[start : start + num_samples]
    if len(chosen) != num_samples:
        raise ReproductionError("rank_mod_fixed_v1 rank slice has the wrong length")
    return chosen


def _per_rank_independent(size: int, seed: int, epoch: int, world_size: int, rank: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(mix_seed("per_rank", seed, rank))
    local = torch.randperm(size, generator=generator).tolist()
    positions = _torch_distributed(size, world_size, rank, epoch, sampler_seed=0)
    return [local[position] for position in positions]
