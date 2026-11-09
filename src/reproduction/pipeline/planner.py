"""Fixed-pair epoch plans.

The plan is a function of the case list, the protocol rule, the seed, the
epoch, and the batch shape. It does not read waveforms.
"""

from __future__ import annotations

from typing import Any

import torch

from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest
from src.reproduction.pipeline.chunking import ChunkEpochPlan, plan_chunk_epoch
from src.reproduction.pipeline.order import order_params, rank_order, resolve_order_policy
from src.reproduction.pipeline.seeds import mix_seed
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.pipeline.types import CropSpec, EpochPlan, MicroBatchPlan, SampleRef, UpdateGroup


def chunk_case_length(store: AssetStore, case_id: str) -> int:
    """Planned sample count from the realization document. The file is not decoded."""

    case = store.case(case_id)
    document = store.docs.get(case_id)
    if document is None:
        raise ReproductionError(f"case {case_id} has no realization document")
    planned = int(document["output"]["length"])
    recorded = int(case["frames"])
    if planned != recorded or planned < 1:
        raise ReproductionError(f"case {case_id} length is not the planned length")
    return planned


def plan_espnet_chunk(
    samples,
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
) -> ChunkEpochPlan:
    """TF-GridNet and TF-Locoformer chunk plan. Fixed-pair planning does not call this."""
    return plan_chunk_epoch(
        samples,
        espnet_epoch=espnet_epoch,
        seed=seed,
        world_size=world_size,
        num_samples=num_samples,
        chunk_length=chunk_length,
        global_batch=global_batch,
        num_cache_chunks=num_cache_chunks,
        shuffle=shuffle,
        shift_ratio=shift_ratio,
    )


def plan_fixed_pair(
    store: AssetStore,
    *,
    seed: int,
    epoch: int,
    world_size: int,
    global_batch: int,
    micro_batch: int,
    accumulation: int,
) -> EpochPlan:
    if epoch < 0 or world_size < 1 or micro_batch < 1 or accumulation < 1 or global_batch < 1:
        raise ReproductionError("epoch plan sizes must be positive")
    if micro_batch * world_size * accumulation != global_batch:
        raise ReproductionError(
            "micro_batch * world_size * accumulation must equal global_batch "
            f"({micro_batch} * {world_size} * {accumulation} != {global_batch})"
        )
    rule = store.dataset["rule_json"]
    policy = resolve_order_policy(rule["order"], world_size)
    window = int(rule["window"])
    normalize = str(rule["normalize"])
    pad = _pad_name(normalize)
    case_ids = store.case_ids("train.cases")
    lengths = [_pair_length(store, case_id, normalize) for case_id in case_ids]
    span = micro_batch * accumulation
    packed = []
    dropped = []
    for rank in range(world_size):
        positions = rank_order(
            policy,
            len(case_ids),
            seed=seed,
            epoch=epoch,
            world_size=world_size,
            rank=rank,
        )
        updates = len(positions) // span
        dropped.append(len(positions) - updates * span)
        groups = []
        kept = positions[: updates * span]
        cursor = 0
        for _ in range(updates):
            micros = []
            for _ in range(accumulation):
                chunk = kept[cursor : cursor + micro_batch]
                cursor += micro_batch
                refs = tuple(
                    _reference(
                        case_ids[position],
                        lengths[position],
                        plan_index=_plan_index(policy, local, rank, world_size, len(case_ids)),
                        window=window,
                        seed=seed,
                        epoch=epoch,
                        pad=pad,
                        full=False,
                    )
                    for local, position in enumerate(chunk, start=cursor - micro_batch)
                )
                micros.append(MicroBatchPlan(refs))
            groups.append(UpdateGroup(tuple(micros)))
        packed.append(tuple(groups))
    counts = [len(groups) for groups in packed]
    if not counts or min(counts) < 1:
        raise ReproductionError(
            f"epoch {epoch} does not contain a full update "
            f"(rank lengths {[len(rank_order(policy, len(case_ids), seed=seed, epoch=epoch, world_size=world_size, rank=rank)) for rank in range(world_size)]})"
        )
    updates = min(counts)
    rank_groups = tuple(tuple(groups[:updates]) for groups in packed)
    repeats = _repeat_count(rank_groups)
    return EpochPlan(
        epoch=epoch,
        world_size=world_size,
        updates=updates,
        rank_groups=rank_groups,
        plan_hash=hash_plan(rank_groups),
        order_policy=policy,
        stats={
            "order_policy": policy,
            "order_params": order_params(policy, seed),
            "dropped_tail": dropped,
            "truncated_updates": [count - updates for count in counts],
            "repeated_case_slots": repeats,
            "train_cases": len(case_ids),
            "torch": torch.__version__,
        },
    )


def hash_plan(rank_groups: tuple[tuple[UpdateGroup, ...], ...]) -> str:
    payload: list[Any] = []
    for groups in rank_groups:
        rank_rows = []
        for group in groups:
            micros = []
            for micro in group.micro_batches:
                micros.append([_ref_payload(sample) for sample in micro.samples])
            rank_rows.append(micros)
        payload.append(rank_rows)
    return digest(payload)


def validation_refs(
    store: AssetStore,
    *,
    seed: int,
    epoch: int,
    world_size: int,
    policy: str,
    rank: int | None = None,
) -> list[SampleRef]:
    """Validation samples for one rank, or the concatenated MossFormer view.

    MP-SENet's validation loader does not shuffle. MossFormerGAN's DDP
    validation sampler is epoch 0 of ``rank_mod_fixed_v1`` and is not moved
    by ``set_epoch``. Pass ``rank`` to take that rank's slice, including the
    official prefix padding and without the dropped tail. Omit ``rank`` to
    concatenate every rank, which is the single-process view of the same
    sampler. ``epoch`` affects ``per_epoch_random`` crops only.
    """
    rule = store.dataset["rule_json"]
    crop_name = str(rule["validation"]["crop"])
    window = int(rule["window"])
    normalize = str(rule["normalize"])
    pad = _pad_name(normalize)
    case_ids = store.case_ids("valid.cases")
    lengths = [_pair_length(store, case_id, normalize) for case_id in case_ids]
    if policy == "rank_mod_fixed_v1" and rank is not None:
        positions = rank_order(
            policy,
            len(case_ids),
            seed=seed,
            epoch=0,
            world_size=world_size,
            rank=rank,
        )
    else:
        positions = _validation_positions(policy, len(case_ids), seed, world_size)
    full = crop_name == "full_length"
    epoch_key: int | str = epoch if crop_name == "per_epoch_random" else "valid"
    if crop_name not in ("full_length", "per_epoch_random", "fixed"):
        raise ReproductionError(f"unknown validation crop {crop_name}")
    refs = []
    for local, position in enumerate(positions):
        refs.append(
            _reference(
                case_ids[position],
                lengths[position],
                plan_index=local,
                window=window,
                seed=seed,
                epoch=epoch_key,
                pad=pad,
                full=full,
            )
        )
    return refs


def _validation_positions(policy: str, size: int, seed: int, world_size: int) -> list[int]:
    if policy != "rank_mod_fixed_v1":
        return list(range(size))
    positions: list[int] = []
    for rank in range(world_size):
        positions.extend(
            rank_order(
                policy,
                size,
                seed=seed,
                epoch=0,
                world_size=world_size,
                rank=rank,
            )
        )
    return positions


def _reference(
    case_id: str,
    length: int,
    *,
    plan_index: int,
    window: int,
    seed: int,
    epoch: int | str,
    pad: str,
    full: bool,
) -> SampleRef:
    return SampleRef(
        case_id=case_id,
        index=plan_index,
        draw=None,
        length=length,
        crop=None if full else _crop(case_id, length, window, seed, epoch, pad),
        chunk=None,
    )


def _crop(case_id: str, length: int, window: int, seed: int, epoch: int | str, pad: str) -> CropSpec:
    if length >= window:
        start = mix_seed("crop", seed, epoch, case_id) % (length - window + 1)
        return CropSpec(start=start, length=window, pad_to=window, pad="none")
    if length < 1:
        raise ReproductionError(f"case {case_id} has no samples")
    return CropSpec(start=0, length=length, pad_to=window, pad=pad)


def _pair_length(store: AssetStore, case_id: str, normalize: str) -> int:
    case = store.case(case_id)
    if (store.dataset.get("rule_json") or {}).get("resample"):
        length = int(case["frames"])
        if length < 1:
            raise ReproductionError(f"case {case_id} has no resampled samples")
        return length
    clean = int(store.asset(case["clean_asset_id"])["frames"])
    noisy = int(store.asset(case["noisy_asset_id"])["frames"])
    if clean < 1 or noisy < 1:
        raise ReproductionError(f"case {case_id} has an empty asset")
    if normalize == "noisy_rms_unit_v1":
        return min(clean, noisy)
    if normalize == "independent_audio_norm_v1":
        if clean != noisy:
            raise ReproductionError(f"case {case_id} clean and noisy durations differ")
        return clean
    raise ReproductionError(f"normalization {normalize} is not implemented")


def _pad_name(normalize: str) -> str:
    if normalize == "noisy_rms_unit_v1":
        return "tail_zero"
    if normalize == "independent_audio_norm_v1":
        return "head_place"
    raise ReproductionError(f"normalization {normalize} is not implemented")


def fixed_pair_crop(
    case_id: str,
    length: int,
    window: int,
    seed: int,
    epoch: int | str,
    pad: str,
) -> CropSpec:
    """Public name for the fixed-pair crop. The formula remains ``_crop``."""

    return _crop(case_id, length, window, seed, epoch, pad)


def fixed_pair_case_length(store: AssetStore, case_id: str, normalize: str) -> int:
    """Public name for the fixed-pair case length. The formula remains ``_pair_length``."""

    return _pair_length(store, case_id, normalize)


def fixed_pair_pad_name(normalize: str) -> str:
    """Public name for the pad mode selected by normalization."""

    return _pad_name(normalize)


def fixed_pair_plan_index(policy: str, local: int, rank: int, world_size: int, size: int) -> int:
    """Public name for a sample's plan index. The formula remains ``_plan_index``."""

    return _plan_index(policy, local, rank, world_size, size)


def _plan_index(policy: str, local: int, rank: int, world_size: int, size: int) -> int:
    if policy == "rank_mod_fixed_v1":
        num_samples = (size + world_size - 1) // world_size
        return rank * num_samples + local
    return rank + local * world_size


def _repeat_count(rank_groups: tuple[tuple[UpdateGroup, ...], ...]) -> int:
    total = 0
    for groups in rank_groups:
        seen: set[str] = set()
        for group in groups:
            for micro in group.micro_batches:
                for sample in micro.samples:
                    if sample.case_id in seen:
                        total += 1
                    elif sample.case_id is not None:
                        seen.add(sample.case_id)
    return total


def _ref_payload(sample: SampleRef) -> dict[str, Any]:
    crop = sample.crop
    return {
        "case_id": sample.case_id,
        "index": sample.index,
        "draw": None if sample.draw is None else list(sample.draw),
        "length": sample.length,
        "crop": None
        if crop is None
        else {"start": crop.start, "length": crop.length, "pad_to": crop.pad_to, "pad": crop.pad},
        "chunk": sample.chunk,
    }
