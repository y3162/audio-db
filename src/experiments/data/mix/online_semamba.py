"""Snapshot-backed SEMamba++ online degradation.

The planner does not open DuckDB and does not write waveforms. Each logical
sample is rendered from ``(protocol, snapshot, seed, split, epoch, index)``.
Rank chooses which speech indexes appear. It does not change a given index.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest
from src.experiments.data.paths import (
    DEFAULT_CORPORA,
    DEFAULT_DERIVED,
    DEFAULT_GENERATED,
    DEFAULT_SNAPSHOTS,
)
from src.experiments.data.render.assemble import assemble
from src.experiments.data.render.clips import SemambaClip as _AssetClip
from src.experiments.data.assets.loader import GroupLoader
from src.experiments.data.sampling.order import order_params, rank_order
from src.experiments.data.snapshot_provenance import snapshot_provenance_parts
from src.experiments.data.sampling.state import require_resume_match
from src.experiments.data.assets.store import AssetStore
from src.experiments.data.fixed_pair_stream import EpochStream, find_snapshot_dir
from src.experiments.data.render.types import EpochPlan, MicroBatchPlan, SampleRef, UpdateGroup
from src.experiments.data.protocols.online_semamba import (
    SAMPLE_RATE,
    SEGMENT_SIZE,
    render_example,
    sample_seed,
)
from src.experiments.data.snapshot import load_snapshot

POLICY = "per_rank_independent_v1"


class OnlineSemambaTrainingData:
    """Train, valid, and test views of one online SEMamba++ snapshot."""

    def __init__(self, store: AssetStore, *, settings: dict[str, Any]) -> None:
        dataset = store.dataset
        if dataset.get("algorithm") != "online.semambapp_gsr" or dataset.get("kind") != "online_mixing":
            raise ReproductionError("this pipeline only opens online.semambapp_gsr snapshots")
        if dataset.get("declared_status") != "ready":
            raise ReproductionError(f"dataset {dataset.get('name')} is not ready")
        self.store = store
        self.rule = dataset["rule_json"]
        self.seed = int(settings["seed"])
        self.world_size = int(settings["world_size"])
        self.global_batch = int(settings["global_batch"])
        self.micro_batch = int(settings["micro_batch"])
        self.accumulation = int(settings["accumulation"])
        self._check_batch()
        segment = settings.get("segment_size")
        if segment is not None and int(segment) != SEGMENT_SIZE:
            raise ReproductionError(f"segment_size {segment} does not match the 48000-sample window")
        model_rate = settings.get("model_sample_rate")
        if model_rate is not None and int(model_rate) != SAMPLE_RATE:
            raise ReproductionError(f"model sampling_rate {model_rate} is not 16000")
        self.allow_test_selection = bool(settings["allow_test_selection"])
        self.selection_uses_test = bool(dataset["selection_uses_test"])
        if self.selection_uses_test and not self.allow_test_selection:
            raise ReproductionError("dataset uses test for selection; allow_test_selection is required")
        self.profile = str(dataset["profile"])
        self.dataset_id = str(dataset["dataset_id"])
        self.name = str(dataset["name"])
        self.version = int(dataset["version"])
        self.snapshot_hash = store.snapshot_hash
        self.protocol_id = str(dataset["protocol_id"])
        self.algorithm = "online.semambapp_gsr"
        self.algorithm_version = int(dataset["algorithm_version"])
        self.loader = GroupLoader(int(settings.get("load_workers", 1)))
        self._clips: dict[str, list[_AssetClip]] = {}
        self._plans: dict[tuple, EpochPlan] = {}
        self.batch_warning = None
        shipped = int(self.rule["global_batch"])
        if self.global_batch != shipped:
            self.batch_warning = (
                f"warning: global_batch_size {self.global_batch} differs from "
                f"the shipped batch {shipped}"
            )

    def set_batch(self, global_batch: int, micro_batch: int, accumulation: int) -> None:
        self.global_batch = int(global_batch)
        self.micro_batch = int(micro_batch)
        self.accumulation = int(accumulation)
        self._plans.clear()
        self._check_batch()

    def identity(self) -> dict[str, Any]:
        return {
            "pipeline": "reproduction",
            "name": self.name,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "snapshot_hash": self.snapshot_hash,
            "protocol_id": self.protocol_id,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "profile": self.profile,
            "selection_uses_test": self.selection_uses_test,
            "seed": self.seed,
            "sample_rate": SAMPLE_RATE,
            "window": SEGMENT_SIZE,
            "normalize": "independent_peak_v1",
            "delivery": "on_demand",
            "order_policy": POLICY,
            "world_size": self.world_size,
            "official_global_batch": int(self.rule["global_batch"]),
            "renderer": self.rule["online"]["renderer_version"],
        }

    def provenance(self) -> dict[str, Any]:
        recorded = self.identity()
        recorded["allow_test_selection"] = self.allow_test_selection
        recorded["online"] = (
            "SEMamba++ degradations are drawn at render time from HEAD config.yaml. "
            "Train changes with epoch. Valid and test ignore epoch. No degraded wav is stored."
        )
        return recorded

    def provenance_parts(self) -> dict[str, Any]:
        cached = getattr(self, "_provenance_parts_cache", None)
        if cached is None:
            cached = snapshot_provenance_parts(self.store)
            self._provenance_parts_cache = cached
        return cached

    def epoch(self, epoch: int, *, rank: int, world_size: int, start_update: int = 0) -> EpochStream:
        self._check_rank(rank, world_size)
        plan = self.plan_for(epoch, "train")
        if start_update < 0 or start_update > plan.updates:
            raise ReproductionError(f"start_update {start_update} is outside epoch {epoch}")

        def render(update_index: int) -> list[dict]:
            group = plan.rank_groups[rank][update_index]
            return [self._render_refs(micro.samples, "train", epoch) for micro in group.micro_batches]

        return EpochStream(plan.updates, plan.plan_hash, render, start_update)

    def validation(self, *, rank: int, world_size: int, epoch: int | None = None) -> Iterator[dict]:
        self._check_rank(rank, world_size)
        del rank
        logical = 0 if epoch is None else int(epoch)
        count = self._length("valid")
        batch = int(self.rule["validation"]["batch"])
        refs = [
            SampleRef(
                case_id=self._case_id("valid", index),
                index=index,
                draw=("valid", 0, index),
                length=SEGMENT_SIZE,
                crop=None,
            )
            for index in range(count)
        ]
        if not refs:
            raise ReproductionError("validation has no samples")
        for start in range(0, len(refs), batch):
            yield self._render_refs(refs[start : start + batch], "valid", logical)

    def plan_for(self, epoch: int, split: str = "train") -> EpochPlan:
        key = (split, int(epoch), self.micro_batch, self.accumulation, self.world_size)
        cached = self._plans.get(key)
        if cached is None:
            cached = plan_semamba_epoch(
                split=split,
                epoch=int(epoch),
                count=self._length(split),
                seed=self.seed,
                world_size=self.world_size,
                micro_batch=self.micro_batch,
                accumulation=self.accumulation,
                protocol_id=self.protocol_id,
                snapshot_hash=self.snapshot_hash,
                case_ids=self._case_ids(split),
            )
            self._plans[key] = cached
        return cached

    def data_state(self, epoch: int, committed_updates: int) -> dict[str, Any]:
        plan = self.plan_for(epoch, "train")
        if committed_updates < 0 or committed_updates > plan.updates:
            raise ReproductionError(
                f"committed updates {committed_updates} exceed epoch {epoch} ({plan.updates})"
            )
        return {
            "pipeline": "reproduction",
            "dataset_id": self.dataset_id,
            "dataset_name": f"{self.name}@{self.version}",
            "snapshot_hash": self.snapshot_hash,
            "protocol_id": self.protocol_id,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "profile": self.profile,
            "selection_uses_test": self.selection_uses_test,
            "seed": self.seed,
            "epoch": int(epoch),
            "update_in_epoch": int(committed_updates),
            "world_size": self.world_size,
            "ranks": list(range(self.world_size)),
            "order_policy": plan.order_policy,
            "order_params": dict(plan.stats["order_params"]),
            "plan_hash": plan.plan_hash,
            "global_batch": self.global_batch,
            "official_global_batch": int(self.rule["global_batch"]),
            "micro_batch": self.micro_batch,
            "accumulation": self.accumulation,
            "normalize": "independent_peak_v1",
            "window": SEGMENT_SIZE,
        }

    def check_resume(self, saved: dict[str, Any]) -> None:
        current = self.data_state(int(saved["epoch"]), int(saved["update_in_epoch"]))
        require_resume_match(saved, current)

    def close(self) -> None:
        self.loader.close()

    def render_index(self, split: str, epoch: int, sample_index: int):
        ref = SampleRef(
            case_id=self._case_id(split, sample_index),
            index=int(sample_index),
            draw=(split, 0 if split != "train" else int(epoch), int(sample_index)),
            length=SEGMENT_SIZE,
            crop=None,
        )
        return self._render_one(ref, split, int(epoch))

    def _render_refs(self, samples, split: str, epoch: int) -> dict:
        rendered = self.loader.map(lambda sample: self._render_one(sample, split, epoch), list(samples))
        return assemble(rendered, dataset_id=self.dataset_id, snapshot_hash=self.snapshot_hash)

    def _render_one(self, sample: SampleRef, split: str, epoch: int):
        from src.experiments.data.render.types import RenderedSample

        if sample.index < 0 or sample.index >= self._length(split):
            raise ReproductionError(f"{split} sample {sample.index} is outside the dataset")
        rng = np.random.Generator(
            np.random.PCG64(
                sample_seed(
                    self.protocol_id,
                    self.snapshot_hash,
                    self.seed,
                    split,
                    epoch,
                    sample.index,
                )
            )
        )
        example = render_example(
            self._clip_list(split, "speech"),
            self._clip_list(split, "noise"),
            self._clip_list(split, "rir"),
            speech_index=int(sample.index),
            split=split,
            rng=rng,
        )
        return RenderedSample(
            noisy=example.degraded,
            clean=example.clean,
            sample_rate=example.sample_rate,
            valid_length=SEGMENT_SIZE,
            case_id=sample.case_id,
            teacher_kind="dry_clean",
            extras={},
        )

    def _clip_list(self, split: str, role: str) -> list[_AssetClip]:
        slot = f"{split}.{role}"
        cached = self._clips.get(slot)
        if cached is None:
            cached = [_AssetClip(self.store, asset_id) for asset_id in _member_ids(self.store, slot)]
            if not cached:
                raise ReproductionError(f"snapshot is missing {slot}")
            self._clips[slot] = cached
        return cached

    def _length(self, split: str) -> int:
        return len(self._clip_list(split, "speech"))

    def _case_ids(self, split: str) -> list[str | None]:
        slot = f"{split}.cases"
        if slot not in self.store.bindings or not self.store.bindings.get(slot):
            return [None] * self._length(split)
        ids = self.store.case_ids(slot)
        if len(ids) != self._length(split):
            raise ReproductionError(f"{slot} count does not match the speech list")
        return ids

    def _case_id(self, split: str, sample_index: int) -> str | None:
        ids = self._case_ids(split)
        if sample_index >= len(ids):
            raise ReproductionError(f"{split} sample {sample_index} has no case")
        return ids[sample_index]

    def _check_batch(self) -> None:
        if self.micro_batch < 1 or self.world_size < 1 or self.accumulation < 1:
            raise ReproductionError("batch sizes must be positive")
        product = self.micro_batch * self.world_size * self.accumulation
        if product != self.global_batch:
            raise ReproductionError(
                "micro_batch * world_size * accumulation must equal global_batch "
                f"({self.micro_batch} * {self.world_size} * {self.accumulation} != {self.global_batch})"
            )

    def _check_rank(self, rank: int, world_size: int) -> None:
        if world_size != self.world_size or rank < 0 or rank >= world_size:
            raise ReproductionError(
                f"rank {rank} world_size {world_size} does not match this training data"
            )


def plan_semamba_epoch(
    *,
    split: str,
    epoch: int,
    count: int,
    seed: int,
    world_size: int,
    micro_batch: int,
    accumulation: int,
    protocol_id: str,
    snapshot_hash: str,
    case_ids: list[str | None] | None = None,
) -> EpochPlan:
    """Per-rank speech order, then drop incomplete micro-batches and updates."""
    if count < 1 or world_size < 1 or micro_batch < 1 or accumulation < 1:
        raise ReproductionError("semamba plan sizes must be positive")
    if split == "train":
        shards = [
            rank_order(
                POLICY,
                count,
                seed=seed,
                epoch=int(epoch),
                world_size=world_size,
                rank=rank,
            )
            for rank in range(world_size)
        ]
    else:
        shards = [list(range(count)) if rank == 0 else [] for rank in range(world_size)]
        if world_size != 1:
            raise ReproductionError("valid and test plans are single-process")
    groups = []
    dropped = []
    for shard in shards:
        usable = (len(shard) // micro_batch) * micro_batch
        dropped.append(len(shard) - usable)
        micros = []
        for start in range(0, usable, micro_batch):
            chunk = shard[start : start + micro_batch]
            refs = tuple(
                SampleRef(
                    case_id=None if not case_ids else case_ids[index],
                    index=int(index),
                    draw=(split, 0 if split != "train" else int(epoch), int(index)),
                    length=SEGMENT_SIZE,
                    crop=None,
                )
                for index in chunk
            )
            micros.append(MicroBatchPlan(refs))
        usable_updates = (len(micros) // accumulation) * accumulation
        dropped[-1] += (len(micros) - usable_updates) * micro_batch
        updates = tuple(
            UpdateGroup(tuple(micros[start : start + accumulation]))
            for start in range(0, usable_updates, accumulation)
        )
        groups.append(updates)
    update_count = len(groups[0])
    if update_count < 1:
        raise ReproductionError(
            f"{split} has no full update for world_size {world_size}, "
            f"micro_batch {micro_batch}, accumulation {accumulation}"
        )
    if any(len(group) != update_count for group in groups):
        raise ReproductionError("semamba ranks do not share an update count")
    covered: list[int] = []
    for shard_groups in groups:
        for group in shard_groups:
            for micro in group.micro_batches:
                covered.extend(sample.index for sample in micro.samples)
    overlap = sorted({index for index in covered if covered.count(index) > 1})
    missing = sorted(set(range(count)) - set(covered))
    stats = {
        "order_policy": POLICY,
        "order_params": {
            **order_params(POLICY, seed),
            "split": split,
            "epoch": int(epoch),
            "count": count,
            "world_size": world_size,
            "micro_batch": micro_batch,
            "accumulation": accumulation,
            "drop_last": True,
        },
        "dropped_samples": dropped,
        "overlap": overlap,
        "missing": missing,
        "updates": update_count,
        "covered_sha256": digest(covered),
    }
    plan_hash = digest(
        {
            "policy": POLICY,
            "split": split,
            "epoch": int(epoch),
            "protocol_id": protocol_id,
            "snapshot_hash": snapshot_hash,
            "seed": int(seed),
            "shards": [
                [sample.index for group in groups[rank] for micro in group.micro_batches for sample in micro.samples]
                for rank in range(world_size)
            ],
            "micro_batch": micro_batch,
            "accumulation": accumulation,
        }
    )
    return EpochPlan(
        epoch=int(epoch),
        world_size=world_size,
        updates=update_count,
        rank_groups=tuple(groups),
        plan_hash=plan_hash,
        order_policy=POLICY,
        stats=stats,
    )


def _member_ids(store: AssetStore, slot: str) -> list[str]:
    material_id = None
    for binding in store.dataset.get("bindings") or []:
        if binding.get("slot") == slot:
            material_id = binding.get("material_set_id")
    if not material_id:
        return []
    rows = [row for row in store.members if row["material_set_id"] == material_id]
    rows.sort(key=lambda row: int(row["position"]))
    return [str(row["asset_id"]) for row in rows]


def open_online_semamba(
    *,
    dataset: str,
    allow_test_selection: bool,
    seed: int,
    global_batch: int,
    micro_batch: int,
    accumulation: int,
    world_size: int,
    segment_size: int | None = None,
    model_sample_rate: int | None = None,
    load_workers: int = 1,
    verify_on_read: bool = False,
    snapshot_root: str | Path | None = None,
    derived_root: str | Path | None = None,
    corpora_root: str | Path | None = None,
    generated_root: str | Path | None = None,
) -> OnlineSemambaTrainingData:
    del verify_on_read
    root = Path(DEFAULT_SNAPSHOTS if snapshot_root is None else snapshot_root)
    loaded = load_snapshot(find_snapshot_dir(root, dataset))
    store = AssetStore.from_snapshot(
        loaded,
        derived=DEFAULT_DERIVED if derived_root is None else derived_root,
        corpora=DEFAULT_CORPORA if corpora_root is None else corpora_root,
        generated=DEFAULT_GENERATED if generated_root is None else generated_root,
    )
    return OnlineSemambaTrainingData(
        store,
        settings={
            "allow_test_selection": allow_test_selection,
            "seed": seed,
            "global_batch": global_batch,
            "micro_batch": micro_batch,
            "accumulation": accumulation,
            "world_size": world_size,
            "segment_size": segment_size,
            "model_sample_rate": model_sample_rate,
            "load_workers": load_workers,
        },
    )
