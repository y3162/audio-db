"""Fixed-pair supply for MP-SENet and MossFormerGAN.

Snapshots and numeric functions stay in ``src.reproduction``. This module
does not open DuckDB and does not construct ``ReproductionTrainingData``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.experiments.data.identity import comparable_identity
from src.experiments.data.sample import Batch, Crop, SampleRef
from src.experiments.errors import DataError, StateError
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest
from src.reproduction.paths import DEFAULT_CORPORA, DEFAULT_DERIVED, DEFAULT_GENERATED
from src.reproduction.pipeline.order import order_params, rank_order, resolve_order_policy
from src.reproduction.pipeline.planner import (
    fixed_pair_case_length,
    fixed_pair_crop,
    fixed_pair_pad_name,
    fixed_pair_plan_index,
    validation_refs,
)
from src.reproduction.pipeline.provenance import snapshot_provenance_parts
from src.reproduction.pipeline.renderer import render_sample
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.pipeline.types import CropSpec
from src.reproduction.pipeline.types import SampleRef as PlannedRef
from src.reproduction.snapshot_reader import LoadedSnapshot, load_snapshot

_IDENTITY_CLASS = "src.experiments.data.SnapshotDataModule"


def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except ReproductionError as exc:
        raise DataError(str(exc)) from exc


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value


def _ref_payload(sample: SampleRef) -> dict[str, Any]:
    crop = None if sample.crop is None else [sample.crop.start, sample.crop.length, sample.crop.pad_to]
    chunk = None if sample.chunk is None else [sample.chunk.start, sample.chunk.stop]
    return {
        "case_id": sample.case_id,
        "chunk": chunk,
        "crop": crop,
        "index": sample.index,
        "realization_id": sample.realization_id,
        "sample_id": sample.sample_id,
    }


def _clone_batch(batch: Batch) -> Batch:
    lengths = None if batch.lengths is None else batch.lengths.clone()
    return replace(batch, inputs=batch.inputs.clone(), targets=batch.targets.clone(), lengths=lengths)


@dataclass(frozen=True)
class _EpochPlan:
    groups: tuple[tuple[tuple[SampleRef, ...], ...], ...]
    plan_hash: str

    def updates(self, rank: int) -> int:
        return len(self.groups[rank])


class SnapshotSource:
    """One ready snapshot directory. The constructor does not read files."""

    def __init__(
        self,
        snapshot: str,
        snapshot_root: str | Path,
        split: str,
        *,
        derived: str | Path | None = None,
        corpora: str | Path | None = None,
        generated: str | Path | None = None,
    ) -> None:
        if not isinstance(snapshot, str) or not snapshot.strip():
            raise ValueError("snapshot must be a non-empty selector")
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test")
        self.snapshot = snapshot
        self.snapshot_root = Path(snapshot_root)
        self.split = split
        self.derived = DEFAULT_DERIVED if derived is None else Path(derived)
        self.corpora = DEFAULT_CORPORA if corpora is None else Path(corpora)
        self.generated = DEFAULT_GENERATED if generated is None else Path(generated)
        self._loaded: LoadedSnapshot | None = None
        self._store: AssetStore | None = None

    @property
    def store(self) -> AssetStore:
        if self._store is None:
            raise DataError("snapshot is not open")
        return self._store

    @property
    def snapshot_hash(self) -> str:
        if self._loaded is None:
            raise DataError("snapshot is not open")
        return self._loaded.snapshot_hash

    def open(self) -> None:
        if self._store is not None:
            return
        directory = self._resolve()
        try:
            loaded = load_snapshot(directory)
            store = AssetStore.from_snapshot(
                loaded,
                derived=self.derived,
                corpora=self.corpora,
                generated=self.generated,
            )
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc
        self._loaded = loaded
        self._store = store

    def close(self) -> None:
        self._loaded = None
        self._store = None

    def dataset(self) -> Mapping[str, Any]:
        return self.store.dataset

    def case_ids(self, slot: str) -> tuple[str, ...]:
        try:
            return tuple(self.store.case_ids(slot))
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc

    def document(self, case_id: str) -> Mapping[str, Any]:
        try:
            row = dict(self.store.case(case_id))
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc
        extra = self.store.docs.get(case_id)
        if extra is not None:
            row["doc_json"] = extra
        return row

    def material_ids(self, slot: str) -> tuple[str, ...]:
        binding = self._binding(slot)
        material_set_id = binding.get("material_set_id")
        if not material_set_id:
            return ()
        rows = [row for row in self.store.members if row.get("material_set_id") == material_set_id]
        rows.sort(key=lambda row: int(row["position"]))
        return tuple(str(row["asset_id"]) for row in rows)

    def asset_path(self, asset_id: str) -> Path:
        return _call(self.store.path, asset_id)

    def _binding(self, slot: str) -> Mapping[str, Any]:
        for binding in self.dataset().get("bindings") or []:
            if binding.get("slot") == slot:
                return binding
        raise DataError(f"snapshot has no {slot} binding")

    def _resolve(self) -> Path:
        selector = Path(self.snapshot)
        if selector.is_absolute() and (selector / "manifest.json").is_file():
            return selector
        direct = self.snapshot_root / self.snapshot
        if (direct / "manifest.json").is_file():
            return direct
        name, version = _selector(self.snapshot)
        if not self.snapshot_root.is_dir():
            raise DataError(f"snapshot root is missing: {self.snapshot_root}")
        matches: list[Path] = []
        for child in sorted(path for path in self.snapshot_root.iterdir() if path.is_dir()):
            document_path = child / "dataset.json"
            if not document_path.is_file():
                continue
            try:
                document = json.loads(document_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise DataError(f"cannot read snapshot dataset: {document_path}") from exc
            if not isinstance(document, dict):
                continue
            if version is None:
                if document.get("dataset_id") == self.snapshot or document.get("name") == self.snapshot:
                    matches.append(child)
            elif document.get("name") == name and document.get("version") == version:
                matches.append(child)
        if len(matches) != 1:
            raise DataError(
                f"snapshot selector {self.snapshot} matched {len(matches)} directories under {self.snapshot_root}"
            )
        return matches[0]


class OrderSampler:
    """Flat fixed-pair order. Crop coordinates come from the protocol."""

    def __init__(self, source: SnapshotSource, protocol: FixedPairProtocol, seed: int = 0) -> None:
        self.source = source
        self.protocol = protocol
        self.seed = _non_negative("seed", seed)

    def policy(self, world_size: int) -> str:
        rule = self.source.dataset().get("rule_json")
        if not isinstance(rule, dict) or "order" not in rule:
            raise DataError("snapshot rule has no order")
        return _call(resolve_order_policy, rule["order"], world_size)

    def sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        return self._sequence(epoch, rank, world_size)

    def all_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        return tuple(self._sequence(epoch, rank, world_size) for rank in range(world_size))

    def _sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        slot = f"{self.source.split}.cases"
        case_ids = self.source.case_ids(slot)
        policy = self.policy(world_size)
        positions = _call(
            rank_order,
            policy,
            len(case_ids),
            seed=self.seed,
            epoch=epoch,
            world_size=world_size,
            rank=rank,
        )
        self.protocol.seed = self.seed
        refs: list[SampleRef] = []
        for local, position in enumerate(positions):
            case_id = case_ids[position]
            length = _call(fixed_pair_case_length, self.source.store, case_id, self.protocol.normalize)
            crop = self.protocol.locate(case_id, length, epoch)
            index = _call(fixed_pair_plan_index, policy, local, rank, world_size, len(case_ids))
            refs.append(
                SampleRef(
                    sample_id=case_id,
                    realization_id=_realization(case_id, epoch, crop),
                    index=index,
                    case_id=case_id,
                    crop=crop,
                )
            )
        return tuple(refs)


class FixedPairProtocol:
    """Crop coordinates and ``render_sample`` for one fixed-pair snapshot."""

    def __init__(self, source: SnapshotSource, verify_on_read: bool = False) -> None:
        if not isinstance(verify_on_read, bool):
            raise ValueError("verify_on_read must be a boolean")
        self.source = source
        self.verify_on_read = verify_on_read
        self.seed: int | None = None

    def locate(self, case_id: str, length: int, epoch: int) -> Crop:
        if self.seed is None:
            raise DataError("fixed-pair protocol has no seed")
        spec = _call(
            fixed_pair_crop,
            case_id,
            length,
            self.window,
            self.seed,
            epoch,
            self.pad_name,
        )
        return Crop(start=int(spec.start), length=int(spec.length), pad_to=int(spec.pad_to))

    def render(self, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
        if not sample.case_id:
            raise DataError("fixed-pair rendering requires a case id")
        crop = None
        if sample.crop is not None:
            pad = "none" if sample.crop.length == sample.crop.pad_to else self.pad_name
            crop = CropSpec(
                start=sample.crop.start,
                length=sample.crop.length,
                pad_to=sample.crop.pad_to,
                pad=pad,
            )
        planned = PlannedRef(
            case_id=sample.case_id,
            index=sample.index,
            draw=None,
            length=None,
            crop=crop,
            chunk=None,
        )
        rule = self._rule()
        resample = rule.get("resample")
        if resample is not None and not isinstance(resample, dict):
            raise DataError("snapshot resample rule is invalid")
        rendered = _call(
            render_sample,
            self.source.store,
            planned,
            normalize=self.normalize,
            sample_rate=int(rule["sample_rate"]),
            verify_on_read=self.verify_on_read,
            resample=resample,
        )
        return rendered.noisy, rendered.clean, int(rendered.valid_length)

    def teacher_kind(self, sample: SampleRef) -> str:
        if not sample.case_id:
            raise DataError("fixed-pair sample has no case id")
        kind = self.source.document(sample.case_id).get("teacher_kind")
        if kind in (None, "none"):
            raise DataError(f"case {sample.case_id} has no teacher")
        return str(kind)

    @property
    def normalize(self) -> str:
        return str(self._rule()["normalize"])

    @property
    def window(self) -> int:
        return int(self._rule()["window"])

    @property
    def pad_name(self) -> str:
        return _call(fixed_pair_pad_name, self.normalize)

    def _rule(self) -> Mapping[str, Any]:
        rule = self.source.dataset().get("rule_json")
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        return rule


class MicroBatcher:
    """Pack one rank's sample list into micro-batches. It does not shard ranks."""

    def __init__(self, micro_batch: int, group_size: int) -> None:
        self.micro_batch = _positive("micro_batch", micro_batch)
        self.group_size = _positive("group_size", group_size)

    def groups(self, samples: Sequence[SampleRef]) -> tuple[tuple[SampleRef, ...], ...]:
        rows = tuple(samples)
        if not rows:
            raise DataError("epoch sample list is empty")
        width = self.micro_batch
        packed = [rows[start : start + width] for start in range(0, len(rows) - len(rows) % width, width)]
        if not packed:
            raise DataError("epoch does not contain a full micro-batch")
        keep = len(packed) - (len(packed) % self.group_size)
        if keep < 1:
            raise DataError("epoch does not contain a full update group")
        return tuple(packed[:keep])

    def collate(self, rendered: Sequence[tuple[np.ndarray, np.ndarray, int, SampleRef]]) -> Batch:
        if not rendered:
            raise DataError("cannot collate an empty micro-batch")
        width = max(int(noisy.shape[0]) for noisy, _clean, _valid, _sample in rendered)
        noises = [_fit(noisy, width) for noisy, _clean, _valid, _sample in rendered]
        cleans = [_fit(clean, width) for _noisy, clean, _valid, _sample in rendered]
        lengths = [int(valid) for _noisy, _clean, valid, _sample in rendered]
        if any(length < 0 or length > width for length in lengths):
            raise DataError("valid length does not fit the collated waveform")
        sample_ids = tuple(sample.sample_id for _noisy, _clean, _valid, sample in rendered)
        realization_ids = tuple(sample.realization_id for _noisy, _clean, _valid, sample in rendered)
        return Batch(
            inputs=torch.from_numpy(np.stack(noises)).contiguous(),
            targets=torch.from_numpy(np.stack(cleans)).contiguous(),
            sample_id=",".join(sample_ids),
            realization_id=",".join(realization_ids),
            lengths=torch.tensor(lengths, dtype=torch.long),
            sample_ids=sample_ids,
            realization_ids=realization_ids,
            teacher_kind=None,
        )


class SnapshotEpochView:
    """One rank's micro-batches. ``batch`` does not move the consumed cursor."""

    def __init__(
        self,
        module: SnapshotDataModule,
        epoch: int,
        start_update: int,
        updates: int,
        groups: tuple[tuple[SampleRef, ...], ...],
        plan_hash: str,
    ) -> None:
        self.updates = updates
        self.plan_hash = plan_hash
        self._module = module
        self._epoch = epoch
        self._start = start_update
        self._groups = groups

    def batch(self, update_index: int) -> list[Batch]:
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise StateError(f"update index {update_index!r} is not an integer")
        if update_index < self._start:
            raise StateError(f"update index {update_index} is before start_update {self._start}")
        if update_index < 0 or update_index >= self.updates:
            raise StateError(f"update index {update_index} is outside 0..{self.updates - 1}")
        return [self._module.materialize(self._epoch, update_index, self._groups)]


class SnapshotDataModule:
    """Assemble one snapshot sampler and protocol into the experiment data contract."""

    def __init__(
        self,
        source: SnapshotSource,
        sampler: OrderSampler,
        protocol: FixedPairProtocol,
        batcher: MicroBatcher,
        require_profile: str | None = None,
        prefetch: bool = False,
    ) -> None:
        if require_profile is not None and not isinstance(require_profile, str):
            raise ValueError("require_profile must be a string")
        if not isinstance(prefetch, bool):
            raise ValueError("prefetch must be a boolean")
        self.source = source
        self.sampler = sampler
        self.protocol = protocol
        self.batcher = batcher
        self.require_profile = require_profile
        self.prefetch = prefetch
        self.epoch = 0
        self.update_index = 0
        self.world_size = 1
        self.rank = 0
        self._ready = False
        self._closed = False
        self._plans: dict[int, _EpochPlan] = {}
        self._cache: dict[tuple[int, int], Batch] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._prefetch_error: BaseException | None = None

    @property
    def group_size(self) -> int:
        return self.batcher.group_size

    @property
    def plan_hash(self) -> str:
        self._require_ready()
        return self._epoch_plan(self.epoch).plan_hash

    def identity(self) -> dict[str, Any]:
        self._require_ready()
        dataset = self.source.dataset()
        rule = dataset.get("rule_json")
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        chunk_length = getattr(self.sampler, "chunk_length", None)
        if chunk_length is not None:
            window = int(chunk_length)
        else:
            window = rule.get("window")
            if window is None:
                window = rule.get("chunk_length")
        return {
            "algorithm": dataset.get("algorithm"),
            "claim_label": dataset.get("claim_label"),
            "class": _IDENTITY_CLASS,
            "dataset_id": dataset.get("dataset_id"),
            "dataset_name": dataset.get("name"),
            "dataset_version": dataset.get("version"),
            "group_size": self.batcher.group_size,
            "micro_batch": self.batcher.micro_batch,
            "normalize": rule.get("normalize"),
            "order_policy": self.sampler.policy(self.world_size),
            "profile": dataset.get("profile"),
            "protocol_id": dataset.get("protocol_id"),
            "seed": self.sampler.seed,
            "selection_uses_test": dataset.get("selection_uses_test"),
            "snapshot_hash": self.source.snapshot_hash,
            "split": self.source.split,
            "window": window,
        }

    def setup(self, runtime: object) -> None:
        if self._closed:
            raise DataError("data module is closed")
        self._accept_runtime(runtime)
        self.source.open()
        self._validate_dataset()
        self._ready = True
        self._plans.clear()

    def train_epoch(self, epoch: int, *, start_update: int) -> SnapshotEpochView:
        self._require_ready()
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        plan = self._epoch_plan(epoch)
        updates = plan.updates(self.rank)
        if isinstance(start_update, bool) or not isinstance(start_update, int):
            raise StateError(f"start_update {start_update!r} is not an integer")
        if start_update < 0 or start_update > updates:
            raise StateError(f"start_update {start_update} is outside 0..{updates}")
        return SnapshotEpochView(
            self,
            epoch,
            start_update,
            updates,
            plan.groups[self.rank],
            plan.plan_hash,
        )

    def validation_epoch(self, epoch: int) -> Iterator[Batch]:
        self._require_ready()
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        validation_ranks = getattr(self.sampler, "validation_ranks", None)
        if callable(validation_ranks):
            self._bind_sampler()
            grouped = self.batcher.groups(validation_ranks(epoch, self.world_size)[self.rank])
            return iter(self._collate(group) for group in grouped)
        algorithm = str(self.source.dataset().get("algorithm"))
        if algorithm == "fixed_pair.mp_senet":
            if self.rank != 0:
                return iter(())
            raw = _call(
                validation_refs,
                self.source.store,
                seed=self.sampler.seed,
                epoch=epoch,
                world_size=self.world_size,
                policy=self.sampler.policy(self.world_size),
                rank=None,
            )
        elif algorithm == "fixed_pair.mossformergan_se_16k":
            raw = _call(
                validation_refs,
                self.source.store,
                seed=self.sampler.seed,
                epoch=epoch,
                world_size=self.world_size,
                policy="rank_mod_fixed_v1",
                rank=self.rank,
            )
        else:
            raise DataError(f"validation for {algorithm} is not implemented in Phase 3-B")
        refs = tuple(_validation_sample(sample, epoch) for sample in raw)
        batches: list[Batch] = []
        width = self.batcher.micro_batch
        for start in range(0, len(refs), width):
            batches.append(self._collate(refs[start : start + width]))
        return iter(batches)

    def state_dict(self) -> dict[str, Any]:
        self._require_ready()
        return {
            "epoch": self.epoch,
            "identity": self.identity(),
            "plan_hash": self.plan_hash,
            "update_index": self.update_index,
            "weights": None,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._require_ready()
        self._stop_prefetch()
        if not isinstance(state, dict):
            raise StateError("data state is not a mapping")
        saved_identity = state.get("identity")
        if not isinstance(saved_identity, dict):
            raise StateError("data identity is missing")
        if comparable_identity(saved_identity) != comparable_identity(self.identity()):
            raise StateError("data state does not match this data module")
        if state.get("world_size") != self.world_size:
            raise StateError("data world_size does not match this data module")
        if "weights" not in state or state.get("weights") is not None:
            raise StateError("data weights do not match this data module")
        epoch = state.get("epoch")
        update_index = state.get("update_index")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError("data epoch is invalid")
        if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
            raise StateError("data update_index is invalid")
        expected = self.plan_hash_for_state(state)
        if state.get("plan_hash") != expected:
            raise StateError("data plan_hash does not match this data module")
        updates = self._epoch_plan(epoch).updates(self.rank)
        if update_index > updates:
            raise StateError(f"data update_index {update_index} is outside 0..{updates}")
        self.epoch = epoch
        self.update_index = update_index

    def plan_hash_for_state(self, state: Mapping[str, Any]) -> str:
        """Recompute the hash for ``state["epoch"]`` before that state is applied."""

        self._require_ready()
        epoch = state.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError("data epoch is invalid")
        return self._epoch_plan(epoch).plan_hash

    def lineage(self) -> dict[str, Any]:
        self._require_ready()
        return _call(snapshot_provenance_parts, self.source.store, workers=1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready = False
        self._stop_prefetch()
        close = getattr(self.source, "close", None)
        if callable(close):
            close()

    def materialize(self, epoch: int, index: int, groups: tuple[tuple[SampleRef, ...], ...]) -> Batch:
        key = (epoch, index)
        with self._lock:
            if self._prefetch_error is not None:
                error = self._prefetch_error
                self._prefetch_error = None
                raise error
            cached = self._cache.get(key)
            if cached is None:
                cached = self._collate(groups[index])
                self._cache[key] = cached
        if self.prefetch and not self._closed:
            self._start_prefetch(epoch, index, groups)
        return _clone_batch(cached)

    def _collate(self, group: tuple[SampleRef, ...]) -> Batch:
        rendered = []
        kinds: list[str] = []
        for sample in group:
            noisy, clean, valid = self.protocol.render(sample)
            rendered.append((noisy, clean, valid, sample))
            kinds.append(self.protocol.teacher_kind(sample))
        if any(kind != kinds[0] for kind in kinds):
            raise DataError("micro-batch mixes teacher kinds")
        return replace(self.batcher.collate(rendered), teacher_kind=kinds[0])

    def _epoch_plan(self, epoch: int) -> _EpochPlan:
        cached = self._plans.get(epoch)
        if cached is not None:
            return cached
        self._bind_sampler()
        grouped = tuple(self.batcher.groups(sequence) for sequence in self.sampler.all_ranks(epoch, self.world_size))
        plan = _EpochPlan(groups=grouped, plan_hash=self._hash(epoch, grouped))
        self._plans[epoch] = plan
        return plan

    def _hash(self, epoch: int, groups: tuple[tuple[tuple[SampleRef, ...], ...], ...]) -> str:
        dataset = self.source.dataset()
        policy = self.sampler.policy(self.world_size)
        parameters = getattr(self.sampler, "order_parameters", None)
        recorded = parameters() if callable(parameters) else _call(order_params, policy, self.sampler.seed)
        payload = {
            "epoch": epoch,
            "group_size": self.batcher.group_size,
            "micro_batch": self.batcher.micro_batch,
            "order_params": recorded,
            "order_policy": policy,
            "protocol_id": dataset.get("protocol_id"),
            "ranks": [[_ref_payload(sample) for group in rank_groups for sample in group] for rank_groups in groups],
            "seed": self.sampler.seed,
            "snapshot_hash": self.source.snapshot_hash,
            "weights_sha256": "none",
            "world_size": self.world_size,
        }
        return _call(digest, payload)

    def _validate_dataset(self) -> None:
        dataset = self.source.dataset()
        if dataset.get("declared_status") != "ready":
            raise DataError(f"snapshot dataset is not ready: {dataset.get('declared_status')}")
        rule = dataset.get("rule_json")
        if not isinstance(rule, dict) or rule.get("delivery") != "on_demand":
            raise DataError("snapshot delivery is not on_demand")
        algorithm = str(dataset.get("algorithm"))
        supply = getattr(self.protocol, "supply", "fixed_pair")
        if supply == "chunk":
            if algorithm not in {"pregenerated.ms_snsd_tfgridnet", "pregenerated.dns_tflocoformer"}:
                raise DataError(f"chunk protocol cannot read algorithm {algorithm}")
            self._bind_sampler()
            prepare = getattr(self.protocol, "prepare", None)
            if callable(prepare):
                prepare()
        elif supply == "fixed_pair":
            if not algorithm.startswith("fixed_pair."):
                raise DataError(f"fixed-pair protocol cannot read algorithm {algorithm}")
        else:
            raise DataError(f"unsupported data supply {supply}")
        if self.require_profile is not None and dataset.get("profile") != self.require_profile:
            raise DataError(f"snapshot profile {dataset.get('profile')} is not {self.require_profile}")
        if dataset.get("profile") == "leakage_safe":
            valid = set(self.source.case_ids("valid.cases"))
            bindings = {binding.get("slot") for binding in dataset.get("bindings") or []}
            test = set(self.source.case_ids("test.cases")) if "test.cases" in bindings else set()
            if valid & test:
                raise DataError("leakage_safe validation shares cases with test")

    def _bind_sampler(self) -> None:
        bind = getattr(self.sampler, "bind_batch", None)
        if callable(bind):
            bind(self.batcher, self.world_size)

    def _accept_runtime(self, runtime: object) -> None:
        world_size = getattr(runtime, "world_size", 1)
        rank = getattr(runtime, "rank", 0)
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise StateError("runtime world_size must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
            raise StateError("runtime rank is outside this world")
        self.world_size = world_size
        self.rank = rank

    def _require_ready(self) -> None:
        if self._closed:
            raise DataError("data module is closed")
        if not self._ready:
            raise DataError("data module is not set up")

    def _start_prefetch(
        self,
        epoch: int,
        index: int,
        groups: tuple[tuple[SampleRef, ...], ...],
    ) -> None:
        nxt = index + 1
        if nxt >= len(groups) or (epoch, nxt) in self._cache:
            return
        if self._thread is not None and self._thread.is_alive():
            return

        def work() -> None:
            if self._closed:
                return
            try:
                with self._lock:
                    if (epoch, nxt) not in self._cache and not self._closed:
                        self._cache[(epoch, nxt)] = self._collate(groups[nxt])
            except Exception as exc:
                self._prefetch_error = exc

        self._thread = threading.Thread(target=work, name="audio-db-prefetch", daemon=True)
        self._thread.start()

    def _stop_prefetch(self) -> None:
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30)
        self._cache.clear()
        self._prefetch_error = None


def _selector(selector: str) -> tuple[str, int | None]:
    if "@" not in selector:
        return selector, None
    name, raw_version = selector.rsplit("@", 1)
    if not name or not raw_version.isdigit():
        return selector, None
    return name, int(raw_version)


def _realization(case_id: str, epoch: int, crop: Crop | None) -> str:
    if crop is None:
        return f"{case_id}:epoch-{epoch}:crop-full"
    return f"{case_id}:epoch-{epoch}:crop-{crop.start}+{crop.length}"


def _validation_sample(sample: PlannedRef, epoch: int) -> SampleRef:
    case_id = str(sample.case_id)
    crop = None if sample.crop is None else Crop(sample.crop.start, sample.crop.length, sample.crop.pad_to)
    return SampleRef(
        sample_id=case_id,
        realization_id=_realization(case_id, epoch, crop),
        index=int(sample.index),
        case_id=case_id,
        crop=crop,
    )


def _fit(wave: np.ndarray, width: int) -> np.ndarray:
    values = np.ascontiguousarray(wave, dtype=np.float32)
    if values.ndim != 1:
        raise DataError("collate expects a mono waveform")
    if int(values.shape[0]) == width:
        return values
    if int(values.shape[0]) > width:
        raise DataError("waveform is longer than the collate width")
    padded = np.zeros(width, dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded
