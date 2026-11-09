"""Fixed-pair supply for MP-SENet and MossFormerGAN.

Snapshots and numeric functions stay in ``src.reproduction``. This module
does not open DuckDB and does not construct ``ReproductionTrainingData``.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.experiments.data.identity import comparable_identity
from src.experiments.data.sample import Batch, Crop, SampleRef
from src.experiments.data.weighted import (
    ALGORITHM as WEIGHTED_ALGORITHM,
    normalize_weight_table,
    rank_sample_positions,
    weighted_order_parameters,
)
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
_MISSING = object()


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
    weights: dict[str, float] | None = None
    weights_sha256: str = "none"

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
    """Rank order for a fixed pair or for SEMamba++ speech indexes.

    Fixed-pair rows call ``protocol.locate``. Online SEMamba++ rows do not.
    """

    def __init__(self, source: SnapshotSource, protocol: FixedPairProtocol, seed: int = 0, global_batch: int | None = None) -> None:
        self.source = source
        self.protocol = protocol
        self.seed = _non_negative("seed", seed)
        self.global_batch = None if global_batch is None else _positive("global_batch", global_batch)
        self._bound_world: int | None = None
        self._micro_batch: int | None = None
        self._group_size: int | None = None
        self._batch_warning: str | None = None
        self._draw_weights: dict[str, float] | None = None

    def policy(self, world_size: int) -> str:
        rule = self.source.dataset().get("rule_json")
        if not isinstance(rule, dict) or "order" not in rule:
            raise DataError("snapshot rule has no order")
        return _call(resolve_order_policy, rule["order"], world_size)

    def order_parameters(self) -> dict[str, Any]:
        """Permutation parameters. Fixed-pair returns the existing ``order_params`` dict."""

        if self._bound_world is None:
            raise DataError("order sampler has no world size")
        recorded = dict(_call(order_params, self.policy(self._bound_world), self.seed))
        if self._supply() != "online_semamba":
            return recorded
        recorded["dataset_id"] = str(self.source.dataset().get("dataset_id"))
        recorded["group_size"] = self._group_size
        recorded["micro_batch"] = self._micro_batch
        return recorded

    def bind_batch(self, batcher: object, world_size: int) -> None:
        """Record the batch the DataModule will apply. Fixed pairs are unchanged."""

        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        self._bound_world = world_size
        if self._supply() != "online_semamba":
            return
        self.protocol.seed = self.seed  # type: ignore[attr-defined]
        group_size = getattr(batcher, "group_size", None)
        micro_batch = getattr(batcher, "micro_batch", None)
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise DataError("group_size must be a positive integer")
        if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
            raise DataError("micro_batch must be a positive integer")
        effective = micro_batch * world_size * group_size
        if self.global_batch is not None and self.global_batch != effective:
            raise DataError(
                f"global_batch {self.global_batch} does not equal "
                f"micro_batch * world_size * group_size ({effective})"
            )
        self._micro_batch = micro_batch
        self._group_size = group_size
        shipped = int(self._rule()["global_batch"])
        if effective != shipped:
            self._batch_warning = (
                f"warning: global_batch_size {effective} differs from the shipped batch {shipped}"
            )
        else:
            self._batch_warning = None

    def batch_warning(self) -> str | None:
        return self._batch_warning

    def validation_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        """SEMamba++ validation is speech order on rank 0. It is not the training shuffle."""

        if self._supply() != "online_semamba":
            raise DataError("validation_ranks on OrderSampler is only for online SEMamba++")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DataError("epoch must be a non-negative integer")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        count = self.protocol.material_count("valid")  # type: ignore[attr-defined]
        rows = tuple(
            self.protocol.reference(index, epoch, "valid")  # type: ignore[attr-defined]
            for index in range(count)
        )
        return tuple(rows if rank == 0 else () for rank in range(world_size))

    def validation_batch(self) -> tuple[int, bool, int]:
        """Validation width, keep the tail, and do not apply the training group drop."""

        if self._supply() != "online_semamba":
            raise DataError("validation_batch on OrderSampler is only for online SEMamba++")
        width = int(self._rule()["validation"]["batch"])
        if width < 1:
            raise DataError("validation batch must be positive")
        return width, True, 1

    def training_ids(self) -> tuple[str, ...]:
        """Canonical training sample ids. Fixed pairs use case ids. SEMamba uses speech ids."""

        if self._supply() == "online_semamba":
            self.protocol.seed = self.seed
            count = self.protocol.material_count(self.source.split)  # type: ignore[attr-defined]
            ids = tuple(
                self.protocol.reference(index, 0, self.source.split).sample_id  # type: ignore[attr-defined]
                for index in range(count)
            )
        elif self._supply() == "fixed_pair":
            ids = tuple(self.source.case_ids(f"{self.source.split}.cases"))
        else:
            raise DataError(f"OrderSampler cannot weight supply {self._supply()}")
        if len(set(ids)) != len(ids):
            raise DataError("training sample ids are not unique")
        return ids

    def normalize_weights(self, weights: object) -> dict[str, float]:
        return normalize_weight_table(weights, self.training_ids())

    def use_draw_weights(self, weights: dict[str, float] | None) -> None:
        self._draw_weights = None if weights is None else dict(weights)

    def sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        return self._sequence(epoch, rank, world_size)

    def all_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        return tuple(self._sequence(epoch, rank, world_size) for rank in range(world_size))

    def _supply(self) -> str:
        return str(getattr(self.protocol, "supply", "fixed_pair"))

    def _rule(self) -> Mapping[str, Any]:
        rule = self.source.dataset().get("rule_json")
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        return rule

    def _sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        if self._supply() == "online_semamba":
            return self._speech_sequence(epoch, rank, world_size)
        if self._supply() != "fixed_pair":
            raise DataError(f"OrderSampler cannot order supply {self._supply()}")
        return self._fixed_sequence(epoch, rank, world_size)

    def _speech_sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        count = self.protocol.material_count(self.source.split)  # type: ignore[attr-defined]
        if self._draw_weights is not None:
            positions = self._weighted_positions(epoch, rank, world_size)
        else:
            positions = _call(
                rank_order,
                self.policy(world_size),
                count,
                seed=self.seed,
                epoch=epoch,
                world_size=world_size,
                rank=rank,
            )
        return tuple(
            self.protocol.reference(int(position), epoch, self.source.split)  # type: ignore[attr-defined]
            for position in positions
        )

    def _fixed_sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        slot = f"{self.source.split}.cases"
        case_ids = self.source.case_ids(slot)
        policy = self.policy(world_size)
        weighted = self._draw_weights is not None
        if weighted:
            positions = self._weighted_positions(epoch, rank, world_size)
        else:
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
            if weighted:
                index = int(position)
            else:
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

    def _weighted_positions(self, epoch: int, rank: int, world_size: int) -> list[int]:
        table = self._draw_weights
        if table is None:
            raise DataError("weighted sampling has no weight table")
        ids = self.training_ids()
        positions = rank_sample_positions(
            [table[sample_id] for sample_id in ids],
            self.seed,
            epoch,
            world_size,
            rank,
        )
        if not positions:
            raise DataError("weighted epoch has no samples for this rank")
        return positions


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

    def __init__(self, micro_batch: int, group_size: int, keep_partial: bool = False) -> None:
        if not isinstance(keep_partial, bool):
            raise ValueError("keep_partial must be a boolean")
        self.micro_batch = _positive("micro_batch", micro_batch)
        self.group_size = _positive("group_size", group_size)
        self.keep_partial = keep_partial

    def groups(self, samples: Sequence[SampleRef]) -> tuple[tuple[SampleRef, ...], ...]:
        return _group_samples(samples, self.micro_batch, self.group_size, self.keep_partial)

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
        self._closed = False
        module._register_view(self)

    def close(self) -> None:
        """Release this view. Garbage collection is not the epoch boundary."""

        if self._closed:
            return
        self._closed = True
        self._module._release_view(self)

    def batch(self, update_index: int) -> list[Batch]:
        if self._closed:
            raise DataError("epoch view is closed")
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
        load_workers: int | None = None,
        worker_join_timeout: float = 30.0,
    ) -> None:
        if require_profile is not None and not isinstance(require_profile, str):
            raise ValueError("require_profile must be a string")
        if not isinstance(prefetch, bool):
            raise ValueError("prefetch must be a boolean")
        if load_workers is None:
            resolved_workers = 1 if prefetch else 0
        elif isinstance(load_workers, bool) or not isinstance(load_workers, int) or load_workers < 0:
            raise ValueError("load_workers must be an integer >= 0")
        else:
            resolved_workers = load_workers
        if prefetch and resolved_workers < 1:
            raise ValueError("prefetch requires load_workers >= 1")
        if not prefetch and resolved_workers != 0:
            raise ValueError("load_workers must be 0 when prefetch is disabled")
        if (
            isinstance(worker_join_timeout, bool)
            or not isinstance(worker_join_timeout, (int, float))
            or float(worker_join_timeout) <= 0
        ):
            raise ValueError("worker_join_timeout must be a positive number")
        self.source = source
        self.sampler = sampler
        self.protocol = protocol
        self.batcher = batcher
        self.require_profile = require_profile
        self.prefetch = prefetch
        self.load_workers = resolved_workers
        self.worker_join_timeout = float(worker_join_timeout)
        self.epoch = 0
        self.update_index = 0
        self.world_size = 1
        self.rank = 0
        self._ready = False
        self._closed = False
        self._source_closed = False
        self._plans: dict[int, _EpochPlan] = {}
        self._cache: dict[tuple[int, int], Batch] = {}
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._inflight: set[tuple[int, int]] = set()
        self._stop = threading.Event()
        self._jobs: queue.Queue[tuple[int, int, tuple[SampleRef, ...]] | None] = queue.Queue()
        self._prefetch_error: BaseException | None = None
        self._batch_warned = False
        self._weights: dict[str, float] | None = None
        self._open_views = 0
        self._runtime: object | None = None

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
        self._runtime = runtime
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

    def set_sampling_weights(self, weights: Mapping[str, float]) -> None:
        """Apply ``weights`` to later epochs of a flat order sampler.

        An open epoch view is refused. Epochs that already have a plan keep
        that plan. Chunk and DeepFilterNet3 samplers are refused.
        """

        self._require_ready()
        if self._open_views:
            raise DataError("sampling weights cannot change while an epoch view is open")
        if not isinstance(self.sampler, OrderSampler):
            raise DataError("set_sampling_weights is only available for a flat order sampler")
        normalized = self.sampler.normalize_weights(weights)
        self._weights = normalized
        self._stop_prefetch(propagate_error=False)

    def validation_epoch(self, epoch: int) -> Iterator[Batch]:
        self._require_ready()
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        validation_ranks = getattr(self.sampler, "validation_ranks", None)
        supply = getattr(self.protocol, "supply", "fixed_pair")
        if callable(validation_ranks) and supply in {"online_dfn3", "online_semamba"}:
            self._bind_sampler()
            samples = validation_ranks(epoch, self.world_size)[self.rank]
            if not samples:
                return iter(())
            width, keep_partial, group_size = self.sampler.validation_batch()
            grouped = _group_samples(samples, width, group_size, keep_partial)
            return iter(self._collate(group) for group in grouped)
        if callable(validation_ranks) and supply == "chunk":
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
        plan = self._epoch_plan(self.epoch)
        weights = None if plan.weights is None else dict(plan.weights)
        return {
            "epoch": self.epoch,
            "identity": self.identity(),
            "plan_hash": plan.plan_hash,
            "update_index": self.update_index,
            "weights": weights,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._require_ready()
        self._stop_prefetch(propagate_error=False)
        if not isinstance(state, dict):
            raise StateError("data state is not a mapping")
        saved_identity = state.get("identity")
        if not isinstance(saved_identity, dict):
            raise StateError("data identity is missing")
        if comparable_identity(saved_identity) != comparable_identity(self.identity()):
            raise StateError("data state does not match this data module")
        if state.get("world_size") != self.world_size:
            raise StateError("data world_size does not match this data module")
        self._require_saved_weights(state.get("weights") if "weights" in state else _MISSING)
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
        if self._closed and not self._workers:
            return
        self._closed = True
        self._ready = False
        try:
            self._stop_prefetch(propagate_error=True)
        finally:
            if not self._source_closed:
                self._source_closed = True
                close = getattr(self.source, "close", None)
                if callable(close):
                    close()

    def materialize(self, epoch: int, index: int, groups: tuple[tuple[SampleRef, ...], ...]) -> Batch:
        key = (epoch, index)
        self._raise_prefetch_error()
        cached = self._cached(key)
        if cached is None and key in self._inflight_snapshot():
            self._wait_for(key)
            self._raise_prefetch_error()
            cached = self._cached(key)
        if cached is None:
            cached = self._collate(groups[index])
            with self._lock:
                existing = self._cache.get(key)
                if existing is None:
                    self._remember_locked(key, cached)
                else:
                    cached = existing
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
        table = self._weights
        sha = "none" if table is None else _call(digest, table)
        self._agree_weights(sha)
        self._bind_sampler()
        applied = False
        try:
            if table is not None:
                use = getattr(self.sampler, "use_draw_weights", None)
                if not callable(use):
                    raise DataError("set_sampling_weights is only available for a flat order sampler")
                use(table)
                applied = True
            grouped = tuple(
                self.batcher.groups(sequence) for sequence in self.sampler.all_ranks(epoch, self.world_size)
            )
        finally:
            if applied:
                self.sampler.use_draw_weights(None)  # type: ignore[attr-defined]
        plan = _EpochPlan(
            groups=grouped,
            plan_hash=self._hash(epoch, grouped, table),
            weights=None if table is None else dict(table),
            weights_sha256=sha,
        )
        self._plans[epoch] = plan
        return plan

    def _hash(
        self,
        epoch: int,
        groups: tuple[tuple[tuple[SampleRef, ...], ...], ...],
        weights: dict[str, float] | None,
    ) -> str:
        dataset = self.source.dataset()
        if weights is None:
            policy = self.sampler.policy(self.world_size)
            parameters = getattr(self.sampler, "order_parameters", None)
            recorded = parameters() if callable(parameters) else _call(order_params, policy, self.sampler.seed)
            sha = "none"
        else:
            policy = WEIGHTED_ALGORITHM
            recorded = weighted_order_parameters(self.sampler.seed)
            sha = _call(digest, weights)
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
            "weights_sha256": sha,
            "world_size": self.world_size,
        }
        return _call(digest, payload)

    def _require_saved_weights(self, saved: object) -> None:
        if saved is _MISSING:
            raise StateError("data weights do not match this data module")
        if saved is None:
            normalized = None
        elif isinstance(self.sampler, OrderSampler):
            try:
                normalized = self.sampler.normalize_weights(saved)
            except DataError as exc:
                raise StateError("data weights do not match this data module") from exc
        else:
            raise StateError("data weights do not match this data module")
        if normalized != self._weights:
            raise StateError("data weights do not match this data module")

    def _agree_weights(self, sha: str) -> None:
        if self.world_size == 1 or self._runtime is None:
            return
        gather = getattr(self._runtime, "gather_objects", None)
        if not callable(gather):
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        gathered = gather(sha)
        if any(item != sha for item in gathered):
            raise DataError("ranks disagree on sampling weights")

    def _register_view(self, view: SnapshotEpochView) -> None:
        del view
        self._open_views += 1

    def _release_view(self, view: SnapshotEpochView) -> None:
        del view
        self._open_views = max(0, self._open_views - 1)

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
        elif supply == "online_dfn3":
            if algorithm != "online.dfn3":
                raise DataError(f"DeepFilterNet3 protocol cannot read algorithm {algorithm}")
            self._bind_sampler()
            self._warn_batch_once()
            prepare = getattr(self.protocol, "prepare", None)
            if callable(prepare):
                prepare()
        elif supply == "online_semamba":
            if algorithm != "online.semambapp_gsr":
                raise DataError(f"SEMamba protocol cannot read algorithm {algorithm}")
            self._bind_sampler()
            self._warn_batch_once()
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

    def _warn_batch_once(self) -> None:
        if self.rank != 0 or self._batch_warned:
            return
        warning = getattr(self.sampler, "batch_warning", None)
        message = warning() if callable(warning) else None
        self._batch_warned = True
        if message:
            logging.getLogger("src.experiments.data").warning("%s", message)

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

    def _cached(self, key: tuple[int, int]) -> Batch | None:
        with self._lock:
            return self._cache.get(key)

    def _inflight_snapshot(self) -> set[tuple[int, int]]:
        with self._lock:
            return set(self._inflight)

    def _raise_prefetch_error(self) -> None:
        with self._lock:
            error = self._prefetch_error
            self._prefetch_error = None
        if error is not None:
            raise error

    def _wait_for(self, key: tuple[int, int]) -> None:
        deadline = time.monotonic() + max(self.worker_join_timeout, 30.0)
        while time.monotonic() < deadline:
            with self._lock:
                if key in self._cache or key not in self._inflight or self._prefetch_error is not None:
                    return
            if self._stop.is_set():
                return
            time.sleep(0.01)
        raise DataError(f"prefetch worker timed out for micro-batch {key[0]}:{key[1]}")

    def _remember_locked(self, key: tuple[int, int], batch: Batch) -> None:
        self._cache[key] = batch
        limit = max(1, self.load_workers + 1)
        while len(self._cache) > limit:
            victims = [item for item in self._cache if item != key]
            if not victims:
                break
            del self._cache[min(victims)]

    def _ensure_workers(self) -> None:
        if self._workers or self._closed or self.load_workers < 1:
            return
        for index in range(self.load_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"audio-db-prefetch-{index}",
                daemon=False,
            )
            thread.start()
            self._workers.append(thread)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None or self._stop.is_set():
                return
            epoch, index, group = item
            key = (epoch, index)
            try:
                if self._closed or self._stop.is_set():
                    return
                batch = self._collate(group)
                with self._lock:
                    if not self._closed and not self._stop.is_set() and key not in self._cache:
                        self._remember_locked(key, batch)
            except Exception as exc:
                with self._lock:
                    if self._prefetch_error is None:
                        self._prefetch_error = exc
            finally:
                with self._lock:
                    self._inflight.discard(key)

    def _start_prefetch(
        self,
        epoch: int,
        index: int,
        groups: tuple[tuple[SampleRef, ...], ...],
    ) -> None:
        if self._closed or self.load_workers < 1:
            return
        self._ensure_workers()
        with self._lock:
            for offset in range(1, self.load_workers + 1):
                nxt = index + offset
                key = (epoch, nxt)
                if nxt >= len(groups) or key in self._cache or key in self._inflight:
                    continue
                self._inflight.add(key)
                self._jobs.put((epoch, nxt, groups[nxt]))

    def _stop_prefetch(self, *, propagate_error: bool) -> None:
        workers = list(self._workers)
        self._stop.set()
        for _ in workers:
            self._jobs.put(None)
        stuck: list[str] = []
        for thread in workers:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=self.worker_join_timeout)
            if thread.is_alive():
                stuck.append(thread.name)
        self._workers = [thread for thread in workers if thread.is_alive() and thread is not threading.current_thread()]
        self._jobs = queue.Queue()
        self._inflight.clear()
        self._cache.clear()
        error = self._prefetch_error
        self._prefetch_error = None
        if not self._workers:
            self._stop.clear()
        if stuck:
            raise DataError("prefetch workers did not exit: " + ", ".join(stuck))
        if propagate_error and error is not None:
            raise error


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


def _group_samples(
    samples: Sequence[SampleRef],
    micro_batch: int,
    group_size: int,
    keep_partial: bool,
) -> tuple[tuple[SampleRef, ...], ...]:
    """Pack one rank. Incomplete micro-batches are dropped unless ``keep_partial``."""

    rows = tuple(samples)
    if not rows:
        raise DataError("epoch sample list is empty")
    if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
        raise DataError("micro_batch must be a positive integer")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise DataError("group_size must be a positive integer")
    if keep_partial:
        packed = [rows[start : start + micro_batch] for start in range(0, len(rows), micro_batch)]
    else:
        packed = [
            rows[start : start + micro_batch]
            for start in range(0, len(rows) - len(rows) % micro_batch, micro_batch)
        ]
    if not packed:
        raise DataError("epoch does not contain a full micro-batch")
    keep = len(packed) - (len(packed) % group_size)
    if keep < 1:
        raise DataError("epoch does not contain a full update group")
    return tuple(packed[:keep])


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
