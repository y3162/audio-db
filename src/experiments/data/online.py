"""Online DeepFilterNet3 mixing and SEMamba++ degradation.

Samplers choose indexes. Protocols call the existing numeric functions.
Neither constructs ``OnlineDFN3TrainingData`` or ``OnlineSemambaTrainingData``.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from src.experiments.data.fixed_pair import SnapshotSource, _non_negative, _positive
from src.experiments.data.sample import SampleRef
from src.experiments.errors import DataError
from src.experiments.data.errors import ReproductionError
from src.experiments.data.render.clips import Dfn3Clip, SemambaClip
from src.experiments.data.mix.online_dfn3 import POLICY as DFN3_POLICY
from src.experiments.data.mix.online_dfn3 import plan_online_epoch
from src.experiments.data.protocols.online_dfn3 import (
    NUM_SAMPLES,
    SAMPLE_RATE as DFN3_RATE,
    DFN3OnlineSpec,
    render_mixture,
    sample_seed as dfn3_sample_seed,
    speech_key_positions,
)
from src.experiments.data.protocols.online_semamba import (
    SAMPLE_RATE as SEMAMBA_RATE,
    SEGMENT_SIZE,
    render_example,
    sample_seed as semamba_sample_seed,
)


def online_sample(
    *,
    protocol_id: str,
    snapshot_hash: str,
    seed: int,
    split: str,
    epoch: int,
    index: int,
    case_id: str | None,
) -> SampleRef:
    """Sample identity from the Phase 3-A online table."""

    if split not in {"train", "valid", "test"}:
        raise DataError(f"unknown split {split}")
    epoch_key = int(epoch) if split == "train" else 0
    if split == "train" or not case_id:
        sample_id = f"{protocol_id}:{split}:{int(index)}"
    else:
        sample_id = str(case_id)
    realization_id = (
        f"{protocol_id}:{snapshot_hash}:{int(seed)}:{split}:{epoch_key}:{int(index)}"
    )
    stored_case = None if not case_id else str(case_id)
    return SampleRef(
        sample_id=sample_id,
        realization_id=realization_id,
        index=int(index),
        case_id=stored_case,
        split=split,
        plan_epoch=int(epoch),
    )


class Dfn3IndexSampler:
    """Sample indexes from ``plan_online_epoch``. Speech keys are not indexes."""

    def __init__(
        self,
        source: SnapshotSource,
        seed: int = 0,
        global_batch: int | None = None,
        *,
        validation_batch: int = 64,
        validation_scope: str = "rank0",
    ) -> None:
        self.source = source
        self.seed = _non_negative("seed", seed)
        self.global_batch = None if global_batch is None else _positive("global_batch", global_batch)
        if validation_scope != "rank0":
            raise ValueError("DeepFilterNet3 validation is read on rank 0")
        self.validation_scope = validation_scope
        self.validation_batch_width = _positive("validation_batch", validation_batch)
        self._micro_batch: int | None = None
        self._world: int | None = None
        self._batch_warning: str | None = None
        self._counts: dict[str, int] = {}
        self._plans: dict[tuple[Any, ...], tuple[tuple[SampleRef, ...], ...]] = {}

    def policy(self, world_size: int) -> str:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        return DFN3_POLICY

    def order_parameters(self) -> dict[str, Any]:
        factor = self._factor()
        fractional = abs(factor - round(factor)) > 1e-9
        return {
            "count": self._count(self._split()),
            "dataset_id": str(self.source.dataset().get("dataset_id")),
            "micro_batch": self._micro_batch,
            "sampling_factor": format(factor, ".6f"),
            "speech_key_epoch": "epoch" if fractional else "zero",
            "world_size": self._world,
        }

    def bind_batch(self, batcher: object, world_size: int) -> None:
        """Require ``group_size == 1`` and keep a short final micro-batch."""

        group_size = getattr(batcher, "group_size", None)
        micro_batch = getattr(batcher, "micro_batch", None)
        if group_size != 1:
            raise DataError("DeepFilterNet3 group_size must be 1")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
            raise DataError("micro_batch must be a positive integer")
        effective = micro_batch * world_size
        if self.global_batch is not None and self.global_batch != effective:
            raise DataError(
                f"global_batch {self.global_batch} does not equal micro_batch * world_size ({effective})"
            )
        if self._micro_batch != micro_batch or self._world != world_size:
            self._plans.clear()
        self._micro_batch = micro_batch
        self._world = world_size
        keep = getattr(batcher, "keep_partial", None)
        if keep is not None:
            batcher.keep_partial = True  # type: ignore[attr-defined]
        shipped = int(self._rule()["global_batch"])
        if effective != shipped:
            self._batch_warning = (
                f"warning: global_batch_size {effective} differs from the shipped cap {shipped}"
            )
        else:
            self._batch_warning = None

    def batch_warning(self) -> str | None:
        return self._batch_warning

    def sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        return self._ranks(self._split(), epoch, world_size)[rank]

    def all_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        return self._ranks(self._split(), epoch, world_size)

    def validation_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        """All validation indexes, ordered for this epoch, returned on rank 0."""

        self._check_epoch(epoch)
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        count = self._count("valid")
        ranks = self._plan("valid", epoch, world_size=1, micro_batch=count)
        return tuple(ranks[0] if rank == 0 else () for rank in range(world_size))

    def validation_batch(self) -> tuple[int, bool, int]:
        return self.validation_batch_width, True, 1

    def _split(self) -> str:
        return self.source.split

    def _ranks(self, split: str, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        self._check_epoch(epoch)
        if self._micro_batch is None or self._world is None:
            raise DataError("DeepFilterNet3 sampler has no batch")
        if world_size != self._world:
            raise DataError(
                f"DeepFilterNet3 plan world_size {world_size} does not match the bound world_size {self._world}"
            )
        return self._plan(split, epoch, world_size=world_size, micro_batch=self._micro_batch)

    def _plan(
        self,
        split: str,
        epoch: int,
        *,
        world_size: int,
        micro_batch: int,
    ) -> tuple[tuple[SampleRef, ...], ...]:
        key = (split, epoch, world_size, micro_batch)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        try:
            plan = plan_online_epoch(
                split=split,
                epoch=epoch,
                count=self._count(split),
                seed=self.seed,
                world_size=world_size,
                micro_batch=micro_batch,
                protocol_id=self._protocol_id(),
                snapshot_hash=self.source.snapshot_hash,
                case_ids=self._case_list(split),
            )
        except (ReproductionError, ValueError) as exc:
            raise DataError(str(exc)) from exc
        ranks = _from_plan(
            plan,
            split=split,
            epoch=epoch,
            protocol_id=self._protocol_id(),
            snapshot_hash=self.source.snapshot_hash,
            seed=self.seed,
        )
        self._plans[key] = ranks
        return ranks

    def _count(self, split: str) -> int:
        cached = self._counts.get(split)
        if cached is not None:
            return cached
        speech = self.source.material_ids(f"{split}.speech")
        if not speech:
            raise DataError(f"snapshot is missing {split}.speech")
        try:
            length = len(
                speech_key_positions(
                    len(speech),
                    self._factor(),
                    protocol_id=self._protocol_id(),
                    snapshot_hash=self.source.snapshot_hash,
                    seed=self.seed,
                    split=split,
                    epoch=0,
                )
            )
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc
        self._counts[split] = length
        return length

    def _case_list(self, split: str) -> list[str | None]:
        count = self._count(split)
        slot = f"{split}.cases"
        try:
            ids = list(self.source.case_ids(slot))
        except DataError:
            return [None] * count
        if len(ids) != count:
            raise DataError(f"{slot} count does not match the speech key list")
        return ids

    def _factor(self) -> float:
        online = self._rule().get("online")
        if not isinstance(online, dict) or "sampling_factor" not in online:
            raise DataError("snapshot has no sampling_factor")
        factor = online["sampling_factor"]
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or float(factor) <= 0:
            raise DataError("sampling_factor must be positive")
        return float(factor)

    def _protocol_id(self) -> str:
        protocol_id = self.source.dataset().get("protocol_id")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise DataError("snapshot has no protocol id")
        return protocol_id

    def _rule(self) -> dict[str, Any]:
        rule = self.source.dataset().get("rule_json")
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        return rule

    def _check_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DataError("epoch must be a non-negative integer")


class Dfn3Protocol:
    """One 48 kHz mixture from ``render_mixture``. The speech row is the key list."""

    supply = "online_dfn3"

    def __init__(
        self,
        source: SnapshotSource,
        verify_on_read: bool = False,
        *,
        sample_rate: int = DFN3_RATE,
        window: int = NUM_SAMPLES,
    ) -> None:
        if not isinstance(verify_on_read, bool):
            raise ValueError("verify_on_read must be a boolean")
        if int(sample_rate) != DFN3_RATE:
            raise ValueError("DeepFilterNet3 pipeline sample_rate must be 48000")
        if int(window) != NUM_SAMPLES:
            raise ValueError("DeepFilterNet3 pipeline window must be 144000")
        self.source = source
        self.verify_on_read = verify_on_read
        self.sample_rate = DFN3_RATE
        self.window = NUM_SAMPLES
        self.normalize = None
        self.teacher: str | None = None
        self.spec = DFN3OnlineSpec.shipped()
        self._clips: dict[str, tuple[Dfn3Clip, ...]] = {}
        self._clip_lock = threading.Lock()

    def prepare(self) -> None:
        rule = self._rule()
        if int(rule.get("sample_rate") or 0) != DFN3_RATE:
            raise DataError("DeepFilterNet3 dataset sample rate is not 48000")
        if rule.get("teacher") != "official_mixed_target":
            raise DataError("DeepFilterNet3 dataset teacher is not official_mixed_target")
        self.teacher = "official_mixed_target"
        self._clips.clear()

    def render(self, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
        split, epoch, seed = _placement(sample)
        speech = self._role(split, "speech")
        if sample.index < 0 or sample.index >= self._key_length(split, seed):
            raise DataError(f"{split} sample {sample.index} is outside the dataset")
        try:
            keys = speech_key_positions(
                len(speech),
                self._factor(),
                protocol_id=self._protocol_id(),
                snapshot_hash=self.source.snapshot_hash,
                seed=seed,
                split=split,
                epoch=epoch,
            )
            if sample.index >= len(keys):
                raise DataError(f"{split} sample {sample.index} is outside the speech key list")
            rng = np.random.Generator(
                np.random.PCG64(
                    dfn3_sample_seed(
                        self._protocol_id(),
                        self.source.snapshot_hash,
                        seed,
                        split,
                        epoch,
                        sample.index,
                    )
                )
            )
            mixture = render_mixture(
                self.spec,
                speech,
                self._role(split, "noise"),
                self._role(split, "rir"),
                speech_index=keys[int(sample.index)],
                split=split,
                rng=rng,
            )
        except DataError:
            raise
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc
        except OSError as exc:
            raise DataError(str(exc)) from exc
        except Exception as exc:
            if exc.__class__.__name__ == "LibsndfileError":
                raise DataError(str(exc)) from exc
            raise
        return np.ascontiguousarray(mixture.noisy, dtype=np.float32), np.ascontiguousarray(mixture.clean, dtype=np.float32), NUM_SAMPLES

    def teacher_kind(self, sample: SampleRef) -> str:
        return _teacher(self.source, sample, "official_mixed_target")

    def _key_length(self, split: str, seed: int) -> int:
        try:
            return len(
                speech_key_positions(
                    len(self._role(split, "speech")),
                    self._factor(),
                    protocol_id=self._protocol_id(),
                    snapshot_hash=self.source.snapshot_hash,
                    seed=seed,
                    split=split,
                    epoch=0,
                )
            )
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc

    def _role(self, split: str, role: str) -> list[Dfn3Clip]:
        slot = f"{split}.{role}"
        cached = self._clips.get(slot)
        if cached is not None:
            return list(cached)
        with self._clip_lock:
            cached = self._clips.get(slot)
            if cached is None:
                if role == "rir":
                    try:
                        ids = self.source.material_ids(slot)
                    except DataError:
                        ids = ()
                else:
                    ids = self.source.material_ids(slot)
                if not ids and role != "rir":
                    raise DataError(f"snapshot is missing {slot}")
                try:
                    cached = tuple(Dfn3Clip(self.source.store, asset_id) for asset_id in ids)
                except ReproductionError as exc:
                    raise DataError(str(exc)) from exc
                self._clips[slot] = cached
        return list(cached)

    def _factor(self) -> float:
        online = self._rule().get("online")
        if not isinstance(online, dict) or "sampling_factor" not in online:
            raise DataError("snapshot has no sampling_factor")
        return float(online["sampling_factor"])

    def _protocol_id(self) -> str:
        protocol_id = self.source.dataset().get("protocol_id")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise DataError("snapshot has no protocol id")
        return protocol_id

    def _rule(self) -> dict[str, Any]:
        rule = self.source.dataset().get("rule_json")
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        return rule


class SemambaProtocol:
    """One 16 kHz degraded pair from ``render_example``."""

    supply = "online_semamba"

    def __init__(
        self,
        source: SnapshotSource,
        verify_on_read: bool = False,
        *,
        sample_rate: int = SEMAMBA_RATE,
        window: int = SEGMENT_SIZE,
    ) -> None:
        if not isinstance(verify_on_read, bool):
            raise ValueError("verify_on_read must be a boolean")
        if int(sample_rate) != SEMAMBA_RATE:
            raise ValueError("SEMamba++ pipeline sample_rate must be 16000")
        if int(window) != SEGMENT_SIZE:
            raise ValueError("SEMamba++ pipeline window must be 48000")
        self.source = source
        self.verify_on_read = verify_on_read
        self.sample_rate = SEMAMBA_RATE
        self.window = SEGMENT_SIZE
        self.normalize = "independent_peak_v1"
        self.teacher: str | None = None
        self.seed: int | None = None
        self._clips: dict[str, tuple[SemambaClip, ...]] = {}
        self._clip_lock = threading.Lock()

    def prepare(self) -> None:
        rule = self._rule()
        if int(rule.get("sample_rate") or 0) != SEMAMBA_RATE:
            raise DataError("SEMamba++ dataset sample rate is not 16000")
        if rule.get("teacher") != "dry_clean":
            raise DataError("SEMamba++ dataset teacher is not dry_clean")
        self.teacher = "dry_clean"
        self._clips.clear()

    def material_count(self, split: str) -> int:
        ids = self.source.material_ids(f"{split}.speech")
        if not ids:
            raise DataError(f"snapshot is missing {split}.speech")
        return len(ids)

    def reference(self, index: int, epoch: int, split: str) -> SampleRef:
        count = self.material_count(split)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count:
            raise DataError(f"{split} sample {index} is outside the dataset")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DataError("epoch must be a non-negative integer")
        return online_sample(
            protocol_id=self._protocol_id(),
            snapshot_hash=self.source.snapshot_hash,
            seed=self._seed(),
            split=split,
            epoch=epoch,
            index=index,
            case_id=self._case(split, index, count),
        )

    def render(self, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
        split, epoch, seed = _placement(sample)
        if sample.index < 0 or sample.index >= self.material_count(split):
            raise DataError(f"{split} sample {sample.index} is outside the dataset")
        try:
            rng = np.random.Generator(
                np.random.PCG64(
                    semamba_sample_seed(
                        self._protocol_id(),
                        self.source.snapshot_hash,
                        seed,
                        split,
                        epoch,
                        sample.index,
                    )
                )
            )
            example = render_example(
                self._role(split, "speech"),
                self._role(split, "noise"),
                self._role(split, "rir"),
                speech_index=int(sample.index),
                split=split,
                rng=rng,
            )
        except DataError:
            raise
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc
        except OSError as exc:
            raise DataError(str(exc)) from exc
        except Exception as exc:
            if exc.__class__.__name__ in {"LibsndfileError", "NoBackendError"}:
                raise DataError(str(exc)) from exc
            raise
        return (
            np.ascontiguousarray(example.degraded, dtype=np.float32),
            np.ascontiguousarray(example.clean, dtype=np.float32),
            SEGMENT_SIZE,
        )

    def teacher_kind(self, sample: SampleRef) -> str:
        return _teacher(self.source, sample, "dry_clean")

    def _seed(self) -> int:
        """Training seed is not on the protocol. Callers that build refs pass it in."""

        seed = getattr(self, "seed", None)
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
        raise DataError("SEMamba protocol has no seed")

    def _case(self, split: str, index: int, count: int) -> str | None:
        slot = f"{split}.cases"
        try:
            ids = self.source.case_ids(slot)
        except DataError:
            return None
        if len(ids) != count:
            raise DataError(f"{slot} count does not match the speech list")
        return ids[index]

    def _role(self, split: str, role: str) -> list[SemambaClip]:
        slot = f"{split}.{role}"
        cached = self._clips.get(slot)
        if cached is not None:
            return list(cached)
        with self._clip_lock:
            cached = self._clips.get(slot)
            if cached is None:
                ids = self.source.material_ids(slot)
                if not ids:
                    raise DataError(f"snapshot is missing {slot}")
                try:
                    cached = tuple(SemambaClip(self.source.store, asset_id) for asset_id in ids)
                except ReproductionError as exc:
                    raise DataError(str(exc)) from exc
                self._clips[slot] = cached
        return list(cached)

    def _protocol_id(self) -> str:
        protocol_id = self.source.dataset().get("protocol_id")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise DataError("snapshot has no protocol id")
        return protocol_id

    def _rule(self) -> dict[str, Any]:
        rule = self.source.dataset().get("rule_json")
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        return rule


def _from_plan(
    plan: Any,
    *,
    split: str,
    epoch: int,
    protocol_id: str,
    snapshot_hash: str,
    seed: int,
) -> tuple[tuple[SampleRef, ...], ...]:
    ranks: list[tuple[SampleRef, ...]] = []
    for groups in plan.rank_groups:
        rows: list[SampleRef] = []
        for group in groups:
            for micro in group.micro_batches:
                for sample in micro.samples:
                    rows.append(
                        online_sample(
                            protocol_id=protocol_id,
                            snapshot_hash=snapshot_hash,
                            seed=seed,
                            split=split,
                            epoch=epoch,
                            index=int(sample.index),
                            case_id=None if sample.case_id is None else str(sample.case_id),
                        )
                    )
        ranks.append(tuple(rows))
    return tuple(ranks)


def _placement(sample: SampleRef) -> tuple[str, int, int]:
    if sample.split is None or sample.plan_epoch is None:
        raise DataError("online sample has no split or epoch")
    pieces = sample.realization_id.rsplit(":", 4)
    if len(pieces) != 5:
        raise DataError("online realization id is invalid")
    try:
        seed = int(pieces[1])
    except ValueError as exc:
        raise DataError("online realization id has no seed") from exc
    if pieces[2] != sample.split:
        raise DataError("online realization split does not match the sample")
    return sample.split, int(sample.plan_epoch), seed


def _teacher(source: SnapshotSource, sample: SampleRef, expected: str) -> str:
    if sample.case_id:
        kind = source.document(sample.case_id).get("teacher_kind")
        if kind != expected:
            raise DataError(f"case {sample.case_id} teacher {kind} is not {expected}")
    return expected
