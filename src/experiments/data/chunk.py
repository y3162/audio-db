"""Chunk supply for models that train on fixed-length spans.

The epoch plan is ``plan_chunk_epoch``. The waveform adapter is chosen while
the protocol is prepared. This module does not branch on a dataset name while
planning or slicing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.experiments.data.adapters import ChunkAdapter, open_chunk_adapter
from src.experiments.data.fixed_pair import SnapshotSource, _call, _non_negative, _positive
from src.experiments.data.sample import ChunkSpan, SampleRef
from src.experiments.errors import DataError
from src.experiments.data.errors import ReproductionError
from src.experiments.data.render.chunking import ChunkEpochPlan, plan_chunk_epoch
from src.experiments.data.sampling.order import order_params
from src.experiments.data.sampling.planner import planned_case_length


class ChunkSampler:
    """Turn ``plan_chunk_epoch`` into ``SampleRef`` rows. It does not render audio."""

    def __init__(
        self,
        source: SnapshotSource,
        seed: int = 0,
        num_samples_per_epoch: int = 1,
        chunk_length: int = 2,
        shift_ratio: float = 0.5,
        num_cache_chunks: int = 1,
        shuffle: bool = True,
        global_batch: int | None = None,
        validation_scope: str = "per_rank",
        max_chunks: int | None = None,
    ) -> None:
        self.source = source
        self.seed = _non_negative("seed", seed)
        self.num_samples_per_epoch = _positive("num_samples_per_epoch", num_samples_per_epoch)
        self.chunk_length = _positive("chunk_length", chunk_length)
        if self.chunk_length < 2:
            raise ValueError("chunk_length must be at least 2")
        if isinstance(shift_ratio, bool) or not isinstance(shift_ratio, (int, float)) or float(shift_ratio) <= 0:
            raise ValueError("shift_ratio must be a positive number")
        self.shift_ratio = float(shift_ratio)
        self.num_cache_chunks = _positive("num_cache_chunks", num_cache_chunks)
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        self.shuffle = shuffle
        if validation_scope not in {"rank0", "per_rank"}:
            raise ValueError("validation_scope must be rank0 or per_rank")
        self.validation_scope = validation_scope
        if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
            raise ValueError("max_chunks must be a positive integer")
        self.max_chunks = max_chunks
        self.global_batch = None if global_batch is None else _positive("global_batch", global_batch)
        self._resolved_global_batch: int | None = None
        self._bound_world: int | None = None
        self._plans: dict[tuple[Any, ...], tuple[tuple[SampleRef, ...], ...]] = {}

    def policy(self, world_size: int) -> str:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        return "espnet_chunk_v1"

    def order_parameters(self) -> dict[str, Any]:
        """Plan inputs that ``order_params`` does not already name."""

        base = _call(order_params, "espnet_chunk_v1", self.seed)
        parameters = {
            "cache_rule": "max(num_cache_chunks // world_size, per_rank_batch)",
            "chunk_length": self.chunk_length,
            "iterator_stop": base["iterator_stop"],
            "num_cache_chunks": self.num_cache_chunks,
            "num_samples_per_epoch": self.num_samples_per_epoch,
            "rng": base["rng"],
            "sampler": base["sampler"],
            "shard": base["shard"],
            "shift": int(self.chunk_length * self.shift_ratio),
            "shift_ratio": format(self.shift_ratio, ".6f"),
            "max_chunks": self.max_chunks,
            "shuffle": self.shuffle,
            "training_seed": self.seed,
        }
        return parameters

    def bind_batch(self, batcher: object, world_size: int) -> None:
        """Bind a logical batch to its micro-batches and world size."""

        group_size = getattr(batcher, "group_size", None)
        micro_batch = getattr(batcher, "micro_batch", None)
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise DataError("chunk group_size must be a positive integer")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
            raise DataError("micro_batch must be a positive integer")
        expected = micro_batch * world_size * group_size
        if self.global_batch is not None and self.global_batch != expected:
            raise DataError(
                f"global_batch {self.global_batch} does not equal micro_batch * world_size * group_size ({expected})"
            )
        if self._resolved_global_batch != expected or self._bound_world != world_size:
            self._plans.clear()
        self._resolved_global_batch = expected
        self._bound_world = world_size

    def sequence(self, epoch: int, rank: int, world_size: int) -> tuple[SampleRef, ...]:
        return self._ranks(epoch, world_size, slot=self._train_slot(), shuffle=self.shuffle, num_samples=self.num_samples_per_epoch)[rank]

    def all_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        return self._ranks(
            epoch,
            world_size,
            slot=self._train_slot(),
            shuffle=self.shuffle,
            num_samples=self.num_samples_per_epoch,
        )

    def validation_ranks(self, epoch: int, world_size: int) -> tuple[tuple[SampleRef, ...], ...]:
        """Validation uses ``shuffle=False`` and every validation case."""

        return self._ranks(epoch, world_size, slot="valid.cases", shuffle=False, num_samples=None)

    def _train_slot(self) -> str:
        return f"{self.source.split}.cases"

    def _ranks(
        self,
        epoch: int,
        world_size: int,
        *,
        slot: str,
        shuffle: bool,
        num_samples: int | None,
    ) -> tuple[tuple[SampleRef, ...], ...]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DataError("epoch must be a non-negative integer")
        if self._resolved_global_batch is None or self._bound_world is None:
            raise DataError("chunk sampler has no global batch")
        if world_size != self._bound_world:
            raise DataError(
                f"chunk plan world_size {world_size} does not match the bound world_size {self._bound_world}"
            )
        key = (slot, epoch, world_size, shuffle, num_samples, self._resolved_global_batch, self.max_chunks)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        case_ids = self.source.case_ids(slot)
        samples = [(case_id, _call(planned_case_length, self.source.store, case_id)) for case_id in case_ids]
        try:
            plan = plan_chunk_epoch(
                samples,
                espnet_epoch=epoch + 1,
                seed=self.seed,
                world_size=world_size,
                num_samples=num_samples,
                chunk_length=self.chunk_length,
                global_batch=self._resolved_global_batch,
                num_cache_chunks=self.num_cache_chunks,
                shuffle=shuffle,
                shift_ratio=self.shift_ratio,
            )
        except (ReproductionError, ValueError) as exc:
            raise DataError(str(exc)) from exc
        ranks = _refs(plan)
        if self.max_chunks is not None:
            ranks = tuple(tuple(rank[: self.max_chunks]) for rank in ranks)
        self._plans[key] = ranks
        return ranks


class ChunkProtocol:
    """Render one planned span through the adapter bound at prepare time."""

    supply = "chunk"

    def __init__(
        self,
        source: SnapshotSource,
        verify_on_read: bool = False,
        *,
        sample_rate: int = 16000,
        resample: dict | None = None,
    ) -> None:
        if not isinstance(verify_on_read, bool):
            raise ValueError("verify_on_read must be a boolean")
        self.source = source
        self.verify_on_read = verify_on_read
        self.sample_rate = _positive("sample_rate", sample_rate)
        self.resample = resample
        self.normalize = None
        self.teacher: str | None = None
        self._adapter: ChunkAdapter | None = None

    def prepare(self) -> None:
        dataset = self.source.dataset()
        rule = dataset.get("rule_json") or {}
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        # The snapshot may record a historical normalize. This pipeline does not
        # apply it. Generated mixers still refuse a normalize of their own.
        teacher = rule.get("teacher")
        if teacher not in {"dry_clean", "reverberant_clean", "official_mixed_target"}:
            raise DataError(f"dataset teacher {teacher!r} is not a training target")
        self.teacher = str(teacher)
        self._adapter = open_chunk_adapter(
            self.source,
            verify_on_read=self.verify_on_read,
            sample_rate=self.sample_rate,
            resample=self.resample,
        )
        self._adapter.prepare()

    def render(self, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
        if self._adapter is None:
            raise DataError("chunk protocol is not prepared")
        if not sample.case_id:
            raise DataError("chunk rendering requires a case id")
        if sample.chunk is None:
            raise DataError("chunk rendering requires a span")
        noisy, clean, valid = self._adapter.render_span(sample.case_id, int(sample.chunk.start), int(sample.chunk.stop))
        if noisy.shape != clean.shape or int(noisy.shape[0]) != valid:
            raise DataError(f"chunk {sample.case_id} lost noisy/target alignment")
        if not np.isfinite(noisy).all() or not np.isfinite(clean).all():
            raise DataError(f"chunk {sample.case_id} rendered a non-finite waveform")
        return noisy, clean, valid

    def teacher_kind(self, sample: SampleRef) -> str:
        if not sample.case_id:
            raise DataError("chunk sample has no case id")
        kind = self.source.document(sample.case_id).get("teacher_kind")
        if kind in (None, "none"):
            raise DataError(f"case {sample.case_id} has no teacher")
        if self.teacher is not None and str(kind) != self.teacher:
            raise DataError(f"case {sample.case_id} teacher {kind} is not the dataset teacher {self.teacher}")
        return str(kind)


def _refs(plan: ChunkEpochPlan) -> tuple[tuple[SampleRef, ...], ...]:
    ranks: list[tuple[SampleRef, ...]] = []
    for batches in plan.rank_batches:
        rows: list[SampleRef] = []
        index = 0
        for batch in batches:
            for chunk in batch:
                case_id = str(chunk.case_id)
                start = int(chunk.start)
                stop = int(chunk.stop)
                rows.append(
                    SampleRef(
                        sample_id=case_id,
                        realization_id=f"{case_id}:{start}:{stop}",
                        index=index,
                        case_id=case_id,
                        chunk=ChunkSpan(start, stop),
                    )
                )
                index += 1
        ranks.append(tuple(rows))
    return tuple(ranks)
