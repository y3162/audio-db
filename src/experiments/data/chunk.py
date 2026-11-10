"""Chunk supply for TF-GridNet and TF-Locoformer.

The epoch plan is ``plan_chunk_epoch``. Waveforms come from the MS-SNSD or
DNS ``render_realization`` path that the on-demand readers already use.
This module does not construct those reader classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.experiments.data.fixed_pair import SnapshotSource, _call, _non_negative, _positive
from src.experiments.data.sample import ChunkSpan, SampleRef
from src.experiments.errors import DataError
from src.experiments.data.errors import ReproductionError
from src.experiments.data.render.chunking import ChunkEpochPlan, plan_chunk_epoch
from src.experiments.data.mix.dns_ondemand import dns_rendering, render_dns_pair
from src.experiments.data.mix.ondemand import ms_snsd_rendering, render_ms_snsd_pair
from src.experiments.data.sampling.order import order_params
from src.experiments.data.sampling.planner import chunk_case_length

_MS = "pregenerated.ms_snsd_tfgridnet"
_DNS = "pregenerated.dns_tflocoformer"


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
        return {
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
            "shuffle": self.shuffle,
            "training_seed": self.seed,
        }

    def bind_batch(self, batcher: object, world_size: int) -> None:
        """Require ``group_size == 1`` and ``global_batch == micro_batch * world_size``."""

        group_size = getattr(batcher, "group_size", None)
        micro_batch = getattr(batcher, "micro_batch", None)
        if group_size != 1:
            raise DataError("chunk group_size must be 1")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise DataError("world_size must be a positive integer")
        if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
            raise DataError("micro_batch must be a positive integer")
        expected = micro_batch * world_size
        if self.global_batch is not None and self.global_batch != expected:
            raise DataError(
                f"global_batch {self.global_batch} does not equal micro_batch * world_size ({expected})"
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
        key = (slot, epoch, world_size, shuffle, num_samples, self._resolved_global_batch)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        case_ids = self.source.case_ids(slot)
        samples = [(case_id, _call(chunk_case_length, self.source.store, case_id)) for case_id in case_ids]
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
        self._plans[key] = ranks
        return ranks


class ChunkProtocol:
    """Render one planned span from the snapshot algorithm's realization function."""

    supply = "chunk"

    def __init__(self, source: SnapshotSource, verify_on_read: bool = False) -> None:
        if not isinstance(verify_on_read, bool):
            raise ValueError("verify_on_read must be a boolean")
        self.source = source
        self.verify_on_read = verify_on_read
        self._kind: str | None = None
        self._speech: dict[str, Any] | None = None
        self._noise: dict[str, Any] | None = None
        self._retry: str | None = None
        self._renderer: str | None = None

    def prepare(self) -> None:
        dataset = self.source.dataset()
        rule = dataset.get("rule_json") or {}
        if not isinstance(rule, dict):
            raise DataError("snapshot rule_json is missing")
        if rule.get("normalize") is not None:
            raise DataError("chunk audio is not RMS-normalized at load time")
        algorithm = str(dataset.get("algorithm"))
        if algorithm == _MS:
            slot = "test.cases" if self.source.split == "test" else "train.cases"
            speech, noise = _call(ms_snsd_rendering, self.source.store, slot)
            self._kind = "ms_snsd"
            self._retry = None
        elif algorithm == _DNS:
            speech, noise, renderer, retry = _call(dns_rendering, self.source.store)
            self._kind = "dns"
            self._renderer = renderer
            self._retry = retry
        else:
            raise DataError(f"chunk protocol cannot read algorithm {algorithm}")
        self._speech = speech
        self._noise = noise

    def render(self, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
        if self._kind is None or self._speech is None or self._noise is None:
            raise DataError("chunk protocol is not prepared")
        if not sample.case_id or sample.chunk is None:
            raise DataError("chunk rendering requires a case id and a span")
        if self._kind == "ms_snsd":
            clean, noisy = _call(
                render_ms_snsd_pair,
                self.source.store,
                sample.case_id,
                speech=self._speech,
                noise=self._noise,
                verify_on_read=self.verify_on_read,
            )
        else:
            clean, noisy = _call(
                render_dns_pair,
                self.source.store,
                sample.case_id,
                speech=self._speech,
                noise=self._noise,
                retry_policy=str(self._retry),
                renderer=str(self._renderer),
                verify_on_read=self.verify_on_read,
            )
        start = int(sample.chunk.start)
        stop = int(sample.chunk.stop)
        if start < 0 or stop > int(clean.shape[0]) or stop <= start:
            raise DataError(f"chunk {start}:{stop} is outside {sample.case_id} ({int(clean.shape[0])} samples)")
        if self._kind == "ms_snsd":
            clean_row = clean[start:stop].copy()
            noisy_row = noisy[start:stop].copy()
        else:
            clean_row = np.ascontiguousarray(clean[start:stop])
            noisy_row = np.ascontiguousarray(noisy[start:stop])
        return noisy_row, clean_row, stop - start

    def teacher_kind(self, sample: SampleRef) -> str:
        if not sample.case_id:
            raise DataError("chunk sample has no case id")
        kind = self.source.document(sample.case_id).get("teacher_kind")
        if kind in (None, "none"):
            raise DataError(f"case {sample.case_id} has no teacher")
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
