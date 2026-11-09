"""Synthetic chunk-shaped batches for the TF-GridNet trainer tests.

This is not a snapshot reader. Algorithm tests use it so epoch length and
validation counts are fixed. The snapshot path is covered separately.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from src.experiments.data import Batch
from src.experiments.errors import DataError, StateError


class ScriptedChunkData:
    """Deterministic waveform batches with an explicit validation split per rank."""

    def __init__(
        self,
        updates_per_epoch: int = 2,
        micro_batch: int = 1,
        length: int = 32,
        seed: int = 17,
        valid_sizes: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if isinstance(updates_per_epoch, bool) or not isinstance(updates_per_epoch, int) or updates_per_epoch < 1:
            raise ValueError("updates_per_epoch must be a positive integer")
        if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
            raise ValueError("micro_batch must be a positive integer")
        if isinstance(length, bool) or not isinstance(length, int) or length < 2:
            raise ValueError("length must be an integer >= 2")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be an integer >= 0")
        sizes = ((2, 1),) if valid_sizes is None else tuple(tuple(int(item) for item in row) for row in valid_sizes)
        if not sizes or any(not row or any(item < 1 for item in row) for row in sizes):
            raise ValueError("valid_sizes must list positive batch sizes for each rank")
        self.updates_per_epoch = updates_per_epoch
        self.micro_batch = micro_batch
        self.length = length
        self.seed = seed
        self.valid_sizes = sizes
        self.group_size = 1
        self.plan_hash = f"scripted-{updates_per_epoch}-{micro_batch}-{length}-{seed}-{len(sizes)}"
        self.epoch = 0
        self.update_index = 0
        self.world_size = 1
        self.rank = 0

    def setup(self, runtime: object) -> None:
        self.world_size = int(getattr(runtime, "world_size"))
        self.rank = int(getattr(runtime, "rank"))
        if self.world_size > len(self.valid_sizes):
            raise DataError(
                f"valid_sizes has {len(self.valid_sizes)} ranks and world size is {self.world_size}"
            )

    def identity(self) -> dict[str, object]:
        return {
            "length": self.length,
            "micro_batch": self.micro_batch,
            "name": "scripted-chunk",
            "seed": self.seed,
            "updates_per_epoch": self.updates_per_epoch,
            "valid_sizes": [list(row) for row in self.valid_sizes],
        }

    def train_epoch(self, epoch: int, *, start_update: int) -> "_ScriptedView":
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        if isinstance(start_update, bool) or not isinstance(start_update, int):
            raise StateError(f"start_update {start_update!r} is not an integer")
        if start_update < 0 or start_update > self.updates_per_epoch:
            raise StateError(f"start_update {start_update} is outside 0..{self.updates_per_epoch}")
        return _ScriptedView(self, epoch, start_update)

    def validation_epoch(self, epoch: int) -> Iterator[Batch]:
        del epoch
        sizes = self.valid_sizes[0] if self.world_size == 1 else self.valid_sizes[self.rank]
        for index, size in enumerate(sizes):
            yield self._waveforms(split=1, epoch=0, index=index, batch=size, rank=self.rank)

    def state_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "identity": self.identity(),
            "plan_hash": self.plan_hash,
            "update_index": self.update_index,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("identity") != self.identity() or state.get("world_size") != self.world_size:
            raise StateError("scripted data identity does not match this run")
        if state.get("plan_hash") != self.plan_hash:
            raise StateError("scripted data plan_hash does not match this run")
        epoch = state.get("epoch")
        index = state.get("update_index")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or isinstance(index, bool) or not isinstance(index, int):
            raise StateError("scripted data position is invalid")
        self.epoch = epoch
        self.update_index = index

    def close(self) -> None:
        return

    def _waveforms(self, *, split: int, epoch: int, index: int, batch: int, rank: int) -> Batch:
        generator = torch.Generator()
        generator.manual_seed(self.seed + split * 100000 + epoch * 1000 + index * 10 + rank)
        noisy = torch.randn(batch, self.length, generator=generator)
        clean = torch.randn(batch, self.length, generator=generator)
        sample_id = f"r{rank}-e{epoch}-i{index}"
        return Batch(
            inputs=noisy,
            targets=clean,
            sample_id=sample_id,
            realization_id=sample_id,
            lengths=torch.full((batch,), self.length, dtype=torch.long),
            sample_ids=tuple(f"{sample_id}-{item}" for item in range(batch)),
            realization_ids=tuple(f"{sample_id}-{item}" for item in range(batch)),
        )


class _ScriptedView:
    def __init__(self, data: ScriptedChunkData, epoch: int, start_update: int) -> None:
        self._data = data
        self._epoch = epoch
        self._start = start_update
        self.updates = data.updates_per_epoch
        self.plan_hash = f"scripted-{data.updates_per_epoch}"

    def batch(self, update_index: int) -> list[Batch]:
        if update_index < self._start or update_index >= self.updates:
            raise StateError(f"update index {update_index} is outside this view")
        return [
            self._data._waveforms(
                split=0,
                epoch=self._epoch,
                index=update_index,
                batch=self._data.micro_batch,
                rank=self._data.rank,
            )
        ]

    def close(self) -> None:
        return
