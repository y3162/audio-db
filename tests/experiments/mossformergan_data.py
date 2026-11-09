"""Fixed-length waveform pairs for MossFormerGAN trainer tests.

This is not a snapshot reader. Algorithm tests use it so an epoch length is
known. The committed YAML still points at the VoiceBank snapshot.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from src.experiments.data import Batch
from src.experiments.errors import StateError


class WavePairData:
    """Deterministic noisy/clean pairs. Every rank has a validation batch."""

    def __init__(
        self,
        updates_per_epoch: int = 2,
        micro_batch: int = 1,
        length: int = 2400,
        seed: int = 17,
        profile: str = "official_faithful",
        selection_uses_test: bool = True,
        group_size: int = 1,
        nan_indexes: tuple[int, ...] | list[int] = (),
    ) -> None:
        if isinstance(updates_per_epoch, bool) or not isinstance(updates_per_epoch, int) or updates_per_epoch < 1:
            raise ValueError("updates_per_epoch must be a positive integer")
        if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
            raise ValueError("micro_batch must be a positive integer")
        if isinstance(length, bool) or not isinstance(length, int) or length < 2:
            raise ValueError("length must be an integer >= 2")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be an integer >= 0")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be a positive integer")
        if not isinstance(selection_uses_test, bool):
            raise ValueError("selection_uses_test must be a boolean")
        if isinstance(nan_indexes, bool) or not isinstance(nan_indexes, (tuple, list)):
            raise ValueError("nan_indexes must be a list")
        self.updates_per_epoch = updates_per_epoch
        self.micro_batch = micro_batch
        self.length = length
        self.seed = seed
        self.profile = profile
        self.selection_uses_test = selection_uses_test
        self.group_size = group_size
        self.nan_indexes = tuple(int(item) for item in nan_indexes)
        self.epoch = 0
        self.update_index = 0
        self.world_size = 1
        self.rank = 0
        self.plan_hash = (
            f"moss-{updates_per_epoch}-{micro_batch}-{length}-{seed}-{profile}-"
            f"{int(selection_uses_test)}-{group_size}-{','.join(str(item) for item in self.nan_indexes)}"
        )

    def setup(self, runtime: object) -> None:
        self.world_size = int(getattr(runtime, "world_size"))
        self.rank = int(getattr(runtime, "rank"))

    def identity(self) -> dict[str, object]:
        return {
            "algorithm": "fixed_pair.mossformergan_se_16k",
            "class": "tests.experiments.mossformergan_data.WavePairData",
            "group_size": self.group_size,
            "length": self.length,
            "micro_batch": self.micro_batch,
            "profile": self.profile,
            "seed": self.seed,
            "selection_uses_test": self.selection_uses_test,
            "updates_per_epoch": self.updates_per_epoch,
        }

    def train_epoch(self, epoch: int, *, start_update: int) -> "_WaveView":
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        if isinstance(start_update, bool) or not isinstance(start_update, int):
            raise StateError(f"start_update {start_update!r} is not an integer")
        if start_update < 0 or start_update > self.updates_per_epoch:
            raise StateError(f"start_update {start_update} is outside 0..{self.updates_per_epoch}")
        return _WaveView(self, epoch, start_update)

    def validation_epoch(self, epoch: int) -> Iterator[Batch]:
        del epoch
        return iter((self._batch(split=1, epoch=0, index=0, rank=self.rank),))

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
            raise StateError("wave data identity does not match this run")
        if state.get("plan_hash") != self.plan_hash:
            raise StateError("wave data plan_hash does not match this run")
        epoch = state.get("epoch")
        index = state.get("update_index")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or isinstance(index, bool) or not isinstance(index, int):
            raise StateError("wave data cursor is invalid")
        if index < 0 or index > self.updates_per_epoch or epoch < 0:
            raise StateError("wave data cursor is outside this epoch")
        self.epoch = epoch
        self.update_index = index

    def close(self) -> None:
        return

    def _batch(self, *, split: int, epoch: int, index: int, rank: int) -> Batch:
        generator = torch.Generator()
        generator.manual_seed(self.seed + split * 100000 + epoch * 1000 + index * 10 + rank)
        noisy = torch.randn(self.micro_batch, self.length, generator=generator) * 0.05
        clean = torch.randn(self.micro_batch, self.length, generator=generator) * 0.05
        if split == 0 and index in self.nan_indexes:
            noisy = torch.full_like(noisy, float("nan"))
        sample_id = f"r{rank}-e{epoch}-i{index}"
        return Batch(
            inputs=noisy,
            targets=clean,
            sample_id=sample_id,
            realization_id=sample_id,
            lengths=torch.full((self.micro_batch,), self.length, dtype=torch.long),
            sample_ids=(sample_id,),
            realization_ids=(sample_id,),
        )


class _WaveView:
    def __init__(self, data: WavePairData, epoch: int, start_update: int) -> None:
        self._data = data
        self._epoch = epoch
        self._start = start_update
        self.updates = data.updates_per_epoch
        self.plan_hash = data.plan_hash

    def batch(self, update_index: int) -> list[Batch]:
        if update_index < self._start or update_index >= self.updates:
            raise StateError(f"update index {update_index} is outside this view")
        return [self._data._batch(split=0, epoch=self._epoch, index=update_index, rank=self._data.rank)]

    def close(self) -> None:
        return
