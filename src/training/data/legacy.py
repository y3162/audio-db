"""Wrap the recipe dataset in the training-data interface.

The wrapped dataset, the sampler, and the collate path are the existing ones.
This module does not change their crop, order, or normalization.
"""

from __future__ import annotations

from typing import Iterator

from src.reproduction.ids import digest
from src.reproduction.pipeline.state import require_resume_match
from src.training.data.dataset import RecipeDataset
from src.training.data.sampler import epoch_indices
from src.training.errors import TrainingError


class LegacyTrainingData:
    """RecipeDataset viewed as ``epoch`` / ``validation`` / ``data_state``."""

    def __init__(
        self,
        train: RecipeDataset,
        validation: RecipeDataset | None,
        *,
        seed: int,
        micro_batch: int,
        accumulation: int,
        world_size: int,
    ) -> None:
        if micro_batch < 1 or accumulation < 1 or world_size < 1:
            raise TrainingError("legacy batch shape must be positive")
        self.train = train
        self.validation_set = validation
        self.seed = int(seed)
        self.micro_batch = int(micro_batch)
        self.accumulation = int(accumulation)
        self.world_size = int(world_size)

    def identity(self) -> dict:
        return self.train.identity()

    def epoch(self, epoch: int, *, rank: int, world_size: int, start_update: int = 0):
        self._check_world(world_size, rank)
        indices, updates = self._order(epoch, rank)
        if start_update < 0 or start_update > updates:
            raise TrainingError(f"start_update {start_update} is outside epoch {epoch}")
        self.train.set_epoch(epoch)

        def render(update_index: int) -> list[dict]:
            self.train.set_epoch(epoch)
            group = []
            base = update_index * self.accumulation
            for step in range(self.accumulation):
                cursor = base + step
                start = cursor * self.micro_batch
                group.append(self.train.collate(indices[start : start + self.micro_batch]))
            return group

        from src.reproduction.pipeline.training_data import EpochStream

        digest_value = self.data_state(epoch, 0)["plan_hash"]
        return EpochStream(updates, digest_value, render, start_update)

    def validation(self, *, rank: int, world_size: int, epoch: int | None = None) -> Iterator[dict]:
        self._check_world(world_size, rank)
        del rank, epoch
        if self.validation_set is None:
            return
        for index in range(len(self.validation_set)):
            yield self.validation_set.collate([index])

    def data_state(self, epoch: int, committed_updates: int) -> dict:
        if epoch < 0 or committed_updates < 0:
            raise TrainingError("legacy data position must be non-negative")
        rows = []
        updates = 0
        for rank in range(self.world_size):
            indices, updates = self._order(epoch, rank)
            rows.append([self.train.case_ids()[index] for index in indices])
        if committed_updates > updates:
            raise TrainingError(
                f"committed updates {committed_updates} exceed epoch {epoch} ({updates})"
            )
        return {
            "pipeline": "legacy",
            "seed": self.seed,
            "epoch": epoch,
            "update_in_epoch": committed_updates,
            "world_size": self.world_size,
            "ranks": list(range(self.world_size)),
            "order_policy": "global_shuffle_stride_v1",
            "micro_batch": self.micro_batch,
            "accumulation": self.accumulation,
            "plan_hash": digest(
                {"rows": rows, "micro": self.micro_batch, "accumulation": self.accumulation}
            ),
            "dataset": self.train.identity(),
        }

    def check_resume(self, saved: dict) -> None:
        current = self.data_state(int(saved["epoch"]), int(saved["update_in_epoch"]))
        require_resume_match(saved, current)

    def close(self) -> None:
        return None

    def _order(self, epoch: int, rank: int) -> tuple[list[int], int]:
        indices = epoch_indices(len(self.train), self.seed, epoch, self.world_size, rank)
        micro_batches = len(indices) // self.micro_batch
        updates = micro_batches // self.accumulation
        usable = updates * self.accumulation * self.micro_batch
        if updates < 1:
            raise TrainingError(
                f"epoch {epoch} on rank {rank} has {len(indices) // self.micro_batch} micro-batches, "
                f"which is fewer than accumulation_steps {self.accumulation}"
            )
        return indices[:usable], updates

    def _check_world(self, world_size: int, rank: int) -> None:
        if world_size != self.world_size or rank < 0 or rank >= world_size:
            raise TrainingError(
                f"rank {rank} world_size {world_size} does not match the legacy data plan"
            )
