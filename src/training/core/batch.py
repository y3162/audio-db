"""Exact global-batch plans.

``global_batch_size = micro_batch_size * world_size * accumulation_steps``.

``micro_batch_size_per_gpu`` is an upper bound. The plan keeps the requested
global batch and uses the largest micro batch that divides it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.training.errors import TrainingError


@dataclass(frozen=True)
class BatchPlan:
    global_batch_size: int
    micro_batch_size: int
    world_size: int
    accumulation_steps: int

    def as_dict(self) -> dict[str, int]:
        return {
            "global_batch_size": self.global_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "world_size": self.world_size,
            "accumulation_steps": self.accumulation_steps,
        }


def plan_batches(
    global_batch_size: int,
    micro_batch_size_per_gpu: int,
    world_size: int,
) -> BatchPlan:
    """Choose a micro batch and an integer accumulation count.

    Raises ``TrainingError`` when the global batch cannot be formed exactly.
    """
    if isinstance(global_batch_size, bool) or not isinstance(global_batch_size, int):
        raise TrainingError("global_batch_size must be a positive integer")
    if isinstance(micro_batch_size_per_gpu, bool) or not isinstance(micro_batch_size_per_gpu, int):
        raise TrainingError("micro_batch_size_per_gpu must be a positive integer")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TrainingError("world_size must be a positive integer")
    if global_batch_size < 1 or micro_batch_size_per_gpu < 1 or world_size < 1:
        raise TrainingError("batch sizes and world_size must be positive")
    if global_batch_size % world_size != 0:
        raise TrainingError(
            f"global_batch_size {global_batch_size} is not divisible by world_size {world_size}"
        )
    per_rank = global_batch_size // world_size
    micro = 0
    for candidate in range(min(micro_batch_size_per_gpu, per_rank), 0, -1):
        if per_rank % candidate == 0:
            micro = candidate
            break
    if micro < 1:
        raise TrainingError(
            f"no micro batch at or below {micro_batch_size_per_gpu} realizes "
            f"global_batch_size {global_batch_size} on {world_size} processes"
        )
    accumulation = per_rank // micro
    if micro * world_size * accumulation != global_batch_size:
        raise TrainingError("batch plan did not preserve global_batch_size")
    return BatchPlan(global_batch_size, micro, world_size, accumulation)
