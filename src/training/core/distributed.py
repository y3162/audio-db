"""Process setup for CPU, one GPU, and DistributedDataParallel.

GPU choice is ``CUDA_VISIBLE_DEVICES`` plus ``torchrun``. This module does not
record a physical GPU number in the model or the checkpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.training.errors import TrainingError


@dataclass(frozen=True)
class ProcessInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    distributed: bool

    @property
    def uses_cuda(self) -> bool:
        return self.device.type == "cuda"


def setup_process(device_name: str) -> ProcessInfo:
    """Initialize a process group only when ``WORLD_SIZE`` is greater than 1.

    A single CPU or GPU process keeps the same training step with no process
    group. Several CPU processes use gloo. Several GPUs use nccl.
    """
    if device_name not in ("cpu", "cuda"):
        raise TrainingError(f"device must be 'cpu' or 'cuda', got {device_name!r}")
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size < 1 or rank < 0 or rank >= world_size or local_rank < 0:
        raise TrainingError(
            f"invalid distributed rank {rank} local_rank {local_rank} world_size {world_size}"
        )
    distributed = world_size > 1
    if device_name == "cpu":
        device = torch.device("cpu")
        if distributed and not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        if not torch.cuda.is_available():
            raise TrainingError("CUDA was requested but is not available")
        count = torch.cuda.device_count()
        if local_rank >= count:
            raise TrainingError(
                f"local_rank {local_rank} is outside the {count} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return ProcessInfo(rank, world_size, local_rank, device, distributed)


def wrap_module(module: torch.nn.Module, process: ProcessInfo) -> torch.nn.Module:
    if not process.distributed:
        return module
    if process.uses_cuda:
        return DistributedDataParallel(
            module,
            device_ids=[process.local_rank],
            output_device=process.local_rank,
        )
    return DistributedDataParallel(module)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def barrier(process: ProcessInfo) -> None:
    if process.distributed and dist.is_initialized():
        dist.barrier()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise TrainingError(f"{name} must be an integer, got {raw!r}") from exc
