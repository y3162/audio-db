"""CUDA global-batch check for two data-parallel layouts.

Configurations, each with global batch 4:

- B: 2 GPUs, micro-batch 2, accumulation 1
- C: 2 GPUs, micro-batch 1, accumulation 2

Rank 0 also computes the single-process full batch. Different GPUs do not
need bitwise equality; atol 1e-5 and rtol 1e-4 are the bound.

    CUDA_VISIBLE_DEVICES=1,2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/lab/bin/python -m torch.distributed.run --nproc_per_node=2 \\
        tests/training/reference/compare_global_batch.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.core.trainer import backward  # noqa: E402

ATOL = 1e-5
RTOL = 1e-4


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(123)
    return torch.randn(4, 8, generator=generator), torch.randn(4, 4, generator=generator)


def _model() -> torch.nn.Sequential:
    torch.manual_seed(5)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))


def _flat(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in module.parameters()])


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", local)
    features, target = _batch()
    features = features.to(device)
    target = target.to(device)
    path = Path("/tmp/audio-db-global-batch-ref.pt")
    if rank == 0:
        reference = _model().to(device)
        F.mse_loss(reference(features), target).backward()
        torch.save(_flat(reference).cpu(), path)
    torch.distributed.barrier()
    expected = torch.load(path, weights_only=True)
    reports = {}
    for mode, micro in (("B", 2), ("C", 1)):
        model = DistributedDataParallel(_model().to(device), device_ids=[local])
        model.zero_grad(set_to_none=True)
        chunks = 4 // world // micro
        for step in range(chunks):
            start = rank * micro * chunks + step * micro
            stop = start + micro
            loss = F.mse_loss(model(features[start:stop]), target[start:stop]) / chunks
            backward(loss, [model], last=step == chunks - 1)
        gap = ( _flat(model.module).cpu() - expected).abs()
        reports[mode] = {"max_abs": float(gap.max()), "ok": bool(torch.allclose(_flat(model.module).cpu(), expected, rtol=RTOL, atol=ATOL))}
        torch.distributed.barrier()
    if rank == 0:
        print(reports)
        if not all(item["ok"] for item in reports.values()):
            raise SystemExit(1)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
