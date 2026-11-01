"""CPU gloo worker used to check a two-process global batch.

Run as a script. The project root is inserted on ``sys.path`` because the
script directory would otherwise hide ``src``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.training.core.trainer import accumulation_scope


def shared_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(123)
    features = torch.randn(4, 3, generator=generator)
    target = torch.randn(4, 1, generator=generator)
    return features, target


def build_model() -> torch.nn.Linear:
    torch.manual_seed(5)
    return torch.nn.Linear(3, 1)


def main() -> None:
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    port = int(sys.argv[3])
    mode = sys.argv[4]
    destination = Path(sys.argv[5])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world)
    features, target = shared_batch()
    model = DistributedDataParallel(build_model())
    flags: list[int] = []
    inner = model.module
    original = inner.forward

    def wrapped(value):
        flags.append(int(bool(model.require_backward_grad_sync)))
        return original(value)

    inner.forward = wrapped
    model.zero_grad(set_to_none=True)
    if mode == "sync":
        local_x = features[rank * 2 : (rank + 1) * 2]
        local_y = target[rank * 2 : (rank + 1) * 2]
        with accumulation_scope([model], last=True):
            loss = F.mse_loss(model(local_x), local_y)
            loss.backward()
    elif mode == "accum":
        # global 4, world 2, micro 1, accumulation 2. Forward and backward stay paired.
        for step in range(2):
            index = rank * 2 + step
            with accumulation_scope([model], last=step == 1):
                loss = F.mse_loss(model(features[index : index + 1]), target[index : index + 1]) / 2
                loss.backward()
    elif mode == "split":
        # Leave the scope before backward, but backward before the next forward.
        for step in range(2):
            index = rank * 2 + step
            with accumulation_scope([model], last=step == 1):
                prediction = model(features[index : index + 1])
            (F.mse_loss(prediction, target[index : index + 1]) / 2).backward()
    elif mode == "delay":
        local_x = features[rank * 2 : (rank + 1) * 2]
        local_y = target[rank * 2 : (rank + 1) * 2]
        with accumulation_scope([model], last=True):
            prediction = model(local_x)
        F.mse_loss(prediction, local_y).backward()
    elif mode == "manual":
        pending = []
        for step in range(2):
            index = rank * 2 + step
            with accumulation_scope([model], last=False):
                prediction = model(features[index : index + 1])
            pending.append(F.mse_loss(prediction, target[index : index + 1]) / 2)
        for loss in pending:
            loss.backward()
        for parameter in model.parameters():
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad.div_(world)
    else:
        raise SystemExit(f"unknown mode {mode}")
    gradient = torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])
    torch.save({"gradient": gradient.cpu(), "flags": torch.tensor(flags, dtype=torch.int32)}, destination)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
