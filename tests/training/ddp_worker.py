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

from src.training.core.trainer import backward


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
    model.zero_grad(set_to_none=True)
    if mode == "sync":
        local_x = features[rank * 2 : (rank + 1) * 2]
        local_y = target[rank * 2 : (rank + 1) * 2]
        loss = F.mse_loss(model(local_x), local_y)
        backward(loss, [model], last=True)
    elif mode == "accum":
        # global 4, world 2, micro 1, accumulation 2
        for step in range(2):
            index = rank * 2 + step
            loss = F.mse_loss(model(features[index : index + 1]), target[index : index + 1]) / 2
            backward(loss, [model], last=step == 1)
    else:
        raise SystemExit(f"unknown mode {mode}")
    gradient = torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])
    torch.save(gradient.cpu(), destination)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
