"""Two-process Trainer run of the linear mock task."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from src.training.core.batch import plan_batches
from src.training.core.distributed import setup_process
from src.training.core.trainer import Trainer
from tests.training.mock_task import LinearPairTask
from tests.training.support import paired_dataset, tiny_config


def main() -> None:
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    port = int(sys.argv[3])
    root = Path(sys.argv[4])
    destination = Path(sys.argv[5])
    micro = int(sys.argv[6])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    process = setup_process("cpu")
    config = tiny_config(
        root / f"ckpt-{rank}",
        max_steps=1,
        seed=7,
        global_batch_size=4,
        micro_batch_size_per_gpu=micro,
        loss="mock",
    )
    dataset = paired_dataset(root / "wav", 4, frames=320, segment_size=80, seed=7, rewrite=False)
    task = LinearPairTask(config, torch.device("cpu"))
    trainer = Trainer(
        task,
        plan_batches(4, micro, world),
        process,
        max_steps=1,
        seed=7,
        checkpoint_dir=root / f"ckpt-{rank}",
        checkpoint_interval=0,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset)
    if rank == 0:
        torch.save(task.state_dict(), destination)


if __name__ == "__main__":
    main()
