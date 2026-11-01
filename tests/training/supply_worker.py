"""One rank of a real-dataset Trainer recording run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training.core.batch import plan_batches
from src.training.core.distributed import setup_process
from src.training.core.trainer import Trainer
from tests.training.supply import AuditedDataset, SupplyTask
from tests.training.support import paired_dataset, tiny_config


def main() -> None:
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    port = int(sys.argv[3])
    root = Path(sys.argv[4])
    destination = Path(sys.argv[5])
    seed = int(sys.argv[6])
    global_batch = int(sys.argv[7])
    micro = int(sys.argv[8])
    steps = int(sys.argv[9])
    count = int(sys.argv[10])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    process = setup_process("cpu")
    dataset = AuditedDataset(
        paired_dataset(root / "wav", count, frames=480, segment_size=160, seed=seed, rewrite=False)
    )
    task = SupplyTask()
    plan = plan_batches(global_batch, micro, world)
    config = tiny_config(root / f"ckpt-{rank}", max_steps=steps, seed=seed, global_batch_size=global_batch)
    trainer = Trainer(
        task,
        plan,
        process,
        max_steps=steps,
        seed=seed,
        checkpoint_dir=root / f"ckpt-{rank}",
        checkpoint_interval=0,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset)
    destination.write_text(
        json.dumps({"rank": rank, "rng_draws": dataset.rng_draws, "records": task.records}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
