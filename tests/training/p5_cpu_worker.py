"""Two-process CPU check for the ESPnet chunk iterator.

The project root is inserted on ``sys.path`` because the script directory
would otherwise hide ``src``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.training.core.batch import plan_batches
from src.training.core.distributed import ProcessInfo, unwrap
from src.training.core.trainer import Trainer
from tests.training.test_p5_chunk import _TinyTask, _Waves


def main() -> None:
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    port = int(sys.argv[3])
    root = Path(sys.argv[4])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world)
    train = _Waves([(f"t{i}", 13) for i in range(4)])
    valid = _Waves([(f"v{i}", 20) for i in range(4)])
    samples = list(zip(valid.case_ids(), valid.waveform_lengths()))
    valid_plan = plan_chunk_epoch(
        samples,
        espnet_epoch=1,
        seed=17,
        world_size=world,
        num_samples=None,
        chunk_length=8,
        global_batch=4,
        num_cache_chunks=6,
        shuffle=False,
    )
    train_plan = plan_chunk_epoch(
        list(zip(train.case_ids(), train.waveform_lengths())),
        espnet_epoch=1,
        seed=17,
        world_size=world,
        num_samples=1,
        chunk_length=8,
        global_batch=4,
        num_cache_chunks=6,
        shuffle=True,
    )
    directory = root / "ckpt"
    config = {
        "seed": 17,
        "deterministic": True,
        "global_batch_size": 4,
        "micro_batch_size_per_gpu": 4,
        "max_steps": 1,
        "segment_size": 8,
        "loss": "official",
        "checkpoint_dir": str(directory),
        "checkpoint_interval": 1,
        "validation_interval": 0,
        "chunk_iterator": {
            "num_samples_per_epoch": 1,
            "chunk_length": 8,
            "chunk_shift_ratio": 0.5,
            "num_cache_chunks": 6,
            "shuffle": True,
            "max_epochs": 1,
            "early_stopping_patience": 5,
            "keep_nbest": 1,
        },
        "model": {"name": "tiny"},
        "data": {
            "database": "unused.duckdb",
            "train_set": "toy",
            "validation_set": "toy_valid",
            "rms_normalize": False,
        },
        "runtime": {"device": "cpu", "allow_busy_gpu": False},
        "logging": {
            "tensorboard": {
                "enabled": False,
                "scalar_interval": 10,
                "audio": {"enabled": False, "max_cases": 0, "max_seconds": 1},
                "spectrogram": False,
            }
        },
    }
    process = ProcessInfo(rank=rank, world_size=world, local_rank=rank, device=torch.device("cpu"), distributed=True)
    task = _TinyTask()
    trainer = Trainer(
        task,
        plan_batches(4, 4, world),
        process,
        max_steps=1,
        seed=17,
        checkpoint_dir=directory,
        checkpoint_interval=1,
        validation_interval=0,
        config=config,
    )
    trainer.fit(train, valid)
    layer = unwrap(task.layer)
    vector = torch.nn.utils.parameters_to_vector(layer.parameters()).detach()
    gathered = [torch.empty_like(vector) for _ in range(world)]
    torch.distributed.all_gather(gathered, vector)
    hashes = [None] * world
    torch.distributed.all_gather_object(hashes, train_plan.plan_hash)
    updates = [None] * world
    torch.distributed.all_gather_object(updates, train_plan.updates)
    numerator = 0.0
    denominator = 0
    for rank_index in range(world):
        for chunks in valid_plan.rank_batches[rank_index]:
            for chunk in chunks:
                numerator += 0.2 + 0.01 * (chunk.start % 7)
                denominator += 1
    record = {
        "plan_hashes": hashes,
        "updates": updates,
        "validation": None,
        "expected_validation": numerator / denominator,
        "weights_equal": all(torch.equal(gathered[0], item) for item in gathered),
        "provenance_writers": [0] if (directory / "data_provenance.json").is_file() else [],
    }
    if rank == 0:
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["type"] == "validation":
                record["validation"] = item["losses"]["generator"]["value"]
        (root / "summary.json").write_text(json.dumps(record), encoding="utf-8")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
