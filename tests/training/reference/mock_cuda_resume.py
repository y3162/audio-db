"""CUDA resume of the linear mock task. Not a SEMamba++ result.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        .venvs/lab/bin/python tests/training/reference/mock_cuda_resume.py \
        data/results/training/mock-cuda-resume-20261101
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint
from src.training.core.distributed import setup_process
from src.training.core.gpu import assert_gpu_idle
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from tests.training.mock_task import LinearPairTask
from tests.training.support import paired_dataset, tiny_config


def main() -> None:
    destination = Path(sys.argv[1])
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    assert_gpu_idle(0, allow_busy=False)
    destination.mkdir(parents=True)
    set_determinism(True, cuda=True)
    _fit(destination / "continuous", steps=20, resume=None)
    _fit(destination / "partial", steps=10, resume=None)
    _fit(destination / "resumed", steps=20, resume=destination / "partial" / "ckpt" / "step_00000010.pt")
    left = load_checkpoint(destination / "continuous" / "ckpt" / "step_00000020.pt", torch.device("cpu"))
    right = load_checkpoint(destination / "resumed" / "ckpt" / "step_00000020.pt", torch.device("cpu"))
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "global_step": [left["global_step"], right["global_step"]],
        "epoch": [left["epoch"], right["epoch"]],
        "cursor": [left["micro_batches_consumed"], right["micro_batches_consumed"]],
        "generator_max_abs": _max_abs(left["task"]["generator"], right["task"]["generator"]),
        "discriminator_max_abs": _max_abs(left["task"]["discriminator"], right["task"]["discriminator"]),
        "optim_max_abs": max(
            _max_abs_state(left["task"]["optim_g"], right["task"]["optim_g"]),
            _max_abs_state(left["task"]["optim_d"], right["task"]["optim_d"]),
        ),
        "scheduler_equal": left["task"]["sched_g"] == right["task"]["sched_d"] or _scheduler_equal(left, right),
        "torch_rng_equal": bool(torch.equal(left["rng"]["torch"].cpu(), right["rng"]["torch"].cpu())),
        "cuda_rng_equal": _cuda_equal(left["rng"].get("cuda"), right["rng"].get("cuda")),
    }
    report["exact"] = (
        report["generator_max_abs"] == 0
        and report["discriminator_max_abs"] == 0
        and report["optim_max_abs"] == 0
        and report["torch_rng_equal"]
        and report["cuda_rng_equal"]
        and left["global_step"] == right["global_step"] == 20
    )
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["exact"]:
        raise SystemExit("cuda mock resume is not exact")


def _fit(root: Path, *, steps: int, resume: Path | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config(
        root / "ckpt",
        max_steps=steps,
        seed=7,
        global_batch_size=4,
        micro_batch_size_per_gpu=4,
        loss="mock",
    )
    config["runtime"] = {"device": "cuda", "allow_busy_gpu": False}
    dataset = paired_dataset(root / "wav", 8, frames=320, segment_size=80, seed=7)
    process = setup_process("cuda")
    task = LinearPairTask(config, process.device)
    trainer = Trainer(
        task,
        plan_batches(4, 4, 1),
        process,
        max_steps=steps,
        seed=7,
        checkpoint_dir=root / "ckpt",
        checkpoint_interval=10,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset, resume=resume)


def _max_abs(left: dict, right: dict) -> float:
    return max(float((left[key] - right[key]).abs().max()) for key in left)


def _max_abs_state(left, right) -> float:
    if isinstance(left, dict):
        return max(_max_abs_state(left[key], right[key]) for key in left)
    if torch.is_tensor(left):
        return float((left.cpu() - right.cpu()).abs().max())
    return 0.0


def _scheduler_equal(left: dict, right: dict) -> bool:
    return left["task"]["sched_g"]["last_epoch"] == right["task"]["sched_g"]["last_epoch"] and left["task"]["sched_d"][
        "last_epoch"
    ] == right["task"]["sched_d"]["last_epoch"]


def _cuda_equal(left, right) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(left, right))


if __name__ == "__main__":
    main()
