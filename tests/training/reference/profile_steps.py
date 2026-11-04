"""Phase timing and a short PyTorch profiler window for one training config.

Steady steps before the profiler window are the timing sample. The profiler
window is reported separately because its overhead is not a training step time.
Checkpoint files land in ``--out`` and are not a full trial.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from src.training.core.batch import plan_batches
from src.training.core.distributed import setup_process
from src.training.core.gpu import assert_gpu_idle
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from src.training.data.dataset import load_recipe_dataset
from src.training.se.config import load_training_config
from src.training.se.run import build_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--profile-last", type=int, default=3)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_training_config(args.config)
    config["max_steps"] = args.steps
    config["checkpoint_dir"] = str(out / "ckpt")
    config["checkpoint_interval"] = 0
    config["validation_interval"] = 0
    if args.allow_busy:
        config["runtime"]["allow_busy_gpu"] = True
    device_name = config["runtime"]["device"]
    if device_name == "cuda":
        assert_gpu_idle(int(os.environ.get("LOCAL_RANK", "0")), allow_busy=config["runtime"]["allow_busy_gpu"])
    set_determinism(config["deterministic"], cuda=device_name == "cuda")
    process = setup_process(device_name)
    plan = plan_batches(config["global_batch_size"], config["micro_batch_size_per_gpu"], process.world_size)
    task = build_task(config, process.device)
    data = config["data"]
    sample_rate = int(config["model"]["sampling_rate"])
    train = load_recipe_dataset(
        data["database"], data["train_set"], role="train", seed=config["seed"],
        sample_rate=sample_rate, segment_size=config["segment_size"], training=True,
    )
    trainer = Trainer(
        task, plan, process, max_steps=config["max_steps"], seed=config["seed"],
        checkpoint_dir=Path(config["checkpoint_dir"]), checkpoint_interval=0,
        validation_interval=0, config=config,
    )
    rows: list[dict[str, Any]] = []
    profile_start = args.steps - args.profile_last
    profiler_holder: dict[str, Any] = {}

    original_collate = train.collate
    original_update = task.train_update

    state = {"collate": 0.0, "items": 0, "mark": None}

    def wrapped_collate(*args, **kwargs):
        if state["mark"] is None and process.uses_cuda:
            torch.cuda.synchronize(process.device)
            state["mark"] = time.perf_counter()
        elif state["mark"] is None:
            state["mark"] = time.perf_counter()
        started = time.perf_counter()
        batch = original_collate(*args, **kwargs)
        state["collate"] += time.perf_counter() - started
        state["items"] += len(batch["case_id"])
        return batch

    def wrapped_update(*args, **kwargs):
        step = trainer.global_step + 1
        use_profiler = step > profile_start
        if process.uses_cuda:
            torch.cuda.reset_peak_memory_stats(process.device)
            torch.cuda.synchronize(process.device)
        collate_seconds = state["collate"]
        started = time.perf_counter()
        prof = None
        if use_profiler:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            prof.__enter__()
        metrics = original_update(*args, **kwargs)
        if prof is not None:
            prof.__exit__(None, None, None)
            profiler_holder.setdefault("tables", []).append(_table(prof, step))
        if process.uses_cuda:
            torch.cuda.synchronize(process.device)
        update_seconds = time.perf_counter() - started
        # Prefetch collates the next step on another thread, so the start mark
        # may already have been consumed. Time from the update in that case.
        if state["mark"] is None:
            state["mark"] = started
        total = time.perf_counter() - state["mark"]
        rows.append({
            "optimizer_step": step,
            "profiled": use_profiler,
            "collate_seconds": collate_seconds,
            "update_seconds": update_seconds,
            "step_seconds": total,
            "items": state["items"],
            "losses": {name: float(value) for name, value in metrics.items()},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(process.device) if process.uses_cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(process.device) if process.uses_cuda else None,
        })
        state["collate"] = 0.0
        state["items"] = 0
        state["mark"] = None
        return metrics

    train.collate = wrapped_collate
    task.train_update = wrapped_update
    wall = time.perf_counter()
    trainer.fit(train, None)
    wall = time.perf_counter() - wall
    clean = [row for row in rows if not row["profiled"]]
    summary = {
        "steps": rows,
        "clean_steady": _dist([row["step_seconds"] for row in clean[1:]]),
        "clean_collate": _dist([row["collate_seconds"] for row in clean[1:]]),
        "clean_update": _dist([row["update_seconds"] for row in clean[1:]]),
        "profiled_step": _dist([row["step_seconds"] for row in rows if row["profiled"]]),
        "wall_seconds": wall,
        "gpu": _gpu(),
        "profiler_tables": profiler_holder.get("tables", []),
    }
    (out / "profile.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "clean_steady": summary["clean_steady"],
        "clean_collate": summary["clean_collate"],
        "clean_update": summary["clean_update"],
        "profiled_step": summary["profiled_step"],
    }), flush=True)


def _table(prof: torch.profiler.profile, step: int) -> dict[str, str]:
    return {
        "step": step,
        "cuda": prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18),
        "cpu": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18),
    }


def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(0.9 * len(ordered)) - 1))
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": ordered[index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _gpu() -> dict[str, Any]:
    commands = {
        "nvidia_smi_l": ["nvidia-smi", "-L"],
        "memory": ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
    }
    found = {}
    for name, command in commands.items():
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        found[name] = (completed.stdout or completed.stderr).strip()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        found["torch_name"] = props.name
        found["total_bytes"] = props.total_memory
    return found


if __name__ == "__main__":
    main()
