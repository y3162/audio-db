"""Measure a short real-audio training run without changing the update rules.

One recorded step is one optimizer update, including in-process recipe loading.
Gradient accumulation counts as one step after all of its micro-batches.
CUDA work is synchronized before the clock stops so the time is not the launch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint
from src.training.core.distributed import setup_process
from src.training.core.gpu import assert_gpu_idle
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from src.training.data.dataset import load_recipe_dataset
from src.training.se.config import load_training_config
from src.training.se.run import build_task


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Time a fixed number of training updates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume")
    parser.add_argument("--validation-items", type=int, default=8)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validation_only:
        if not args.resume or not args.checkpoint_dir:
            parser.error("--validation-only requires --resume and --checkpoint-dir")
        started = time.perf_counter()
        config = load_training_config(args.config)
        directory = Path(args.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        summary = validate_only(config, directory, Path(args.resume), args.validation_items)
        summary["process_seconds"] = time.perf_counter() - started
        path = directory / "measurement.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"measurement": str(path), "result": summary["result"]}), flush=True)
        return
    started = time.perf_counter()
    config = load_training_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    directory = Path(config["checkpoint_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    summary = run(config, directory, resume=None if args.resume is None else Path(args.resume), validation_items=args.validation_items)
    summary["process_seconds"] = time.perf_counter() - started
    path = directory / "measurement.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"measurement": str(path), "optimizer_steps": summary["optimizer_steps"], "result": summary["result"]}), flush=True)


def run(config: dict[str, Any], directory: Path, *, resume: Path | None, validation_items: int) -> dict[str, Any]:
    device_name = config["runtime"]["device"]
    if device_name == "cuda":
        assert_gpu_idle(int(os.environ.get("LOCAL_RANK", "0")), allow_busy=config["runtime"]["allow_busy_gpu"])
    set_determinism(config["deterministic"], cuda=device_name == "cuda")
    process = setup_process(device_name)
    plan = plan_batches(config["global_batch_size"], config["micro_batch_size_per_gpu"], process.world_size)
    model_started = time.perf_counter()
    task = build_task(config, process.device)
    model_seconds = time.perf_counter() - model_started
    data = config["data"]
    sample_rate = int(config["model"]["sampling_rate"])
    data_started = time.perf_counter()
    train = load_recipe_dataset(
        data["database"], data["train_set"], role="train", seed=config["seed"],
        sample_rate=sample_rate, segment_size=config["segment_size"], training=True,
    )
    validation = None
    if data["validation_set"]:
        validation = load_recipe_dataset(
            data["database"], data["validation_set"], role="validation", seed=config["seed"],
            sample_rate=sample_rate, segment_size=None, training=False,
        )
    data_seconds = time.perf_counter() - data_started
    resume_before = _resume_snapshot(resume, process.device) if resume is not None else None
    trainer = Trainer(
        task, plan, process, max_steps=config["max_steps"], seed=config["seed"],
        checkpoint_dir=directory, checkpoint_interval=config["checkpoint_interval"],
        validation_interval=config["validation_interval"], config=config,
    )
    meter = _Meter(trainer, train, plan.accumulation_steps, sample_rate)
    meter.install()
    if process.uses_cuda:
        torch.cuda.reset_peak_memory_stats(process.device)
    wall_started = time.perf_counter()
    outcome = "ok"
    error = None
    try:
        trainer.fit(train, validation, resume=resume)
    except Exception as exc:
        outcome = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process.uses_cuda:
            torch.cuda.synchronize(process.device)
        wall_seconds = time.perf_counter() - wall_started
        steps = _summarize_steps(meter.steps)
        validation_report = None
        if outcome == "ok" and validation is not None and validation_items > 0 and resume is None:
            try:
                validation_report = _time_validation(
                    task, validation, process, validation_items, int(config["segment_size"])
                )
            except Exception as exc:
                validation_report = {"error": f"{type(exc).__name__}: {exc}"}
        payload = {
            "result": outcome,
            "error": error,
            "optimizer_steps": len(meter.steps),
            "global_step": trainer.global_step,
            "model_init_seconds": model_seconds,
            "dataset_init_seconds": data_seconds,
            "train_wall_seconds": wall_seconds,
            "accumulation_steps": plan.accumulation_steps,
            "micro_batch_size": plan.micro_batch_size,
            "global_batch_size": plan.global_batch_size,
            "segment_size": config["segment_size"],
            "sample_rate": sample_rate,
            "seed": config["seed"],
            "deterministic": config["deterministic"],
            "dataloader_workers": 0,
            "mixed_precision": False,
            "steps": steps,
            "checkpoint_seconds": meter.checkpoint_seconds,
            "validation_subset": validation_report,
            "resume_before": resume_before,
            "gpu": _gpu(process),
            "nonfinite_steps": sum(1 for step in meter.steps if step["nonfinite_grads"] or step["nonfinite_loss"]),
        }
        (directory / "steps.jsonl").write_text("".join(json.dumps(step, sort_keys=True) + "\n" for step in meter.steps), encoding="utf-8")
        if outcome != "ok":
            (directory / "measurement.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


class _Meter:
    def __init__(self, trainer: Trainer, dataset, accumulation: int, sample_rate: int) -> None:
        self.trainer = trainer
        self.dataset = dataset
        self.accumulation = accumulation
        self.sample_rate = sample_rate
        self.steps: list[dict[str, Any]] = []
        self.checkpoint_seconds: list[float] = []
        self._collate_seconds = 0.0
        self._collate_count = 0
        self._started: float | None = None
        self._case_ids: list[str] = []
        self._audio_samples = 0

    def install(self) -> None:
        dataset_collate = self.dataset.collate
        update = self.trainer.task.train_update
        save = self.trainer._save

        def collate(*args, **kwargs):
            if self._collate_count % self.accumulation == 0:
                if self.trainer.process.uses_cuda:
                    torch.cuda.synchronize(self.trainer.process.device)
                self._started = time.perf_counter()
                self._collate_seconds = 0.0
                self._case_ids = []
                self._audio_samples = 0
            started = time.perf_counter()
            batch = dataset_collate(*args, **kwargs)
            self._collate_seconds += time.perf_counter() - started
            self._collate_count += 1
            self._case_ids.extend(str(item) for item in batch["case_id"])
            self._audio_samples += int(batch["clean"].shape[0] * batch["clean"].shape[-1])
            return batch

        def train_update(*args, **kwargs):
            if self.trainer.process.uses_cuda:
                torch.cuda.reset_peak_memory_stats(self.trainer.process.device)
                torch.cuda.synchronize(self.trainer.process.device)
            update_started = time.perf_counter()
            metrics = update(*args, **kwargs)
            if self.trainer.process.uses_cuda:
                torch.cuda.synchronize(self.trainer.process.device)
            update_seconds = time.perf_counter() - update_started
            total = time.perf_counter() - (self._started if self._started is not None else update_started)
            grad_l2, nonfinite = _grad_stats(self.trainer.task)
            allocated = reserved = None
            if self.trainer.process.uses_cuda:
                device = self.trainer.process.device
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
            self.steps.append({
                "optimizer_step": self.trainer.global_step + 1,
                "step_seconds": total,
                "collate_seconds": self._collate_seconds,
                "update_seconds": update_seconds,
                "losses": {name: float(value) for name, value in metrics.items()},
                "nonfinite_loss": any(not math.isfinite(float(value)) for value in metrics.values()),
                "grad_l2": grad_l2,
                "nonfinite_grads": nonfinite,
                "learning_rate": self.trainer.task.learning_rates(),
                "case_ids": list(self._case_ids),
                "audio_seconds": self._audio_samples / self.sample_rate,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
            })
            return metrics

        def wrapped_save(*args, **kwargs):
            if self.trainer.process.uses_cuda:
                torch.cuda.synchronize(self.trainer.process.device)
            started = time.perf_counter()
            path = save(*args, **kwargs)
            if self.trainer.process.uses_cuda:
                torch.cuda.synchronize(self.trainer.process.device)
            self.checkpoint_seconds.append(time.perf_counter() - started)
            return path

        self.dataset.collate = collate
        self.trainer.task.train_update = train_update
        self.trainer._save = wrapped_save


def _grad_stats(task) -> tuple[float, int]:
    total = 0.0
    nonfinite = 0
    for module in vars(task).values():
        if not isinstance(module, nn.Module):
            continue
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all():
                nonfinite += 1
                continue
            total += float(gradient.float().pow(2).sum())
    return math.sqrt(total), nonfinite


def _summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {"count": 0}
    times = [step["step_seconds"] for step in steps]
    steady = times[5:] if len(times) > 5 else times
    losses = {}
    for name in steps[-1]["losses"]:
        values = [step["losses"][name] for step in steps if name in step["losses"]]
        losses[name] = {"first": values[0], "last": values[-1], "min": min(values), "max": max(values)}
    allocated = [step["peak_allocated_bytes"] for step in steps if step["peak_allocated_bytes"] is not None]
    reserved = [step["peak_reserved_bytes"] for step in steps if step["peak_reserved_bytes"] is not None]
    return {
        "count": len(steps),
        "first_step_seconds": times[0],
        "warmup_step_seconds": _distribution(times[1:5] if len(times) > 1 else []),
        "steady_step_seconds": _distribution(steady),
        "all_step_seconds": _distribution(times),
        "losses": losses,
        "audio_seconds": sum(step["audio_seconds"] for step in steps),
        "utterances": sum(len(step["case_ids"]) for step in steps),
        "peak_allocated_bytes": max(allocated) if allocated else None,
        "peak_reserved_bytes": max(reserved) if reserved else None,
        "last_learning_rate": steps[-1]["learning_rate"],
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def validate_only(config: dict[str, Any], directory: Path, resume: Path, validation_items: int) -> dict[str, Any]:
    device_name = config["runtime"]["device"]
    if device_name == "cuda":
        assert_gpu_idle(int(os.environ.get("LOCAL_RANK", "0")), allow_busy=config["runtime"]["allow_busy_gpu"])
    set_determinism(config["deterministic"], cuda=device_name == "cuda")
    process = setup_process(device_name)
    data = config["data"]
    sample_rate = int(config["model"]["sampling_rate"])
    task = build_task(config, process.device)
    payload = load_checkpoint(resume, process.device)
    task.load_state_dict(payload["task"])
    validation = load_recipe_dataset(
        data["database"], data["validation_set"], role="validation", seed=config["seed"],
        sample_rate=sample_rate, segment_size=None, training=False,
    )
    report = _time_validation(task, validation, process, validation_items, int(config["segment_size"]))
    return {"result": "ok" if "error" not in report else "failed", "validation_subset": report, "gpu": _gpu(process), "resume": str(resume)}


def _time_validation(task, validation, process, count: int, segment_size: int | None = None) -> dict[str, Any]:
    count = min(count, len(validation))
    task.eval_mode()
    durations = []
    finite = True
    mode = "full_utterance"
    try:
        for index in range(count):
            batch = validation.collate([index])
            if process.uses_cuda:
                torch.cuda.synchronize(process.device)
            started = time.perf_counter()
            try:
                report = task.validation_report(batch)
            except Exception:
                if segment_size is None:
                    raise
                mode = "center_crop_segment"
                report = task.validation_report(_center_crop(batch, segment_size))
            if process.uses_cuda:
                torch.cuda.synchronize(process.device)
            durations.append(time.perf_counter() - started)
            finite = finite and _report_finite(report)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "items_completed": len(durations), "mode": mode}
    finally:
        task.train_mode()
    mean = statistics.fmean(durations)
    return {
        "items": count,
        "set_size": len(validation),
        "seconds": sum(durations),
        "seconds_per_item": mean,
        "extrapolated_full_seconds": mean * len(validation),
        "finite": finite,
        "mode": mode,
        "extrapolation": "first N registered validation items; utterance length is not stratified",
    }


def _center_crop(batch: dict[str, Any], segment_size: int) -> dict[str, Any]:
    length = int(batch["clean"].shape[-1])
    if length < segment_size:
        raise RuntimeError(f"utterance length {length} is shorter than segment {segment_size}")
    start = (length - segment_size) // 2
    stop = start + segment_size
    cropped = dict(batch)
    cropped["clean"] = batch["clean"][..., start:stop]
    cropped["noisy"] = batch["noisy"][..., start:stop]
    return cropped


def _report_finite(report: dict[str, Any]) -> bool:
    losses = report.get("losses") or {}
    for item in losses.values():
        value = item.get("value") if isinstance(item, dict) else item
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def _resume_snapshot(path: Path, device) -> dict[str, Any]:
    payload = load_checkpoint(path, device)
    return {
        "path": str(path),
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "micro_batches_consumed": int(payload["micro_batches_consumed"]),
        "rng_kinds": sorted(payload["rng"]),
        "task_keys": sorted(payload["task"]),
    }


def _gpu(process) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(process.device)}
    if process.uses_cuda:
        props = torch.cuda.get_device_properties(process.device)
        info.update({
            "name": props.name,
            "total_bytes": props.total_memory,
            "capability": f"{props.major}.{props.minor}",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(process.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(process.device),
        })
    completed = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
    info["nvidia_smi_l"] = completed.stdout.strip()
    used = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    info["nvidia_smi_memory"] = used.stdout.strip()
    return info


if __name__ == "__main__":
    main()
