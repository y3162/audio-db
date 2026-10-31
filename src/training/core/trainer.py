"""Optimizer-step loop shared by speech-enhancement tasks.

The trainer does not know about a particular model. A task owns its modules,
losses, and the order of optimizer updates. One ``train_update`` consumes
``accumulation_steps`` micro-batches and counts as one global step. The
learning-rate scheduler advances once when that epoch's usable micro-batches
are finished, matching the official MP-SENet loop. Gradient accumulation
changes how many optimizer updates fit in an epoch, not how often the
scheduler steps.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.evaluation.envinfo import capture_environment
from src.training.core.batch import BatchPlan
from src.training.core.checkpoint import load_checkpoint, save_atomic
from src.training.core.distributed import ProcessInfo, barrier, unwrap
from src.training.core.metrics_log import append_record, truncate_after, write_json
from src.training.core.rng import capture_rng, restore_rng, seed_training_stream
from src.training.data.dataset import RecipeDataset
from src.training.data.sampler import epoch_indices
from src.training.errors import TrainingError

FORMAT = "audio-db-training-v1"


def _inference_payload(task: Any) -> dict[str, Any] | None:
    """File contents that ``load_enhancer`` can read back.

    Tasks that need a different layout (ESPnet ``separator.`` keys, a raw
    state dictionary) implement ``export_for_inference``. The default matches
    the MP-SENet ``generator`` checkpoint.
    """
    export = getattr(task, "export_for_inference", None)
    if export is not None:
        return export()
    generator = task.generator_state_dict()
    if generator is None:
        return None
    return {"generator": generator}


def backward(loss: torch.Tensor, modules: list[nn.Module], *, last: bool) -> None:
    """Backpropagate one scaled micro-batch.

    On every micro-batch except the last, DistributedDataParallel skips the
    gradient all-reduce. The last backward reduces the accumulated gradients.
    The caller divides the loss by the accumulation count so the reduced
    gradient matches the mean over the global batch.
    """
    with contextlib.ExitStack() as stack:
        for module in modules:
            if isinstance(module, DistributedDataParallel) and not last:
                stack.enter_context(module.no_sync())
        loss.backward()


class Trainer:
    def __init__(
        self,
        task: Any,
        plan: BatchPlan,
        process: ProcessInfo,
        *,
        max_steps: int,
        seed: int,
        checkpoint_dir: Path,
        checkpoint_interval: int,
        validation_interval: int,
        config: dict[str, Any],
    ) -> None:
        if max_steps < 1:
            raise TrainingError("max_steps must be positive")
        if checkpoint_interval < 0 or validation_interval < 0:
            raise TrainingError("checkpoint and validation intervals must be zero or positive")
        self.task = task
        self.plan = plan
        self.process = process
        self.max_steps = max_steps
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.config = config
        self.global_step = 0
        self.best_pesq: float | None = None
        start = config.get("best_checkpoint_start_epoch", 40)
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise TrainingError("best_checkpoint_start_epoch must be an integer >= 0")
        self.best_checkpoint_start_epoch = start

    def fit(
        self,
        dataset: RecipeDataset,
        validation: RecipeDataset | None = None,
        resume: Path | None = None,
    ) -> None:
        epoch = 0
        skip = 0
        if resume is not None:
            payload = load_checkpoint(resume, self.process.device)
            self._check_resume(payload, dataset)
            self.task.load_state_dict(payload["task"])
            restore_rng(payload["rng"], cuda=self.process.uses_cuda)
            self.global_step = int(payload["global_step"])
            epoch = int(payload["epoch"])
            skip = int(payload["micro_batches_consumed"])
            self.best_pesq = _stored_best_pesq(payload, self.checkpoint_dir)
            if self.global_step >= self.max_steps:
                raise TrainingError("checkpoint global_step is already at max_steps")
            if self.process.rank == 0:
                truncate_after(self._log_path(), self.global_step)
        else:
            self._reject_existing_checkpoints()
            seed_training_stream(self.seed, cuda=self.process.uses_cuda)
            if self.process.rank == 0:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                append_record(self._log_path(), _run_header(self))
        self.task.distribute(self.process)
        if self.process.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        barrier(self.process)
        while self.global_step < self.max_steps:
            indices = epoch_indices(
                len(dataset),
                self.seed,
                epoch,
                self.process.world_size,
                self.process.rank,
            )
            micro = self.plan.micro_batch_size
            accumulation = self.plan.accumulation_steps
            n_batches = len(indices) // micro
            usable_batches = (n_batches // accumulation) * accumulation
            if usable_batches < accumulation:
                raise TrainingError(
                    f"epoch {epoch} on rank {self.process.rank} has {n_batches} micro-batches, "
                    f"which is fewer than accumulation_steps {accumulation}"
                )
            if skip > usable_batches or skip % accumulation != 0:
                raise TrainingError(
                    f"checkpoint micro-batch position {skip} is not an optimizer boundary "
                    f"in an epoch of {usable_batches} micro-batches"
                )
            if skip == usable_batches:
                epoch += 1
                skip = 0
                continue
            dataset.set_epoch(epoch)
            cursor = skip
            skip = 0
            while cursor < usable_batches and self.global_step < self.max_steps:
                group = []
                for _ in range(accumulation):
                    start = cursor * micro
                    group.append(dataset.collate(indices[start : start + micro]))
                    cursor += 1
                metrics = self.task.train_update(group, accumulation)
                self._require_finite(metrics)
                self.global_step += 1
                epoch_done = cursor >= usable_batches
                validation_summary = None
                if self.validation_interval and self.global_step % self.validation_interval == 0:
                    validation_summary = self._validate(validation, epoch)
                if epoch_done:
                    self.task.step_schedulers(validation_summary)
                checkpoint_path = None
                if self._should_save(self.global_step):
                    checkpoint_path = self._save(dataset, epoch, cursor, usable_batches)
                if self.process.rank == 0:
                    self._log_train(epoch, metrics, checkpoint_path, validation_summary)
                    if (
                        self.global_step == 1
                        or self.global_step % 10 == 0
                        or self.global_step == self.max_steps
                    ):
                        rendered = " ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
                        print(f"step {self.global_step} {rendered}", flush=True)
                barrier(self.process)
            if self.global_step >= self.max_steps:
                break
            epoch += 1
        barrier(self.process)

    def _should_save(self, step: int) -> bool:
        if self.process.rank != 0:
            return False
        if step == self.max_steps:
            return True
        return self.checkpoint_interval > 0 and step % self.checkpoint_interval == 0

    def _save(self, dataset: RecipeDataset, epoch: int, cursor: int, usable_batches: int) -> None:
        if cursor >= usable_batches:
            stored_epoch, stored_cursor = epoch + 1, 0
        else:
            stored_epoch, stored_cursor = epoch, cursor
        payload = {
            "format": FORMAT,
            "global_step": self.global_step,
            "epoch": stored_epoch,
            "micro_batches_consumed": stored_cursor,
            "task": self.task.state_dict(),
            "rng": capture_rng(cuda=self.process.uses_cuda),
            "config": self.config,
            "batch_plan": self.plan.as_dict(),
            "dataset": dataset.identity(),
            "environment": _environment(self.process),
            "best_pesq": self.best_pesq,
        }
        path = self.checkpoint_dir / f"step_{self.global_step:08d}.pt"
        save_atomic(path, payload)
        exported = _inference_payload(self.task)
        if exported is not None:
            save_atomic(self.checkpoint_dir / "generator.pt", exported)
        return str(path)

    def _validate(self, validation: RecipeDataset | None, epoch: int) -> dict[str, Any] | None:
        if validation is None:
            raise TrainingError("validation_interval is set but no validation set was given")
        rng_state = capture_rng(cuda=self.process.uses_cuda)
        summary = None
        try:
            if self.process.rank == 0:
                self.task.eval_mode()
                reports = [
                    self.task.validation_report(validation.collate([index]))
                    for index in range(len(validation))
                ]
                summary = _aggregate_validation(reports)
                best_path = self._consider_best(epoch, summary)
                summary["best_checkpoint"] = best_path
                append_record(
                    self._log_path(),
                    {
                        "type": "validation",
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "losses": summary["losses"],
                        "pesq": summary["pesq"],
                        "stoi": summary["stoi"],
                        "best_checkpoint": best_path,
                        "seed": self.seed,
                    },
                )
                pesq = summary["pesq"]
                pesq_text = "na" if pesq["value"] is None else f"{pesq['value']:.6f}"
                print(
                    f"validation step {self.global_step} pesq={pesq_text} ({pesq['status']})",
                    flush=True,
                )
        finally:
            restore_rng(rng_state, cuda=self.process.uses_cuda)
            self.task.train_mode()
        barrier(self.process)
        return summary

    def _check_resume(self, payload: dict[str, Any], dataset: RecipeDataset) -> None:
        if payload["batch_plan"] != self.plan.as_dict():
            raise TrainingError("checkpoint batch plan does not match this run")
        if payload["dataset"] != dataset.identity():
            raise TrainingError("checkpoint dataset identity does not match this run")
        saved = payload["config"]
        if saved["seed"] != self.config["seed"] or saved["loss"] != self.config["loss"]:
            raise TrainingError("checkpoint seed or loss does not match this run")
        if saved["model"] != self.config["model"]:
            raise TrainingError("checkpoint model settings do not match this run")

    def _log_path(self) -> Path:
        return self.checkpoint_dir / "metrics.jsonl"

    def _log_train(
        self,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_path: str | None,
        validation_summary: dict[str, Any] | None,
    ) -> None:
        append_record(
            self._log_path(),
            {
                "type": "train",
                "global_step": self.global_step,
                "epoch": epoch,
                "losses": metrics,
                "learning_rate": self.task.learning_rates(),
                "global_batch_size": self.plan.global_batch_size,
                "micro_batch_size": self.plan.micro_batch_size,
                "gradient_accumulation_steps": self.plan.accumulation_steps,
                "seed": self.seed,
                "checkpoint": checkpoint_path,
                "validated": validation_summary is not None,
            },
        )

    def _consider_best(self, epoch: int, summary: dict[str, Any]) -> str | None:
        """Save the generator when validation PESQ improves after the official start epoch."""
        if epoch < self.best_checkpoint_start_epoch:
            return None
        pesq = summary["pesq"]
        if pesq["status"] != "ok" or pesq["value"] is None:
            return None
        score = float(pesq["value"])
        if self.best_pesq is not None and score <= self.best_pesq:
            return None
        exported = _inference_payload(self.task)
        if exported is None:
            return None
        self.best_pesq = score
        path = self.checkpoint_dir / "best_generator.pt"
        save_atomic(path, exported)
        write_json(
            self.checkpoint_dir / "best_generator.json",
            {
                "metric": "pesq",
                "pesq": score,
                "global_step": self.global_step,
                "epoch": epoch,
                "validation": summary,
                "config": self.config,
                "model": self.config.get("model"),
                "checkpoint": str(path),
            },
        )
        return str(path)

    def _reject_existing_checkpoints(self) -> None:
        if self.process.rank != 0 or not self.checkpoint_dir.exists():
            return
        existing = [path for path in self.checkpoint_dir.iterdir() if path.name != ".gitkeep"]
        if existing:
            raise TrainingError(
                f"checkpoint directory is not empty: {self.checkpoint_dir}. "
                "Choose another directory or resume from a checkpoint."
            )

    @staticmethod
    def _require_finite(metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            if value != value or value in (float("inf"), float("-inf")):
                raise TrainingError(f"non-finite metric {name}")


def _environment(process: ProcessInfo) -> dict[str, Any]:
    info = capture_environment()
    info["torch_cuda"] = torch.version.cuda
    info["device"] = str(process.device)
    info["world_size"] = process.world_size
    info["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    if process.uses_cuda:
        info["gpu_name"] = torch.cuda.get_device_name(process.device)
    return info


def _run_header(trainer: Trainer) -> dict[str, Any]:
    return {
        "type": "run",
        "seed": trainer.seed,
        "config": trainer.config,
        "global_batch_size": trainer.plan.global_batch_size,
        "micro_batch_size": trainer.plan.micro_batch_size,
        "gradient_accumulation_steps": trainer.plan.accumulation_steps,
        "best_checkpoint_start_epoch": trainer.best_checkpoint_start_epoch,
        "scheduler": "epoch",
    }


def _stored_best_pesq(payload: dict[str, Any], directory: Path) -> float | None:
    if "best_pesq" in payload:
        value = payload["best_pesq"]
        return None if value is None else float(value)
    meta = directory / "best_generator.json"
    if not meta.is_file():
        return None
    stored = json.loads(meta.read_text(encoding="utf-8"))
    value = stored.get("pesq")
    return None if value is None else float(value)


def _aggregate_validation(reports: list[dict[str, Any]]) -> dict[str, Any]:
    names: list[str] = []
    for report in reports:
        for name in report["losses"]:
            if name not in names:
                names.append(name)
    losses = {name: _combine_metric([report["losses"][name] for report in reports if name in report["losses"]]) for name in names}
    return {
        "losses": losses,
        "pesq": _combine_metric([report["pesq"] for report in reports]),
        "stoi": _combine_metric([report["stoi"] for report in reports]),
    }


def _combine_metric(items: list[dict[str, Any]]) -> dict[str, Any]:
    if any(item["status"] == "unavailable" for item in items):
        reason = next(item["reason"] for item in items if item["status"] == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in items if item["status"] == "ok" and item["value"] is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item.get("reason") for item in items if item.get("reason")), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}
