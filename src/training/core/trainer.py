"""Optimizer-step loop shared by speech-enhancement tasks.

The trainer does not know about a particular model. A task owns its modules,
losses, and the order of optimizer updates. One ``train_update`` consumes
``accumulation_steps`` micro-batches and counts as one global step. Schedulers
advance on that boundary, not on each micro-batch.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.evaluation.envinfo import capture_environment
from src.training.core.batch import BatchPlan
from src.training.core.checkpoint import load_checkpoint, save_atomic
from src.training.core.distributed import ProcessInfo, barrier, unwrap
from src.training.core.rng import capture_rng, restore_rng, seed_training_stream
from src.training.data.dataset import RecipeDataset
from src.training.data.sampler import epoch_indices
from src.training.errors import TrainingError

FORMAT = "audio-db-training-v1"


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
            if self.global_step >= self.max_steps:
                raise TrainingError("checkpoint global_step is already at max_steps")
        else:
            self._reject_existing_checkpoints()
            seed_training_stream(self.seed, cuda=self.process.uses_cuda)
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
                self.task.step_schedulers()
                self.global_step += 1
                if self.process.rank == 0 and (
                    self.global_step == 1 or self.global_step % 10 == 0 or self.global_step == self.max_steps
                ):
                    rendered = " ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
                    print(f"step {self.global_step} {rendered}", flush=True)
                if self.validation_interval and self.global_step % self.validation_interval == 0:
                    self._validate(validation)
                if self._should_save(self.global_step):
                    self._save(dataset, epoch, cursor, usable_batches)
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
        }
        path = self.checkpoint_dir / f"step_{self.global_step:08d}.pt"
        save_atomic(path, payload)
        generator = self.task.generator_state_dict()
        if generator is not None:
            save_atomic(self.checkpoint_dir / "generator.pt", {"generator": generator})

    def _validate(self, validation: RecipeDataset | None) -> None:
        if validation is None:
            raise TrainingError("validation_interval is set but no validation set was given")
        rng_state = capture_rng(cuda=self.process.uses_cuda)
        try:
            if self.process.rank == 0:
                self.task.eval_mode()
                totals: dict[str, float] = {}
                counts: dict[str, int] = {}
                for index in range(len(validation)):
                    metrics = self.task.validation_metrics(validation.collate([index]))
                    self._require_finite(metrics)
                    for name, value in metrics.items():
                        totals[name] = totals.get(name, 0.0) + value
                        counts[name] = counts.get(name, 0) + 1
                rendered = " ".join(
                    f"{name}={totals[name] / counts[name]:.6f} (n={counts[name]})"
                    for name in sorted(totals)
                )
                print(f"validation step {self.global_step} {rendered}", flush=True)
        finally:
            restore_rng(rng_state, cuda=self.process.uses_cuda)
            self.task.train_mode()
        barrier(self.process)

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
