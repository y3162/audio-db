"""Optimizer-step loop shared by speech-enhancement tasks.

The trainer does not know about a particular model. A task owns its modules,
losses, and the order of optimizer updates. One ``train_update`` consumes
``accumulation_steps`` micro-batches and counts as one global step. The
learning-rate scheduler normally advances when the usable micro-batches in a
data pass are finished. An optional fixed-update logical epoch instead triggers
validation and scheduler updates at fixed optimizer-step boundaries. Gradient accumulation
changes how many optimizer updates fit in an epoch, not how often the
scheduler steps.
"""

from __future__ import annotations

import contextlib
import json
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.evaluation.envinfo import capture_environment
from src.training.core.batch import BatchPlan, plan_batches
from src.training.core.batch_schedule import scheduled_batch_size
from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.training.data.chunk_plan import epoch_chunk_batches
from src.training.core.average import average_checkpoint_files, replace_symlink
from src.training.core.checkpoint import load_checkpoint, save_atomic
from src.training.core.distributed import ProcessInfo, barrier, unwrap
from src.training.core.metrics_log import append_record, truncate_after, write_json
from src.training.core.rng import capture_rng, restore_rng, seed_training_stream
from src.training.data.dataset import RecipeDataset
from src.training.data.sampler import epoch_indices
from src.training.errors import TrainingError

FORMAT = "audio-db-training-v1"
_STOP_REQUESTED = False


def request_training_stop(signum, _frame) -> None:
    """Ask the trainer to finish the current optimizer update and save."""
    del signum, _frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def install_training_stop() -> None:
    signal.signal(signal.SIGTERM, request_training_stop)


def reset_training_stop() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False


def training_stop_requested() -> bool:
    return _STOP_REQUESTED


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


@contextlib.contextmanager
def accumulation_scope(modules: list[nn.Module], *, last: bool):
    """Skip the DDP all-reduce on every micro-batch except the last.

    DistributedDataParallel reads ``require_backward_grad_sync`` during
    forward and installs the all-reduce there. A ``no_sync`` that starts only
    at ``loss.backward()`` does not cancel that reduction. Enter this context
    before the forward that should participate in the optimizer step. The
    matching backward may run after the context exits: the reducer already
    recorded the decision. Non-distributed modules are unchanged. The caller
    divides the loss by the accumulation count so the reduced gradient matches
    the mean over the global batch.
    """
    with contextlib.ExitStack() as stack:
        if not last:
            for module in modules:
                if isinstance(module, DistributedDataParallel):
                    stack.enter_context(module.no_sync())
        yield


def backward(loss: torch.Tensor, modules: list[nn.Module], *, last: bool) -> None:
    """Backpropagate one scaled micro-batch.

    Prefer :func:`accumulation_scope` around the forward. This function only
    runs ``loss.backward()``. Wrapping it in ``no_sync`` cannot undo an
    all-reduce that the forward already scheduled, so the arguments are
    retained for existing callers and are not a synchronization point.
    """
    del modules, last
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
        self.steps_per_epoch = config.get("logical_steps_per_epoch")
        self.best_loss = None
        self.best_pesq: float | None = None
        start = config.get("best_checkpoint_start_epoch", 40)
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise TrainingError("best_checkpoint_start_epoch must be an integer >= 0")
        self.best_checkpoint_start_epoch = start
        self.chunk_history: list[dict[str, Any]] = []
        self.plateau_count = 0
        self.plateau_metric: float | None = None
        self.stop_reason: str | None = None
        self.last_checkpoint: str | None = None
        self._tb = None
        self._provenance_raw: bytes | None = None
        self._provenance_sha: str | None = None
        self._run_id: str | None = None

    def fit(
        self,
        dataset: RecipeDataset,
        validation: RecipeDataset | None = None,
        resume: Path | None = None,
        *,
        deadline: float | None = None,
        segment_steps: int | None = None,
    ) -> None:
        self._data_provenance = _provenance_of(dataset)
        if (deadline is not None or segment_steps is not None) and self.process.world_size != 1:
            raise TrainingError("managed stops require one training process")
        if segment_steps is not None and segment_steps < 1:
            raise TrainingError("segment_steps must be positive")
        epoch = 0
        skip = 0
        if self.process.rank == 0:
            reset_training_stop()
            install_training_stop()
            if (self.checkpoint_dir / "complete.json").is_file():
                raise TrainingError(
                    f"training is already complete: {self.checkpoint_dir}. "
                    "Refusing to resume a finished experiment."
                )
        if resume is not None:
            payload = load_checkpoint(resume, self.process.device)
            self._check_resume(payload, dataset, resume)
            self.task.load_state_dict(payload["task"])
            restore_rng(payload["rng"], cuda=self.process.uses_cuda)
            self.global_step = int(payload["global_step"])
            if payload.get("logical_steps_per_epoch") != self.steps_per_epoch:
                raise TrainingError("checkpoint logical epoch settings do not match this run")
            self.best_loss = payload.get("best_loss")
            epoch = int(payload["epoch"])
            skip = int(payload["micro_batches_consumed"])
            self.best_pesq = _stored_best_pesq(payload, self.checkpoint_dir)
            self.chunk_history = list(payload.get("chunk_history") or [])
            self.plateau_count = int(payload.get("plateau_count") or 0)
            stored_metric = payload.get("plateau_metric")
            self.plateau_metric = None if stored_metric is None else float(stored_metric)
            if self.global_step >= self.max_steps:
                raise TrainingError("checkpoint global_step is already at max_steps")
            if self.process.rank == 0:
                truncate_after(self._log_path(), self.global_step)
                self._bind_provenance(dataset)
        else:
            self._reject_existing_checkpoints()
            seed_training_stream(self.seed, cuda=self.process.uses_cuda)
            if self.process.rank == 0:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self._bind_provenance(dataset)
                append_record(self._log_path(), _run_header(self))
        if self.config.get("logging", {}).get("tensorboard", {}).get("enabled"):
            state = capture_rng(cuda=self.process.uses_cuda)
            try:
                import tensorboard  # noqa: F401
            except ImportError as exc:
                raise TrainingError("TensorBoard is enabled but tensorboard is not installed") from exc
            finally:
                restore_rng(state, cuda=self.process.uses_cuda)
        self.task.distribute(self.process)
        self._deadline = deadline
        self._managed = deadline is not None or segment_steps is not None
        self._segment_end = None if segment_steps is None else self.global_step + segment_steps
        if self.process.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        barrier(self.process)
        workers = int(self.config.get("data", {}).get("load_workers", 1))
        if _is_training_data(dataset):
            wrapped = getattr(dataset, "train", None)
            if wrapped is not None and hasattr(wrapped, "load_workers"):
                wrapped.load_workers = workers
        else:
            dataset.load_workers = workers
        prefetch = bool(self.config.get("data", {}).get("prefetch", False))
        halt = False
        loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-db-prefetch") if prefetch else None
        try:
            tb_config = self.config.get("logging", {}).get("tensorboard", {})
            if self.process.rank == 0 and tb_config.get("enabled"):
                from src.training.core.tensorboard import open_writer
                state = capture_rng(cuda=self.process.uses_cuda)
                try:
                    self._tb = open_writer(self.checkpoint_dir / "tensorboard", self.global_step if resume else None)
                    from src.training.core.tensorboard import write_run_info
                    write_run_info(self._tb, self.config, self.process.world_size, self.global_step)
                finally:
                    restore_rng(state, cuda=self.process.uses_cuda)
            if _is_training_data(dataset):
                self._run_data_epochs(dataset, skip, epoch, halt, loader, prefetch)
            elif self.config.get("chunk_iterator"):
                self._run_chunk_epochs(dataset, validation, skip, epoch, halt)
            else:
                self._run_epochs(dataset, validation, skip, epoch, halt, loader, prefetch)
        finally:
            if loader is not None:
                loader.shutdown(wait=True, cancel_futures=True)
            closer = getattr(dataset, "close", None)
            if _is_training_data(dataset) and callable(closer):
                closer()
            if self._tb is not None:
                self._tb.close()

    def _run_epochs(self, dataset, validation, skip, epoch, halt, loader, prefetch) -> None:
        while self.global_step < self.max_steps and not halt:
            self._apply_batch_schedule(epoch)
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
            self._retarget_schedule(usable_batches // accumulation)
            cursor = skip
            skip = 0
            pending: Future | None = None
            while cursor < usable_batches and self.global_step < self.max_steps:
                self._note_schedule(epoch, cursor // accumulation, usable_batches // accumulation)
                if pending is not None:
                    group, cursor = pending.result()
                    pending = None
                else:
                    group, cursor = _load_group(dataset, indices, cursor, micro, accumulation)
                self._batch_ready = time.perf_counter()
                if (
                    prefetch
                    and loader is not None
                    and cursor < usable_batches
                    and self.global_step + 1 < self.max_steps
                ):
                    pending = loader.submit(
                        _load_group, dataset, indices, cursor, micro, accumulation
                    )
                update_started = time.perf_counter()
                metrics = self.task.train_update(group, accumulation)
                self._update_seconds = time.perf_counter() - update_started
                self._require_finite(metrics)
                self.global_step += 1
                epoch_done = (self.global_step % self.steps_per_epoch == 0) if self.steps_per_epoch else cursor >= usable_batches
                logical_epoch = (self.global_step - 1) // self.steps_per_epoch if self.steps_per_epoch else epoch
                validation_summary = None
                scheduled_epoch = bool(self.config.get("batch_schedule") and epoch_done and validation is not None)
                step_validation = bool(
                    not self.steps_per_epoch
                    and not self.config.get("batch_schedule")
                    and self.validation_interval
                    and self.global_step % self.validation_interval == 0
                )
                if (self.steps_per_epoch and epoch_done) or scheduled_epoch or step_validation:
                    validation_summary = self._validate(validation, logical_epoch)
                if epoch_done:
                    self.task.step_schedulers(self._scheduler_validation(validation_summary))
                    halt = bool(getattr(self.task, "stop_training", False)) or self._batch_schedule_should_stop(validation_summary)
                timed_stop = not halt and self.global_step < self.max_steps and self._stop_requested()
                save = self._should_save(self.global_step) or timed_stop or ((self.steps_per_epoch or self.config.get("batch_schedule")) and epoch_done and self.process.rank == 0)
                checkpoint_path = str(self.checkpoint_dir / f"step_{self.global_step:08d}.pt") if save else None
                if self.process.rank == 0:
                    self._log_train(logical_epoch, metrics, checkpoint_path, validation_summary, group)
                    if save:
                        if self._tb is not None:
                            self._tb.flush()
                        self.last_checkpoint = self._save(dataset, epoch, cursor, usable_batches)
                    if (
                        self.global_step == 1
                        or self.global_step % 10 == 0
                        or self.global_step == self.max_steps
                    ):
                        rendered = " ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
                        print(f"step {self.global_step} {rendered}", flush=True)
                barrier(self.process)
                if timed_stop:
                    self.stop_reason = "CHECKPOINTED"
                    halt = True
                    break
            if pending is not None:
                pending.result()
                pending = None
            if halt or self.global_step >= self.max_steps:
                break
            epoch += 1
        barrier(self.process)
        if self.stop_reason is None:
            self.stop_reason = "COMPLETED"
            self._mark_complete()

    def _run_data_epochs(self, dataset, skip, epoch, halt, loader, prefetch) -> None:
        """Same optimizer loop as ``_run_epochs``, fed by ``TrainingData``."""
        accumulation = self.plan.accumulation_steps
        while self.global_step < self.max_steps and not halt:
            self._apply_batch_schedule(epoch)
            stream = dataset.epoch(
                epoch,
                rank=self.process.rank,
                world_size=self.process.world_size,
                start_update=0,
            )
            self._broadcast_plan_hash(stream.plan_hash)
            updates = stream.updates
            usable_batches = updates * accumulation
            if skip > usable_batches or skip % accumulation != 0:
                raise TrainingError(
                    f"checkpoint micro-batch position {skip} is not an optimizer boundary "
                    f"in an epoch of {usable_batches} micro-batches"
                )
            if skip == usable_batches:
                epoch += 1
                skip = 0
                continue
            self._retarget_schedule(usable_batches // accumulation)
            cursor = skip
            skip = 0
            pending: Future | None = None
            while cursor < usable_batches and self.global_step < self.max_steps:
                self._note_schedule(epoch, cursor // accumulation, usable_batches // accumulation)
                if pending is not None:
                    group, cursor = pending.result()
                    pending = None
                else:
                    group, cursor = _load_update(stream.render, cursor // accumulation, accumulation)
                self._batch_ready = time.perf_counter()
                if (
                    prefetch
                    and loader is not None
                    and cursor < usable_batches
                    and self.global_step + 1 < self.max_steps
                ):
                    pending = loader.submit(
                        _load_update, stream.render, cursor // accumulation, accumulation
                    )
                update_started = time.perf_counter()
                metrics = self.task.train_update(group, accumulation)
                self._update_seconds = time.perf_counter() - update_started
                self._require_finite(metrics)
                self.global_step += 1
                epoch_done = (self.global_step % self.steps_per_epoch == 0) if self.steps_per_epoch else cursor >= usable_batches
                logical_epoch = (self.global_step - 1) // self.steps_per_epoch if self.steps_per_epoch else epoch
                validation_summary = None
                scheduled_epoch = bool(self.config.get("batch_schedule") and epoch_done and _has_validation(dataset))
                step_validation = bool(
                    not self.steps_per_epoch
                    and not self.config.get("batch_schedule")
                    and self.validation_interval
                    and self.global_step % self.validation_interval == 0
                )
                if (self.steps_per_epoch and epoch_done) or scheduled_epoch or step_validation:
                    validation_summary = self._validate_training_data(dataset, logical_epoch)
                if epoch_done:
                    self.task.step_schedulers(self._scheduler_validation(validation_summary))
                    halt = bool(getattr(self.task, "stop_training", False)) or self._batch_schedule_should_stop(validation_summary)
                timed_stop = not halt and self.global_step < self.max_steps and self._stop_requested()
                save = self._should_save(self.global_step) or timed_stop or ((self.steps_per_epoch or self.config.get("batch_schedule")) and epoch_done and self.process.rank == 0)
                checkpoint_path = str(self.checkpoint_dir / f"step_{self.global_step:08d}.pt") if save else None
                if self.process.rank == 0:
                    self._log_train(logical_epoch, metrics, checkpoint_path, validation_summary, group)
                    if save:
                        if self._tb is not None:
                            self._tb.flush()
                        if cursor >= usable_batches:
                            stored_epoch, stored_updates = epoch + 1, 0
                        else:
                            stored_epoch, stored_updates = epoch, cursor // accumulation
                        self.last_checkpoint = self._save(
                            dataset,
                            epoch,
                            cursor,
                            usable_batches,
                            dataset.data_state(stored_epoch, stored_updates),
                        )
                    if (
                        self.global_step == 1
                        or self.global_step % 10 == 0
                        or self.global_step == self.max_steps
                    ):
                        rendered = " ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
                        print(f"step {self.global_step} {rendered}", flush=True)
                barrier(self.process)
                if timed_stop:
                    self.stop_reason = "CHECKPOINTED"
                    halt = True
                    break
            if pending is not None:
                pending.result()
                pending = None
            if halt or self.global_step >= self.max_steps:
                break
            epoch += 1
        barrier(self.process)
        if self.stop_reason is None:
            self.stop_reason = "COMPLETED"
            self._mark_complete()

    def _run_chunk_epochs(self, dataset, validation, skip, epoch, halt) -> None:
        """One epoch is one pass of the ESPnet chunk iterator, not a fixed update count."""
        world, mode = self._chunk_alignment()
        if mode != "single":
            self._run_aligned_chunk_epochs(dataset, validation, skip, epoch, halt, world, mode)
            return
        if self.process.world_size != 1:
            raise TrainingError("the chunk iterator is implemented for one process")
        spec = self.config["chunk_iterator"]
        max_epochs = int(spec["max_epochs"])
        while epoch < max_epochs and self.global_step < self.max_steps and not halt:
            micro = self.plan.micro_batch_size
            accumulation = self.plan.accumulation_steps
            if accumulation != 1:
                raise TrainingError("chunk iterator updates use one official batch and no gradient accumulation")
            batches = self._planned_chunks(dataset, epoch + 1, train=True)
            usable = len(batches)
            if usable < 1:
                raise TrainingError(f"chunk epoch {epoch + 1} produced no batches")
            if skip > usable:
                raise TrainingError(
                    f"checkpoint chunk position {skip} is past the {usable} batches in epoch {epoch + 1}"
                )
            if skip == usable:
                epoch += 1
                skip = 0
                continue
            dataset.set_epoch(epoch)
            cursor = skip
            skip = 0
            while cursor < usable and self.global_step < self.max_steps:
                group = [dataset.collate_chunks(batches[cursor])]
                cursor += 1
                update_started = time.perf_counter()
                metrics = self.task.train_update(group, accumulation)
                self._update_seconds = time.perf_counter() - update_started
                self._require_finite(metrics)
                self.global_step += 1
                epoch_done = cursor >= usable
                espnet_epoch = epoch + 1
                validation_summary = None
                if epoch_done:
                    validation_summary = self._validate_chunks(validation, espnet_epoch)
                    self.task.step_schedulers(self._scheduler_validation(validation_summary))
                    if self._chunk_should_stop(espnet_epoch, validation_summary):
                        halt = True
                complete = halt or (epoch_done and espnet_epoch >= max_epochs)
                timed_stop = not complete and self.global_step < self.max_steps and self._stop_requested()
                save = self._should_save(self.global_step) or timed_stop or (epoch_done and self.process.rank == 0)
                checkpoint_path = str(self.checkpoint_dir / f"step_{self.global_step:08d}.pt") if save else None
                if self.process.rank == 0:
                    self._log_train(espnet_epoch, metrics, checkpoint_path, validation_summary, group)
                    if save:
                        if self._tb is not None:
                            self._tb.flush()
                        state = self._chunk_state(
                            dataset,
                            epoch if cursor < usable else epoch + 1,
                            0 if cursor >= usable else cursor,
                            1,
                            "single",
                        )
                        self.last_checkpoint = self._save(dataset, epoch, cursor, usable, state)
                barrier(self.process)
                if timed_stop:
                    self.stop_reason = "CHECKPOINTED"
                    halt = True
                    break
            if halt or self.global_step >= self.max_steps:
                break
            epoch += 1
        barrier(self.process)
        if self.stop_reason is None:
            if self._managed and self.global_step >= self.max_steps and epoch < max_epochs and not halt:
                raise TrainingError("chunk iterator reached max_steps before completing training")
            self.stop_reason = "COMPLETED"
            self._mark_complete()

    def _mark_complete(self) -> None:
        if self.process.rank != 0:
            return
        write_json(self.checkpoint_dir / "complete.json", {
            "status": "COMPLETED",
            "global_step": self.global_step,
            "checkpoint": self.last_checkpoint,
        })

    def _stop_requested(self) -> bool:
        return ((self._deadline is not None and time.time() >= self._deadline)
                or (self._segment_end is not None and self.global_step >= self._segment_end)
                or training_stop_requested())

    def _planned_chunks(self, dataset, espnet_epoch: int, *, train: bool):
        spec = self.config["chunk_iterator"]
        num_samples = int(spec["num_samples_per_epoch"]) if train else None
        return epoch_chunk_batches(
            dataset.case_ids(),
            dataset.waveform_lengths(),
            espnet_epoch=espnet_epoch,
            seed=self.seed,
            num_samples=num_samples,
            chunk_length=int(spec["chunk_length"]),
            batch_size=self.plan.micro_batch_size,
            cache_size=int(spec["num_cache_chunks"]),
            shuffle=bool(spec["shuffle"]) if train else False,
            shift_ratio=float(spec["chunk_shift_ratio"]),
        )

    def _chunk_alignment(self) -> tuple[int, str]:
        virtual = (self.config.get("data") or {}).get("virtual_world_size")
        if virtual is not None:
            if self.process.world_size != 1:
                raise TrainingError("data.virtual_world_size cannot be combined with a distributed process")
            world = int(virtual)
            if world < 1:
                raise TrainingError("data.virtual_world_size must be positive")
            return world, "virtual"
        if self.process.world_size != 1:
            return self.process.world_size, "real"
        return 1, "single"

    def _run_aligned_chunk_epochs(self, dataset, validation, skip, epoch, halt, world: int, mode: str) -> None:
        """DDP or virtual-rank chunk epochs. Every rank takes the same number of updates."""
        if self.plan.accumulation_steps != 1:
            raise TrainingError("chunk iterator updates use one official batch and no gradient accumulation")
        if self.plan.global_batch_size % world != 0:
            raise TrainingError(
                f"global_batch_size {self.plan.global_batch_size} is not divisible by chunk world {world}"
            )
        if self.plan.micro_batch_size != self.plan.global_batch_size // world:
            raise TrainingError("chunk iterator requires micro_batch_size == global_batch_size / world")
        spec = self.config["chunk_iterator"]
        max_epochs = int(spec["max_epochs"])
        while epoch < max_epochs and self.global_step < self.max_steps and not halt:
            plan = self._aligned_chunk_plan(dataset, epoch + 1, train=True, world=world)
            self._broadcast_plan_hash(plan.plan_hash)
            usable = plan.updates
            if usable < 1:
                raise TrainingError(f"chunk epoch {epoch + 1} produced no batches")
            if skip > usable:
                raise TrainingError(
                    f"checkpoint chunk position {skip} is past the {usable} batches in epoch {epoch + 1}"
                )
            if skip == usable:
                epoch += 1
                skip = 0
                continue
            dataset.set_epoch(epoch)
            cursor = skip
            skip = 0
            while cursor < usable and self.global_step < self.max_steps:
                update_started = time.perf_counter()
                if mode == "virtual":
                    group = [
                        dataset.collate_chunks(plan.rank_batches[rank][cursor]) for rank in range(world)
                    ]
                    # Sequential 1/N average of the rank batches. This is not a DDP all-reduce.
                    metrics = self.task.train_update(group, len(group))
                else:
                    group = [dataset.collate_chunks(plan.rank_batches[self.process.rank][cursor])]
                    metrics = self.task.train_update(group, 1)
                self._update_seconds = time.perf_counter() - update_started
                cursor += 1
                self._require_finite(metrics)
                self.global_step += 1
                epoch_done = cursor >= usable
                espnet_epoch = epoch + 1
                validation_summary = None
                if epoch_done:
                    validation_summary = self._validate_aligned_chunks(validation, espnet_epoch, world, mode)
                    self.task.step_schedulers(self._scheduler_validation(validation_summary))
                    if self._chunk_should_stop(espnet_epoch, validation_summary):
                        halt = True
                complete = halt or (epoch_done and espnet_epoch >= max_epochs)
                timed_stop = not complete and self.global_step < self.max_steps and self._stop_requested()
                save = self._should_save(self.global_step) or timed_stop or (epoch_done and self.process.rank == 0)
                checkpoint_path = str(self.checkpoint_dir / f"step_{self.global_step:08d}.pt") if save else None
                if self.process.rank == 0:
                    self._log_train(espnet_epoch, metrics, checkpoint_path, validation_summary, group)
                    if save:
                        if self._tb is not None:
                            self._tb.flush()
                        state = self._chunk_state(
                            dataset,
                            epoch if cursor < usable else epoch + 1,
                            0 if cursor >= usable else cursor,
                            world,
                            mode,
                        )
                        self.last_checkpoint = self._save(dataset, epoch, cursor, usable, state)
                barrier(self.process)
                if timed_stop:
                    self.stop_reason = "CHECKPOINTED"
                    halt = True
                    break
            if halt or self.global_step >= self.max_steps:
                break
            epoch += 1
        barrier(self.process)
        if self.stop_reason is None:
            if self._managed and self.global_step >= self.max_steps and epoch < max_epochs and not halt:
                raise TrainingError("chunk iterator reached max_steps before completing training")
            self.stop_reason = "COMPLETED"
            self._mark_complete()

    def _aligned_chunk_plan(self, dataset, espnet_epoch: int, *, train: bool, world: int):
        spec = self.config["chunk_iterator"]
        samples = list(zip(dataset.case_ids(), (int(length) for length in dataset.waveform_lengths())))
        from src.reproduction.errors import ReproductionError

        try:
            return plan_chunk_epoch(
                samples,
                espnet_epoch=espnet_epoch,
                seed=self.seed,
                world_size=world,
                num_samples=int(spec["num_samples_per_epoch"]) if train else None,
                chunk_length=int(spec["chunk_length"]),
                global_batch=self.plan.global_batch_size,
                num_cache_chunks=int(spec["num_cache_chunks"]),
                shuffle=bool(spec["shuffle"]) if train else False,
                shift_ratio=float(spec["chunk_shift_ratio"]),
            )
        except ReproductionError as exc:
            raise TrainingError(str(exc)) from exc

    def _chunk_state(self, dataset, epoch: int, update: int, world: int, mode: str) -> dict[str, Any]:
        plan = self._aligned_chunk_plan(dataset, epoch + 1, train=True, world=world)
        if update < 0 or update > plan.updates:
            raise TrainingError(
                f"chunk update {update} is outside epoch {epoch + 1} ({plan.updates} updates)"
            )
        spec = self.config["chunk_iterator"]
        return {
            "pipeline": "espnet_chunk_v1",
            "order_policy": "espnet_chunk_v1",
            "seed": self.seed,
            "epoch": epoch,
            "update_in_epoch": update,
            "world_size": world,
            "process_world_size": self.process.world_size,
            "virtual_world_size": world if mode == "virtual" else None,
            "ddp_mode": mode,
            "ranks": list(range(world)),
            "plan_hash": plan.plan_hash,
            "global_batch": self.plan.global_batch_size,
            "micro_batch": self.plan.micro_batch_size,
            "accumulation": 1,
            "per_rank_batch": plan.per_rank_batch,
            "per_rank_cache": plan.per_rank_cache,
            "num_samples_per_epoch": int(spec["num_samples_per_epoch"]),
            "chunk_length": int(spec["chunk_length"]),
            "chunk_shift_ratio": float(spec["chunk_shift_ratio"]),
            "num_cache_chunks": int(spec["num_cache_chunks"]),
            "shuffle": bool(spec["shuffle"]),
            "iterator_stop": "min_rank_batches",
            "dataset": dataset.identity(),
        }

    def _check_chunk_state(self, saved: dict[str, Any], dataset) -> None:
        world, mode = self._chunk_alignment()
        current = self._chunk_state(dataset, int(saved["epoch"]), int(saved["update_in_epoch"]), world, mode)
        if saved != current:
            keys = sorted(set(saved) | set(current))
            changed = [key for key in keys if saved.get(key) != current.get(key)]
            raise TrainingError(
                "checkpoint chunk plan does not match this run: " + ", ".join(changed)
            )

    def _validate_aligned_chunks(self, validation, espnet_epoch: int, world: int, mode: str) -> dict[str, Any] | None:
        if validation is None:
            raise TrainingError("chunk_iterator requires a validation set")
        rng_state = capture_rng(cuda=self.process.uses_cuda)
        summary = None
        try:
            self.task.eval_mode()
            plan = self._aligned_chunk_plan(validation, espnet_epoch, train=False, world=world)
            self._broadcast_plan_hash(plan.plan_hash)
            if plan.updates < 1:
                raise TrainingError("validation chunk iterator produced no batches")
            if mode == "virtual":
                ranks = range(world)
            else:
                ranks = (self.process.rank,)
            reports = []
            weights = []
            for rank in ranks:
                for chunks in plan.rank_batches[rank]:
                    batch = validation.collate_chunks(chunks)
                    reports.append(self.task.validation_report(batch))
                    weights.append(len(batch["case_id"]))
            summary = _reduce_weighted_reports(reports, weights, self.process)
            if self.process.rank == 0 and summary is not None:
                if self._tb is not None:
                    from src.training.se.preview import write_preview
                    write_preview(
                        self._tb,
                        self.task,
                        validation,
                        self.global_step,
                        self.config["logging"]["tensorboard"],
                        chunks=plan.rank_batches[0],
                    )
                summary["best_checkpoint"] = self._retain_chunk_models(espnet_epoch, summary)
                append_record(self._log_path(), {
                    "type": "validation",
                    "global_step": self.global_step,
                    "epoch": espnet_epoch,
                    "losses": summary["losses"],
                    "pesq": summary["pesq"],
                    "stoi": summary["stoi"],
                    "si_snr": summary.get("si_snr"),
                    "best_checkpoint": summary["best_checkpoint"],
                    "seed": self.seed,
                    "weight": summary.get("weight"),
                })
        finally:
            restore_rng(rng_state, cuda=self.process.uses_cuda)
            self.task.train_mode()
        barrier(self.process)
        return summary if self.process.rank == 0 else None

    def _validate_chunks(self, validation, espnet_epoch: int) -> dict[str, Any] | None:
        if validation is None:
            raise TrainingError("chunk_iterator requires a validation set")
        rng_state = capture_rng(cuda=self.process.uses_cuda)
        summary = None
        try:
            if self.process.rank == 0:
                self.task.eval_mode()
                batches = self._planned_chunks(validation, espnet_epoch, train=False)
                if not batches:
                    raise TrainingError("validation chunk iterator produced no batches")
                reports = [self.task.validation_report(validation.collate_chunks(chunks)) for chunks in batches]
                if self._tb is not None:
                    from src.training.se.preview import write_preview
                    write_preview(self._tb, self.task, validation, self.global_step, self.config["logging"]["tensorboard"], chunks=batches)
                summary = _aggregate_validation(reports)
                summary["best_checkpoint"] = self._retain_chunk_models(espnet_epoch, summary)
                append_record(self._log_path(), {
                    "type": "validation",
                    "global_step": self.global_step,
                    "epoch": espnet_epoch,
                    "losses": summary["losses"],
                    "pesq": summary["pesq"],
                    "stoi": summary["stoi"],
                    "si_snr": summary.get("si_snr"),
                    "best_checkpoint": summary["best_checkpoint"],
                    "seed": self.seed,
                })
        finally:
            restore_rng(rng_state, cuda=self.process.uses_cuda)
            self.task.train_mode()
        barrier(self.process)
        return summary

    def _chunk_should_stop(self, espnet_epoch: int, summary: dict[str, Any] | None) -> bool:
        spec = self.config["chunk_iterator"]
        patience = spec.get("early_stopping_patience")
        if patience is None or summary is None or self.process.rank != 0:
            stopped = False
        else:
            loss = _loss_value(summary)
            if loss is None:
                stopped = False
            else:
                best = min(self.chunk_history, key=lambda row: (row["loss"], row["epoch"]))
                stopped = espnet_epoch - int(best["epoch"]) > int(patience)
        if self.process.distributed:
            flag = torch.tensor([1 if stopped else 0], dtype=torch.int64, device=self.process.device)
            dist.broadcast(flag, src=0)
            stopped = bool(int(flag.item()))
        return stopped

    def _retain_chunk_models(self, espnet_epoch: int, summary: dict[str, Any]) -> str | None:
        loss = _loss_value(summary)
        si = summary.get("si_snr") or {}
        si_value = float(si["value"]) if si.get("status") == "ok" and si.get("value") is not None else None
        if loss is None or loss != loss or loss in (float("inf"), float("-inf")):
            return None
        self.chunk_history.append({"epoch": espnet_epoch, "loss": loss, "si_snr": si_value})
        keep = int(self.config["chunk_iterator"]["keep_nbest"])
        by_loss = sorted(self.chunk_history, key=lambda row: (row["loss"], row["epoch"]))[:keep]
        ranked_si = [row for row in self.chunk_history if row["si_snr"] is not None]
        by_si = sorted(ranked_si, key=lambda row: (-row["si_snr"], row["epoch"]))[:keep]
        kept = {int(row["epoch"]) for row in by_loss + by_si}
        exported = _inference_payload(self.task)
        path = None
        if exported is not None:
            path = self.checkpoint_dir / f"{espnet_epoch}epoch.pt"
            save_atomic(path, exported)
            self._attach_provenance(path)
            for row, name, metric in (
                (by_loss[0], "valid.loss.best.pt", "loss"),
                (by_si[0] if by_si else None, "valid.si_snr.best.pt", "si_snr"),
            ):
                if row is not None and int(row["epoch"]) == espnet_epoch:
                    best_path = self.checkpoint_dir / name
                    save_atomic(best_path, exported)
                    self._attach_provenance(best_path)
                    write_json(self.checkpoint_dir / f"{name}.json", {
                        "metric": metric, "epoch": espnet_epoch, "value": row[metric],
                        "global_step": self.global_step,
                    })
        for old in self.checkpoint_dir.glob("*epoch.pt"):
            if old.name[0].isdigit() and int(old.name.split("epoch")[0]) not in kept:
                old.unlink()
        if exported is not None and by_loss:
            best_name = f"{int(by_loss[0]['epoch'])}epoch.pt"
            if keep > 1 and len(by_loss) >= keep:
                epochs = [int(row["epoch"]) for row in by_loss]
                averaged = average_checkpoint_files(
                    [self.checkpoint_dir / f"{epoch}epoch.pt" for epoch in epochs]
                )
                average_path = self.checkpoint_dir / f"valid.loss.ave_{keep}best.pt"
                save_atomic(average_path, averaged)
                self._attach_provenance(average_path)
                write_json(self.checkpoint_dir / f"{average_path.name}.json", {
                    "metric": "loss",
                    "criterion": "valid.loss.min",
                    "epochs": epochs,
                    "values": [row["loss"] for row in by_loss],
                    "files": [f"{epoch}epoch.pt" for epoch in epochs],
                    "global_step": self.global_step,
                    "average": "espnet_nbest_float_mean_int_sum",
                })
                replace_symlink(self.checkpoint_dir / "valid.loss.ave.pt", average_path.name)
            else:
                # Pinned ESPnet average_nbest_models: n larger than the history
                # writes valid.loss.ave_1best as a symlink to the best epoch,
                # not an average of the shorter list. ave_1best is left in place
                # once a real average exists.
                replace_symlink(self.checkpoint_dir / "valid.loss.ave_1best.pt", best_name)
                replace_symlink(self.checkpoint_dir / "valid.loss.ave.pt", "valid.loss.ave_1best.pt")
        return None if path is None else str(path)

    def _batch_schedule_should_stop(self, summary: dict[str, Any] | None) -> bool:
        """DeepFilterNet resets its patience counter when validation loss improves."""
        patience = self.config.get("early_stopping_patience")
        if not self.config.get("batch_schedule") or patience is None or summary is None:
            return False
        if self.process.rank != 0:
            return False
        loss = _loss_value(summary)
        if loss is None:
            return False
        if self.plateau_metric is None or loss < self.plateau_metric:
            self.plateau_count = 0
            self.plateau_metric = loss
            return False
        self.plateau_count += 1
        return self.plateau_count >= int(patience)

    def _apply_batch_schedule(self, epoch: int) -> None:
        schedule = self.config.get("batch_schedule")
        if not schedule:
            return
        if self.process.world_size != 1:
            raise TrainingError("batch_schedule is implemented for one process")
        limit = int(self.config["batch_size_limit"])
        size = scheduled_batch_size(epoch, limit, [tuple(item) for item in schedule])
        cap = int(self.config["micro_batch_size_per_gpu"])
        if size > cap:
            raise TrainingError(
                f"scheduled batch {size} at epoch {epoch} exceeds micro_batch_size_per_gpu {cap}"
            )
        self.plan = plan_batches(size, size, self.process.world_size)

    def _retarget_schedule(self, steps_per_epoch: int) -> None:
        retarget = getattr(self.task, "retarget_deepfilter_schedule", None)
        if retarget is not None and self.config.get("batch_schedule"):
            retarget(steps_per_epoch)

    def _note_schedule(self, epoch: int, index_in_epoch: int, steps_per_epoch: int) -> None:
        note = getattr(self.task, "note_schedule_position", None)
        if note is not None and self.config.get("batch_schedule"):
            note(epoch * steps_per_epoch + index_in_epoch)

    def _scheduler_validation(self, summary: dict[str, Any] | None) -> dict[str, Any] | None:
        metric = None
        if self.process.rank == 0 and summary:
            generator = (summary.get("losses") or {}).get("generator") or {}
            value = generator.get("value")
            if generator.get("status") == "ok" and value is not None:
                metric = float(value)
        if self.process.distributed:
            packed = torch.tensor([float("nan") if metric is None else metric], dtype=torch.float64, device=self.process.device)
            dist.broadcast(packed, src=0)
            metric = None if torch.isnan(packed).item() else float(packed.item())
        if metric is None:
            return None
        return {"losses": {"generator": {"value": metric, "status": "ok"}}}

    def _should_save(self, step: int) -> bool:
        if self.process.rank != 0:
            return False
        if step == self.max_steps:
            return True
        return self.checkpoint_interval > 0 and step % self.checkpoint_interval == 0

    def _save(self, dataset, epoch: int, cursor: int, usable_batches: int, data_state: dict[str, Any] | None = None) -> None:
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
            "best_loss": self.best_loss,
            "logical_steps_per_epoch": self.steps_per_epoch,
            "logical_epoch": self.global_step // self.steps_per_epoch if self.steps_per_epoch else stored_epoch,
            "chunk_history": self.chunk_history,
            "plateau_count": self.plateau_count,
            "plateau_metric": self.plateau_metric,
        }
        if data_state is not None:
            payload["data_state"] = data_state
            payload["update_in_epoch"] = data_state["update_in_epoch"]
        if self._provenance_raw is None:
            self._bind_provenance(dataset)
        payload["run_id"] = self._run_id
        payload["provenance_sha256"] = self._provenance_sha
        payload["provenance_file"] = "data_provenance.json"
        path = self.checkpoint_dir / f"step_{self.global_step:08d}.pt"
        save_atomic(path, payload, verify_training=True)
        self._attach_provenance(path)
        if data_state is not None:
            write_json(self.checkpoint_dir / "data_state.json", data_state)
        exported = _inference_payload(self.task)
        if exported is not None:
            generator = self.checkpoint_dir / "generator.pt"
            save_atomic(generator, exported)
            self._attach_provenance(generator)
        return str(path)

    def _validate(self, validation: RecipeDataset | None, epoch: int) -> dict[str, Any] | None:
        if validation is None:
            raise TrainingError("validation_interval is set but no validation set was given")
        rng_state = capture_rng(cuda=self.process.uses_cuda)
        summary = None
        try:
            if self.process.rank == 0:
                self.task.eval_mode()
                reports = []
                for index in range(len(validation)):
                    batch = validation.collate([index])
                    report = self.task.validation_report(batch)
                    reports.append(report)
                    if "si_sdr" in report:
                        append_record(self._log_path(), {
                            "type": "validation_case", "global_step": self.global_step,
                            "set": validation.name, "case_id": batch["case_id"][0],
                            **report,
                        })
                if self._tb is not None:
                    from src.training.se.preview import write_preview
                    write_preview(self._tb, self.task, validation, self.global_step, self.config["logging"]["tensorboard"])
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
                        "si_sdr": summary.get("si_sdr"),
                        "si_snr": summary.get("si_snr"),
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

    def _validate_training_data(self, dataset, epoch: int) -> dict[str, Any] | None:
        if hasattr(dataset, "validation_set") and dataset.validation_set is None:
            raise TrainingError("validation_interval is set but no validation set was given")
        rng_state = capture_rng(cuda=self.process.uses_cuda)
        summary = None
        try:
            if self.process.rank == 0:
                self.task.eval_mode()
                reports = []
                set_name = _validation_name(dataset)
                batches = dataset.validation(
                    rank=self.process.rank,
                    world_size=self.process.world_size,
                    epoch=epoch,
                )
                for batch in batches:
                    report = self.task.validation_report(batch)
                    reports.append(report)
                    if "si_sdr" in report:
                        append_record(self._log_path(), {
                            "type": "validation_case", "global_step": self.global_step,
                            "set": set_name, "case_id": batch["case_id"][0],
                            **report,
                        })
                if not reports:
                    raise TrainingError("validation produced no batches")
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
                        "si_sdr": summary.get("si_sdr"),
                        "si_snr": summary.get("si_snr"),
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

    def _broadcast_plan_hash(self, plan_hash: str) -> None:
        if not self.process.distributed:
            return
        raw = bytes.fromhex(plan_hash)
        if len(raw) != 32:
            raise TrainingError("plan_hash is not a sha256 hex digest")
        local = torch.tensor(list(raw), dtype=torch.int64, device=self.process.device)
        gathered = [torch.empty_like(local) for _ in range(self.process.world_size)]
        dist.all_gather(gathered, local)
        if any(not torch.equal(item, gathered[0]) for item in gathered):
            raise TrainingError("data plan_hash differs across ranks")

    def _bind_provenance(self, dataset) -> None:
        from src.training.provenance import build_provenance, provenance_sha256, publish_provenance

        document = build_provenance(dataset, self.config, self.process)
        raw = publish_provenance(self.checkpoint_dir, document, rank=self.process.rank)
        self._provenance_raw = raw
        self._provenance_sha = provenance_sha256(raw)
        self._run_id = json.loads(raw.decode("utf-8"))["run_id"]

    def _attach_provenance(self, checkpoint: Path) -> None:
        if not self._provenance_raw:
            return
        from src.training.provenance import write_provenance_sidecar

        write_provenance_sidecar(checkpoint, self._provenance_raw)

    def _check_resume(self, payload: dict[str, Any], dataset: RecipeDataset, checkpoint: Path | None = None) -> None:
        if self.config.get("batch_schedule"):
            if payload["config"].get("batch_schedule") != self.config.get("batch_schedule"):
                raise TrainingError("checkpoint batch schedule does not match this run")
            if payload["config"].get("batch_size_limit") != self.config.get("batch_size_limit"):
                raise TrainingError("checkpoint batch size limit does not match this run")
        elif payload["batch_plan"] != self.plan.as_dict():
            raise TrainingError("checkpoint batch plan does not match this run")
        if payload["config"].get("chunk_iterator") != self.config.get("chunk_iterator"):
            raise TrainingError("checkpoint chunk iterator does not match this run")
        if payload["dataset"] != dataset.identity():
            raise TrainingError("checkpoint dataset identity does not match this run")
        saved = payload["config"]
        if saved["seed"] != self.config["seed"] or saved["loss"] != self.config["loss"]:
            raise TrainingError("checkpoint seed or loss does not match this run")
        if saved["model"] != self.config["model"]:
            raise TrainingError("checkpoint model settings do not match this run")
        if payload.get("provenance_sha256") is not None:
            if checkpoint is None:
                raise TrainingError("checkpoint path is required to check data provenance")
            from src.training.provenance import assert_checkpoint_provenance

            assert_checkpoint_provenance(payload, checkpoint, dataset)
        saved_state = payload.get("data_state")
        checker = getattr(dataset, "check_resume", None)
        if checker is None:
            if isinstance(saved_state, dict) and saved_state.get("pipeline") == "espnet_chunk_v1":
                self._check_chunk_state(saved_state, dataset)
                return
            if saved_state is not None:
                raise TrainingError("checkpoint data_state does not match this dataset")
            return
        if not isinstance(saved_state, dict):
            raise TrainingError("checkpoint is missing data_state")
        accumulation = self.plan.accumulation_steps
        consumed = int(payload["micro_batches_consumed"])
        if consumed % accumulation != 0:
            raise TrainingError("checkpoint micro-batch position is not an optimizer boundary")
        if int(saved_state["update_in_epoch"]) != consumed // accumulation:
            raise TrainingError("checkpoint update_in_epoch does not match micro_batches_consumed")
        if int(saved_state["epoch"]) != int(payload["epoch"]):
            raise TrainingError("checkpoint data_state epoch does not match")
        from src.reproduction.errors import ReproductionError

        try:
            checker(saved_state)
        except ReproductionError as exc:
            raise TrainingError(str(exc)) from exc

    def _log_path(self) -> Path:
        return self.checkpoint_dir / "metrics.jsonl"

    def _log_train(
        self,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_path: str | None,
        validation_summary: dict[str, Any] | None,
        group: list[dict[str, Any]] | None = None,
    ) -> None:
        record = {
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
        }
        if group is not None:
            record["case_ids"] = [case_id for batch in group for case_id in batch["case_id"]]
        append_record(self._log_path(), record)
        if self._tb is not None:
            from src.training.core.tensorboard import write_scalars
            interval = self.config["logging"]["tensorboard"]["scalar_interval"]
            if self.global_step % interval == 0:
                write_scalars(self._tb, record)
                if validation_summary is not None:
                    write_scalars(self._tb, {"type": "validation", "global_step": self.global_step, **validation_summary})
                self._tb.add_scalar("progress/world_size", self.process.world_size, self.global_step)
                self._tb.add_scalar("performance/update_seconds_host", self._update_seconds, self.global_step)
                if group is not None and self._update_seconds > 0:
                    self._tb.add_scalar("performance/utterances_per_second_rank0_host", sum(len(batch["case_id"]) for batch in group) / self._update_seconds, self.global_step)
                if self.process.uses_cuda:
                    self._tb.add_scalar("performance/cuda_allocated_bytes_rank0", torch.cuda.memory_allocated(self.process.device), self.global_step)
                    self._tb.add_scalar("performance/cuda_reserved_bytes_rank0", torch.cuda.memory_reserved(self.process.device), self.global_step)
                    self._tb.add_scalar("performance/cuda_peak_allocated_bytes_rank0", torch.cuda.max_memory_allocated(self.process.device), self.global_step)
            elif validation_summary is not None:
                write_scalars(self._tb, {"type": "validation", "global_step": self.global_step, **validation_summary})

    def _consider_best(self, epoch: int, summary: dict[str, Any]) -> str | None:
        """Save the generator when validation PESQ improves after the official start epoch."""
        if epoch < self.best_checkpoint_start_epoch:
            return None
        if self.config.get("model", {}).get("name") == "semambapp":
            return None
        if self.steps_per_epoch or self.config.get("batch_schedule"):
            loss = summary["losses"].get("generator", {})
            if loss.get("status") != "ok" or loss.get("value") is None:
                return None
            score = float(loss["value"])
            if self.best_loss is not None and score >= self.best_loss:
                return None
            exported = _inference_payload(self.task)
            if exported is None:
                return None
            self.best_loss = score
            path = self.checkpoint_dir / "best_generator.pt"
            save_atomic(path, exported)
            self._attach_provenance(path)
            write_json(self.checkpoint_dir / "best_generator.json", {
                "metric": "validation_loss", "loss": score, "global_step": self.global_step,
                "epoch": epoch, "validation": summary, "config": self.config, "checkpoint": str(path),
            })
            return str(path)
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
        self._attach_provenance(path)
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
        existing = [
            path for path in self.checkpoint_dir.iterdir()
            if path.name != ".gitkeep" and not path.name.startswith(".")
        ]
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


def _is_training_data(dataset) -> bool:
    return (
        not isinstance(dataset, RecipeDataset)
        and callable(getattr(dataset, "epoch", None))
        and callable(getattr(dataset, "data_state", None))
    )


def _provenance_of(dataset):
    source = getattr(dataset, "provenance", None)
    if not callable(source):
        return None
    return source()


def _has_validation(dataset) -> bool:
    if hasattr(dataset, "validation_set"):
        return dataset.validation_set is not None
    return True


def _validation_name(dataset) -> str:
    validation = getattr(dataset, "validation_set", None)
    if validation is not None and hasattr(validation, "name"):
        return validation.name
    identity = dataset.identity()
    return str(identity.get("name", "validation"))


def _load_update(render, update_index: int, accumulation: int):
    """Load one optimizer step. The returned cursor is the next micro-batch."""
    group = render(update_index)
    return group, (update_index + 1) * accumulation


def _load_group(dataset, indices, cursor: int, micro: int, accumulation: int):
    """Load the next optimizer step. The returned cursor is the following step."""
    group = []
    for _ in range(accumulation):
        start = cursor * micro
        group.append(dataset.collate(indices[start : start + micro]))
        cursor += 1
    return group, cursor


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
    record = {
        "type": "run",
        "seed": trainer.seed,
        "config": trainer.config,
        "global_batch_size": trainer.plan.global_batch_size,
        "micro_batch_size": trainer.plan.micro_batch_size,
        "gradient_accumulation_steps": trainer.plan.accumulation_steps,
        "best_checkpoint_start_epoch": trainer.best_checkpoint_start_epoch,
        "scheduler": "epoch",
    }
    provenance = getattr(trainer, "_data_provenance", None)
    if provenance:
        record["data_provenance"] = provenance
    if trainer._run_id:
        record["run_id"] = trainer._run_id
    if trainer._provenance_sha:
        record["provenance_sha256"] = trainer._provenance_sha
    return record


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


_WEIGHT_SLOTS = (
    ("losses", "generator"),
    ("losses", "mr_l1_tfd"),
    ("losses", "discriminator"),
    ("pesq", None),
    ("stoi", None),
    ("si_sdr", None),
    ("si_snr", None),
)


def _reduce_weighted_reports(reports: list[dict[str, Any]], weights: list[int], process: ProcessInfo) -> dict[str, Any]:
    """Weighted mean of batch-mean metrics. Distributed ranks contribute sums and counts."""
    sums: list[float] = []
    counts: list[float] = []
    unavailable: list[int] = []
    present: list[int] = []
    reasons: list[str | None] = []
    for group, name in _WEIGHT_SLOTS:
        total = 0.0
        count = 0.0
        bad = 0
        seen = 0
        reason = None
        for report, weight in zip(reports, weights):
            item = report.get(group) if name is None else (report.get(group) or {}).get(name)
            if not isinstance(item, dict):
                continue
            seen = 1
            if item.get("status") == "unavailable":
                bad = 1
                reason = item.get("reason")
            elif item.get("status") == "ok" and item.get("value") is not None:
                total += float(item["value"]) * float(weight)
                count += float(weight)
            elif reason is None:
                reason = item.get("reason")
        sums.append(total)
        counts.append(count)
        unavailable.append(bad)
        present.append(seen)
        reasons.append(reason)
    sample_weight = float(sum(weights))
    if process.distributed and dist.is_initialized():
        device = process.device
        sum_t = torch.tensor(sums, dtype=torch.float64, device=device)
        count_t = torch.tensor(counts, dtype=torch.float64, device=device)
        bad_t = torch.tensor(unavailable, dtype=torch.int64, device=device)
        present_t = torch.tensor(present, dtype=torch.int64, device=device)
        weight_t = torch.tensor([sample_weight], dtype=torch.float64, device=device)
        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(bad_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(present_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(weight_t, op=dist.ReduceOp.SUM)
        sums = [float(value) for value in sum_t.tolist()]
        counts = [float(value) for value in count_t.tolist()]
        unavailable = [int(value) for value in bad_t.tolist()]
        present = [int(value) for value in present_t.tolist()]
        sample_weight = float(weight_t.item())

    def metric(index: int) -> dict[str, Any] | None:
        if present[index] == 0:
            return None
        if unavailable[index]:
            return {"status": "unavailable", "value": None, "reason": reasons[index], "count": 0}
        if counts[index] > 0:
            return {
                "status": "ok",
                "value": sums[index] / counts[index],
                "reason": None,
                "count": counts[index],
            }
        return {"status": "na", "value": None, "reason": reasons[index] or "no score", "count": 0}

    losses = {}
    for index, (group, name) in enumerate(_WEIGHT_SLOTS):
        if group != "losses" or name is None:
            continue
        item = metric(index)
        if item is not None:
            losses[name] = item
    result: dict[str, Any] = {
        "losses": losses,
        "pesq": metric(3) or {"status": "na", "value": None, "reason": "no score", "count": 0},
        "stoi": metric(4) or {"status": "na", "value": None, "reason": "no score", "count": 0},
        "weight": sample_weight,
    }
    for index, key in ((5, "si_sdr"), (6, "si_snr")):
        item = metric(index)
        if item is not None:
            result[key] = item
    return result


def _aggregate_validation(reports: list[dict[str, Any]]) -> dict[str, Any]:
    names: list[str] = []
    for report in reports:
        for name in report["losses"]:
            if name not in names:
                names.append(name)
    losses = {name: _combine_metric([report["losses"][name] for report in reports if name in report["losses"]]) for name in names}
    result = {
        "losses": losses,
        "pesq": _combine_metric([report["pesq"] for report in reports]),
        "stoi": _combine_metric([report["stoi"] for report in reports]),
    }
    if reports and all("si_sdr" in report for report in reports):
        result["si_sdr"] = _combine_metric([report["si_sdr"] for report in reports])
    if reports and all("si_snr" in report for report in reports):
        result["si_snr"] = _combine_metric([report["si_snr"] for report in reports])
    return result


def _loss_value(summary: dict[str, Any]) -> float | None:
    loss = (summary.get("losses") or {}).get("generator") or {}
    if loss.get("status") != "ok" or loss.get("value") is None:
        return None
    return float(loss["value"])


def _combine_metric(items: list[dict[str, Any]]) -> dict[str, Any]:
    if any(item["status"] == "unavailable" for item in items):
        reason = next(item["reason"] for item in items if item["status"] == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in items if item["status"] == "ok" and item["value"] is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item.get("reason") for item in items if item.get("reason")), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}
