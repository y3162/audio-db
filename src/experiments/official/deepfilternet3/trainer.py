"""DeepFilterNet3 training loop.

Read this file from ``DeepFilterNet3Trainer.fit`` downward.

``fit`` orders start, a resumed pending boundary, then each optimizer step.
When the epoch index is 0, the batch schedule is applied. ``CosineTable.apply``
writes the learning rate and weight decay, and only then does
``run_accumulation`` backward and step AdamW. A partial epoch offers a step
save. The last step opens a pending boundary, offers that save, then validates
with equal utterance weight, writes best, and applies patience. Counters live
on ``DFProgress`` in this file. Clean audio is ``Batch.targets``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.experiments.calc.batch_schedule import scheduled_batch_size
from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.deepfilternet3.dataset import clean_noisy

_clean_noisy = clean_noisy
from src.experiments.official.deepfilternet3.evaluator import evaluate, note_early_stop, write_best, write_rfft
from src.experiments.official.deepfilternet3.loss import component_losses, loss_tensor
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.stopping import observe_validation, should_stop
from src.experiments.trainers import _close_epoch_view, _rank
from src.experiments.training.control import finish_training_run, stop_requested
from src.se.models.deepfilternet3.dsp import DFStream

# --- Trainer. ``fit`` is the timeline. The sections below are the steps it calls. ---

class DeepFilterNet3Trainer:
    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        model: PreparedModel,
        data: Any,
        optimizer: torch.optim.Optimizer,
        *,
        objective: torch.nn.Module,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float = 1.0,
        max_epochs: int = 120,
        early_stopping_patience: int = 25,
        steps_per_epoch: int = 1,
        epochs: int = 120,
        lr_min: float = 1e-6,
        lr_warmup: float = 1e-4,
        warmup_epochs: int = 3,
        weight_decay_end: float = 0.01,
        batch_schedule: list[list[int]] | None = None,
        batch_size_limit: int = 64,
        stop_after_logical_steps: int | None = None,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        require_prepared(model, DeepFilterNet3Model, "model must be a prepared DeepFilterNet3Model")
        if not isinstance(objective, torch.nn.Module):
            raise BuildError("objective must be a loss module")
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        self.optimizer_id = require_adamw(optimizer)
        if model.policy.find_unused_parameters is not True:
            raise BuildError("DeepFilterNet3 requires find_unused_parameters")
        self.objective = objective
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.audio_sample_rate = int(model.unwrapped().generator.params.sr)
        self.audio_teachers = ("official_mixed_target",)
        self.max_logical_steps = positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = positive_int("accumulation", accumulation)
        self.grad_clip = positive_float("grad_clip", grad_clip)
        self.max_epochs = positive_int("max_epochs", max_epochs)
        self.early_stopping_patience = positive_int("early_stopping_patience", early_stopping_patience)
        self.batch_schedule = parse_schedule(batch_schedule)
        self.batch_size_limit = positive_int("batch_size_limit", batch_size_limit)
        self.stop_after_logical_steps = (
            None if stop_after_logical_steps is None else positive_int("stop_after_logical_steps", stop_after_logical_steps)
        )
        steps = positive_int("steps_per_epoch", steps_per_epoch)
        self._cosine = CosineTable(
            optimizer,
            schedule_epochs=positive_int("epochs", epochs),
            lr_min=float(lr_min),
            lr_warmup=float(lr_warmup),
            warmup_epochs=non_negative_int("warmup_epochs", warmup_epochs),
            weight_decay_end=float(weight_decay_end),
            steps_per_epoch=steps,
        )
        self.progress = DFProgress.initial(self.optimizer_id, steps)
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators,
            model_selection,
            data=self.data,
            module=self.model,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection
        self._restored = False
        self._stream: DFStream | None = None

    @property
    def logical_step(self) -> int:
        return self.progress.logical_step

    @property
    def optimizer_updates(self) -> dict[str, int]:
        return self.progress.optimizer_updates

    @property
    def losses(self) -> list[float]:
        return self.progress.losses

    @losses.setter
    def losses(self, value: list[float]) -> None:
        self.progress.losses = value

    @property
    def stop_reason(self) -> str | None:
        return self.progress.stop_reason

    @stop_reason.setter
    def stop_reason(self, value: str) -> None:
        self.progress.stop_reason = value

    def fit(self) -> RunResult:
        """Train until the step budget, the epoch cap, or early stopping. Cosine runs before the optimizer."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        if self.progress.phase == PENDING_VALIDATION:
            self._finish_boundary(runtime, component_id)
            if should_stop(self) or self.progress.early_stopped or self._epoch_cap_reached():
                return self._result("completed")
        while self.progress.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                if self.progress.logical_step > 0:
                    self._offer(runtime, "stop")
                return self._result("interrupted")
            if self._epoch_cap_reached():
                return self._result("completed")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            if index == 0:
                apply_batch_schedule(self.data, epoch, self.batch_schedule, self.batch_size_limit)
            view, steps = open_epoch(
                self.data, self._cosine, self.progress,
                epoch=epoch, index=index, accumulation=self.accumulation, component_id=component_id,
            )
            try:
                position = schedule_index(epoch, index // self.accumulation, steps)
                learning_rate, decay = self._cosine.apply(position)
                local = run_accumulation(
                    self.model, self.optimizer, runtime, view, index,
                    accumulation=self.accumulation, grad_clip=self.grad_clip, component_id=component_id,
                    stream=self._stream, objective=self.objective,
                )
                self.progress.record_step(
                    applied=local.applied, loss=local.loss, learning_rate=learning_rate, decay=decay,
                )
                index += self.accumulation
                self.data.epoch = epoch
                self.data.update_index = index
                note_step(
                    self.journal, self.progress, applied=local.applied, epoch=epoch, loss=local.loss,
                    learning_rate=learning_rate, sample_id=local.sample_id, weight_decay=decay,
                    schedule_index=position,
                )
                if index != view.updates:
                    self._offer_step(runtime, epoch)
                    continue
                self.progress.open_boundary(epoch, steps)
                self._offer(runtime, "boundary_pending")
                if self._stop_requested(runtime):
                    return self._result("interrupted")
                self._finish_boundary(runtime, component_id)
                if should_stop(self) or self.progress.early_stopped or self._epoch_cap_reached():
                    return self._result("completed")
            finally:
                _close_epoch_view(view)
        return self._result("completed")

    def state_dict(self) -> dict[str, Any]:
        return self.progress.export()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        decoded = self.progress.decode(state, self._cosine.tables)
        self._cosine.install(decoded.schedule_steps, decoded.rate_table, decoded.decay_table)
        self.progress.apply(decoded)
        self._restored = True

    def research_predict(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enhanced waveform for research metrics. Official validation does not call this."""

        if self._stream is None:
            raise RuntimeFailure("DeepFilterNet3 research predict has no DF stream")
        parts = component_losses(self._stream, self.model.unwrapped(), targets, inputs, self.objective)
        wave = parts.get("enhanced_wave")
        if not isinstance(wave, torch.Tensor):
            raise RuntimeFailure("DeepFilterNet3 research predict did not return a waveform")
        return wave

    def _finish_boundary(self, runtime: ProcessRuntime, component_id: str) -> None:
        loss = finish_boundary(
            self.progress, runtime, self.data, self.model, self._stream, self.objective,
            patience=self.early_stopping_patience, component_id=component_id,
            output_directory=self.lifecycle.output_directory,
        )
        official = {"loss": float(loss)}
        observe_validation(
            getattr(self, "training_control", None), official,
            step=self.progress.logical_step, scoped=self.research.attached,
        )
        extra = self.research.score(
            runtime, output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity, kind="boundary",
            epoch=int(self.data.epoch) - 1, logical_step=self.progress.logical_step,
            official=official, stopping=getattr(self, "training_control", None),
        )
        self._offer(runtime, "boundary", metrics={**official, **extra} if extra else official)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> float:
        return evaluate(
            runtime, self.data, self.model, self._stream, self.objective, trainer_epoch, component_id,
        )

    def _forward_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        return forward_loss(self._stream, self.model, self.objective, batches, component_id, runtime)

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        self.research.rewind(
            self.lifecycle.output_directory, restored=self._restored, phase=self.progress.phase,
            progress={"logical_step": int(self.progress.logical_step)},
        )
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self.batch_schedule is not None and runtime.world_size != 1:
            raise RuntimeFailure("batch_schedule is implemented for one process")
        self.losses = self.journal.begin(
            restored=self._restored, logical_step=self.progress.logical_step, rewind_samples_on_rank0=False,
        )
        if not self._restored:
            self.progress.reset_fresh()
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        require_group_size(self.data, self.accumulation, str(component_id))
        self._stream = stream_for(self.model.unwrapped(), runtime.device)
        record_rfft(runtime, self.lifecycle.output_directory)
        runtime.barrier()
        return str(component_id)

    def _offer_step(self, runtime: ProcessRuntime, epoch: int) -> None:
        step_metrics = self.research.score(
            runtime, output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity, kind="step", epoch=epoch,
            logical_step=self.progress.logical_step, stopping=getattr(self, "training_control", None),
        )
        if step_metrics:
            self._offer(runtime, "step", metrics=step_metrics)
        else:
            self._offer(runtime, "step")

    def _offer(self, runtime: ProcessRuntime, event: str, metrics: dict[str, float] | None = None) -> None:
        if event == "step":
            self.lifecycle.save_if_needed(
                event, logical_step=self.progress.logical_step, metrics=metrics, prepare=self._sync_ranks,
            )
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.progress.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        runtime.broadcast_buffers(self.model)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        return stop_requested(self, runtime)

    def _epoch_cap_reached(self) -> bool:
        return int(self.data.epoch) >= self.max_epochs

    def _result(self, reason: str) -> RunResult:
        return finish_training_run(self, reason, prepare=self._sync_ranks)


# --- Checkpoint progress. This object is the only live counter owner. ---

IN_EPOCH = "in_epoch"

PENDING_VALIDATION = "pending_validation"

_PHASES = frozenset({IN_EPOCH, PENDING_VALIDATION})

@dataclass
class _DecodedProgress:
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    schedule_steps: int
    rate_table: list[float]
    decay_table: list[float]
    history: list[Any]
    learning_rates: list[float]
    weight_decays: list[float]
    plateau_count: int
    plateau_metric: float | None
    best_loss: float | None
    best_epoch: int | None
    early_stopped: bool
    boundary: dict[str, Any] | None
    finished_epoch: int | None
    stop_reason: str | None

@dataclass
class DFProgress:
    """Checkpoint fields for one DeepFilterNet3 run, plus the unsaved loss list."""

    best_epoch: int | None
    best_loss: float | None
    boundary: dict[str, Any] | None
    early_stopped: bool
    finished_epoch: int | None
    history: list[dict[str, Any]]
    learning_rates: list[float]
    logical_step: int
    losses: list[float]
    optimizer_updates: dict[str, int]
    phase: str
    plateau_count: int
    plateau_metric: float | None
    schedule_steps: int
    stop_reason: str | None
    weight_decays: list[float]
    _optimizer_id: str

    @classmethod
    def initial(cls, optimizer_id: str, schedule_steps: int) -> DFProgress:
        return cls(
            best_epoch=None,
            best_loss=None,
            boundary=None,
            early_stopped=False,
            finished_epoch=None,
            history=[],
            learning_rates=[],
            logical_step=0,
            losses=[],
            optimizer_updates={optimizer_id: 0},
            phase=IN_EPOCH,
            plateau_count=0,
            plateau_metric=None,
            schedule_steps=schedule_steps,
            stop_reason="initialized",
            weight_decays=[],
            _optimizer_id=optimizer_id,
        )

    def export(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "boundary": self.boundary,
            "early_stopped": self.early_stopped,
            "finished_epoch": self.finished_epoch,
            "history": list(self.history),
            "learning_rates": list(self.learning_rates),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "plateau_count": self.plateau_count,
            "plateau_metric": self.plateau_metric,
            "schedule_steps": self.schedule_steps,
            "stop_reason": self.stop_reason,
            "weight_decays": list(self.weight_decays),
        }

    def decode(self, state: dict[str, Any], tables: Callable[[int], tuple[list[float], list[float]]]) -> _DecodedProgress:
        return _decode_progress(state, self._optimizer_id, tables)

    def apply(self, decoded: _DecodedProgress) -> None:
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.schedule_steps = decoded.schedule_steps
        self.history = decoded.history
        self.learning_rates = decoded.learning_rates
        self.weight_decays = decoded.weight_decays
        self.plateau_count = decoded.plateau_count
        self.plateau_metric = decoded.plateau_metric
        self.best_loss = decoded.best_loss
        self.best_epoch = decoded.best_epoch
        self.early_stopped = decoded.early_stopped
        self.boundary = decoded.boundary
        self.finished_epoch = decoded.finished_epoch
        if decoded.stop_reason is not None:
            self.stop_reason = decoded.stop_reason

    def reset_fresh(self) -> None:
        self.logical_step = 0
        self.optimizer_updates = {self._optimizer_id: 0}
        self.losses = []
        self.learning_rates = []
        self.weight_decays = []
        self.history = []
        self.best_loss = None
        self.best_epoch = None
        self.plateau_metric = None
        self.plateau_count = 0
        self.early_stopped = False
        self.phase = IN_EPOCH
        self.finished_epoch = None
        self.boundary = None

    def record_step(self, *, applied: bool, loss: float, learning_rate: float, decay: float) -> None:
        self.logical_step += 1
        if applied:
            self.optimizer_updates[self._optimizer_id] += 1
        self.losses.append(loss)
        self.learning_rates.append(learning_rate)
        self.weight_decays.append(decay)

    def open_boundary(self, epoch: int, schedule_steps: int) -> None:
        self.phase = PENDING_VALIDATION
        self.finished_epoch = epoch
        self.boundary = {"epoch": epoch, "schedule_steps": schedule_steps}

def _decode_progress(
    state: dict[str, Any],
    optimizer_id: str,
    tables: Callable[[int], tuple[list[float], list[float]]],
) -> _DecodedProgress:
    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    steps = state.get("schedule_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise RuntimeFailure("schedule_steps is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    if optimizer_id not in converted:
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer ids")
    rate_table, decay_table = tables(steps)
    history = list(state.get("history") or [])
    learning_rates = [float(value) for value in state.get("learning_rates") or []]
    weight_decays = [float(value) for value in state.get("weight_decays") or []]
    plateau_count = int(state.get("plateau_count") or 0)
    metric = state.get("plateau_metric")
    plateau_metric = None if metric is None else float(metric)
    best = state.get("best_loss")
    best_loss = None if best is None else float(best)
    best_epoch = state.get("best_epoch")
    decoded_epoch = None if best_epoch is None else int(best_epoch)
    boundary = state.get("boundary")
    decoded_boundary = None if boundary is None else dict(boundary)
    finished = state.get("finished_epoch")
    finished_epoch = None if finished is None else int(finished)
    stop_reason = state.get("stop_reason")
    return _DecodedProgress(
        logical_step=logical_step,
        optimizer_updates=converted,
        phase=str(phase),
        schedule_steps=steps,
        rate_table=rate_table,
        decay_table=decay_table,
        history=history,
        learning_rates=learning_rates,
        weight_decays=weight_decays,
        plateau_count=plateau_count,
        plateau_metric=plateau_metric,
        best_loss=best_loss,
        best_epoch=decoded_epoch,
        early_stopped=bool(state.get("early_stopped", False)),
        boundary=decoded_boundary,
        finished_epoch=finished_epoch,
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )


# --- Constructor checks and the epoch-0 batch schedule. ---

def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be an integer >= 1")
    return value

def non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value

def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)

def require_adamw(optimizer: torch.optim.Optimizer) -> str:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError("DeepFilterNet3 optimizer must be AdamW")
    group = optimizer.param_groups[0]
    if float(group["lr"]) != 0.001:
        raise BuildError("DeepFilterNet3 learning rate must be 0.001")
    betas = tuple(group["betas"])
    if betas != (0.9, 0.999):
        raise BuildError("DeepFilterNet3 betas must be (0.9, 0.999)")
    if float(group["weight_decay"]) != 1e-12:
        raise BuildError("DeepFilterNet3 weight decay must be 1e-12")
    if group.get("amsgrad") is not True:
        raise BuildError("DeepFilterNet3 AdamW requires amsgrad")
    component_id = getattr(optimizer, "_experiment_component_id", None)
    if not isinstance(component_id, str):
        raise BuildError("optimizer has no component id")
    return component_id

def parse_schedule(raw: list[list[int]] | None) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise BuildError("batch_schedule must be a non-empty list of [epoch, batch_size]")
    schedule = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise BuildError("batch_schedule entries must be [epoch, batch_size]")
        schedule.append(
            (non_negative_int("batch_schedule epoch", item[0]), positive_int("batch_schedule batch_size", item[1]))
        )
    if schedule[0][0] != 0:
        raise BuildError("batch_schedule must start at epoch 0")
    return schedule

def apply_batch_schedule(data: object, epoch: int, schedule: list[tuple[int, int]] | None, limit: int) -> None:
    if schedule is None:
        return
    size = scheduled_batch_size(epoch, limit, schedule)
    current = getattr(getattr(data, "batcher", None), "micro_batch", None)
    if current == size:
        return
    rebind = getattr(data, "rebind_micro_batch", None)
    if not callable(rebind):
        raise RuntimeFailure("data cannot rebind a DeepFilterNet3 micro-batch")
    rebind(size)

def require_group_size(data: object, accumulation: int, component_id: str) -> None:
    group_size = getattr(data, "group_size", None)
    if group_size is not None and group_size != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} "
            f"does not match accumulation {accumulation}"
        )


# --- Global schedule index used by the cosine table. ---

def schedule_index(epoch: int, index_in_epoch: int, steps_per_epoch: int) -> int:
    """DeepFilterNet's table index. Past epochs are not summed."""

    if epoch < 0 or index_in_epoch < 0 or steps_per_epoch < 1:
        raise RuntimeFailure("schedule position is invalid")
    if index_in_epoch >= steps_per_epoch:
        raise RuntimeFailure(
            f"index {index_in_epoch} is outside an epoch of {steps_per_epoch} updates"
        )
    return epoch * steps_per_epoch + index_in_epoch


# --- Cosine and weight decay. ``apply`` runs before the optimizer step. ---

class CosineTable:
    """Own the rebuilt learning-rate and weight-decay tables for one epoch length."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        schedule_epochs: int,
        lr_min: float,
        lr_warmup: float,
        warmup_epochs: int,
        weight_decay_end: float,
        steps_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self.schedule_epochs = schedule_epochs
        self.lr_min = lr_min
        self.lr_warmup = lr_warmup
        self.warmup_epochs = warmup_epochs
        self.weight_decay_end = weight_decay_end
        self.steps = steps_per_epoch
        self.rates, self.decays = self.tables(steps_per_epoch)

    def tables(self, steps_per_epoch: int) -> tuple[list[float], list[float]]:
        config = {
            "optimizer": {"lr": 0.001, "weight_decay": 1e-12},
            "scheduler": {
                "epochs": self.schedule_epochs,
                "lr_min": self.lr_min,
                "lr_warmup": self.lr_warmup,
                "name": "deepfilter_cosine",
                "steps_per_epoch": steps_per_epoch,
                "warmup_epochs": self.warmup_epochs,
                "weight_decay_end": self.weight_decay_end,
            },
        }
        try:
            return deepfilter_cosine_schedules(config)
        except TrainingError as exc:
            raise BuildError(str(exc)) from exc

    def retarget(self, steps_per_epoch: int) -> None:
        if steps_per_epoch == self.steps and self.rates:
            return
        self.rates, self.decays = self.tables(steps_per_epoch)
        self.steps = steps_per_epoch

    def install(self, steps_per_epoch: int, rates: list[float], decays: list[float]) -> None:
        self.steps = steps_per_epoch
        self.rates = rates
        self.decays = decays

    def apply(self, index: int) -> tuple[float, float]:
        """Copy one table row onto the optimizer before it steps."""

        if index < 0 or index >= len(self.rates):
            raise RuntimeFailure(
                f"deepfilter cosine step {index} is outside the schedule of {len(self.rates)}"
            )
        learning_rate = self.rates[index]
        decay = self.decays[index]
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
            if float(group["weight_decay"]) > 0.0:
                group["weight_decay"] = decay
        return learning_rate, decay

def open_epoch(
    data: Any,
    cosine: CosineTable,
    progress: Any,
    *,
    epoch: int,
    index: int,
    accumulation: int,
    component_id: str,
) -> tuple[Any, int]:
    """Open the epoch view and rebuild the cosine table for its optimizer-step count."""

    from src.experiments.trainers import _close_epoch_view

    view = data.train_epoch(epoch, start_update=index)
    try:
        if view.updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if view.updates % accumulation != 0:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} has {view.updates} micro-batches, "
                f"which is not divisible by accumulation {accumulation}"
            )
        steps = view.updates // accumulation
        cosine.retarget(steps)
        progress.schedule_steps = cosine.steps
        if index > view.updates or index % accumulation != 0:
            raise RuntimeFailure(f"component {component_id}: update {index} is not an optimizer boundary")
        if index == view.updates:
            raise RuntimeFailure(
                f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
            )
    except Exception:
        _close_epoch_view(view)
        raise
    return view, steps


def deepfilter_cosine_schedules(config: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Shipped per-step cosine from ``df.lr.cosine_scheduler``.

    Warmup is ``numpy.linspace(lr_warmup, lr, warmup_epochs * steps_per_epoch)``,
    including the endpoint. The remaining steps are one cosine cycle from the
    base learning rate down to ``lr_min``. Weight decay is one cosine from the
    optimizer value to ``weight_decay_end``, with no warmup. A linspace of one
    point is the stop value, matching NumPy.
    """
    schedule = config["scheduler"]
    steps = int(schedule["steps_per_epoch"])
    epochs = int(schedule["epochs"])
    total = steps * epochs
    warmup = int(schedule["warmup_epochs"]) * steps
    if warmup > total:
        raise TrainingError("deepfilter cosine warmup is longer than the run")
    base = float(config["optimizer"]["lr"])
    final = float(schedule["lr_min"])
    start = float(schedule["lr_warmup"])
    learning_rates = _linspace(start, base, warmup) + _cosine_cycle(base, final, total - warmup)
    if len(learning_rates) != total:
        raise TrainingError("deepfilter cosine learning-rate schedule has the wrong length")
    decay = _cosine_cycle(
        float(config["optimizer"]["weight_decay"]),
        float(schedule["weight_decay_end"]),
        total,
    )
    return learning_rates, decay

def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [stop]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]

def _cosine_cycle(base: float, final: float, count: int) -> list[float]:
    if count <= 0:
        return []
    return [
        final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * index / count))
        for index in range(count)
    ]


# --- Step journal. The step save itself stays in ``_offer_step``. ---

def note_step(
    journal: StepJournal,
    progress: DFProgress,
    *,
    applied: bool,
    epoch: int,
    loss: float,
    learning_rate: float,
    sample_id: str,
    weight_decay: float,
    schedule_index: int,
) -> None:
    journal.write_metrics(
        {
            "applied": applied,
            "epoch": epoch,
            "logical_step": progress.logical_step,
            "loss": loss,
            "lr": learning_rate,
            "optimizer_updates": dict(progress.optimizer_updates),
            "sample_id": sample_id,
            "weight_decay": weight_decay,
        }
    )
    journal.write_sample(
        {
            "epoch": epoch,
            "logical_step": progress.logical_step,
            "loss": loss,
            "lr": learning_rate,
            "sample_id": sample_id,
            "schedule_index": schedule_index,
            "weight_decay": weight_decay,
        }
    )

def record_rfft(runtime: ProcessRuntime, output_directory: Any) -> None:
    """Write the kernel selected on this host. Rank 0 writes the file."""

    problem = ""
    if _rank(runtime) == 0:
        try:
            from src.se.models.deepfilternet3.dsp import rfft_runtime_record

            write_rfft(Path(output_directory), rfft_runtime_record())
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
    if runtime.world_size > 1:
        reported = runtime.gather_objects(problem)
        problem = next((item for item in reported if item), "")
    if problem:
        raise RuntimeFailure(f"DeepFilterNet3 realfft kernel: {problem}")


# --- Accumulation. Cosine is already on the optimizer. ---

@dataclass(frozen=True)
class DFStep:
    loss: float
    applied: bool
    sample_id: str

def run_accumulation(
    model: Any,
    optimizer: torch.optim.Optimizer,
    runtime: ProcessRuntime,
    view: Any,
    index: int,
    *,
    accumulation: int,
    grad_clip: float,
    component_id: str,
    stream: DFStream | None,
    objective: torch.nn.Module,
) -> DFStep:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total: torch.Tensor | None = None
    sample_ids: list[str] = []
    for offset in range(accumulation):
        sync = offset == accumulation - 1
        batches = view.batch(index + offset)
        sample_ids.extend(batch.sample_id for batch in batches)
        with runtime.sync_gradients([model], enabled=sync):
            with runtime.autocast():
                micro = forward_loss(stream, model, objective, batches, component_id, runtime)
            runtime.backward(micro / accumulation)
        detached = (micro / accumulation).detach()
        total = detached if total is None else total + detached
    runtime.clip(model, grad_clip, optimizer=optimizer)
    result = runtime.step(optimizer)
    runtime.end_step()
    if total is None or not bool(torch.isfinite(total).all()):
        raise RuntimeFailure(f"component {component_id}: loss is not finite")
    return DFStep(float(total), bool(result.applied), ",".join(sample_ids))

def forward_loss(
    stream: DFStream | None,
    model: Any,
    objective: torch.nn.Module,
    batches: list[Batch],
    component_id: str,
    runtime: ProcessRuntime,
) -> torch.Tensor:
    if not batches:
        raise RuntimeFailure(f"component {component_id}: empty micro-batch")
    total: torch.Tensor | None = None
    for batch in batches:
        batch = runtime.move(batch)
        clean, noisy = clean_noisy(batch)
        value = loss_tensor(stream, clean, noisy, model.execution(), objective)
        total = value if total is None else total + value
    assert total is not None
    return total / len(batches)

def stream_for(model: Any, device: torch.device) -> DFStream:
    """Open a DFStream from the model's own sample rate and FFT sizes."""

    params = model.generator.params
    return DFStream(
        params.sr,
        params.fft_size,
        params.hop_size,
        params.nb_erb,
        params.nb_df,
        params.min_nb_freqs,
        params.norm_tau,
        device,
    )


# --- Epoch boundary: equal-weight validation, strict best, patience. ---

def finish(
    progress: DFProgress,
    runtime: ProcessRuntime,
    data: Any,
    model: Any,
    stream: DFStream | None,
    objective: torch.nn.Module,
    *,
    patience: int,
    component_id: str,
    output_directory: Any,
) -> float:
    if progress.phase != PENDING_VALIDATION or progress.finished_epoch is None:
        raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
    trainer_epoch = int(progress.finished_epoch)
    loss = evaluate(runtime, data, model, stream, objective, trainer_epoch, component_id)
    progress.plateau_metric, progress.plateau_count, stop = note_early_stop(
        progress.plateau_metric, progress.plateau_count, loss, patience
    )
    progress.history.append({"epoch": trainer_epoch, "loss": loss})
    improved = progress.best_loss is None or loss < progress.best_loss
    if improved:
        progress.best_loss = loss
        progress.best_epoch = trainer_epoch
    _write_best(runtime, model, output_directory, progress.best_epoch, progress.best_loss, progress.logical_step, improved)
    progress.phase = "in_epoch"
    progress.finished_epoch = None
    progress.boundary = None
    data.epoch = trainer_epoch + 1
    data.update_index = 0
    if stop:
        progress.early_stopped = True
    return float(loss)

def _write_best(
    runtime: ProcessRuntime,
    model: Any,
    directory: Any,
    best_epoch: int | None,
    best_loss: float | None,
    logical_step: int,
    improved: bool,
) -> None:
    if not improved or _rank(runtime) != 0:
        runtime.barrier()
        return
    write_best(
        directory,
        {
            "epoch": best_epoch,
            "generator": model.unwrapped().export_for_inference(),
            "logical_step": logical_step,
            "loss": best_loss,
        },
    )
    runtime.barrier()

finish_boundary = finish
