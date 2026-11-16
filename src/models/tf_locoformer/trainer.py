"""TF-Locoformer DNS training loop.

Read this file from ``TFLocoformerTrainer.fit`` downward.

``fit`` orders start, a resumed pending boundary, one grouped accumulation
update, ``step_batch`` immediately after an applied optimizer update, the
deferred step log, then either a later step save or a pending save,
validation, one ``step_validation``, n-best, research, and the boundary
save. ``scheduler.step`` is not called. Counters live on
``LocoformerProgress`` in this file. ``learning_rates`` are the rates that
were used. ``scheduled_rates`` are the rates ``step_batch`` left for the next
update. SI-SDR is a monitor, not the objective.

``LocoformerUpdate.run`` averages the injected objective, keeps detached
SI-SDR, clips, and takes one AdamW step. ``LocoformerBoundary.run`` validates,
calls the trainer's ``step_validation`` once, selects, then early-stops.
``DeferredStepLog.drain`` offers a continuing step save, and ``fit`` calls
that drain before the next forward and again before ``boundary_pending``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.experiments.calc.optim import WarmupReduceLROnPlateau
from src.experiments.calc.spectral import si_sdr_loss
from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.models.tf_gridnet.evaluator import early_stop, rank_validation_loss
from src.models.tf_locoformer.evaluator import (
    LocoformerValidator,
    rank_validation_si_snr,
    write_selection,
)
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.stopping import observe_validation, should_stop
from src.experiments.training.control import (
    close_training_run,
    finite_gradient_report,
    connected_zero,
    gather_row_counts,
    micro_rows,
    open_grouped_window,
    stop_after_step,
    weight_loss,
    window_scale,
)
from src.experiments.training.objectives import call_objective, coerce_spectral_objective, term_floats
from src.experiments.trainers import _close_epoch_view, _rank

# --- Trainer. ``fit`` is the timeline. The sections below are the steps it calls. ---

RESEARCH_VALIDATION_PROFILE = "tf_locoformer.dns.accept_recipe.research_valid_fixed@1"


class TFLocoformerTrainer:
    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        model: PreparedModel,
        objective: torch.nn.Module,
        data: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupReduceLROnPlateau,
        *,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float = 5.0,
        keep_nbest: int = 5,
        max_epochs: int | None = None,
        early_stopping_patience: int | None = None,
        stop_after_logical_steps: int | None = None,
        validation_profile: str = RESEARCH_VALIDATION_PROFILE,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        objective, optimizer_id = require_locoformer_setup(
            model, objective, data, optimizer, scheduler, validation_profile
        )
        self.model, self.objective, self.data = model, objective, data
        self.optimizer, self.scheduler = optimizer, scheduler
        self.audio_sample_rate, self.audio_teachers = 16000, ("dry_clean",)
        self.max_logical_steps = positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = positive_int("accumulation", accumulation)
        self.grad_clip = positive_float("grad_clip", grad_clip)
        self.keep_nbest = positive_int("keep_nbest", keep_nbest)
        self.max_epochs = None if max_epochs is None else positive_int("max_epochs", max_epochs)
        self.early_stopping_patience = optional_patience(early_stopping_patience)
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self.progress = LocoformerProgress.initial(optimizer_id, validation_profile)
        self._log = DeferredStepLog()
        self._restored = False
        self._update = LocoformerUpdate(
            model, objective, optimizer, accumulation=self.accumulation, grad_clip=self.grad_clip,
        )
        self._boundary = LocoformerBoundary(
            LocoformerValidator(model, objective),
            early_stopping_patience=early_stopping_patience,
        )
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators, model_selection, data=data, module=model, predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection

    def preflight_probe(self) -> dict[str, Any]:
        """One real window of forward, loss, and backward. The optimizer does not step."""

        runtime = getattr(self, "runtime", None)
        if not isinstance(runtime, ProcessRuntime):
            raise BuildError("preflight has no runtime")
        component_id = str(getattr(self, "_experiment_component_id", "trainer"))
        was_training = bool(self.model.training)
        self.model.train()
        _epoch, _index, view, planned = open_grouped_window(self, component_id)
        try:
            local = self._update.run(planned, runtime, component_id, apply_optimizer=False)
            return finite_gradient_report(
                [self.model], loss=local.loss, sample_id=local.sample_id, component_id=component_id,
            )
        finally:
            if not was_training:
                self.model.eval()
            _close_epoch_view(view)

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or early stopping."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        progress = self.progress
        if should_stop(self):
            return self._finish("completed")
        if self._restored and (
            len(progress.learning_rates) != progress.logical_step
            or len(progress.scheduled_rates) != progress.logical_step
        ):
            raise RuntimeFailure(
                f"component {component_id}: learning-rate history does not match logical step"
            )
        if progress.pending:
            self._finish_pending(runtime, component_id)
            if should_stop(self) or progress.early_stopped or self._epoch_capped():
                return self._finish("completed")
        while progress.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                self._drain()
                if progress.logical_step > 0:
                    self.lifecycle.save_if_needed("stop", logical_step=progress.logical_step)
                return self._finish("interrupted")
            if self._epoch_capped():
                self._drain()
                return self._finish("completed")
            epoch, index, view, planned = open_grouped_window(self, component_id)
            try:
                self._drain()
                local = self._update.run(planned, runtime, component_id)
                scheduled_lr = local.used_lr
                if local.applied:
                    progress.count_applied_optimizer()
                    self.scheduler.step_batch()
                    scheduled_lr = float(self.optimizer.param_groups[0]["lr"])
                progress.record_step(loss=local.loss, used_lr=local.used_lr, scheduled_lr=scheduled_lr)
                index += len(planned)
                progress.place_cursor(self.data, epoch, index)
                epoch_done = index == view.updates
                step_metrics: dict[str, float] = {}
                if not epoch_done:
                    step_metrics = self._research(runtime, "step", epoch)
                self._log.queue(
                    {
                        "applied": local.applied,
                        "epoch": epoch,
                        "logical_step": progress.logical_step,
                        "loss": local.loss,
                        "offer_step": not epoch_done,
                        "sample_id": local.sample_id,
                        "scheduled_lr": scheduled_lr,
                        "si_sdr": local.si_sdr,
                        "step_metrics": step_metrics,
                        "terms": local.terms,
                        "used_lr": local.used_lr,
                    }
                )
                if not epoch_done:
                    continue
                self._drain()
                progress.open_pending(epoch, warmup_updated=local.applied)
                self._on_boundary_phase(progress.phase)
                self.lifecycle.save_if_needed("boundary_pending", logical_step=progress.logical_step)
                if self._stop_requested(runtime):
                    return self._finish("interrupted")
                self._finish_pending(runtime, component_id)
                if should_stop(self) or progress.early_stopped or self._epoch_capped():
                    return self._finish("completed")
            finally:
                _close_epoch_view(view)
        self._drain()
        return self._finish("completed")

    def state_dict(self) -> dict[str, Any]:
        return self.progress.export()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.progress.restore(state)
        self._restored = True

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        progress = self.progress
        self.research.rewind(
            self.lifecycle.output_directory,
            restored=self._restored,
            phase=progress.phase,
            progress={"logical_step": progress.logical_step},
        )
        component_id = str(getattr(self, "_experiment_component_id", "trainer"))
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        progress.losses = self.journal.begin(restored=self._restored, logical_step=progress.logical_step)
        if not self._restored:
            progress.reset_fresh()
            progress.reset_cursor(self.data)
        bind_trainer_data(self)
        require_group_size(self.data, self.accumulation, component_id)
        runtime.barrier()
        return component_id

    def _finish_pending(self, runtime: ProcessRuntime, component_id: str) -> None:
        done = self._boundary.run(
            self.progress,
            runtime,
            self.data,
            self.lifecycle.output_directory,
            component_id,
            self._on_boundary_phase,
            self._step_validation,
            self._select,
        )
        official = {"loss": done.loss, "si_snr": done.si_snr}
        observe_validation(
            getattr(self, "training_control", None),
            official,
            step=self.progress.logical_step,
            scoped=self.research.attached,
        )
        extra = self._research(runtime, "boundary", done.trainer_epoch, official=official)
        metrics = {**official, **extra} if extra else official
        self.lifecycle.save_if_needed("boundary", logical_step=self.progress.logical_step, metrics=metrics)

    def _step_validation(self, loss: float) -> float:
        """Step the plateau side once. Warmup ``step_num`` must not move."""

        step_num_before = int(self.scheduler.step_num)
        self.scheduler.step_validation(float(loss))
        if int(self.scheduler.step_num) != step_num_before:
            component_id = getattr(self, "_experiment_component_id", "trainer")
            raise RuntimeFailure(f"component {component_id}: validation advanced the warmup step")
        return float(self.optimizer.param_groups[0]["lr"])

    def _select(
        self,
        runtime: ProcessRuntime,
        directory: Path,
        espnet_epoch: int,
        loss: float,
        si_snr: float,
    ) -> None:
        """Rank valid.loss and valid.si_snr, then write the official selection files."""

        select_checkpoints(
            self.progress,
            runtime,
            directory,
            self.model,
            espnet_epoch,
            loss,
            si_snr,
            self.keep_nbest,
            self.progress.validation_profile,
        )

    def _research(
        self,
        runtime: ProcessRuntime,
        kind: str,
        epoch: int,
        official: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return self.research.score(
            runtime,
            output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity,
            kind=kind,
            epoch=epoch,
            logical_step=self.progress.logical_step,
            official=official,
            stopping=getattr(self, "training_control", None),
        )

    def _on_boundary_phase(self, phase: str) -> None:
        del phase

    def _drain(self) -> None:
        self._log.drain(
            self.journal,
            self.lifecycle,
            self.progress.optimizer_updates,
            int(self.scheduler.step_num),
        )

    def _finish(self, reason: str) -> RunResult:
        return close_training_run(
            reason,
            research=self.research,
            runtime=self.runtime,
            lifecycle=self.lifecycle,
            data=self.data,
            logical_step=self.progress.logical_step,
            optimizer_updates=self.progress.optimizer_updates,
            training_control=getattr(self, "training_control", None),
            set_stop_reason=self.progress.set_stop_reason,
        )

    def _stop_requested(self, runtime: object | None = None) -> bool:
        return stop_after_step(self.progress.logical_step, self.stop_after_logical_steps, runtime)

    def _epoch_capped(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

# --- Checkpoint progress. This object is the only live counter owner. ---

IN_EPOCH = "in_epoch"
PENDING_VALIDATION = "pending_validation"
_PHASES = frozenset({IN_EPOCH, PENDING_VALIDATION})


def open_boundary(trainer_epoch: int, *, warmup_updated: bool) -> dict[str, Any]:
    """Record that validation, the plateau update, and selection have not run."""

    return {
        "espnet_epoch": trainer_epoch + 1,
        "learning_rate": None,
        "optimizer_updated": True,
        "scheduler_updated": False,
        "selection_done": False,
        "trainer_epoch": trainer_epoch,
        "validation_done": False,
        "validation_loss": None,
        "validation_si_snr": None,
        "validation_weight": None,
        "warmup_updated": bool(warmup_updated),
    }


@dataclass(frozen=True)
class _DecodedProgress:
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[Any]
    learning_rates: list[float]
    scheduled_rates: list[float]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str | None


def _decode_progress(state: dict[str, Any], *, optimizer_id: str, validation_profile: str) -> _DecodedProgress:
    """Validate and copy a payload. This does not touch the live progress."""

    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    if state.get("validation_profile") != validation_profile:
        raise RuntimeFailure("validation profile does not match this trainer")
    history = state.get("history")
    rates = state.get("learning_rates")
    scheduled = state.get("scheduled_rates")
    completed = state.get("completed_boundaries")
    if (
        not isinstance(history, list)
        or not isinstance(rates, list)
        or not isinstance(scheduled, list)
        or not isinstance(completed, list)
    ):
        raise RuntimeFailure("trainer history is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    copied_rates = [float(value) for value in rates]
    copied_scheduled = [float(value) for value in scheduled]
    boundary = state.get("boundary")
    copied_boundary = None if boundary is None else dict(boundary)
    finished = state.get("finished_epoch")
    if finished is not None and (isinstance(finished, bool) or not isinstance(finished, int)):
        raise RuntimeFailure("finished_epoch is invalid")
    if optimizer_id not in converted:
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer id")
    stop_reason = state.get("stop_reason")
    return _DecodedProgress(
        logical_step=logical_step,
        optimizer_updates=converted,
        phase=phase,
        history=list(history),
        learning_rates=copied_rates,
        scheduled_rates=copied_scheduled,
        completed_boundaries=list(completed),
        boundary=copied_boundary,
        finished_epoch=finished,
        early_stopped=bool(state.get("early_stopped", False)),
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )


@dataclass
class LocoformerProgress:
    """Live TF-Locoformer counters. Checkpoint code sees them only through ``export``."""

    logical_step: int
    optimizer_updates: dict[str, int]
    learning_rates: list[float]
    scheduled_rates: list[float]
    history: list[Any]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    phase: str
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str
    validation_profile: str
    losses: list[float]
    _optimizer_id: str = field(repr=False)

    @classmethod
    def initial(cls, optimizer_id: str, validation_profile: str) -> LocoformerProgress:
        return cls(
            logical_step=0,
            optimizer_updates={optimizer_id: 0},
            learning_rates=[],
            scheduled_rates=[],
            history=[],
            completed_boundaries=[],
            boundary=None,
            phase=IN_EPOCH,
            finished_epoch=None,
            early_stopped=False,
            stop_reason="initialized",
            validation_profile=validation_profile,
            losses=[],
            _optimizer_id=optimizer_id,
        )

    @property
    def pending(self) -> bool:
        return self.phase == PENDING_VALIDATION

    def reset_fresh(self) -> None:
        """Zero a run that is not resuming. ``stop_reason`` and the profile stay."""

        self.logical_step = 0
        self.optimizer_updates = {self._optimizer_id: 0}
        self.losses = []
        self.learning_rates = []
        self.scheduled_rates = []
        self.history = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None
        self.early_stopped = False

    def count_applied_optimizer(self) -> None:
        """Count one optimizer step that ``runtime.step`` applied. Does not move the logical step."""

        self.optimizer_updates[self._optimizer_id] += 1

    def record_step(self, *, loss: float, used_lr: float, scheduled_lr: float) -> None:
        """Count the logical step and both learning-rate histories."""

        self.logical_step += 1
        self.losses.append(loss)
        self.learning_rates.append(used_lr)
        self.scheduled_rates.append(scheduled_lr)

    def place_cursor(self, data: Any, epoch: int, index: int) -> None:
        data.epoch = int(epoch)
        data.update_index = int(index)

    def reset_cursor(self, data: Any) -> None:
        data.epoch = 0
        data.update_index = 0

    def open_pending(self, trainer_epoch: int, *, warmup_updated: bool) -> None:
        self.phase = PENDING_VALIDATION
        self.finished_epoch = int(trainer_epoch)
        self.boundary = open_boundary(trainer_epoch, warmup_updated=warmup_updated)

    def note_validation(self, loss: float, si_snr: float, weight: int) -> None:
        boundary = self._open_boundary()
        boundary["validation_done"] = True
        boundary["validation_loss"] = loss
        boundary["validation_si_snr"] = si_snr
        boundary["validation_weight"] = weight

    def note_scheduler(self, learning_rate: float) -> None:
        boundary = self._open_boundary()
        boundary["scheduler_updated"] = True
        boundary["learning_rate"] = float(learning_rate)

    def append_history(self, row: dict[str, Any]) -> None:
        self.history.append(row)

    def close_boundary(self) -> None:
        boundary = self._open_boundary()
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None

    def roll_epoch(self, data: Any, trainer_epoch: int) -> None:
        data.epoch = int(trainer_epoch) + 1
        data.update_index = 0

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def export(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "early_stopped": self.early_stopped,
            "finished_epoch": self.finished_epoch,
            "history": list(self.history),
            "learning_rates": list(self.learning_rates),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "scheduled_rates": list(self.scheduled_rates),
            "stop_reason": self.stop_reason,
            "validation_profile": self.validation_profile,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Replace the counters after the payload has been checked. ``losses`` stays."""

        decoded = _decode_progress(
            state,
            optimizer_id=self._optimizer_id,
            validation_profile=self.validation_profile,
        )
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.history = decoded.history
        self.learning_rates = decoded.learning_rates
        self.scheduled_rates = decoded.scheduled_rates
        self.completed_boundaries = decoded.completed_boundaries
        self.boundary = decoded.boundary
        self.finished_epoch = decoded.finished_epoch
        self.early_stopped = decoded.early_stopped
        if decoded.stop_reason is not None:
            self.stop_reason = decoded.stop_reason

    def _open_boundary(self) -> dict[str, Any]:
        boundary = self.boundary
        if boundary is None or self.finished_epoch is None:
            raise RuntimeFailure("there is no pending epoch boundary")
        return boundary

# --- Constructor checks. They do not train or save. ---

def require_locoformer_setup(
    model: PreparedModel,
    objective: torch.nn.Module,
    data: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupReduceLROnPlateau,
    validation_profile: str,
) -> tuple[torch.nn.Module, str]:
    """Refuse a loop that cannot run the DNS warmup schedule. Return the objective and optimizer id."""

    require_prepared(model)
    objective = coerce_spectral_objective(objective, label="DNS")
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    if not isinstance(scheduler, WarmupReduceLROnPlateau):
        raise BuildError("scheduler must be WarmupReduceLROnPlateau")
    if scheduler.mode != "min":
        raise BuildError("DNS scheduler mode must be min")
    if not isinstance(validation_profile, str) or not validation_profile:
        raise BuildError("validation_profile must be a non-empty string")
    optimizer_id = getattr(optimizer, "_experiment_component_id", None)
    if not isinstance(optimizer_id, str):
        raise BuildError("optimizer has no component id")
    return objective, optimizer_id


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def optional_patience(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError("early_stopping_patience must be an integer >= 0")
    return value


def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)


def require_group_size(data: Any, accumulation: int, component_id: str) -> None:
    group_size = getattr(data, "group_size", None)
    if group_size is None:
        return
    if group_size != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} does not match accumulation {accumulation}"
        )

# --- Deferred step log. A continuing step is offered here, before the next forward. ---

class DeferredStepLog:
    """At most one unflushed step row."""

    def __init__(self) -> None:
        self._item: dict[str, Any] | None = None

    def queue(self, record: dict[str, Any]) -> None:
        self._item = record

    def drain(
        self,
        journal: StepJournal,
        lifecycle: Any,
        optimizer_updates: Mapping[str, int],
        step_num: int,
    ) -> None:
        item = self._item
        if item is None:
            return
        self._item = None
        journal.write_metrics(
            {
                "applied": item["applied"],
                "epoch": item["epoch"],
                "logical_step": item["logical_step"],
                "loss": item["loss"],
                "lr": item["used_lr"],
                "optimizer_updates": dict(optimizer_updates),
                "sample_id": item["sample_id"],
                "scheduled_lr": item["scheduled_lr"],
                "si_sdr": item["si_sdr"],
                "step_num": int(step_num),
                "terms": item["terms"],
            }
        )
        journal.write_sample(
            {
                "epoch": item["epoch"],
                "logical_step": item["logical_step"],
                "loss": item["loss"],
                "lr": item["used_lr"],
                "sample_id": item["sample_id"],
                "scheduled_lr": item["scheduled_lr"],
                "si_sdr": item["si_sdr"],
            }
        )
        if not item["offer_step"]:
            return
        step_metrics = item["step_metrics"]
        logical_step = item["logical_step"]
        if step_metrics:
            lifecycle.save_if_needed("step", logical_step=logical_step, metrics=step_metrics)
        else:
            lifecycle.save_if_needed("step", logical_step=logical_step)

# --- Grouped accumulation. One AdamW step, then the trainer calls step_batch. ---

@dataclass(frozen=True)
class LocoformerStep:
    """Scalars from one finished accumulation window, before ``step_batch``."""

    loss: float
    applied: bool
    used_lr: float
    si_sdr: float
    sample_id: str
    terms: dict[str, float]


class LocoformerUpdate:
    """Forward, backward, clip, and one AdamW step for a grouped window."""

    def __init__(
        self,
        model: PreparedModel,
        objective: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        accumulation: int,
        grad_clip: float,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.accumulation = accumulation
        self.grad_clip = grad_clip

    def run(
        self,
        planned: list[list[Batch]],
        runtime: ProcessRuntime,
        component_id: str,
        *,
        apply_optimizer: bool = True,
    ) -> LocoformerStep:
        """Run one window. The learning rate is the one used for this step."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total: torch.Tensor | None = None
        sample_ids: list[str] = []
        monitor_sum: torch.Tensor | None = None
        monitor_count = 0
        term_acc: dict[str, float] = {}
        counts = [micro_rows(batches) for batches in planned]
        scale = window_scale(
            counts,
            gather_row_counts(runtime, counts),
            accumulation=self.accumulation,
            world_size=int(runtime.world_size),
        )
        for offset, batches in enumerate(planned):
            sync = offset == len(planned) - 1
            sample_ids.extend(batch.sample_id for batch in batches)
            with runtime.sync_gradients([self.model], enabled=sync):
                with runtime.autocast():
                    if counts[offset] == 0:
                        micro = connected_zero(self.model.parameters, component_id)
                        monitor = None
                        count = 0
                        terms = {}
                    else:
                        micro, monitor, count, terms = _forward_loss(
                            self.model, self.objective, batches, component_id, runtime
                        )
                micro = weight_loss(micro, scale, offset)
                runtime.backward(micro)
            detached = micro.detach()
            total = detached if total is None else total + detached
            if monitor is not None and count:
                monitor_sum = monitor if monitor_sum is None else monitor_sum + monitor
                monitor_count += count
            logged = scale if isinstance(scale, int) else scale[offset]
            for name, value in terms.items():
                term_acc[name] = term_acc.get(name, 0.0) + (
                    value / logged if isinstance(scale, int) else value * logged
                )
        if total is None:
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        if monitor_sum is None or monitor_count < 1:
            if sum(counts) != 0:
                raise RuntimeFailure(f"component {component_id}: SI-SDR monitor has no batches")
            monitor_sum = total.detach() * 0
            monitor_count = 1
        if not apply_optimizer:
            used_lr = float(self.optimizer.param_groups[0]["lr"])
            loss_value = runtime.read_host_scalar(total)
            monitor_value = runtime.read_host_scalar(monitor_sum / monitor_count)
            if not math.isfinite(loss_value):
                raise RuntimeFailure(f"component {component_id}: loss is not finite")
            return LocoformerStep(
                loss=loss_value,
                applied=False,
                used_lr=used_lr,
                si_sdr=monitor_value,
                sample_id=",".join(sample_ids),
                terms=term_acc,
            )
        if self.grad_clip is not None:
            runtime.clip(self.model, self.grad_clip, optimizer=self.optimizer)
        used_lr = float(self.optimizer.param_groups[0]["lr"])
        host_loss = runtime.host_scalar(total)
        host_monitor = runtime.host_scalar(monitor_sum / monitor_count)
        result = runtime.step(self.optimizer)
        runtime.end_step()
        loss_value = float(host_loss)
        if not math.isfinite(loss_value):
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        return LocoformerStep(
            loss=loss_value,
            applied=bool(result.applied),
            used_lr=used_lr,
            si_sdr=float(host_monitor),
            sample_id=",".join(sample_ids),
            terms=term_acc,
        )


def _forward_loss(
    model: PreparedModel,
    objective: torch.nn.Module,
    batches: list[Batch],
    component_id: str,
    runtime: ProcessRuntime,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float]]:
    """Average the injected objective and keep SI-SDR as a detached monitor."""

    if not batches:
        raise RuntimeFailure(f"component {component_id}: empty micro-batch")
    total: torch.Tensor | None = None
    monitor: torch.Tensor | None = None
    term_sums: dict[str, float] = {}
    for batch in batches:
        batch = runtime.move(batch)
        try:
            prediction = model(batch.inputs)
            result = call_objective(objective, _objective_context(prediction, batch, component_id))
        except ValueError as exc:
            raise RuntimeFailure(str(exc)) from exc
        value = result.total
        if not isinstance(value, torch.Tensor):
            raise RuntimeFailure(f"component {component_id}: loss did not return a tensor")
        for name, amount in term_floats(result).items():
            term_sums[name] = term_sums.get(name, 0.0) + amount
        total = value if total is None else total + value
        with torch.no_grad():
            term = si_sdr_loss(prediction, batch.targets).detach()
            monitor = term if monitor is None else monitor + term
    assert total is not None and monitor is not None
    count = len(batches)
    return total / count, monitor, count, {name: amount / count for name, amount in term_sums.items()}


def _objective_context(prediction: object, batch: Batch, component_id: str) -> dict[str, object]:
    """Waveform models keep ``estimate``. A named mapping may add one feature."""

    if isinstance(prediction, torch.Tensor):
        return {"estimate": prediction, "target": batch.targets}
    if not isinstance(prediction, Mapping) or "enhanced" not in prediction:
        raise RuntimeFailure(
            f"component {component_id}: model must return a waveform or a mapping with enhanced"
        )
    context: dict[str, object] = {"estimate": prediction["enhanced"], "target": batch.targets}
    for key, value in prediction.items():
        if key == "enhanced":
            continue
        if key in context:
            raise RuntimeFailure(
                f"component {component_id}: model output {key!r} collides with a batch field"
            )
        context[str(key)] = value
    return context


# --- Epoch boundary: validation, one step_validation, n-best, early stop. ---

@dataclass(frozen=True)
class BoundaryOutcome:
    trainer_epoch: int
    loss: float
    si_snr: float


class LocoformerBoundary:
    """One pending epoch. The trainer supplies the plateau step and ``_select``."""

    def __init__(self, validator: LocoformerValidator, *, early_stopping_patience: int | None) -> None:
        self.validator = validator
        self.early_stopping_patience = early_stopping_patience

    def run(
        self,
        progress: LocoformerProgress,
        runtime: ProcessRuntime,
        data: Any,
        output_directory: Path,
        component_id: str,
        on_phase: Callable[[str], None],
        step_validation: Callable[[float], float],
        select: Callable[..., None],
    ) -> BoundaryOutcome:
        if progress.phase != PENDING_VALIDATION or progress.finished_epoch is None or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        trainer_epoch = int(progress.finished_epoch)
        espnet_epoch = trainer_epoch + 1
        summary = self.validator.evaluate(data, runtime, trainer_epoch, component_id)
        loss = float(summary["loss"])
        si_snr = float(summary["si_snr"])
        progress.note_validation(loss, si_snr, int(summary["weight"]))
        progress.phase = "pending_scheduler"
        on_phase(progress.phase)
        learning_rate = step_validation(loss)
        progress.note_scheduler(learning_rate)
        progress.phase = "pending_selection"
        on_phase(progress.phase)
        select(runtime, output_directory, espnet_epoch, loss, si_snr)
        boundary = progress.boundary
        if boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        boundary["selection_done"] = True
        progress.close_boundary()
        progress.roll_epoch(data, trainer_epoch)
        on_phase(progress.phase)
        if self.early_stopping_patience is not None and early_stop(
            progress.history, espnet_epoch, self.early_stopping_patience
        ):
            progress.early_stopped = True
        return BoundaryOutcome(trainer_epoch=trainer_epoch, loss=loss, si_snr=si_snr)

# --- n-best. ``_select`` calls this so a test can replace the whole selection. ---

def select_checkpoints(
    progress: LocoformerProgress,
    runtime: ProcessRuntime,
    directory: Path,
    model: Any,
    espnet_epoch: int,
    loss: float,
    si_snr: float,
    keep_nbest: int,
    validation_profile: str,
) -> None:
    """Rank both criteria and write the official selection files."""

    progress.append_history({"epoch": int(espnet_epoch), "loss": float(loss), "si_snr": float(si_snr)})
    by_loss = rank_validation_loss(progress.history, keep_nbest)
    by_si = rank_validation_si_snr(progress.history, keep_nbest)
    error: str | None = None
    if _rank(runtime) == 0:
        try:
            write_selection(
                directory,
                model.unwrapped().export_for_inference(),
                espnet_epoch,
                by_loss,
                by_si,
                keep_nbest,
                progress.logical_step,
                validation_profile,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = runtime.gather_objects(error)
    failure = next((item for item in errors if item), None)
    if failure:
        raise RuntimeFailure(f"n-best selection failed: {failure}")
    runtime.barrier()
