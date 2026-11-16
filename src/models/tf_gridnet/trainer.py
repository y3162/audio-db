"""TF-GridNet MS-SNSD training loop.

Read this file from ``TFGridNetTrainer.fit`` downward.

``fit`` orders start, a resumed pending boundary, one grouped accumulation
update, the deferred step log, then either a later step save or a pending
save, validation, ``ReduceLROnPlateau``, n-best, research, and the boundary
save. Counters live on ``GridNetProgress`` in this file. ``valid.si_snr`` is
not a selection criterion. ``amp`` may skip the optimizer; the logical step
still advances.

``GridNetUpdate.run`` is the accumulation order: each micro-batch averages the
injected spectral objective, the last micro-batch synchronizes, then clip and
one Adam step. ``GridNetBoundary.run`` is validation, the plateau scheduler,
n-best, then early stop. The step save is offered from ``DeferredStepLog.drain``
only for a continuing step, and ``fit`` calls that drain before the next
forward and again before ``boundary_pending``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.models.tf_gridnet.evaluator import (
    GridNetValidator,
    early_stop,
    rank_validation_loss,
    write_nbest,
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

class TFGridNetTrainer:
    """Orders one update, the epoch boundary, and checkpoint safe points."""

    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        model: PreparedModel,
        objective: torch.nn.Module,
        data: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        *,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float = 5.0,
        keep_nbest: int = 5,
        max_epochs: int | None = None,
        early_stopping_patience: int | None = None,
        stop_after_logical_steps: int | None = None,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        objective, optimizer_id = require_gridnet_setup(model, objective, data, optimizer, scheduler)
        self.model, self.objective, self.data = model, objective, data
        self.optimizer, self.scheduler = optimizer, scheduler
        self.audio_sample_rate, self.audio_teachers = 16000, ("dry_clean",)
        self.max_logical_steps = positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = positive_int("accumulation", accumulation)
        self.grad_clip = positive_float("grad_clip", grad_clip)
        self.keep_nbest = positive_int("keep_nbest", keep_nbest)
        self.max_epochs = None if max_epochs is None else positive_int("max_epochs", max_epochs)
        if early_stopping_patience is not None and (
            isinstance(early_stopping_patience, bool)
            or not isinstance(early_stopping_patience, int)
            or early_stopping_patience < 0
        ):
            raise BuildError("early_stopping_patience must be an integer >= 0")
        self.early_stopping_patience = early_stopping_patience
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self.progress = GridNetProgress.initial(optimizer_id)
        self._log = DeferredStepLog()
        self._restored = False
        self._update = GridNetUpdate(
            model, objective, optimizer, accumulation=self.accumulation, grad_clip=self.grad_clip,
        )
        self._boundary = GridNetBoundary(
            GridNetValidator(model, objective),
            scheduler,
            optimizer,
            model,
            keep_nbest=self.keep_nbest,
            early_stopping_patience=early_stopping_patience,
        )
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators, model_selection, data=data, module=model, predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection

    @property
    def logical_step(self) -> int:
        """Live step counter. ``GridNetProgress`` owns the value."""

        return self.progress.logical_step

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

    @property
    def phase(self) -> str:
        """Live phase. ``GridNetProgress`` owns the value."""

        return self.progress.phase

    @property
    def boundary(self) -> dict[str, Any] | None:
        """Open boundary, or None. The progress object owns the dict."""

        return self.progress.boundary

    @property
    def losses(self) -> list[float]:
        """Journal-backed loss list. It is not stored in the checkpoint."""

        return self.progress.losses

    @losses.setter
    def losses(self, value: list[float]) -> None:
        self.progress.losses = value

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or early stopping."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        progress = self.progress
        if should_stop(self):
            return self._finish("completed")
        if self._restored and len(progress.learning_rates) != progress.logical_step:
            raise RuntimeFailure(
                f"component {component_id}: learning-rate history does not match logical step"
            )
        if progress.pending:
            self._finish_pending(runtime, component_id)
            if should_stop(self) or progress.early_stopped or self._epoch_capped():
                return self._finish("completed")
        while progress.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                self._log.drain(self.journal, self.lifecycle, progress.optimizer_updates)
                if progress.logical_step > 0:
                    self.lifecycle.save_if_needed("stop", logical_step=progress.logical_step)
                return self._finish("interrupted")
            if self._epoch_capped():
                self._log.drain(self.journal, self.lifecycle, progress.optimizer_updates)
                return self._finish("completed")
            epoch, index, view, planned = open_grouped_window(self, component_id)
            try:
                self._log.drain(self.journal, self.lifecycle, progress.optimizer_updates)
                local = self._update.run(planned, runtime, component_id)
                progress.record_step(loss=local.loss, applied=local.applied, learning_rate=local.learning_rate)
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
                        "learning_rate": local.learning_rate,
                        "logical_step": progress.logical_step,
                        "loss": local.loss,
                        "offer_step": not epoch_done,
                        "sample_id": local.sample_id,
                        "step_metrics": step_metrics,
                        "terms": local.terms,
                    }
                )
                if not epoch_done:
                    continue
                self._log.drain(self.journal, self.lifecycle, progress.optimizer_updates)
                progress.open_pending(epoch)
                self._on_boundary_phase(progress.phase)
                self.lifecycle.save_if_needed("boundary_pending", logical_step=progress.logical_step)
                if self._stop_requested(runtime):
                    return self._finish("interrupted")
                self._finish_pending(runtime, component_id)
                if should_stop(self) or progress.early_stopped or self._epoch_capped():
                    return self._finish("completed")
            finally:
                _close_epoch_view(view)
        self._log.drain(self.journal, self.lifecycle, progress.optimizer_updates)
        return self._finish("completed")

    def _forward_loss(
        self,
        batches: list[Any],
        component_id: str,
        runtime: ProcessRuntime,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return _micro_loss(self.model, self.objective, batches, component_id, runtime)

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
        )
        official = {"loss": done.loss}
        observe_validation(
            getattr(self, "training_control", None),
            official,
            step=self.progress.logical_step,
            scoped=self.research.attached,
        )
        extra = self._research(runtime, "boundary", done.trainer_epoch, official=official)
        metrics = {**official, **extra} if extra else official
        self.lifecycle.save_if_needed("boundary", logical_step=self.progress.logical_step, metrics=metrics)

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
        """Hook for tests that record validation, scheduler, and selection order."""

        del phase

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
        """Agree a logical-step cap with a rank interrupt."""

        return stop_after_step(self.progress.logical_step, self.stop_after_logical_steps, runtime)

    def _epoch_capped(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

# --- Checkpoint progress. This object is the only live counter owner. ---

IN_EPOCH = "in_epoch"
PENDING_VALIDATION = "pending_validation"
_PHASES = frozenset({IN_EPOCH, PENDING_VALIDATION})


def open_boundary(trainer_epoch: int) -> dict[str, Any]:
    """Record that validation, the scheduler, and n-best selection have not run."""

    return {
        "espnet_epoch": trainer_epoch + 1,
        "learning_rate": None,
        "optimizer_updated": True,
        "scheduler_updated": False,
        "selection_done": False,
        "trainer_epoch": trainer_epoch,
        "validation_done": False,
        "validation_loss": None,
        "validation_weight": None,
    }


@dataclass(frozen=True)
class _DecodedProgress:
    """Temporary decode of one payload. It is not stored on the live progress."""

    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[Any]
    learning_rates: list[float]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str | None


def _decode_progress(
    state: dict[str, Any],
    *,
    optimizer_id: str,
) -> _DecodedProgress:
    """Validate and copy a payload. This does not touch the live progress."""

    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    history = state.get("history")
    rates = state.get("learning_rates")
    completed = state.get("completed_boundaries")
    if not isinstance(history, list) or not isinstance(rates, list) or not isinstance(completed, list):
        raise RuntimeFailure("trainer history is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    copied_rates = [float(value) for value in rates]
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
        completed_boundaries=list(completed),
        boundary=copied_boundary,
        finished_epoch=finished,
        early_stopped=bool(state.get("early_stopped", False)),
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )


@dataclass
class GridNetProgress:
    """Live TF-GridNet counters. Checkpoint code sees them only through ``export``."""

    logical_step: int
    optimizer_updates: dict[str, int]
    learning_rates: list[float]
    history: list[Any]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    phase: str
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str
    losses: list[float]
    _optimizer_id: str = field(repr=False)

    @classmethod
    def initial(cls, optimizer_id: str) -> GridNetProgress:
        """Zero counters for a trainer that has not restored a checkpoint."""

        return cls(
            logical_step=0,
            optimizer_updates={optimizer_id: 0},
            learning_rates=[],
            history=[],
            completed_boundaries=[],
            boundary=None,
            phase=IN_EPOCH,
            finished_epoch=None,
            early_stopped=False,
            stop_reason="initialized",
            losses=[],
            _optimizer_id=optimizer_id,
        )

    @property
    def pending(self) -> bool:
        """True after a pending save and before that boundary has been closed."""

        return self.phase == PENDING_VALIDATION

    def reset_fresh(self) -> None:
        """Zero a run that is not resuming. ``stop_reason`` stays as it is."""

        self.logical_step = 0
        self.optimizer_updates = {self._optimizer_id: 0}
        self.losses = []
        self.learning_rates = []
        self.history = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None
        self.early_stopped = False

    def record_step(self, *, loss: float, applied: bool, learning_rate: float) -> None:
        """Count one logical step. Only an applied optimizer step increments its id."""

        self.logical_step += 1
        if applied:
            self.optimizer_updates[self._optimizer_id] += 1
        self.losses.append(loss)
        self.learning_rates.append(learning_rate)

    def place_cursor(self, data: Any, epoch: int, index: int) -> None:
        """Write the data cursor after a finished update. Data owns the fields."""

        data.epoch = int(epoch)
        data.update_index = int(index)

    def reset_cursor(self, data: Any) -> None:
        """Put a fresh run at the start of epoch 0."""

        data.epoch = 0
        data.update_index = 0

    def open_pending(self, trainer_epoch: int) -> None:
        """Open the boundary that a pending checkpoint is about to store."""

        self.phase = PENDING_VALIDATION
        self.finished_epoch = int(trainer_epoch)
        self.boundary = open_boundary(trainer_epoch)

    def note_validation(self, loss: float, weight: int) -> None:
        """Record the validation totals on the open boundary."""

        boundary = self._open_boundary()
        boundary["validation_done"] = True
        boundary["validation_loss"] = loss
        boundary["validation_weight"] = weight

    def note_scheduler(self, learning_rate: float) -> None:
        """Record that ReduceLROnPlateau has stepped for this boundary."""

        boundary = self._open_boundary()
        boundary["scheduler_updated"] = True
        boundary["learning_rate"] = float(learning_rate)

    def append_history(self, row: dict[str, Any]) -> None:
        self.history.append(row)

    def close_boundary(self) -> None:
        """Append the open boundary and return to the epoch."""

        boundary = self._open_boundary()
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None

    def roll_epoch(self, data: Any, trainer_epoch: int) -> None:
        """Advance past an epoch that the closed boundary has finished."""

        data.epoch = int(trainer_epoch) + 1
        data.update_index = 0

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def export(self) -> dict[str, Any]:
        """Copy the counters into the existing ``states["trainer"]`` dict."""

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
            "stop_reason": self.stop_reason,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Replace the counters from a stored trainer dict.

        Decoding finishes before any checkpoint field is written. ``losses``
        is not in the payload and is left untouched.
        """

        decoded = _decode_progress(state, optimizer_id=self._optimizer_id)
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.history = decoded.history
        self.learning_rates = decoded.learning_rates
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

def require_gridnet_setup(
    model: PreparedModel,
    objective: torch.nn.Module,
    data: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
) -> tuple[torch.nn.Module, str]:
    """Refuse a loop that cannot run the MS-SNSD boundary. Return the objective and optimizer id."""

    require_prepared(model)
    objective = coerce_spectral_objective(objective, label="MS-SNSD")
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        raise BuildError("scheduler must be ReduceLROnPlateau")
    if scheduler.mode != "min":
        raise BuildError("MS-SNSD scheduler mode must be min")
    optimizer_id = getattr(optimizer, "_experiment_component_id", None)
    if not isinstance(optimizer_id, str):
        raise BuildError("optimizer has no component id")
    return objective, optimizer_id


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
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

    def drain(self, journal: StepJournal, lifecycle: Any, optimizer_updates: Mapping[str, int]) -> None:
        """Write the held row and offer its step save when the trainer asked."""

        item = self._item
        if item is None:
            return
        self._item = None
        loss_value = item["loss"]
        applied = item["applied"]
        sample_id = item["sample_id"]
        learning_rate = item["learning_rate"]
        epoch = item["epoch"]
        logical_step = item["logical_step"]
        offer_step = item["offer_step"]
        terms = item["terms"]
        step_metrics = item["step_metrics"]
        journal.write_metrics(
            {
                "applied": applied,
                "epoch": epoch,
                "logical_step": logical_step,
                "loss": loss_value,
                "lr": learning_rate,
                "optimizer_updates": dict(optimizer_updates),
                "sample_id": sample_id,
                "terms": terms,
            }
        )
        journal.write_sample(
            {
                "epoch": epoch,
                "logical_step": logical_step,
                "loss": loss_value,
                "lr": learning_rate,
                "sample_id": sample_id,
            }
        )
        if not offer_step:
            return
        if step_metrics:
            lifecycle.save_if_needed("step", logical_step=logical_step, metrics=step_metrics)
        else:
            lifecycle.save_if_needed("step", logical_step=logical_step)

# --- Grouped accumulation. One Adam step after the last synchronized micro-batch. ---

@dataclass(frozen=True)
class GridNetStep:
    """Scalars from one finished accumulation window."""

    loss: float
    applied: bool
    learning_rate: float
    sample_id: str
    terms: dict[str, float]


class GridNetUpdate:
    """Forward, backward, clip, and one Adam step for a grouped window."""

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
    ) -> GridNetStep:
        """Run one window. The last micro-batch is the one that synchronizes."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total: torch.Tensor | None = None
        sample_ids: list[str] = []
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
                        terms = {}
                    else:
                        micro, terms = _forward_loss(self.model, self.objective, batches, component_id, runtime)
                micro = weight_loss(micro, scale, offset)
                runtime.backward(micro)
            detached = micro.detach()
            total = detached if total is None else total + detached
            logged = scale if isinstance(scale, int) else scale[offset]
            for name, value in terms.items():
                term_acc[name] = term_acc.get(name, 0.0) + (
                    value / logged if isinstance(scale, int) else value * logged
                )
        if total is None:
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        if not apply_optimizer:
            loss_value = runtime.read_host_scalar(total)
            if not math.isfinite(loss_value):
                raise RuntimeFailure(f"component {component_id}: loss is not finite")
            return GridNetStep(
                loss=loss_value,
                applied=False,
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                sample_id=",".join(sample_ids),
                terms=term_acc,
            )
        if self.grad_clip is not None:
            runtime.clip(self.model, self.grad_clip, optimizer=self.optimizer)
        host_loss = runtime.host_scalar(total)
        result = runtime.step(self.optimizer)
        runtime.end_step()
        loss_value = float(host_loss)
        if not math.isfinite(loss_value):
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        return GridNetStep(
            loss=loss_value,
            applied=bool(result.applied),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            sample_id=",".join(sample_ids),
            terms=term_acc,
        )


def _forward_loss(
    model: PreparedModel,
    objective: torch.nn.Module,
    batches: list[Batch],
    component_id: str,
    runtime: ProcessRuntime,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average the injected spectral objective over one micro-batch group."""

    if not batches:
        raise RuntimeFailure(f"component {component_id}: empty micro-batch")
    total: torch.Tensor | None = None
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
    assert total is not None
    count = len(batches)
    return total / count, {name: amount / count for name, amount in term_sums.items()}

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


_micro_loss = _forward_loss

# --- Epoch boundary: validation, ReduceLROnPlateau, n-best, early stop. ---

@dataclass(frozen=True)
class BoundaryOutcome:
    """Validation loss after the boundary has closed and the cursor has moved."""

    trainer_epoch: int
    loss: float


class GridNetBoundary:
    """One pending epoch: validation, scheduler, n-best, then early stop."""

    def __init__(
        self,
        validator: GridNetValidator,
        scheduler: Any,
        optimizer: Any,
        model: Any,
        *,
        keep_nbest: int,
        early_stopping_patience: int | None,
    ) -> None:
        self.validator = validator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.keep_nbest = keep_nbest
        self.early_stopping_patience = early_stopping_patience

    def run(
        self,
        progress: GridNetProgress,
        runtime: ProcessRuntime,
        data: Any,
        output_directory: Path,
        component_id: str,
        on_phase: Callable[[str], None],
    ) -> BoundaryOutcome:
        """Close one pending epoch. ``on_phase`` sees each intermediate phase."""

        if progress.phase != PENDING_VALIDATION or progress.finished_epoch is None or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        trainer_epoch = int(progress.finished_epoch)
        espnet_epoch = trainer_epoch + 1
        summary = self.validator.evaluate(data, runtime, trainer_epoch, component_id)
        progress.note_validation(float(summary["loss"]), int(summary["weight"]))
        progress.phase = "pending_scheduler"
        on_phase(progress.phase)
        self.scheduler.step(summary["loss"])
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        progress.note_scheduler(learning_rate)
        progress.phase = "pending_selection"
        on_phase(progress.phase)
        _select(
            progress,
            runtime,
            output_directory,
            self.model,
            espnet_epoch,
            float(summary["loss"]),
            self.keep_nbest,
        )
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
        return BoundaryOutcome(trainer_epoch=trainer_epoch, loss=float(summary["loss"]))


def _select(
    progress: GridNetProgress,
    runtime: ProcessRuntime,
    directory: Path,
    model: Any,
    espnet_epoch: int,
    loss: float,
    keep_nbest: int,
) -> None:
    """Rank valid.loss and write the official n-best files."""

    progress.append_history({"epoch": int(espnet_epoch), "loss": float(loss)})
    ranked = rank_validation_loss(progress.history, keep_nbest)
    error: str | None = None
    if _rank(runtime) == 0:
        try:
            write_nbest(
                directory,
                model.unwrapped().export_for_inference(),
                espnet_epoch,
                ranked,
                keep_nbest,
                progress.logical_step,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = runtime.gather_objects(error)
    failure = next((item for item in errors if item), None)
    if failure:
        raise RuntimeFailure(f"n-best selection failed: {failure}")
    runtime.barrier()
