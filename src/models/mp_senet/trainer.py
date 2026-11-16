"""Official MP-SENet GAN loop.

Read this file from ``MPSENetGANTrainer.fit`` downward.

``fit`` orders start, a resumed pending boundary, one update, the step log,
then either a step save or a pending save, validation, best, schedulers,
research, and the boundary save. Counters live on ``MPTrainingProgress`` in
this file. ``OfficialBoundary.run`` is validation, best, then the schedulers.
A step save is offered after both optimizers, before the next forward. A CUDA
log flush is not a save. ``fp16`` is refused.

The update order is ``MPSENetUpdate.run`` in this file. CPU, more than one
rank, and CUDA with accumulation above 1 use ``accumulated_step``: discriminator
micro-batches, one discriminator step, then the generator on the stored graph.
CUDA with one rank and one micro-batch uses ``cuda_overlapped_step``: signal
backward, previous-step log flush, PESQ, discriminator step, metric backward,
generator step. Neither path offers a checkpoint.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.data.pair import bind_declared_audio
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.models.mp_senet.dataset import require_discriminator_resolution
from src.models.mp_senet.evaluator import (
    MPSENetOfficialValidator,
    apply_pesq_selection,
    publish_best,
)
from src.models.mp_senet.model import (
    MPSENetGANGenerator,
    MPSENetMetricDiscriminator,
    pack_forward,
    require_official_architecture,
)
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.selectors import scalar_metrics
from src.experiments.stopping import control_requests_stop, observe_validation
from src.experiments.training.control import (
    close_training_run,
    finite_gradient_report,
    connected_zero,
    gather_row_counts,
    micro_rows,
    open_flat_window_at,
    restore_fields,
    snapshot_fields,
    stop_after_step,
    weight_loss,
    window_scale,
)
from src.experiments.training.objectives import call_objective
from src.experiments.training.official_objectives import require_mp_senet
from src.experiments.trainers import _close_epoch_view

# --- Trainer. ``fit`` is the timeline. The sections below are the steps it calls. ---

OFFICIAL_VALIDATION_INTERVAL = 5000
LEAKAGE_SAFE_VALIDATION_INTERVAL = 2675
OFFICIAL_CHECKPOINT_INTERVAL = 5000
LEAKAGE_SAFE_CHECKPOINT_INTERVAL = 2675
BEST_CHECKPOINT_START_EPOCH = 40
OFFICIAL_LEARNING_RATE = 0.0005
OFFICIAL_BETAS = (0.8, 0.99)
OFFICIAL_WEIGHT_DECAY = 0.01
OFFICIAL_GAMMA = 0.99


class MPSENetGANTrainer:
    """Orders updates, official validation, and checkpoint safe points."""

    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        generator: PreparedModel,
        discriminator: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
        scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
        *,
        signal_objective: torch.nn.Module,
        metric_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        max_logical_steps: int,
        experiment_claim: str,
        accumulation: int = 1,
        validation_interval: int = OFFICIAL_VALIDATION_INTERVAL,
        best_checkpoint_start_epoch: int = BEST_CHECKPOINT_START_EPOCH,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        optim_g_id, optim_d_id = require_mp_gan_setup(
            generator, discriminator, data, optim_g, optim_d, scheduler_g, scheduler_d,
            signal_objective, metric_objective, discriminator_objective, experiment_claim,
        )
        self.generator, self.discriminator, self.data = generator, discriminator, data
        self.optim_g, self.optim_d = optim_g, optim_d
        self.experiment_claim = experiment_claim
        self.audio_sample_rate = 16000
        self.audio_teachers = ("dry_clean",)
        self.max_logical_steps = positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = positive_int("accumulation", accumulation)
        self.validation_interval = non_negative_int("validation_interval", validation_interval)
        self.best_checkpoint_start_epoch = non_negative_int(
            "best_checkpoint_start_epoch", best_checkpoint_start_epoch
        )
        self.max_epochs = None if max_epochs is None else positive_int("max_epochs", max_epochs)
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self.progress = MPTrainingProgress.initial(optim_g_id, optim_d_id)
        self._recorder = StepRecorder()
        self._restored = False
        self._in_official_step = False
        self._update = MPSENetUpdate(
            generator=generator, discriminator=discriminator, optim_g=optim_g, optim_d=optim_d,
            signal_objective=signal_objective, metric_objective=metric_objective,
            discriminator_objective=discriminator_objective, accumulation=self.accumulation,
        )
        self._validator = MPSENetOfficialValidator(
            generator=generator, discriminator=discriminator, data=data,
            optim_g=optim_g, scheduler_g=scheduler_g,
        )
        self._boundary = OfficialBoundary(
            self._validator, scheduler_g, scheduler_d, optim_g, optim_d, generator, experiment_claim,
        )
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators, model_selection, data=data, module=generator,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or an interrupt.

        A continuing step is logged and may be saved. A validation or epoch
        edge saves ``boundary_pending`` first, then validation, best, the
        epoch-end schedulers, research, and the boundary checkpoint.
        """

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        progress = self.progress
        if self._control_stopped():
            return self._finish("completed")
        if progress.pending:
            self._finish_pending(runtime, component_id)
            if self._control_stopped() or self._epoch_capped():
                return self._finish("completed")
        while progress.logical_step < self.max_logical_steps:
            if self._stop_between_steps(runtime):
                return self._finish("interrupted")
            if self._epoch_capped():
                return self._finish("completed")
            epoch, index, view, batches = open_flat_window_at(self.data, self.accumulation, component_id)
            try:
                local = self.official_step(batches, runtime)
                progress.complete_update()
                index += len(batches)
                progress.place_cursor(self.data, epoch, index)
                epoch_done = index == view.updates
                validate = self.validation_interval > 0 and progress.logical_step % self.validation_interval == 0
                if self._record_step(
                    runtime, local, epoch=epoch, epoch_done=epoch_done, will_validate=validate,
                ):
                    continue
                progress.open_pending(epoch, validate=validate, epoch_done=epoch_done)
                self._on_boundary_phase(progress.phase)
                self._offer("boundary_pending")
                if stop_after_step(progress.logical_step, self.stop_after_logical_steps, runtime):
                    return self._finish("interrupted")
                self._finish_pending(runtime, component_id)
                if self._control_stopped() or self._epoch_capped():
                    return self._finish("completed")
            finally:
                _close_epoch_view(view)
        return self._finish("completed")

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        progress = self.progress
        self.research.rewind(
            self.lifecycle.output_directory, restored=self._restored, phase=progress.phase,
            progress={"logical_step": progress.logical_step},
        )
        refuse_gan_amp(runtime)
        component_id = str(getattr(self, "_experiment_component_id", "trainer"))
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.journal.begin(
            restored=self._restored, logical_step=progress.logical_step,
            expect_rows=len(progress.step_log) if self._restored else None,
        )
        if not self._restored:
            progress.reset_fresh()
            progress.reset_cursor(self.data)
        bind_declared_audio(
            self, data=self.data, sample_rate=self.audio_sample_rate,
            teachers=self.audio_teachers, experiment_claim=self.experiment_claim,
        )
        require_group_size(self.data, self.accumulation, component_id)
        runtime.barrier()
        return component_id

    def _record_step(
        self, runtime: ProcessRuntime, local: dict[str, Any], *, epoch: int, epoch_done: bool, will_validate: bool,
    ) -> bool:
        """Log the finished step. Offer ``step`` only when the epoch continues."""

        progress = self.progress
        entry = self._recorder.entry(
            self._recorder.share(runtime, local), epoch=epoch, logical_step=progress.logical_step,
        )
        progress.append_step(entry)
        continuing = not will_validate and not epoch_done
        defer = continuing and runtime.device.type == "cuda" and runtime.world_size == 1 and self.accumulation == 1
        self._recorder.publish(self.journal, entry, local, defer=defer)
        if not continuing:
            return False
        self._offer("step", metrics=self._research(runtime, "step", epoch) or None)
        return True

    def _finish_pending(self, runtime: ProcessRuntime, component_id: str) -> None:
        done = self._boundary.run(
            self.progress, runtime, self.lifecycle.output_directory, component_id,
            start_epoch=self.best_checkpoint_start_epoch, on_phase=self._on_boundary_phase,
        )
        if done.roll_epoch:
            self.progress.roll_epoch(self.data, done.trainer_epoch)
        if done.validate:
            observe_validation(
                getattr(self, "training_control", None), done.metrics,
                step=self.progress.logical_step, scoped=self.research.attached,
            )
        official = done.metrics if done.validate else {}
        extra = self._research(runtime, "boundary", done.trainer_epoch, official=official)
        self._offer("boundary", metrics={**done.metrics, **extra} if extra else done.metrics)

    def preflight_probe(self) -> dict[str, Any]:
        """One real window of both backwards. Neither optimizer steps."""

        runtime = getattr(self, "runtime", None)
        if not isinstance(runtime, ProcessRuntime):
            raise BuildError("preflight has no runtime")
        component_id = str(getattr(self, "_experiment_component_id", "trainer"))
        saved = snapshot_fields(self._update)
        was_training = bool(self.generator.training)
        _epoch, _index, view, batches = open_flat_window_at(self.data, self.accumulation, component_id)
        try:
            local = self._update.run(
                batches,
                runtime,
                phase_hook=self._on_update_phase,
                drain_previous=lambda: None,
                note_generator_sync=self._update.note_generator_sync,
                note_gradients_averaged=self._update.note_gradients_averaged,
                apply_optimizer=False,
            )
            return finite_gradient_report(
                [self.generator, self.discriminator],
                loss=float(local["generator"]),
                sample_id=str(local["sample_id"]),
                component_id=component_id,
            )
        finally:
            restore_fields(self._update, saved)
            if not was_training:
                self.generator.eval()
            _close_epoch_view(view)

    def official_step(self, batches: list[Any], runtime: ProcessRuntime) -> dict[str, float]:
        self._in_official_step = True
        try:
            return self._update.run(
                batches, runtime, phase_hook=self._on_update_phase, drain_previous=self._drain_mp_io,
                note_generator_sync=self._update.note_generator_sync,
                note_gradients_averaged=self._update.note_gradients_averaged,
            )
        finally:
            self._in_official_step = False

    def state_dict(self) -> dict[str, Any]:
        return self.progress.export(experiment_claim=self.experiment_claim)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.progress.restore(state, experiment_claim=self.experiment_claim)
        self._validator.invalidate()
        self._restored = True

    def _research(self, runtime: ProcessRuntime, kind: str, epoch: int, official: dict[str, float] | None = None) -> dict[str, float]:
        return self.research.score(
            runtime, output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity, kind=kind, epoch=epoch,
            logical_step=self.progress.logical_step, official=official,
            stopping=getattr(self, "training_control", None),
        )

    def _on_update_phase(self, phase: str, payload: object = None) -> None:
        del phase, payload

    def _on_boundary_phase(self, phase: str) -> None:
        del phase

    def _offer(self, event: str, metrics: dict[str, float] | None = None) -> None:
        if self._in_official_step:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")

        def prepare(_runtime: object) -> None:
            self._recorder.flush(self.journal)

        step = self.progress.logical_step
        if metrics:
            self.lifecycle.save_if_needed(event, logical_step=step, metrics=metrics, prepare=prepare)
        else:
            self.lifecycle.save_if_needed(event, logical_step=step, prepare=prepare)

    def _finish(self, reason: str) -> RunResult:
        self._recorder.flush(self.journal)
        return close_training_run(
            reason, research=self.research, runtime=self.runtime, lifecycle=self.lifecycle,
            data=self.data, logical_step=self.progress.logical_step,
            optimizer_updates=self.progress.optimizer_updates,
            training_control=getattr(self, "training_control", None),
            set_stop_reason=self.progress.set_stop_reason,
        )

    def _stop_between_steps(self, runtime: ProcessRuntime) -> bool:
        if not stop_after_step(self.progress.logical_step, self.stop_after_logical_steps, runtime):
            return False
        self._recorder.flush(self.journal)
        if self.progress.logical_step > 0 and not self._in_official_step:
            self._offer("stop")
        return True

    def _control_stopped(self) -> bool:
        return control_requests_stop(getattr(self, "training_control", None))

    def _epoch_capped(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

    def _drain_mp_io(self) -> None:
        self._recorder.flush(self.journal)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        return stop_after_step(self.progress.logical_step, self.stop_after_logical_steps, runtime)

# --- Checkpoint progress. This object is the only live counter owner. ---

IN_EPOCH = "in_epoch"
PENDING_BOUNDARY = "pending_boundary"
_PHASES = frozenset({IN_EPOCH, PENDING_BOUNDARY})


def open_boundary(trainer_epoch: int, *, validate: bool, epoch_done: bool) -> dict[str, Any]:
    """Record whether this safe point still owes validation or a scheduler step."""

    return {
        "closed": False,
        "epoch_done": bool(epoch_done),
        "lr_d": None,
        "lr_g": None,
        "pesq": None,
        "pesq_status": None,
        "scheduler": bool(epoch_done),
        "scheduler_done": False,
        "selected": False,
        "selection_done": False,
        "trainer_epoch": int(trainer_epoch),
        "validate": bool(validate),
        "validation_done": False,
    }


@dataclass(frozen=True)
class _DecodedProgress:
    """Temporary decode of one payload. It is not stored on the live progress."""

    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[Any]
    completed_boundaries: list[Any]
    step_log: list[Any]
    boundary: dict[str, Any] | None
    scheduler_steps: int
    best_pesq: float | None
    best_epoch: int | None
    best_logical_step: int | None
    stop_reason: str | None


def _decode_progress(
    state: dict[str, Any],
    *,
    experiment_claim: str,
    optimizer_ids: tuple[str, str],
) -> _DecodedProgress:
    """Validate and copy a payload. This does not touch the live progress.

    Optimizer counts stay on the ids stored in the payload. A payload that
    omits a registered id is rejected instead of being lined up by position.
    Extra ids are kept so an old payload still round-trips; ``complete_update``
    does not add to them. Value conversion matches the pre-``8f54ab8``
    ``int`` / ``float`` / ``dict`` calls, including their exception types.
    """

    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    if state.get("experiment_claim") != experiment_claim:
        raise RuntimeFailure("experiment claim does not match this trainer")
    history = state.get("history")
    completed = state.get("completed_boundaries")
    step_log = state.get("step_log")
    if not isinstance(history, list) or not isinstance(completed, list) or not isinstance(step_log, list):
        raise RuntimeFailure("trainer history is invalid")
    boundary = state.get("boundary")
    best = state.get("best_pesq")
    best_epoch = state.get("best_epoch")
    best_step = state.get("best_logical_step")
    if best_epoch is not None and (isinstance(best_epoch, bool) or not isinstance(best_epoch, int)):
        raise RuntimeFailure("best_epoch is invalid")
    if best_step is not None and (isinstance(best_step, bool) or not isinstance(best_step, int)):
        raise RuntimeFailure("best_logical_step is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    copied_boundary = None if boundary is None else dict(boundary)
    scheduler_steps = int(state.get("scheduler_steps", 0))
    best_pesq = None if best is None else float(best)
    missing = [name for name in optimizer_ids if name not in converted]
    if missing:
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer ids")
    stop_reason = state.get("stop_reason")
    return _DecodedProgress(
        logical_step=logical_step,
        optimizer_updates=converted,
        phase=phase,
        history=list(history),
        completed_boundaries=list(completed),
        step_log=list(step_log),
        boundary=copied_boundary,
        scheduler_steps=scheduler_steps,
        best_pesq=best_pesq,
        best_epoch=best_epoch,
        best_logical_step=best_step,
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )


@dataclass
class MPTrainingProgress:
    """Live MP-SENet counters. Checkpoint code sees them only through ``export``."""

    logical_step: int
    optimizer_updates: dict[str, int]
    scheduler_steps: int
    phase: str
    boundary: dict[str, Any] | None
    history: list[Any]
    completed_boundaries: list[Any]
    step_log: list[Any]
    best_pesq: float | None
    best_epoch: int | None
    best_logical_step: int | None
    stop_reason: str
    _optimizer_ids: tuple[str, str] = field(repr=False)

    @classmethod
    def initial(cls, optim_g_id: str, optim_d_id: str) -> MPTrainingProgress:
        """Zero counters for a trainer that has not restored a checkpoint."""

        return cls(
            logical_step=0,
            optimizer_updates={optim_g_id: 0, optim_d_id: 0},
            scheduler_steps=0,
            phase=IN_EPOCH,
            boundary=None,
            history=[],
            completed_boundaries=[],
            step_log=[],
            best_pesq=None,
            best_epoch=None,
            best_logical_step=None,
            stop_reason="initialized",
            _optimizer_ids=(optim_g_id, optim_d_id),
        )

    @property
    def pending(self) -> bool:
        """True after a pending save and before that boundary has been closed."""

        return self.phase == PENDING_BOUNDARY

    def reset_fresh(self) -> None:
        """Zero a run that is not resuming. ``stop_reason`` stays as it is."""

        optim_g_id, optim_d_id = self._optimizer_ids
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.scheduler_steps = 0
        self.phase = IN_EPOCH
        self.boundary = None
        self.history = []
        self.completed_boundaries = []
        self.step_log = []
        self.best_pesq = None
        self.best_epoch = None
        self.best_logical_step = None

    def complete_update(self) -> None:
        """Count one finished GAN pair on the logical step and the registered optimizers.

        Extra ids restored from an old payload stay in the dict and are not
        incremented. The two registered ids are addressed by name, so payload
        order cannot swap their counts.
        """

        self.logical_step += 1
        generator_id, discriminator_id = self._optimizer_ids
        self.optimizer_updates[generator_id] += 1
        self.optimizer_updates[discriminator_id] += 1

    def append_step(self, entry: dict[str, Any]) -> None:
        self.step_log.append(entry)

    def append_history(self, decision: dict[str, Any]) -> None:
        self.history.append(decision)

    def open_pending(self, trainer_epoch: int, *, validate: bool, epoch_done: bool) -> None:
        """Open the boundary that a pending checkpoint is about to store."""

        self.phase = PENDING_BOUNDARY
        self.boundary = open_boundary(trainer_epoch, validate=validate, epoch_done=epoch_done)

    def remember_best(self, decision: dict[str, Any]) -> None:
        """Store a strict PESQ improvement. A rejection leaves the current best."""

        if not decision["selected"]:
            return
        self.best_pesq = float(decision["pesq"])
        self.best_epoch = int(decision["epoch"])
        self.best_logical_step = int(decision["logical_step"])

    def mark_validation(self, decision: dict[str, Any]) -> None:
        """Record that this open boundary has finished validation and selection."""

        boundary = self._open_boundary()
        boundary["validation_done"] = True
        boundary["selection_done"] = True
        boundary["pesq"] = decision["pesq"]
        boundary["pesq_status"] = decision["pesq_status"]
        boundary["selected"] = decision["selected"]

    def mark_phase(self, phase: str) -> None:
        """Set the in-memory phase. ``pending_scheduler`` is not a checkpoint phase."""

        self.phase = phase

    def mark_scheduler_done(self, lr_g: float, lr_d: float) -> None:
        """Record one epoch-end step of both schedulers."""

        boundary = self._open_boundary()
        boundary["scheduler_done"] = True
        boundary["lr_g"] = float(lr_g)
        boundary["lr_d"] = float(lr_d)
        self.scheduler_steps += 1

    def close_boundary(self) -> None:
        """Append the open boundary and return to the epoch."""

        boundary = self._open_boundary()
        boundary["closed"] = True
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = IN_EPOCH

    def place_cursor(self, data: Any, epoch: int, index: int) -> None:
        """Write the data cursor after a finished update. Data owns the fields."""

        data.epoch = int(epoch)
        data.update_index = int(index)

    def reset_cursor(self, data: Any) -> None:
        """Put a fresh run at the start of epoch 0."""

        data.epoch = 0
        data.update_index = 0

    def roll_epoch(self, data: Any, trainer_epoch: int) -> None:
        """Advance past an epoch that the closed boundary has finished."""

        data.epoch = int(trainer_epoch) + 1
        data.update_index = 0

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def export(self, *, experiment_claim: str) -> dict[str, Any]:
        """Copy the counters into the existing ``states["trainer"]`` dict."""

        return {
            "best_epoch": self.best_epoch,
            "best_logical_step": self.best_logical_step,
            "best_pesq": self.best_pesq,
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "experiment_claim": experiment_claim,
            "history": list(self.history),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "scheduler_steps": self.scheduler_steps,
            "step_log": list(self.step_log),
            "stop_reason": self.stop_reason,
        }

    def restore(self, state: dict[str, Any], *, experiment_claim: str) -> None:
        """Replace the counters from a stored trainer dict. Invalid payloads raise.

        Decoding finishes before any field of this object is written. A
        conversion or consistency error leaves the live counters and their
        nested containers as they were.
        """

        decoded = _decode_progress(
            state,
            experiment_claim=experiment_claim,
            optimizer_ids=self._optimizer_ids,
        )
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.history = decoded.history
        self.completed_boundaries = decoded.completed_boundaries
        self.step_log = decoded.step_log
        self.boundary = decoded.boundary
        self.scheduler_steps = decoded.scheduler_steps
        self.best_pesq = decoded.best_pesq
        self.best_epoch = decoded.best_epoch
        self.best_logical_step = decoded.best_logical_step
        if decoded.stop_reason is not None:
            self.stop_reason = decoded.stop_reason

    def _open_boundary(self) -> dict[str, Any]:
        boundary = self.boundary
        if boundary is None:
            raise RuntimeFailure("there is no pending GAN boundary")
        return boundary

# --- Constructor checks. They do not train or save. ---

_CLAIMS = frozenset({"official_faithful", "leakage_safe", "research"})


def require_mp_gan_setup(
    generator: PreparedModel,
    discriminator: PreparedModel,
    data: Any,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
    scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
    signal_objective: torch.nn.Module,
    metric_objective: torch.nn.Module,
    discriminator_objective: torch.nn.Module,
    experiment_claim: str,
) -> tuple[str, str]:
    """Refuse a pair that cannot run the official GAN loop. Return optimizer ids."""

    require_prepared(generator, MPSENetGANGenerator, "generator must be a prepared MPSENetGANGenerator")
    require_prepared(
        discriminator,
        MPSENetMetricDiscriminator,
        "discriminator must be a prepared MPSENetMetricDiscriminator",
    )
    if generator.unwrapped().architecture != discriminator.unwrapped().architecture:
        raise BuildError("discriminator initialization architecture does not match the generator")
    if generator.unwrapped().seed != discriminator.unwrapped().seed:
        raise BuildError("generator and discriminator seeds differ")
    if not isinstance(optim_g, torch.optim.AdamW) or not isinstance(optim_d, torch.optim.AdamW):
        raise BuildError("MP-SENet GAN optimizers must be AdamW")
    if not isinstance(scheduler_g, torch.optim.lr_scheduler.ExponentialLR):
        raise BuildError("scheduler_g must be ExponentialLR")
    if not isinstance(scheduler_d, torch.optim.lr_scheduler.ExponentialLR):
        raise BuildError("scheduler_d must be ExponentialLR")
    _require_disjoint(optim_g, optim_d)
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    if experiment_claim not in _CLAIMS:
        raise BuildError("experiment_claim must be official_faithful, leakage_safe, or research")
    generator.unwrapped().official_differences = (
        *require_official_architecture(generator.unwrapped().architecture, experiment_claim),
        *require_mp_senet(signal_objective, metric_objective, discriminator_objective, experiment_claim),
    )
    optim_g_id = getattr(optim_g, "_experiment_component_id", None)
    optim_d_id = getattr(optim_d, "_experiment_component_id", None)
    if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
        raise BuildError("optimizers have no component id")
    return optim_g_id, optim_d_id


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value


def require_group_size(data: Any, accumulation: int, component_id: str) -> None:
    """Refuse a data group whose width is not the GAN accumulation."""

    group_size = getattr(data, "group_size", None)
    if group_size is None:
        return
    if int(group_size) != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} "
            f"does not match accumulation {accumulation}"
        )


def _require_disjoint(optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer) -> None:
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("MP-SENet GAN optimizers must each own parameters")

# --- Step log. A flush is not a checkpoint offer. ---

class StepRecorder:
    """One unflushed metrics row. Empty at every save that actually writes."""

    def __init__(self) -> None:
        self._pending: tuple[dict[str, Any], dict[str, Any]] | None = None

    @property
    def pending_logical_step(self) -> int | None:
        if self._pending is None:
            return None
        return int(self._pending[0]["logical_step"])

    def share(self, runtime: ProcessRuntime, local: dict[str, Any]) -> dict[str, Any]:
        """Rank 0's scalars, copied onto every rank so the checkpoint matches."""

        gathered = runtime.gather_objects(dict(local))
        shared = gathered[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure("rank 0 did not return GAN metrics")
        return shared

    def entry(self, shared: dict[str, Any], *, epoch: int, logical_step: int) -> dict[str, Any]:
        return {
            "applied_d": True,
            "applied_g": True,
            "complex": shared["complex"],
            "consistency": shared["consistency"],
            "discriminator": shared["discriminator"],
            "epoch": epoch,
            "generator": shared["generator"],
            "logical_step": logical_step,
            "loss": shared["generator"],
            "lr_d": shared["lr_d"],
            "lr_g": shared["lr_g"],
            "magnitude": shared["magnitude"],
            "metric": shared["metric"],
            "phase": shared["phase"],
            "sample_id": shared["sample_id"],
            "time": shared["time"],
        }

    def stage(self, journal: StepJournal, entry: dict[str, Any], local: dict[str, Any]) -> None:
        """Keep this row. Write an older row first so it is not replaced."""

        if self._pending is not None:
            self.flush(journal)
        self._pending = (entry, local)

    def write_now(self, journal: StepJournal, entry: dict[str, Any], local: dict[str, Any]) -> None:
        """Write any older row, then this one. Nothing stays pending."""

        self.flush(journal)
        _write(journal, entry, local)

    def flush(self, journal: StepJournal) -> None:
        """Write the held row, if there is one. This does not offer a checkpoint."""

        if self._pending is None:
            return
        entry, local = self._pending
        self._pending = None
        _write(journal, entry, local)

    def publish(self, journal: StepJournal, entry: dict[str, Any], local: dict[str, Any], *, defer: bool) -> None:
        """Stage the row when the next backward may overlap its write. Otherwise write it now."""

        if defer:
            self.stage(journal, entry, local)
        else:
            self.write_now(journal, entry, local)


def _write(journal: StepJournal, entry: dict[str, Any], local: dict[str, Any]) -> None:
    journal.write_metrics(entry)
    journal.write_sample({**local, "logical_step": entry["logical_step"], "loss": local["generator"]})

# --- Tensors shared by the CPU and CUDA updates. They do not choose the order. ---

LOSS_NAMES = ("total", "magnitude", "phase", "complex", "consistency", "metric", "time")


def signal_context(parts: dict[str, torch.Tensor], clean: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the signal-term context from one generator forward."""

    return {
        "clean_audio": clean,
        "clean_complex": parts["clean_complex"],
        "clean_mag": parts["clean_mag"],
        "clean_phase": parts["clean_phase"],
        "generated_audio": parts["audio"],
        "generated_complex": parts["complex"],
        "generated_mag": parts["mag"],
        "generated_phase": parts["phase"],
        "recomputed_complex": parts["recomputed_complex"],
    }


def metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    """Look up PESQ targets on the losses module so tests can patch that function."""

    from src.experiments.calc import losses as loss_lib

    return loss_lib.metric_targets(clean, enhanced, sample_rate)


def inputs_ready(clean: torch.Tensor, noisy: torch.Tensor, n_fft: int) -> bool:
    """Accept a matching pair long enough for the STFT. CPU also requires finite values."""

    if clean.ndim != 2 or clean.shape != noisy.shape:
        return False
    if int(clean.shape[-1]) <= n_fft // 2:
        return False
    if clean.is_cuda:
        # A non-finite waveform makes the loss non-finite, which rejects the step.
        return True
    return bool(torch.isfinite(clean).all()) and bool(torch.isfinite(noisy).all())


def waveforms(batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
    batch = runtime.move(batch)
    clean = batch.targets
    noisy = batch.inputs
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    return clean, noisy


def backward_if_finite(
    runtime: ProcessRuntime,
    loss: torch.Tensor,
    what: str,
    *,
    retain_graph: bool = False,
) -> None:
    if runtime.device.type != "cuda":
        finite = bool(torch.isfinite(loss.detach()).all())
        if not runtime.agree(finite, policy="all"):
            raise RuntimeFailure(f"{what} is not finite on every rank")
    runtime.backward(loss, retain_graph=retain_graph)

# --- CPU, DDP, and CUDA accumulation above 1. Discriminator, then generator. ---

def accumulated_step(
    update: Any,
    batches: list[Batch],
    runtime: ProcessRuntime,
    *,
    phase_hook: Callable[..., None],
    note_generator_sync: Callable[[bool], None],
    note_gradients_averaged: Callable[[], None],
    apply_optimizer: bool = True,
) -> dict[str, float]:
    """Discriminator micro-batches, one discriminator step, then the generator."""

    row_counts = [micro_rows([batch]) for batch in batches]
    scale = window_scale(
        row_counts,
        gather_row_counts(runtime, row_counts),
        accumulation=update.accumulation,
        world_size=int(runtime.world_size),
    )
    phase_hook("zero_grad")
    update.optim_d.zero_grad(set_to_none=True)
    update.optim_g.zero_grad(set_to_none=True)
    stored: list[tuple[torch.Tensor, dict[str, torch.Tensor]] | None] = []
    discriminator_total = 0.0
    sample_ids: list[str] = []
    with runtime.sync_gradients([update.generator], enabled=False):
        for index, batch in enumerate(batches):
            last = index == len(batches) - 1
            if row_counts[index] == 0:
                sample_ids.append(batch.sample_id)
                if not runtime.agree(True, policy="all"):
                    raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
                phase_hook("before_generator_forward")
                errors = runtime.gather_objects(None)
                if any(errors):
                    raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
                stored.append(None)
                with runtime.sync_gradients([update.discriminator], enabled=last):
                    disc_errors = runtime.gather_objects(None)
                    if any(disc_errors):
                        raise RuntimeFailure(
                            "discriminator forward failed: " + "; ".join(item for item in disc_errors if item)
                        )
                    backward_if_finite(
                        runtime,
                        connected_zero(update.discriminator.parameters, "trainer"),
                        "discriminator loss",
                    )
                continue
            clean, noisy = waveforms(batch, runtime)
            sample_ids.append(batch.sample_id)
            ready = inputs_ready(clean, noisy, update.generator.unwrapped().n_fft)
            if not runtime.agree(ready, policy="all"):
                raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
            phase_hook("before_generator_forward")
            error: str | None = None
            parts: dict[str, torch.Tensor] | None = None
            try:
                parts = pack_forward(update.generator(noisy, clean))
                require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
            except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors = runtime.gather_objects(error)
            if any(errors):
                raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
            if parts is None:
                raise RuntimeFailure("generator forward produced no tensors")
            stored.append((clean, parts))
            ones = torch.ones(clean.shape[0], device=clean.device)
            with runtime.sync_gradients([update.discriminator], enabled=last):
                disc_error: str | None = None
                loss: torch.Tensor | None = None
                try:
                    real = update.discriminator(parts["clean_mag"], parts["clean_mag"])
                    fake = update.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
                    targets = metric_targets(
                        clean, parts["audio"], int(update.generator.unwrapped().architecture["sampling_rate"])
                    )
                    loss = call_objective(
                        update.discriminator_objective,
                        {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
                    ).total
                    loss = weight_loss(loss, scale, index)
                except (TrainingError, RuntimeError, ValueError) as exc:
                    disc_error = f"{type(exc).__name__}: {exc}"
                disc_errors = runtime.gather_objects(disc_error)
                if any(disc_errors):
                    raise RuntimeFailure(
                        "discriminator forward failed: " + "; ".join(item for item in disc_errors if item)
                    )
                if loss is None:
                    raise RuntimeFailure("discriminator loss is missing")
                backward_if_finite(runtime, loss, "discriminator loss")
                raw_disc = loss.detach()
                if isinstance(scale, int):
                    discriminator_total += float(raw_disc) * scale
                else:
                    factor = scale[index]
                    discriminator_total += float(raw_disc) / factor if factor else 0.0
        if apply_optimizer:
            discriminator_result = runtime.step(update.optim_d)
        phase_hook("after_discriminator_step", stored)
        raw = update.discriminator.unwrapped().discriminator
        frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
        component_totals = {name: 0.0 for name in LOSS_NAMES}
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            for index, item in enumerate(stored):
                if item is None:
                    gen_errors = runtime.gather_objects(None)
                    if any(gen_errors):
                        raise RuntimeFailure(
                            "generator loss failed: " + "; ".join(item for item in gen_errors if item)
                        )
                    synced = update.generator.gradient_sync_enabled()
                    if synced is not None:
                        note_generator_sync(synced)
                    backward_if_finite(
                        runtime,
                        connected_zero(update.generator.parameters, "trainer"),
                        "generator loss",
                    )
                    continue
                clean, parts = item
                ones = torch.ones(clean.shape[0], device=clean.device)
                gen_error: str | None = None
                total: torch.Tensor | None = None
                try:
                    metric = raw(parts["clean_mag"], parts["mag_hat"])
                    signal = call_objective(update.signal_objective, signal_context(parts, clean))
                    metric_out = call_objective(update.metric_objective, {"metric": metric, "ones": ones})
                    total_tensor = signal.total + metric_out.total
                    losses = {
                        "total": total_tensor,
                        "magnitude": signal.terms["magnitude"],
                        "phase": signal.terms["phase"],
                        "complex": signal.terms["complex"],
                        "consistency": signal.terms["consistency"],
                        "metric": metric_out.terms["metric"],
                        "time": signal.terms["time"],
                    }
                    total = weight_loss(total_tensor, scale, index)
                except (TrainingError, RuntimeError, ValueError) as exc:
                    gen_error = f"{type(exc).__name__}: {exc}"
                gen_errors = runtime.gather_objects(gen_error)
                if any(gen_errors):
                    raise RuntimeFailure(
                        "generator loss failed: " + "; ".join(item for item in gen_errors if item)
                    )
                if total is None:
                    raise RuntimeFailure("generator loss is missing")
                synced = update.generator.gradient_sync_enabled()
                if synced is not None:
                    note_generator_sync(synced)
                backward_if_finite(runtime, total, "generator loss")
                for name in LOSS_NAMES:
                    component_totals[name] += float(losses[name].detach())
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        note_gradients_averaged()
        runtime.average_gradients(update.generator)
    if apply_optimizer:
        generator_result = runtime.step(update.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(update.generator)
        runtime.broadcast_buffers(update.discriminator)
        if not discriminator_result.applied or not generator_result.applied:
            raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
    count = len(batches)
    phase_hook("end_step")
    return {
        "complex": component_totals["complex"] / count,
        "consistency": component_totals["consistency"] / count,
        "discriminator": discriminator_total / count,
        "generator": component_totals["total"] / count,
        "lr_d": float(update.optim_d.param_groups[0]["lr"]),
        "lr_g": float(update.optim_g.param_groups[0]["lr"]),
        "magnitude": component_totals["magnitude"] / count,
        "metric": component_totals["metric"] / count,
        "phase": component_totals["phase"] / count,
        "sample_id": ",".join(sample_ids),
        "time": component_totals["time"] / count,
    }

# --- CUDA, one rank, accumulation 1. PESQ overlaps the signal backward. ---

def cuda_overlapped_step(
    update: Any,
    batch: Batch,
    runtime: ProcessRuntime,
    *,
    phase_hook: Callable[..., None],
    drain_previous: Callable[[], None],
    note_gradients_averaged: Callable[[], None],
    apply_optimizer: bool = True,
) -> dict[str, float]:
    """PESQ runs while the signal backward uses the same generator graph."""

    phase_hook("zero_grad")
    update.optim_d.zero_grad(set_to_none=True)
    update.optim_g.zero_grad(set_to_none=True)
    rows = [micro_rows([batch])]
    scale = window_scale(
        rows,
        gather_row_counts(runtime, rows),
        accumulation=update.accumulation,
        world_size=int(runtime.world_size),
    )
    if rows[0] == 0:
        phase_hook("before_generator_forward")
        backward_if_finite(
            runtime, connected_zero(update.generator.parameters, "trainer"), "generator signal loss",
        )
        drain_previous()
        backward_if_finite(
            runtime, connected_zero(update.discriminator.parameters, "trainer"), "discriminator loss",
        )
        if apply_optimizer:
            discriminator_result = runtime.step(update.optim_d)
        phase_hook("after_discriminator_step", [])
        backward_if_finite(
            runtime, connected_zero(update.generator.parameters, "trainer"), "generator metric loss",
        )
        note_gradients_averaged()
        runtime.average_gradients(update.generator)
        if apply_optimizer:
            generator_result = runtime.step(update.optim_g)
            runtime.end_step()
            runtime.broadcast_buffers(update.generator)
            runtime.broadcast_buffers(update.discriminator)
            if not discriminator_result.applied or not generator_result.applied:
                raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
        phase_hook("end_step")
        return {
            "complex": 0.0,
            "consistency": 0.0,
            "discriminator": 0.0,
            "generator": 0.0,
            "lr_d": float(update.optim_d.param_groups[0]["lr"]),
            "lr_g": float(update.optim_g.param_groups[0]["lr"]),
            "magnitude": 0.0,
            "metric": 0.0,
            "phase": 0.0,
            "sample_id": str(batch.sample_id),
            "time": 0.0,
        }
    clean, noisy = waveforms(batch, runtime)
    if not inputs_ready(clean, noisy, update.generator.unwrapped().n_fft):
        raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
    phase_hook("before_generator_forward")
    try:
        parts = pack_forward(update.generator(noisy, clean))
        require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
    except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
        raise RuntimeFailure(f"generator forward failed: {type(exc).__name__}: {exc}") from exc
    pending = _schedule_metric_targets(
        clean,
        parts["audio"],
        int(update.generator.unwrapped().architecture["sampling_rate"]),
    )
    ones = torch.ones(clean.shape[0], device=clean.device)
    signal = call_objective(update.signal_objective, signal_context(parts, clean))
    backward_if_finite(runtime, weight_loss(signal.total, scale, 0), "generator signal loss", retain_graph=True)
    # Previous-step log I/O overlaps the rest of the PESQ thread.
    # The callback must not offer a checkpoint.
    drain_previous()
    try:
        targets = pending.result()
        real = update.discriminator(parts["clean_mag"], parts["clean_mag"])
        fake = update.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
        loss_d = call_objective(
            update.discriminator_objective,
            {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
        ).total
        loss_d = weight_loss(loss_d, scale, 0)
    except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
        raise RuntimeFailure(f"discriminator forward failed: {type(exc).__name__}: {exc}") from exc
    backward_if_finite(runtime, loss_d, "discriminator loss")
    if apply_optimizer:
        discriminator_result = runtime.step(update.optim_d)
    phase_hook("after_discriminator_step", [(clean, parts)])
    raw = update.discriminator.unwrapped().discriminator
    frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    try:
        for parameter in frozen:
            parameter.requires_grad_(False)
        try:
            metric = raw(parts["clean_mag"], parts["mag_hat"])
            metric_out = call_objective(update.metric_objective, {"metric": metric, "ones": ones})
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"generator loss failed: {type(exc).__name__}: {exc}") from exc
        backward_if_finite(runtime, weight_loss(metric_out.total, scale, 0), "generator metric loss")
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
    note_gradients_averaged()
    runtime.average_gradients(update.generator)
    total = signal.total + metric_out.total
    logged = torch.stack(
        [
            signal.terms["complex"].detach(),
            signal.terms["consistency"].detach(),
            (loss_d.detach() * scale) if isinstance(scale, int) else (
                loss_d.detach() / scale[0] if scale[0] else loss_d.detach()
            ),
            total.detach(),
            signal.terms["magnitude"].detach(),
            metric_out.terms["metric"].detach(),
            signal.terms["phase"].detach(),
            signal.terms["time"].detach(),
        ]
    )
    host = torch.empty(logged.shape, dtype=torch.float32, pin_memory=True)
    host.copy_(logged.detach().float(), non_blocking=True)
    if apply_optimizer:
        generator_result = runtime.step(update.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(update.generator)
        runtime.broadcast_buffers(update.discriminator)
        if not discriminator_result.applied or not generator_result.applied:
            raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
    complex_v, consistency_v, disc_v, total_v, magnitude_v, metric_v, phase_v, time_v = (
        float(item) for item in host.tolist()
    )
    phase_hook("end_step")
    return {
        "complex": complex_v,
        "consistency": consistency_v,
        "discriminator": disc_v,
        "generator": total_v,
        "lr_d": float(update.optim_d.param_groups[0]["lr"]),
        "lr_g": float(update.optim_g.param_groups[0]["lr"]),
        "magnitude": magnitude_v,
        "metric": metric_v,
        "phase": phase_v,
        "sample_id": batch.sample_id,
        "time": time_v,
    }


def _schedule_metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> Any:
    """Copy waveforms without blocking, then score PESQ on a side thread."""

    from src.experiments.calc.losses import metric_targets

    clean_cpu = torch.empty(clean.shape, dtype=clean.dtype, pin_memory=True)
    enhanced_cpu = torch.empty(enhanced.shape, dtype=enhanced.dtype, pin_memory=True)
    clean_cpu.copy_(clean.detach(), non_blocking=True)
    enhanced_cpu.copy_(enhanced.detach(), non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    box: dict[str, object] = {}
    done = threading.Event()

    def run() -> None:
        try:
            event.synchronize()
            box["value"] = metric_targets(clean_cpu, enhanced_cpu, sample_rate)
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run, name="mp-pesq", daemon=True)
    thread.start()

    class _Pending:
        def result(self) -> object:
            done.wait()
            if "error" in box:
                raise box["error"]
            return box.get("value")

    return _Pending()

# --- Which of the two updates runs. Both finish before a save is possible. ---

def refuse_gan_amp(runtime: ProcessRuntime) -> None:
    """Refuse FP16 AMP. The two AdamW steps are not one scaler decision."""

    if getattr(runtime, "precision", "fp32") == "fp16":
        raise RuntimeFailure(
            "MP-SENet official GAN refuses fp16. "
            "The current scaler decides overflow per optimizer, so the discriminator "
            "step and the generator step are not one atomic update. FP32 is unchanged."
        )


class MPSENetUpdate:
    """Discriminator-then-generator step for the injected MP-SENet objectives."""

    def __init__(
        self,
        *,
        generator: PreparedModel,
        discriminator: PreparedModel,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        signal_objective: torch.nn.Module,
        metric_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        accumulation: int,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.signal_objective = signal_objective
        self.metric_objective = metric_objective
        self.discriminator_objective = discriminator_objective
        self.accumulation = accumulation
        self.average_calls = 0
        self.generator_synced_during_backward = False

    def note_generator_sync(self, enabled: bool) -> None:
        """Remember the last DDP sync flag seen during generator backward.

        The CUDA overlap path does not call this. The flag is not stored in
        the checkpoint.
        """

        self.generator_synced_during_backward = bool(enabled)

    def note_gradients_averaged(self) -> None:
        """Count one gradient reduction, immediately before average_gradients."""

        self.average_calls += 1

    def run(
        self,
        batches: list[Batch],
        runtime: ProcessRuntime,
        *,
        phase_hook: Callable[..., None],
        drain_previous: Callable[[], None],
        note_generator_sync: Callable[[bool], None],
        note_gradients_averaged: Callable[[], None],
        apply_optimizer: bool = True,
    ) -> dict[str, float]:
        """Apply one official pair, or raise before either optimizer is a safe point."""

        refuse_gan_amp(runtime)
        if len(batches) < 1 or len(batches) > self.accumulation:
            raise RuntimeFailure(
                f"official step received {len(batches)} micro-batches, accumulation is {self.accumulation}"
            )
        counts = runtime.gather_objects(len(batches))
        if any(item != len(batches) for item in counts):
            raise RuntimeFailure("ranks disagree on the number of micro-batches")
        self.generator.train()
        self.discriminator.train()
        if runtime.device.type == "cuda" and runtime.world_size == 1 and self.accumulation == 1 and len(batches) == 1:
            return cuda_overlapped_step(
                self,
                batches[0],
                runtime,
                phase_hook=phase_hook,
                drain_previous=drain_previous,
                note_gradients_averaged=note_gradients_averaged,
                apply_optimizer=apply_optimizer,
            )
        return accumulated_step(
            self,
            batches,
            runtime,
            phase_hook=phase_hook,
            note_generator_sync=note_generator_sync,
            note_gradients_averaged=note_gradients_averaged,
            apply_optimizer=apply_optimizer,
        )

# --- Pending boundary: validation, best, then both ExponentialLR steps. ---

@dataclass(frozen=True)
class BoundaryResult:
    """What remains after validation and the epoch-end schedulers.

    ``metrics`` are the finite official scalars. ``roll_epoch`` tells the
    progress owner to advance the data cursor. ``validate`` is the flag that
    was stored with the pending boundary, including when validation was
    already done and this call did not score again.
    """

    metrics: dict[str, float]
    roll_epoch: bool
    validate: bool
    trainer_epoch: int


class OfficialBoundary:
    """One pending boundary: validation, best PESQ, then both schedulers."""

    def __init__(
        self,
        validator: Any,
        scheduler_g: Any,
        scheduler_d: Any,
        optim_g: Any,
        optim_d: Any,
        generator: Any,
        experiment_claim: str,
    ) -> None:
        self.validator = validator
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.generator = generator
        self.experiment_claim = experiment_claim

    def run(
        self,
        progress: MPTrainingProgress,
        runtime: Any,
        output_directory: Any,
        component_id: str,
        *,
        start_epoch: int,
        on_phase: Callable[[str], None],
    ) -> BoundaryResult:
        """Validate, select, then step both schedulers when the epoch is over.

        A validation interval does not step the schedulers. When the same
        update validates and ends the epoch, selection finishes before either
        ``ExponentialLR`` moves. ``on_phase("pending_scheduler")`` runs only
        in that case, before the scheduler steps.
        """

        if progress.phase != PENDING_BOUNDARY or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending GAN boundary")
        boundary = progress.boundary
        metrics: dict[str, float] = {}
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self.validator.evaluate(runtime, int(boundary["trainer_epoch"]), component_id)
            metrics = scalar_metrics(summary)
            decision = apply_pesq_selection(
                progress,
                int(boundary["trainer_epoch"]),
                summary,
                start_epoch=start_epoch,
            )
            progress.append_history(decision)
            publish_best(
                runtime,
                output_directory,
                decision,
                self.generator.unwrapped().export_for_inference(),
                self.experiment_claim,
            )
            progress.mark_validation(decision)
            if boundary["scheduler"]:
                progress.mark_phase("pending_scheduler")
                on_phase("pending_scheduler")
        if boundary["scheduler"] and not boundary["scheduler_done"]:
            self.scheduler_g.step()
            self.scheduler_d.step()
            progress.mark_scheduler_done(
                float(self.optim_g.param_groups[0]["lr"]),
                float(self.optim_d.param_groups[0]["lr"]),
            )
        result = BoundaryResult(
            metrics=metrics,
            roll_epoch=bool(boundary["epoch_done"]),
            validate=bool(boundary["validate"]),
            trainer_epoch=int(boundary["trainer_epoch"]),
        )
        progress.close_boundary()
        on_phase(progress.phase)
        return result
