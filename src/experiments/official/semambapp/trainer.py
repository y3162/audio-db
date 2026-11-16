"""SEMamba++ GAN training loop.

Read this file from ``SEMambaPPTrainer.fit`` downward. ``UpdateView`` is
defined first because the trainer class extends it.

``fit`` orders start, a resumed pending boundary, one flat window, then
``official_step``. The step refuses AMP, runs the discriminator AdamW, then
the generator AdamW. One micro-batch reuses the generator graph. Two or more
reforward the generator after the discriminator step and do not keep every
graph. A checkpoint is offered only after both optimizers have stepped.
Validation, when the interval says so, runs before both ExponentialLR
schedulers. Validation does not select a best generator. Counters live on
``SEMambaProgress``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.semambapp.evaluator import SEMambaValidator
from src.experiments.official.semambapp.model import (
    SEMambaPPCQTDiscriminator,
    SEMambaPPGenerator,
    SEMambaPPMRD,
    pack_forward,
    scan_execution,
)
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.selectors import scalar_metrics
from src.experiments.stopping import observe_validation, should_stop
from src.experiments.trainers import _close_epoch_view
from src.experiments.training.control import (
    finish_training_run,
    finite_gradient_report,
    gather_row_counts,
    micro_rows,
    open_flat_window,
    restore_fields,
    snapshot_fields,
    stop_after_step,
    weight_loss,
    window_scale,
)
from src.experiments.training.objectives import call_objective

# --- Observations the trainer exposes from the update it owns. ---

class UpdateView:
    """Read the non-persistent GAN observations from the update that owns them."""

    @property
    def capture_gradients(self) -> bool:
        return self._update.capture_gradients

    @capture_gradients.setter
    def capture_gradients(self, value: bool) -> None:
        self._update.capture_gradients = value

    @property
    def frozen_during_generator(self) -> bool:
        return self._update.frozen_during_generator

    @property
    def generator_graph_forwards(self) -> int:
        return self._update.generator_graph_forwards

    @property
    def generator_preview_forwards(self) -> int:
        return self._update.generator_preview_forwards

    @property
    def generator_reforward_forwards(self) -> int:
        return self._update.generator_reforward_forwards

    @property
    def sync_trace(self) -> list[str]:
        return self._update.sync_trace

    @property
    def last_generator_grads(self) -> dict[str, torch.Tensor]:
        return self._update.last_generator_grads

    @property
    def last_discriminator_grads(self) -> dict[str, torch.Tensor]:
        return self._update.last_discriminator_grads


# --- Trainer. ``fit`` is the timeline. ---

SUBSTITUTE_CLAIM = "protocol_conformant_substitute_materials"


class SEMambaPPTrainer(UpdateView):
    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        generator: PreparedModel,
        mrd: PreparedModel,
        cqt: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
        scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
        *,
        generator_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float | None = None,
        validation_interval: int = 0,
        best_checkpoint_start_epoch: int = 40,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
        claim_label: str = SUBSTITUTE_CLAIM,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        setup = require_semamba_setup(
            generator, mrd, cqt, data, optim_g, optim_d, scheduler_g, scheduler_d,
            generator_objective=generator_objective,
            discriminator_objective=discriminator_objective,
            max_logical_steps=max_logical_steps,
            accumulation=accumulation,
            grad_clip=grad_clip,
            validation_interval=validation_interval,
            best_checkpoint_start_epoch=best_checkpoint_start_epoch,
            max_epochs=max_epochs,
            stop_after_logical_steps=stop_after_logical_steps,
            claim_label=claim_label,
        )
        self.generator = generator
        self.mrd = mrd
        self.cqt = cqt
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.optim_g_id = setup.optim_g_id
        self.optim_d_id = setup.optim_d_id
        self.claim_label = setup.claim_label
        self.audio_sample_rate = int(generator.unwrapped().sampling_rate)
        self.audio_teachers = ("dry_clean",)
        self.max_logical_steps = setup.max_logical_steps
        self.accumulation = setup.accumulation
        self.validation_interval = setup.validation_interval
        self.best_checkpoint_start_epoch = setup.best_checkpoint_start_epoch
        self.max_epochs = setup.max_epochs
        self.stop_after_logical_steps = setup.stop_after_logical_steps
        self.progress = SEMambaProgress.initial(setup.optim_g_id, setup.optim_d_id, setup.claim_label)
        self._update = SEMambaUpdate(
            generator=generator,
            mrd=mrd,
            cqt=cqt,
            optim_g=optim_g,
            optim_d=optim_d,
            generator_objective=generator_objective,
            discriminator_objective=discriminator_objective,
            accumulation=self.accumulation,
            grad_clip=setup.grad_clip,
        )
        self._validator = SEMambaValidator(
            generator=generator,
            mrd=mrd,
            cqt=cqt,
            optim_g=optim_g,
            scheduler_g=scheduler_g,
            grad_clip=setup.grad_clip,
        )
        self._boundary = SEMambaBoundary(
            self._validator,
            optim_g=optim_g,
            optim_d=optim_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
        )
        self._log = StepLog()
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators,
            model_selection,
            data=self.data,
            module=self.generator,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection
        self._restored = False
        self._in_official_step = False

    @property
    def logical_step(self) -> int:
        return self.progress.logical_step

    @property
    def optimizer_updates(self) -> dict[str, int]:
        return self.progress.optimizer_updates

    @property
    def stop_reason(self) -> str | None:
        return self.progress.stop_reason

    @stop_reason.setter
    def stop_reason(self, value: str) -> None:
        self.progress.stop_reason = value

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or an interrupt."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        if self.progress.phase == PENDING_BOUNDARY:
            self._finish_boundary(runtime, component_id)
            if should_stop(self) or self._epoch_cap_reached():
                return self._result("completed")
        while self.progress.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                if self.progress.logical_step > 0 and not self._in_official_step:
                    self._offer(runtime, "stop")
                return self._result("interrupted")
            if self._epoch_cap_reached():
                return self._result("completed")
            epoch, index, view, batches = open_flat_window(self, component_id)
            try:
                local = self.official_step(batches, runtime)
                self.progress.complete_update()
                index += len(batches)
                self.data.epoch = epoch
                self.data.update_index = index
                epoch_done = index == view.updates
                will_validate = self.validation_interval > 0 and self.progress.logical_step % self.validation_interval == 0
                self._log.write(self.progress, runtime, self.journal, local, epoch=epoch)
                if not will_validate and not epoch_done:
                    self._offer_step(runtime, epoch)
                    continue
                self.progress.open_boundary(
                    open_boundary_payload(epoch, validate=will_validate, epoch_done=epoch_done)
                )
                self._on_boundary_phase(self.progress.phase)
                self._offer(runtime, "boundary_pending")
                if self._stop_requested(runtime):
                    return self._result("interrupted")
                self._finish_boundary(runtime, component_id)
                if should_stop(self) or self._epoch_cap_reached():
                    return self._result("completed")
            finally:
                _close_epoch_view(view)
        return self._result("completed")

    def _refuse_amp(self, runtime: object) -> None:
        self._update.refuse_amp(runtime)

    def preflight_probe(self) -> dict[str, Any]:
        """One real window of both backwards. Neither optimizer steps."""

        runtime = getattr(self, "runtime", None)
        if not isinstance(runtime, ProcessRuntime):
            raise BuildError("preflight has no runtime")
        component_id = str(getattr(self, "_experiment_component_id", "trainer"))
        saved = snapshot_fields(self._update)
        was_training = bool(self.generator.training)
        _epoch, _index, view, batches = open_flat_window(self, component_id)
        try:
            local = self._update.run(batches, runtime, self._on_update_phase, apply_optimizer=False)
            loss = float(local["generator"])
            return finite_gradient_report(
                [self.generator, self.mrd, self.cqt],
                loss=loss,
                sample_id=str(local["sample_id"]),
                component_id=component_id,
            )
        finally:
            restore_fields(self._update, saved)
            if not was_training:
                self.generator.eval()
            _close_epoch_view(view)

    def official_step(self, batches: list[Any], runtime: ProcessRuntime) -> dict[str, float | str]:
        """Run one official GAN update. Both optimizers step, or neither is checkpointed."""
        self._in_official_step = True
        try:
            return self._update.run(batches, runtime, self._on_update_phase)
        finally:
            self._in_official_step = False

    def state_dict(self) -> dict[str, Any]:
        return self.progress.export()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.progress.restore(state)
        self._validator.invalidate()
        self._restored = True

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        self.research.rewind(
            self.lifecycle.output_directory,
            restored=self._restored,
            phase=self.progress.phase,
            progress={"logical_step": int(self.progress.logical_step)},
        )
        self._update.refuse_amp(runtime)
        self._update.require_scan(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.journal.begin(
            restored=self._restored,
            logical_step=self.progress.logical_step,
            expect_rows=len(self.progress.step_log) if self._restored else None,
        )
        if not self._restored:
            self.progress.reset_fresh()
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        require_group_size(self.data, self.accumulation, str(component_id))
        runtime.barrier()
        return str(component_id)

    def _finish_boundary(self, runtime: ProcessRuntime, component_id: str) -> None:
        outcome = self._boundary.run(
            self.progress, runtime, self.data, component_id, self._on_boundary_phase,
        )
        metrics = published_metrics(
            outcome,
            runtime,
            research=self.research,
            lifecycle=self.lifecycle,
            logical_step=self.progress.logical_step,
            training_control=getattr(self, "training_control", None),
        )
        self._offer(runtime, "boundary", metrics=metrics)

    def _offer_step(self, runtime: ProcessRuntime, epoch: int) -> None:
        step_metrics = self.research.score(
            runtime,
            output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity,
            kind="step",
            epoch=epoch,
            logical_step=self.progress.logical_step,
            stopping=getattr(self, "training_control", None),
        )
        if step_metrics:
            self._offer(runtime, "step", metrics=step_metrics)
        else:
            self._offer(runtime, "step")

    def _on_update_phase(self, phase: str, payload: object = None) -> None:
        del phase, payload

    def _on_boundary_phase(self, phase: str) -> None:
        del phase

    def _offer(self, runtime: ProcessRuntime, event: str, metrics: dict[str, float] | None = None) -> None:
        if self._in_official_step:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")
        if event == "step":
            self.lifecycle.save_if_needed(
                event, logical_step=self.progress.logical_step, metrics=metrics, prepare=self._sync_ranks
            )
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.progress.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.mrd)
        runtime.broadcast_buffers(self.cqt)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        return stop_after_step(self.progress.logical_step, self.stop_after_logical_steps, runtime)

    def _epoch_cap_reached(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

    def _result(self, reason: str) -> RunResult:
        return finish_training_run(self, reason, prepare=self._sync_ranks)


# --- Checkpoint progress. This object is the only live counter owner. ---

IN_EPOCH = "in_epoch"

PENDING_BOUNDARY = "pending_boundary"

_PHASES = frozenset({IN_EPOCH, PENDING_BOUNDARY})

@dataclass
class _DecodedProgress:
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[dict[str, Any]]
    completed_boundaries: list[dict[str, Any]]
    step_log: list[dict[str, Any]]
    boundary: dict[str, Any] | None
    scheduler_steps: int
    stop_reason: str | None

@dataclass
class SEMambaProgress:
    """The checkpoint fields for one SEMamba++ run."""

    boundary: dict[str, Any] | None
    claim_label: str
    completed_boundaries: list[dict[str, Any]]
    history: list[dict[str, Any]]
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    scheduler_steps: int
    step_log: list[dict[str, Any]]
    stop_reason: str | None
    _optimizer_ids: tuple[str, str]

    @classmethod
    def initial(cls, optim_g_id: str, optim_d_id: str, claim_label: str) -> SEMambaProgress:
        return cls(
            boundary=None,
            claim_label=claim_label,
            completed_boundaries=[],
            history=[],
            logical_step=0,
            optimizer_updates={optim_g_id: 0, optim_d_id: 0},
            phase=IN_EPOCH,
            scheduler_steps=0,
            step_log=[],
            stop_reason="initialized",
            _optimizer_ids=(optim_g_id, optim_d_id),
        )

    def export(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "claim_label": self.claim_label,
            "completed_boundaries": list(self.completed_boundaries),
            "history": list(self.history),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "scheduler_steps": self.scheduler_steps,
            "step_log": list(self.step_log),
            "stop_reason": self.stop_reason,
        }

    def restore(self, state: dict[str, Any]) -> None:
        decoded = _decode_progress(state, self.claim_label, self._optimizer_ids)
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.history = decoded.history
        self.completed_boundaries = decoded.completed_boundaries
        self.step_log = decoded.step_log
        self.boundary = decoded.boundary
        self.scheduler_steps = decoded.scheduler_steps
        if decoded.stop_reason is not None:
            self.stop_reason = decoded.stop_reason

    def reset_fresh(self) -> None:
        optim_g_id, optim_d_id = self._optimizer_ids
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.scheduler_steps = 0
        self.step_log = []
        self.history = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH

    def complete_update(self) -> None:
        self.logical_step += 1
        for name in self._optimizer_ids:
            self.optimizer_updates[name] += 1

    def open_boundary(self, payload: dict[str, Any]) -> None:
        self.boundary = payload
        self.phase = PENDING_BOUNDARY

    def close_boundary(self, payload: dict[str, Any]) -> None:
        self.completed_boundaries.append(dict(payload))
        self.boundary = None
        self.phase = IN_EPOCH

def open_boundary_payload(trainer_epoch: int, *, validate: bool, epoch_done: bool) -> dict[str, Any]:
    return {
        "closed": False,
        "epoch_done": bool(epoch_done),
        "lr_d": None,
        "lr_g": None,
        "scheduler": bool(epoch_done),
        "scheduler_done": False,
        "selected": False,
        "selection_done": False,
        "trainer_epoch": int(trainer_epoch),
        "validate": bool(validate),
        "validation_done": False,
    }

def _decode_progress(
    state: dict[str, Any],
    claim_label: str,
    optimizer_ids: tuple[str, str],
) -> _DecodedProgress:
    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    if state.get("claim_label") != claim_label:
        raise RuntimeFailure("claim_label does not match this trainer")
    history = state.get("history")
    completed = state.get("completed_boundaries")
    step_log = state.get("step_log")
    if not isinstance(history, list) or not isinstance(completed, list) or not isinstance(step_log, list):
        raise RuntimeFailure("trainer history is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    missing = [name for name in optimizer_ids if name not in converted]
    if missing:
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer ids")
    boundary = state.get("boundary")
    decoded_boundary = None if boundary is None else dict(boundary)
    scheduler_steps = int(state.get("scheduler_steps", 0))
    stop_reason = state.get("stop_reason")
    kept_reason = stop_reason if isinstance(stop_reason, str) else None
    return _DecodedProgress(
        logical_step=logical_step,
        optimizer_updates=converted,
        phase=str(phase),
        history=list(history),
        completed_boundaries=list(completed),
        step_log=list(step_log),
        boundary=decoded_boundary,
        scheduler_steps=scheduler_steps,
        stop_reason=kept_reason,
    )


# --- Constructor checks. AMP and the official-claim label are refused here. ---

OFFICIAL_LR = 0.0002

OFFICIAL_BETAS = (0.8, 0.99)

OFFICIAL_WEIGHT_DECAY = 0.01

OFFICIAL_GAMMA = 0.99

SUBSTITUTE_CLAIM = "protocol_conformant_substitute_materials"

def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value

def non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be a non-negative integer")
    return value

def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{name} must be a positive number")
    number = float(value)
    if number <= 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise BuildError(f"{name} must be a positive finite number")
    return number

def require_adamw(optimizer: torch.optim.Optimizer, name: str) -> tuple[str, ...]:
    """Require AdamW. Record published hyperparameter differences and keep running."""

    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError(f"{name} must be AdamW")
    if not optimizer.param_groups:
        raise BuildError(f"{name} has no parameters")
    differences: list[str] = []
    for index, group in enumerate(optimizer.param_groups):
        label = f"{name} group {index}"
        if float(group["lr"]) != OFFICIAL_LR:
            differences.append(f"{label} learning rate is {group['lr']}, published value is {OFFICIAL_LR}")
        betas = tuple(float(value) for value in group["betas"])
        if betas != OFFICIAL_BETAS:
            differences.append(f"{label} betas are {betas}, published value is {OFFICIAL_BETAS}")
        if float(group["weight_decay"]) != OFFICIAL_WEIGHT_DECAY:
            differences.append(
                f"{label} weight decay is {group['weight_decay']}, published value is {OFFICIAL_WEIGHT_DECAY}"
            )
        if float(group["eps"]) != 1e-8:
            differences.append(f"{label} eps is {group['eps']}, published value is 1e-8")
        if bool(group.get("amsgrad", False)):
            differences.append(f"{label} amsgrad is true, published value is false")
    optimizer.official_differences = tuple(differences)  # type: ignore[attr-defined]
    return tuple(differences)

def require_scheduler(scheduler: torch.optim.lr_scheduler.ExponentialLR, name: str) -> tuple[str, ...]:
    if not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
        raise BuildError(f"{name} must be ExponentialLR")
    if float(scheduler.gamma) != OFFICIAL_GAMMA:
        difference = (f"{name} gamma is {scheduler.gamma}, published value is {OFFICIAL_GAMMA}",)
        scheduler.official_differences = difference  # type: ignore[attr-defined]
        return difference
    scheduler.official_differences = ()  # type: ignore[attr-defined]
    return ()

def require_disjoint(optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer) -> None:
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("SEMamba++ optimizers must each own parameters")

@dataclass(frozen=True)
class SEMambaSetup:
    optim_g_id: str
    optim_d_id: str
    claim_label: str
    max_logical_steps: int
    accumulation: int
    grad_clip: float | None
    validation_interval: int
    best_checkpoint_start_epoch: int
    max_epochs: int | None
    stop_after_logical_steps: int | None

def require_semamba_setup(
    generator: Any,
    mrd: Any,
    cqt: Any,
    data: Any,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
    scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
    *,
    generator_objective: torch.nn.Module,
    discriminator_objective: torch.nn.Module,
    max_logical_steps: int,
    accumulation: int,
    grad_clip: float | None,
    validation_interval: int,
    best_checkpoint_start_epoch: int,
    max_epochs: int | None,
    stop_after_logical_steps: int | None,
    claim_label: str,
) -> SEMambaSetup:
    require_prepared(generator, SEMambaPPGenerator, "generator must be a prepared SEMambaPPGenerator")
    require_prepared(mrd, SEMambaPPMRD, "mrd must be a prepared SEMambaPPMRD")
    require_prepared(cqt, SEMambaPPCQTDiscriminator, "cqt must be a prepared SEMambaPPCQTDiscriminator")
    if generator.unwrapped().architecture != mrd.unwrapped().architecture:
        raise BuildError("MRD initialization architecture does not match the generator")
    if generator.unwrapped().architecture != cqt.unwrapped().architecture:
        raise BuildError("CQT initialization architecture does not match the generator")
    if generator.unwrapped().seed != mrd.unwrapped().seed or generator.unwrapped().seed != cqt.unwrapped().seed:
        raise BuildError("generator, MRD, and CQT seeds differ")
    require_adamw(optim_g, "optim_g")
    require_adamw(optim_d, "optim_d")
    require_scheduler(scheduler_g, "scheduler_g")
    require_scheduler(scheduler_d, "scheduler_d")
    if not isinstance(generator_objective, torch.nn.Module) or not isinstance(discriminator_objective, torch.nn.Module):
        raise BuildError("SEMamba++ objectives must be loss modules")
    require_disjoint(optim_g, optim_d)
    if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
        raise BuildError("data must provide train_epoch and validation_epoch")
    optim_g_id = getattr(optim_g, "_experiment_component_id", None)
    optim_d_id = getattr(optim_d, "_experiment_component_id", None)
    if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
        raise BuildError("optimizers have no component id")
    if not isinstance(claim_label, str) or not claim_label:
        raise BuildError("claim_label must be a non-empty string")
    if claim_label == "official_reproduction":
        raise BuildError("semambapp.gsr.official is blocked")
    if stop_after_logical_steps is not None:
        stop_after_logical_steps = positive_int("stop_after_logical_steps", stop_after_logical_steps)
    return SEMambaSetup(
        optim_g_id=optim_g_id,
        optim_d_id=optim_d_id,
        claim_label=claim_label,
        max_logical_steps=positive_int("max_logical_steps", max_logical_steps),
        accumulation=positive_int("accumulation", accumulation),
        grad_clip=None if grad_clip is None else positive_float("grad_clip", grad_clip),
        validation_interval=non_negative_int("validation_interval", validation_interval),
        best_checkpoint_start_epoch=non_negative_int(
            "best_checkpoint_start_epoch", best_checkpoint_start_epoch
        ),
        max_epochs=None if max_epochs is None else positive_int("max_epochs", max_epochs),
        stop_after_logical_steps=stop_after_logical_steps,
    )

def require_group_size(data: object, accumulation: int, component_id: str) -> None:
    from src.experiments.errors import RuntimeFailure

    group_size = getattr(data, "group_size", None)
    if group_size is None:
        return
    if int(group_size) != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} "
            f"does not match accumulation {accumulation}"
        )


# --- Step journal. The step save stays in ``_offer_step``. ---

class StepLog:
    def write(
        self,
        progress: SEMambaProgress,
        runtime: ProcessRuntime,
        journal: StepJournal,
        local: dict[str, Any],
        *,
        epoch: int,
    ) -> None:
        shared = _shared_log(runtime, local)
        entry = {
            "adversarial": shared["adversarial"],
            "applied_d": True,
            "applied_g": True,
            "complex": shared["complex"],
            "consistency": shared["consistency"],
            "discriminator": shared["discriminator"],
            "epoch": epoch,
            "generator": shared["generator"],
            "generator_forwards": shared["generator_forwards"],
            "generator_previews": shared["generator_previews"],
            "generator_reforwards": shared["generator_reforwards"],
            "logical_step": progress.logical_step,
            "loss": shared["generator"],
            "lr_d": shared["lr_d"],
            "lr_g": shared["lr_g"],
            "magnitude": shared["magnitude"],
            "matching": shared["matching"],
            "mel": shared["mel"],
            "phase": shared["phase"],
            "sample_id": shared["sample_id"],
            "scan_configured": shared["scan_configured"],
            "scan_executed": shared["scan_executed"],
        }
        progress.step_log.append(entry)
        journal.write_metrics(entry)
        journal.write_sample(
            {
                "logical_step": progress.logical_step,
                "loss": local["generator"],
                "realization_id": local["realization_id"],
                "sample_id": local["sample_id"],
            }
        )

def _shared_log(runtime: ProcessRuntime, local: dict[str, float | str]) -> dict[str, float | str]:
    gathered = runtime.gather_objects(dict(local))
    shared = gathered[0]
    if not isinstance(shared, dict):
        raise RuntimeFailure("rank 0 did not return GAN metrics")
    return shared


# --- Discriminator AdamW, then generator AdamW. ---

class SEMambaUpdate:
    """Own the discriminator-then-generator step and its non-persistent observations."""

    def __init__(
        self,
        *,
        generator: Any,
        mrd: Any,
        cqt: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        generator_objective: Any,
        discriminator_objective: Any,
        accumulation: int,
        grad_clip: float | None,
    ) -> None:
        self.generator = generator
        self.mrd = mrd
        self.cqt = cqt
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.generator_objective = generator_objective
        self.discriminator_objective = discriminator_objective
        self.accumulation = accumulation
        self.grad_clip = grad_clip
        self.capture_gradients = False
        self.frozen_during_generator = False
        self.generator_graph_forwards = 0
        self.generator_preview_forwards = 0
        self.generator_reforward_forwards = 0
        self.mrd_forwards = 0
        self.cqt_forwards = 0
        self.sync_trace: list[str] = []
        self.last_generator_grads: dict[str, torch.Tensor] = {}
        self.last_discriminator_grads: dict[str, torch.Tensor] = {}
        self.scan_report: dict[str, str] | None = None
        self._phase: Any = None

    def refuse_amp(self, runtime: ProcessRuntime) -> None:
        if getattr(runtime, "precision", "fp32") == "fp16":
            raise RuntimeFailure(
                "SEMamba++ official GAN refuses fp16. "
                "The current scaler decides overflow per optimizer, so the discriminator "
                "step and the generator step are not one atomic update. FP32 is unchanged."
            )

    def require_scan(self, runtime: ProcessRuntime) -> None:
        if self.scan_report is None:
            self.scan_report = scan_execution(self.generator.unwrapped().generator, runtime.device)
        report = self.scan_report
        if report["configured"] == "fused" and report["executed"] == "unavailable":
            raise RuntimeFailure(
                "scan_backend fused was requested but mamba-ssm is not importable. "
                "Refusing to run the reference kernel instead."
            )
        if report["configured"] == "fused" and runtime.device.type == "cuda" and report["executed"] != "fused":
            raise RuntimeFailure(
                f"scan_backend fused on CUDA resolved to {report['executed']}, not the fused kernel"
            )

    def run(
        self,
        batches: list[Batch],
        runtime: ProcessRuntime,
        phase_hook: Any,
        *,
        apply_optimizer: bool = True,
    ) -> dict[str, float | str]:
        """Step both optimizers. The caller commits the logical step only after this returns."""

        self._phase = phase_hook
        self.refuse_amp(runtime)
        self.require_scan(runtime)
        if len(batches) < 1 or len(batches) > self.accumulation:
            raise RuntimeFailure(
                f"official step received {len(batches)} micro-batches, accumulation is {self.accumulation}"
            )
        row_counts = [micro_rows([batch]) for batch in batches]
        scale = window_scale(
            row_counts,
            gather_row_counts(runtime, row_counts),
            accumulation=self.accumulation,
            world_size=int(runtime.world_size),
        )
        counts = runtime.gather_objects(len(batches))
        if any(item != len(batches) for item in counts):
            raise RuntimeFailure("ranks disagree on the number of micro-batches")
        self.generator.train()
        self.mrd.train()
        self.cqt.train()
        graph_before = self.generator_graph_forwards
        preview_before = self.generator_preview_forwards
        reforward_before = self.generator_reforward_forwards
        self._phase("zero_grad")
        self.optim_d.zero_grad(set_to_none=True)
        reuse_graph = len(batches) == 1
        cached: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]] = []
        discriminator_total = None
        sample_ids: list[str] = []
        realization_ids: list[str] = []
        for index, batch in enumerate(batches):
            last = index == len(batches) - 1
            clean, noisy = _waveforms(batch, runtime)
            sample_ids.append(batch.sample_id)
            realization_ids.append(batch.realization_id)
            ready = _inputs_ready(clean, noisy)
            if not runtime.agree(ready, policy="all"):
                raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
            if reuse_graph:
                self.sync_trace.append("generator_forward_sync")
                with runtime.sync_gradients([self.generator], enabled=True):
                    parts = self._generator_forward(runtime, clean, noisy, preview=False)
                self.generator_graph_forwards += 1
                audio = parts["audio"]
                cached.append((clean, noisy, parts))
            else:
                self.sync_trace.append("generator_preview_nograd")
                with torch.no_grad():
                    preview = self._generator_forward(runtime, clean, noisy, preview=True)
                self.generator_preview_forwards += 1
                audio = preview["audio"]
                cached.append((clean, noisy, None))
            self.sync_trace.append("discriminator_sync" if last else "discriminator_nosync")
            with runtime.sync_gradients([self.mrd, self.cqt], enabled=last):
                loss = self._discriminator_loss(runtime, clean, audio.detach())
                _backward_if_finite(runtime, weight_loss(loss, scale, index), "discriminator loss")
            term = loss.detach()
            discriminator_total = term if discriminator_total is None else discriminator_total + term
            self.mrd_forwards += 1
            self.cqt_forwards += 1
        if self.grad_clip is not None:
            runtime.clip(self.mrd, self.grad_clip)
            runtime.clip(self.cqt, self.grad_clip)
        disc_norm = _grad_norm(self.mrd.unwrapped().discriminator, self.cqt.unwrapped().discriminator)
        if self.capture_gradients:
            self.last_discriminator_grads = _clone_grads(
                self.mrd.unwrapped().discriminator, self.cqt.unwrapped().discriminator
            )
        self._phase("before_discriminator_step")
        if apply_optimizer:
            discriminator_result = runtime.step(self.optim_d)
        self._phase("after_discriminator_step")
        frozen = [
            parameter
            for module in (self.mrd.unwrapped(), self.cqt.unwrapped())
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.optim_g.zero_grad(set_to_none=True)
        component_totals: dict[str, Any] = {}
        generator_total: Any = 0.0
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            self.frozen_during_generator = True
            self._phase("before_generator_backward")
            for index, (clean, noisy, parts) in enumerate(cached):
                last = index == len(cached) - 1
                if parts is None:
                    self.sync_trace.append("generator_reforward_sync" if last else "generator_reforward_nosync")
                    with runtime.sync_gradients([self.generator], enabled=last):
                        parts = self._generator_forward(runtime, clean, noisy, preview=False)
                        self.generator_reforward_forwards += 1
                        generator_total += self._generator_backward(
                            runtime, clean, parts, component_totals, scale, index
                        )
                else:
                    self.sync_trace.append("generator_backward_retained")
                    generator_total += self._generator_backward(
                        runtime, clean, parts, component_totals, scale, index
                    )
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
            self.frozen_during_generator = False
        if self.grad_clip is not None:
            runtime.clip(self.generator, self.grad_clip)
        log_terms = torch.stack(
            [
                value.detach() if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=runtime.device)
                for value in component_totals.values()
            ]
        )
        if discriminator_total is None or not isinstance(generator_total, torch.Tensor):
            raise RuntimeFailure("generator loss is missing")
        log_pair = torch.stack([discriminator_total.detach(), generator_total.detach()])
        if runtime.device.type == "cuda":
            host_terms = torch.empty(log_terms.shape, dtype=torch.float32, pin_memory=True)
            host_pair = torch.empty(log_pair.shape, dtype=torch.float32, pin_memory=True)
            host_terms.copy_(log_terms.detach().float(), non_blocking=True)
            host_pair.copy_(log_pair.detach().float(), non_blocking=True)
        else:
            host_terms = log_terms.detach().float().cpu()
            host_pair = log_pair.detach().float().cpu()
        gen_norm = _grad_norm(self.generator.unwrapped().generator)
        if self.capture_gradients:
            self.last_generator_grads = _clone_grads(self.generator.unwrapped().generator)
        self._phase("before_generator_step")
        if not apply_optimizer:
            return {
                "generator": float(generator_total.detach()),
                "sample_id": ",".join(sample_ids),
            }
        generator_result = runtime.step(self.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.mrd)
        runtime.broadcast_buffers(self.cqt)
        if not discriminator_result.applied or not generator_result.applied:
            raise RuntimeFailure("SEMamba++ requires both optimizers to apply in one logical step")
        count = len(batches)
        report = self.scan_report or {"configured": "", "executed": ""}
        self._phase("end_step")
        named = {
            name: float(number) for name, number in zip(component_totals, host_terms.tolist(), strict=True)
        }
        discriminator_value = float(host_pair[0])
        generator_value = float(host_pair[1])
        return {
            "adversarial": named["adversarial"] / count,
            "complex": named["complex"] / count,
            "consistency": named["consistency"] / count,
            "discriminator": discriminator_value / count,
            "discriminator_grad_norm": disc_norm,
            "generator": generator_value / count,
            "generator_forwards": float(self.generator_graph_forwards - graph_before),
            "generator_grad_norm": gen_norm,
            "generator_previews": float(self.generator_preview_forwards - preview_before),
            "generator_reforwards": float(self.generator_reforward_forwards - reforward_before),
            "lr_d": float(self.optim_d.param_groups[0]["lr"]),
            "lr_g": float(self.optim_g.param_groups[0]["lr"]),
            "magnitude": named["magnitude"] / count,
            "matching": named["matching"] / count,
            "mel": named["mel"] / count,
            "phase": named["phase"] / count,
            "realization_id": ",".join(realization_ids),
            "sample_id": ",".join(sample_ids),
            "scan_configured": report["configured"],
            "scan_executed": report["executed"],
        }

    def _generator_forward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        *,
        preview: bool,
    ) -> dict[str, torch.Tensor]:
        module = self.generator.unwrapped() if preview else self.generator.execution()
        error: str | None = None
        parts: dict[str, torch.Tensor] | None = None
        try:
            parts = pack_forward(module(clean, noisy))
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
        if parts is None:
            raise RuntimeFailure("generator forward produced no tensors")
        return parts

    def _discriminator_loss(self, runtime: ProcessRuntime, clean: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        error: str | None = None
        loss: torch.Tensor | None = None
        try:
            real_q, fake_q, _real_qf, _fake_qf = self.cqt(clean, audio)
            real_r, fake_r, _real_rf, _fake_rf = self.mrd(clean, audio)
            loss = call_objective(
                self.discriminator_objective,
                {
                    "fake_cqt": fake_q,
                    "fake_mrd": fake_r,
                    "real_cqt": real_q,
                    "real_mrd": real_r,
                },
            ).total
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("discriminator forward failed: " + "; ".join(item for item in errors if item))
        if loss is None:
            raise RuntimeFailure("discriminator loss is missing")
        return loss

    def _generator_backward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        parts: dict[str, torch.Tensor],
        components: dict[str, Any],
        scale: int | list[float],
        offset: int,
    ) -> torch.Tensor:
        error: str | None = None
        total: torch.Tensor | None = None
        terms: dict[str, torch.Tensor] | None = None
        try:
            band = self.cqt.unwrapped().discriminator
            mrd = self.mrd.unwrapped().discriminator
            _real_q, fake_q, real_qf, fake_qf = band(clean, parts["audio"])
            _real_r, fake_r, real_rf, fake_rf = mrd(clean, parts["audio"])
            result = call_objective(
                self.generator_objective,
                {
                    "clean_complex": parts["clean_complex"],
                    "clean_mag": parts["clean_mag"],
                    "clean_phase": parts["clean_phase"],
                    "estimate": parts["audio"],
                    "fake_cqt": fake_q,
                    "fake_cqt_features": fake_qf,
                    "fake_mrd": fake_r,
                    "fake_mrd_features": fake_rf,
                    "generated_complex": parts["complex"],
                    "generated_mag": parts["mag"],
                    "generated_phase": parts["phase"],
                    "real_cqt_features": real_qf,
                    "real_mrd_features": real_rf,
                    "recomputed_complex": parts["recomputed"],
                    "reference": clean,
                    "sample_rate": int(self.generator.unwrapped().sampling_rate),
                },
            )
            total = result.total
            terms = result.terms
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator loss failed: " + "; ".join(item for item in errors if item))
        if total is None or terms is None:
            raise RuntimeFailure("generator loss is missing")
        _backward_if_finite(runtime, weight_loss(total, scale, offset), "generator loss")
        for name, value in terms.items():
            current = components.get(name, 0.0)
            components[name] = current + value.detach()
        return total.detach()

def _waveforms(batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
    batch = runtime.move(batch)
    clean = batch.targets
    noisy = batch.inputs
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    return clean, noisy

def _backward_if_finite(runtime: ProcessRuntime, loss: torch.Tensor, what: str) -> None:
    if runtime.device.type != "cuda":
        finite = bool(torch.isfinite(loss.detach()).all())
        if not runtime.agree(finite, policy="all"):
            raise RuntimeFailure(f"{what} is not finite on every rank")
    runtime.backward(loss)

def _grad_norm(*modules: torch.nn.Module) -> float:
    pieces = [
        parameter.grad.detach().square().sum()
        for module in modules
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not pieces:
        return 0.0
    values = torch.stack(pieces).detach().cpu().tolist()
    return sum(float(value) for value in values) ** 0.5

def _clone_grads(*modules: torch.nn.Module) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for module in modules:
        prefix = module.__class__.__name__
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            cloned[f"{prefix}.{name}"] = parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
    return cloned

def _inputs_ready(clean: torch.Tensor, noisy: torch.Tensor) -> bool:
    if clean.ndim != 2 or clean.shape != noisy.shape:
        return False
    if int(clean.shape[-1]) <= 1024:
        return False
    if clean.is_cuda:
        return True
    return bool(torch.isfinite(clean).all()) and bool(torch.isfinite(noisy).all())


# --- Boundary: optional validation, then both ExponentialLR steps. ---

@dataclass(frozen=True)
class BoundaryOutcome:
    trainer_epoch: int
    metrics: dict[str, float]
    validated: bool

class SEMambaBoundary:
    """Validate when the boundary asks, then step both ExponentialLR schedulers together."""

    def __init__(self, validator: SEMambaValidator, *, optim_g: Any, optim_d: Any, scheduler_g: Any, scheduler_d: Any) -> None:
        self.validator = validator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d

    def run(
        self,
        progress: SEMambaProgress,
        runtime: ProcessRuntime,
        data: Any,
        component_id: str,
        on_phase: Any,
    ) -> BoundaryOutcome:
        if progress.phase != PENDING_BOUNDARY or progress.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending GAN boundary")
        boundary = progress.boundary
        trainer_epoch = int(boundary["trainer_epoch"])
        metrics: dict[str, float] = {}
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self.validator.evaluate(runtime, data, trainer_epoch, component_id)
            metrics = scalar_metrics(summary)
            progress.history.append(
                {
                    "epoch": trainer_epoch,
                    "logical_step": int(progress.logical_step),
                    "reason": "semambapp_does_not_select_best_generator",
                    "selected": False,
                    "summary": summary,
                }
            )
            boundary["validation_done"] = True
            boundary["selection_done"] = True
            if boundary["scheduler"]:
                progress.phase = "pending_scheduler"
                on_phase(progress.phase)
        if boundary["scheduler"] and not boundary["scheduler_done"]:
            self.scheduler_g.step()
            self.scheduler_d.step()
            progress.scheduler_steps += 1
            boundary["scheduler_done"] = True
            boundary["lr_g"] = float(self.optim_g.param_groups[0]["lr"])
            boundary["lr_d"] = float(self.optim_d.param_groups[0]["lr"])
        if boundary["epoch_done"]:
            data.epoch = trainer_epoch + 1
            data.update_index = 0
        boundary["closed"] = True
        progress.close_boundary(boundary)
        on_phase(progress.phase)
        return BoundaryOutcome(trainer_epoch, metrics, bool(boundary["validate"]))

def published_metrics(
    outcome: BoundaryOutcome,
    runtime: ProcessRuntime,
    *,
    research: Any,
    lifecycle: Any,
    logical_step: int,
    training_control: Any,
) -> dict[str, float]:
    """Attach the research score after validation and both schedulers have finished."""

    if outcome.validated:
        observe_validation(
            training_control,
            outcome.metrics,
            step=logical_step,
            scoped=research.attached,
        )
    extra = research.score(
        runtime,
        output_directory=lifecycle.output_directory,
        config_identity=lifecycle.config_identity,
        kind="boundary",
        epoch=outcome.trainer_epoch,
        logical_step=logical_step,
        official=outcome.metrics if outcome.validated else {},
        stopping=training_control,
    )
    return {**outcome.metrics, **extra} if extra else outcome.metrics
