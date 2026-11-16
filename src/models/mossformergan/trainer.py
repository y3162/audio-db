"""MossFormerGAN training loop.

Read this file from ``MossFormerGANTrainer.fit`` downward. ``ProgressView``
is defined first because the trainer class extends it.

``fit`` walks micro-batches. ``consume_micro_batch`` is the entry. A missing
PESQ target skips that micro-batch and still advances the cursor. Both
backwards finish before the generator AdamW and then the discriminator AdamW.
Gradients are checkpointed only while the accumulation window is open. At the
epoch boundary the equal-weight validation loss drives the plateau: five
stalls restore the best weights and halve the generator learning rate, and
ten stalls set ``halt_reason``. Counters live on ``MossProgress``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data, check_publication
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.models.mossformergan import loss as moss_loss
from src.models.mossformergan.evaluator import (
    OfficialCall,
    evaluate,
    mossformer_validation_decision,
    validation_proxy,
)
from src.models.mossformergan.loss import OFFICIAL_ADAMW_WEIGHT_DECAY, generator_parts
from src.models.mossformergan.model import (
    MossFormerGANDiscriminator,
    MossFormerGANGenerator,
    require_official_architecture,
)
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.selectors import scalar_metrics
from src.experiments.stopping import control_requests_stop, observe_validation, should_stop
from src.experiments.trainers import _close_epoch_view
from src.experiments.training.control import (
    connected_zero,
    finite_gradient_report,
    gather_row_counts,
    planned_row_counts,
    restore_fields,
    snapshot_fields,
    stop_requested,
    weight_loss,
    window_divisor,
    window_scale,
)
from src.experiments.training.objectives import call_objective
from src.experiments.training.official_objectives import require_mossformer

# --- Counters the trainer exposes from the progress object it owns. ---

class ProgressView:
    """Read the persistent counters from the progress object that owns them."""

    @property
    def logical_step(self) -> int:
        return self.progress.logical_step

    @property
    def optimizer_updates(self) -> dict[str, int]:
        return self.progress.optimizer_updates

    @property
    def phase(self) -> str:
        return self.progress.phase

    @property
    def halt_reason(self) -> str:
        return self.progress.halt_reason

    @property
    def stop_training(self) -> bool:
        return self.progress.stop_training

    @property
    def counters(self) -> dict[str, int]:
        return self.progress.counters

    @property
    def _accu_count(self) -> int:
        return self.progress.accu_count

    @property
    def _average_calls(self) -> int:
        return self._micro.average_calls

    @property
    def _stalls(self) -> int:
        return self.progress.stalls

    @property
    def _in_optimizer_pair(self) -> bool:
        return self._guard.active

    @_in_optimizer_pair.setter
    def _in_optimizer_pair(self, value: bool) -> None:
        self._guard.active = bool(value)


OFFICIAL_FAITHFUL = "official_faithful"
OFFICIAL_CLIP = 10.0

# --- Trainer. ``fit`` is the timeline. ``consume_micro_batch`` is the entry. ---

class MossFormerGANTrainer(ProgressView):
    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        generator: PreparedModel,
        discriminator: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        *,
        generator_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        max_logical_steps: int,
        accumulation: int = 2,
        grad_clip: float = OFFICIAL_CLIP,
        experiment_claim: str = OFFICIAL_FAITHFUL,
        validate_on_epoch_end: bool = True,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        require_prepared(generator, MossFormerGANGenerator, "generator must be a prepared MossFormerGANGenerator")
        require_prepared(discriminator, MossFormerGANDiscriminator, "discriminator must be a prepared MossFormerGANDiscriminator")
        if generator.unwrapped().architecture != discriminator.unwrapped().architecture:
            raise BuildError("discriminator initialization architecture does not match the generator")
        if generator.unwrapped().seed != discriminator.unwrapped().seed:
            raise BuildError("generator and discriminator seeds differ")
        require_adamw(optim_g, "optim_g", lr=OFFICIAL_LEARNING_RATE)
        require_adamw(optim_d, "optim_d", lr=OFFICIAL_DISCRIMINATOR_LEARNING_RATE)
        require_disjoint(optim_g, optim_d)
        if experiment_claim not in CLAIMS:
            raise BuildError("experiment_claim must be official_faithful, leakage_safe, or research")
        generator.unwrapped().official_differences = (
            *require_official_architecture(generator.unwrapped().architecture, experiment_claim),
            *require_mossformer(generator_objective, discriminator_objective, experiment_claim),
        )
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        optim_g_id = getattr(optim_g, "_experiment_component_id", None)
        optim_d_id = getattr(optim_d, "_experiment_component_id", None)
        if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
            raise BuildError("optimizers have no component id")
        if not isinstance(validate_on_epoch_end, bool):
            raise BuildError("validate_on_epoch_end must be a boolean")
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.experiment_claim = experiment_claim
        self.max_logical_steps = positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = positive_int("accumulation", accumulation)
        self.grad_clip = positive_float("grad_clip", grad_clip)
        self.validate_on_epoch_end = validate_on_epoch_end
        self.max_epochs = None if max_epochs is None else positive_int("max_epochs", max_epochs)
        self.stop_after_logical_steps = (
            None if stop_after_logical_steps is None else positive_int("stop_after_logical_steps", stop_after_logical_steps)
        )
        self.n_fft = int(generator.unwrapped().n_fft)
        self.win = int(generator.unwrapped().win)
        self.hop = int(generator.unwrapped().hop)
        self.sample_rate = int(generator.unwrapped().sampling_rate)
        self.audio_sample_rate = self.sample_rate
        self.audio_teachers = ("dry_clean",)
        self.progress = MossProgress.initial(optim_g_id, optim_d_id, experiment_claim)
        self._guard = PairGuard()
        self._micro = MicroBatch(
            generator=generator, discriminator=discriminator, optim_g=optim_g, optim_d=optim_d,
            generator_objective=generator_objective, discriminator_objective=discriminator_objective,
            progress=self.progress, guard=self._guard, accumulation=self.accumulation, grad_clip=self.grad_clip,
            n_fft=self.n_fft, hop=self.hop, win=self.win, sample_rate=self.sample_rate,
        )
        self._restored = False
        self._skip_path: Path | None = None
        self.fail_phase: str | None = None
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators, model_selection, data=self.data, module=self.generator,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection

    def fit(self) -> RunResult:
        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        while True:
            if self.progress.phase == "pending_boundary":
                self._finish_boundary(runtime, component_id)
                if should_stop(self) or self.progress.stop_training:
                    return self._result("completed")
                continue
            if self.progress.logical_step >= self.max_logical_steps:
                return self._result("completed")
            if self._stop_requested(runtime):
                self._publish_interrupt(runtime)
                return self._result("interrupted")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            if self.max_epochs is not None and epoch >= self.max_epochs and index == 0:
                self.progress.halt_reason = "max_epochs"
                return self._result("completed")
            opened = open_micro_batch(
                self.data, runtime, self.progress, accumulation=self.accumulation, component_id=component_id,
                epoch=epoch, index=index, open_boundary=lambda: self._open_boundary(runtime, epoch),
            )
            if opened is None:
                continue
            view, batch, updates = opened
            try:
                if int(self.progress.accu_count) == 0:
                    remaining = int(updates) - index
                    if remaining >= self.accumulation or remaining < 1:
                        planned = self.accumulation
                    else:
                        planned = remaining
                    counts = planned_row_counts(view, index, planned)
                    if counts is None:
                        self._micro.loss_scale = None
                    else:
                        resolved = window_scale(
                            counts,
                            gather_row_counts(runtime, counts),
                            accumulation=self.accumulation,
                            world_size=int(runtime.world_size),
                        )
                        self._micro.loss_scale = None if isinstance(resolved, int) else resolved
                stepped = self.consume_micro_batch(
                    batch, runtime, update_index=index, updates=updates,
                )
                index += 1
                self.data.epoch = epoch
                self.data.update_index = index
                note_micro_batch(
                    self.progress, runtime, self.journal, stepped, epoch=epoch, research=self.research,
                    lifecycle=self.lifecycle, training_control=getattr(self, "training_control", None),
                    offer=lambda event, metrics=None: self._offer(runtime, event, metrics),
                )
                if index == updates:
                    self._open_boundary(runtime, epoch)
            finally:
                _close_epoch_view(view)

    def preflight_probe(self) -> dict[str, Any]:
        """One real micro-batch backward. The optimizer pair does not step."""

        runtime = getattr(self, "runtime", None)
        if not isinstance(runtime, ProcessRuntime):
            raise BuildError("preflight has no runtime")
        component_id = str(getattr(self, "_experiment_component_id", "trainer"))
        saved_progress = snapshot_fields(self.progress)
        saved_micro = snapshot_fields(self._micro)
        was_training = bool(self.generator.training)
        view = None
        try:
            index = int(self.data.update_index)
            view = self.data.train_epoch(int(self.data.epoch), start_update=index)
            updates = int(view.updates)
            if updates < 1 or index >= updates:
                raise RuntimeFailure(f"component {component_id}: preflight has no micro-batch")
            report = None
            while index < updates and report is None:
                batches = view.batch(index)
                if len(batches) != 1 or not isinstance(batches[0], Batch):
                    raise RuntimeFailure(f"component {component_id}: expected one micro-batch")
                batch = batches[0]
                self.consume_micro_batch(
                    batch, runtime, update_index=index, updates=updates, apply_optimizer=False,
                )
                components = self._micro.last_components
                if components:
                    report = finite_gradient_report(
                        [self.generator, self.discriminator],
                        loss=float(components["generator"]),
                        sample_id=str(batch.sample_id),
                        component_id=component_id,
                    )
                index += 1
            if report is None:
                # Every micro-batch in this epoch was skipped, so there is no backward.
                # The run itself records that; the probe must not turn it into a failure.
                return {
                    "finite": False,
                    "loss": None,
                    "parameters_missing_grad": 0,
                    "parameters_with_grad": 0,
                    "sample_id": "",
                    "status": "skipped",
                }
            return report
        finally:
            restore_fields(self.progress, saved_progress)
            restore_fields(self._micro, saved_micro)
            if not was_training:
                self.generator.eval()
            if view is not None:
                _close_epoch_view(view)

    def consume_micro_batch(
        self,
        batch: Batch,
        runtime: ProcessRuntime,
        *,
        update_index: int,
        updates: int | None = None,
        apply_optimizer: bool = True,
    ) -> dict[str, Any] | None:
        """Run one micro-batch. Return the shared log when both optimizers stepped."""

        return self._micro.consume(
            batch, runtime, update_index=update_index, updates=updates, apply_optimizer=apply_optimizer,
            phase_hook=self._on_update_phase,
            step_pair=self._step_pair, journal=getattr(self, "journal", None), skip_path=self._skip_path,
        )

    def _step_pair(self, runtime: ProcessRuntime) -> dict[str, Any]:
        return self._micro._step_pair(runtime, self._on_update_phase)

    def _terms_view(self) -> Any:
        return self._micro._terms_view()

    def _validation_proxy(self, runtime: ProcessRuntime) -> Any:
        return validation_proxy(
            runtime, self.generator, self.discriminator,
            n_fft=self.n_fft, hop=self.hop, win=self.win, sample_rate=self.sample_rate,
        )

    def _apply_plateau(self, summary: dict[str, Any]) -> str:
        return apply_plateau(
            self.progress, summary, generator=self.generator, discriminator=self.discriminator,
            optim_g=self.optim_g, optim_d=self.optim_d,
        )

    def _check_claim(self) -> None:
        check_publication(self)

    def state_dict(self) -> dict[str, Any]:
        return self.progress.export(grad_state(self.generator), grad_state(self.discriminator))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        decoded = self.progress.decode(state, self.accumulation)
        generator_grads = check_grads(self.generator, decoded.generator_grads)
        discriminator_grads = check_grads(self.discriminator, decoded.discriminator_grads)
        self.progress.apply(decoded)
        write_grads(self.generator, generator_grads)
        write_grads(self.discriminator, discriminator_grads)
        self._restored = True

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        self.research.rewind(
            self.lifecycle.output_directory, restored=self._restored, phase=self.progress.phase,
            progress={"logical_step": int(self.progress.logical_step)},
        )
        refuse_amp(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.journal.begin(
            restored=self._restored, logical_step=self.progress.logical_step,
            expect_rows=len(self.progress.step_log) if self._restored else None,
        )
        self._skip_path = self.journal.begin_sidecar(
            "skip_log.jsonl", restored=self._restored,
            keep_lines=len(self.progress.skip_log) if self._restored else None,
        )
        if not self._restored:
            self.progress.reset_fresh()
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        require_group_size(self.data, self.accumulation, str(component_id))
        runtime.barrier()
        return str(component_id)

    def _open_boundary(self, runtime: ProcessRuntime, epoch: int) -> None:
        open_boundary(self.progress, epoch=epoch, validate=self.validate_on_epoch_end)
        self._offer(runtime, "boundary_pending")

    def _finish_boundary(self, runtime: ProcessRuntime, component_id: str) -> None:
        finish_boundary(
            self.progress, runtime, self.data, self._micro, component_id=component_id,
            max_logical_steps=self.max_logical_steps, n_fft=self.n_fft, hop=self.hop, win=self.win,
            sample_rate=self.sample_rate, offer=lambda event, metrics=None: self._offer(runtime, event, metrics),
            research=self.research, lifecycle=self.lifecycle,
            training_control=getattr(self, "training_control", None),
        )

    def _on_update_phase(self, phase: str) -> None:
        if self.fail_phase == phase:
            raise RuntimeFailure(f"injected failure at {phase}")

    def _refuse_amp(self, runtime: ProcessRuntime) -> None:
        refuse_amp(runtime)

    def _publish_interrupt(self, runtime: ProcessRuntime) -> None:
        if self._guard.active:
            raise RuntimeFailure("refusing to checkpoint between the generator step and the discriminator step")
        if self.progress.logical_step <= 0 and self.progress.accu_count <= 0:
            return
        self._offer(runtime, "stop")

    def _offer(self, runtime: ProcessRuntime, event: str, metrics: dict[str, float] | None = None) -> None:
        if self._guard.active:
            raise RuntimeFailure("refusing to checkpoint between the generator step and the discriminator step")
        if event in ("step", "partial"):
            self.lifecycle.save_if_needed(
                event, logical_step=self.progress.logical_step,
                metrics=metrics if event == "step" else None, prepare=self._sync_ranks,
            )
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.progress.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        share_window(runtime, self.generator, self.discriminator, self.progress)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        return stop_requested(self, runtime)

    def _result(self, reason: str) -> RunResult:
        return finish_run(
            reason, research=self.research, runtime=self.runtime, lifecycle=self.lifecycle, data=self.data,
            progress=self.progress, training_control=getattr(self, "training_control", None),
            prepare=self._sync_ranks,
        )


# --- Checkpoint progress, including gradients captured inside an open window. ---

IN_EPOCH = "in_epoch"

PENDING_BOUNDARY = "pending_boundary"

_PHASES = frozenset({IN_EPOCH, PENDING_BOUNDARY})

@dataclass
class _Decoded:
    logical_step: int
    optimizer_updates: dict[str, int]
    counters: dict[str, int]
    accu_count: int
    phase: str
    boundary: dict[str, Any] | None
    plateau_history: list[Any]
    completed_boundaries: list[Any]
    step_log: list[Any]
    skip_log: list[Any]
    best_val: float
    stalls: int
    best_snapshot: Any
    stop_training: bool
    halt_reason: str
    epoch_success_start: int
    epoch_update_start: int
    epochs_without_step: int
    window_generator: list[float]
    window_discriminator: list[float]
    window_samples: list[str]
    generator_grads: dict[str, Any]
    discriminator_grads: dict[str, Any]

@dataclass
class MossProgress:
    """The checkpoint fields for one MossFormerGAN run. Partial gradients stay on the parameters."""

    accu_count: int
    best_snapshot: dict[str, Any] | None
    best_val: float
    boundary: dict[str, Any] | None
    completed_boundaries: list[dict[str, Any]]
    counters: dict[str, int]
    epoch_success_start: int
    epoch_update_start: int
    epochs_without_step: int
    experiment_claim: str
    halt_reason: str
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    plateau_history: list[dict[str, Any]]
    skip_log: list[dict[str, Any]]
    stalls: int
    step_log: list[dict[str, Any]]
    stop_training: bool
    window_discriminator: list[float]
    window_generator: list[float]
    window_samples: list[str]
    _optimizer_ids: tuple[str, str]
    last_skip: dict[str, Any] | None = None

    @classmethod
    def initial(cls, optim_g_id: str, optim_d_id: str, experiment_claim: str) -> MossProgress:
        return cls(
            accu_count=0,
            best_snapshot=None,
            best_val=float("inf"),
            boundary=None,
            completed_boundaries=[],
            counters=empty_counters(),
            epoch_success_start=0,
            epoch_update_start=0,
            epochs_without_step=0,
            experiment_claim=experiment_claim,
            halt_reason="initialized",
            logical_step=0,
            optimizer_updates={optim_g_id: 0, optim_d_id: 0},
            phase=IN_EPOCH,
            plateau_history=[],
            skip_log=[],
            stalls=0,
            step_log=[],
            stop_training=False,
            window_discriminator=[],
            window_generator=[],
            window_samples=[],
            _optimizer_ids=(optim_g_id, optim_d_id),
        )

    def export(self, generator_grads: dict[str, torch.Tensor], discriminator_grads: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            "accu_count": int(self.accu_count),
            "best_snapshot": self.best_snapshot,
            "best_val": self.best_val,
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "counters": dict(self.counters),
            "discriminator_grads": discriminator_grads if self.accu_count else {},
            "epoch_success_start": int(self.epoch_success_start),
            "epoch_update_start": int(self.epoch_update_start),
            "epochs_without_step": int(self.epochs_without_step),
            "experiment_claim": self.experiment_claim,
            "generator_grads": generator_grads if self.accu_count else {},
            "halt_reason": self.halt_reason,
            "logical_step": int(self.logical_step),
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "plateau_history": list(self.plateau_history),
            "skip_log": list(self.skip_log),
            "stalls": int(self.stalls),
            "step_log": list(self.step_log),
            "stop_training": bool(self.stop_training),
            "window_discriminator": list(self.window_discriminator),
            "window_generator": list(self.window_generator),
            "window_samples": list(self.window_samples),
        }

    def decode(self, state: dict[str, Any], accumulation: int) -> _Decoded:
        return _decode(state, self.experiment_claim, self._optimizer_ids, accumulation)

    def apply(self, decoded: _Decoded) -> None:
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.counters = decoded.counters
        self.accu_count = decoded.accu_count
        self.phase = decoded.phase
        self.boundary = decoded.boundary
        self.plateau_history = decoded.plateau_history
        self.completed_boundaries = decoded.completed_boundaries
        self.step_log = decoded.step_log
        self.skip_log = decoded.skip_log
        self.last_skip = None if not decoded.skip_log else dict(decoded.skip_log[-1])
        self.best_val = decoded.best_val
        self.stalls = decoded.stalls
        self.best_snapshot = decoded.best_snapshot
        self.stop_training = decoded.stop_training
        self.halt_reason = decoded.halt_reason
        self.epoch_success_start = decoded.epoch_success_start
        self.epoch_update_start = decoded.epoch_update_start
        self.epochs_without_step = decoded.epochs_without_step
        self.window_generator = decoded.window_generator
        self.window_discriminator = decoded.window_discriminator
        self.window_samples = decoded.window_samples

    def reset_fresh(self) -> None:
        optim_g_id, optim_d_id = self._optimizer_ids
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.counters = empty_counters()
        self.accu_count = 0
        self.skip_log = []
        self.last_skip = None
        self.step_log = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH
        self.best_val = float("inf")
        self.stalls = 0
        self.best_snapshot = None
        self.stop_training = False
        self.halt_reason = "running"
        self.plateau_history = []
        self.epoch_success_start = 0
        self.epoch_update_start = 0
        self.epochs_without_step = 0
        self.clear_window()

    def clear_window(self) -> None:
        self.window_generator = []
        self.window_discriminator = []
        self.window_samples = []

    def note_closed_pair(self) -> None:
        self.logical_step += 1

def grad_state(prepared: PreparedModel) -> dict[str, torch.Tensor]:
    grads = {}
    for name, parameter in prepared.unwrapped().named_parameters():
        if parameter.grad is not None:
            grads[name] = parameter.grad.detach().cpu().clone()
    return grads

def check_grads(prepared: PreparedModel, grads: object) -> dict[str, torch.Tensor]:
    if not isinstance(grads, dict):
        raise RuntimeFailure("checkpoint gradients are not a mapping")
    named = dict(prepared.unwrapped().named_parameters())
    checked: dict[str, torch.Tensor] = {}
    for name, value in grads.items():
        if name not in named or not isinstance(value, torch.Tensor):
            raise RuntimeFailure(f"checkpoint gradient {name!r} does not match a parameter")
        parameter = named[name]
        if tuple(value.shape) != tuple(parameter.shape):
            raise RuntimeFailure(f"checkpoint gradient {name} has the wrong shape")
        checked[name] = value
    return checked

def write_grads(prepared: PreparedModel, grads: dict[str, torch.Tensor]) -> None:
    named = dict(prepared.unwrapped().named_parameters())
    for parameter in named.values():
        parameter.grad = None
    for name, value in grads.items():
        parameter = named[name]
        parameter.grad = value.detach().to(device=parameter.device, dtype=parameter.dtype)

def _decode(
    state: dict[str, Any],
    experiment_claim: str,
    optimizer_ids: tuple[str, str],
    accumulation: int,
) -> _Decoded:
    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    counters = state.get("counters")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if not isinstance(counters, dict) or phase not in _PHASES:
        raise RuntimeFailure("trainer counters or phase are invalid")
    if state.get("experiment_claim") != experiment_claim:
        raise RuntimeFailure("experiment claim does not match this trainer")
    if state.get("stop_training") not in {True, False}:
        raise RuntimeFailure("trainer stop flag is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    if any(name not in converted for name in optimizer_ids):
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer ids")
    restored_counters = empty_counters()
    for key in COUNTER_KEYS:
        restored_counters[key] = int(counters.get(key) or 0)
    accu = state.get("accu_count")
    if isinstance(accu, bool) or not isinstance(accu, int) or accu < 0 or accu >= accumulation:
        raise RuntimeFailure("accumulator count is invalid")
    generator_grads = state.get("generator_grads") or {}
    discriminator_grads = state.get("discriminator_grads") or {}
    if accu == 0 and (generator_grads or discriminator_grads):
        raise RuntimeFailure("a closed window cannot carry gradients")
    boundary = state.get("boundary")
    history = state.get("plateau_history")
    completed = state.get("completed_boundaries")
    step_log = state.get("step_log")
    skip_log = state.get("skip_log")
    if not all(isinstance(item, list) for item in (history, completed, step_log, skip_log)):
        raise RuntimeFailure("trainer history is invalid")
    window_generator = [float(item) for item in state.get("window_generator") or []]
    window_discriminator = [float(item) for item in state.get("window_discriminator") or []]
    samples = state.get("window_samples") or []
    if not isinstance(samples, list) or not all(isinstance(item, str) for item in samples):
        raise RuntimeFailure("window sample ids are invalid")
    if accu != len(window_generator) or accu != len(window_discriminator):
        raise RuntimeFailure("window losses do not match the accumulator")
    if len(samples) != accu:
        raise RuntimeFailure("window sample ids do not match the accumulator")
    halt = state.get("halt_reason")
    return _Decoded(
        logical_step=logical_step,
        optimizer_updates=converted,
        counters=restored_counters,
        accu_count=accu,
        phase=str(phase),
        boundary=None if boundary is None else dict(boundary),
        plateau_history=list(history),
        completed_boundaries=list(completed),
        step_log=list(step_log),
        skip_log=list(skip_log),
        best_val=float(state.get("best_val", float("inf"))),
        stalls=int(state.get("stalls") or 0),
        best_snapshot=state.get("best_snapshot"),
        stop_training=bool(state.get("stop_training")),
        halt_reason=halt if isinstance(halt, str) else "resumed",
        epoch_success_start=int(state.get("epoch_success_start") or 0),
        epoch_update_start=int(state.get("epoch_update_start") or 0),
        epochs_without_step=int(state.get("epochs_without_step") or 0),
        window_generator=window_generator,
        window_discriminator=window_discriminator,
        window_samples=list(samples),
        generator_grads=generator_grads,
        discriminator_grads=discriminator_grads,
    )


# --- Constructor checks. The two AdamW learning rates stay distinct. ---

OFFICIAL_FAITHFUL = "official_faithful"

LEAKAGE_SAFE = "leakage_safe"

CLAIMS = frozenset({OFFICIAL_FAITHFUL, LEAKAGE_SAFE, "research"})

OFFICIAL_LEARNING_RATE = 0.0005

OFFICIAL_DISCRIMINATOR_LEARNING_RATE = 0.001

OFFICIAL_BETAS = (0.9, 0.999)

OFFICIAL_CLIP = 10.0

COUNTER_KEYS = (
    "micro_batches_processed",
    "micro_batches_succeeded",
    "micro_batches_failed",
    "optimizer_updates",
    "official_step",
)

def empty_counters() -> dict[str, int]:
    return {key: 0 for key in COUNTER_KEYS}

def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value

def positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)

def require_adamw(optimizer: torch.optim.Optimizer, name: str, *, lr: float) -> tuple[str, ...]:
    """Require AdamW. Record published hyperparameter differences and keep running.

    More than one parameter group is allowed. Each group is compared with the
    published learning rate, betas, weight decay, eps, and amsgrad.
    """

    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError(f"{name} must be AdamW")
    if not optimizer.param_groups:
        raise BuildError(f"{name} has no parameters")
    differences: list[str] = []
    for index, group in enumerate(optimizer.param_groups):
        label = f"{name} group {index}"
        if float(group["lr"]) != float(lr):
            differences.append(f"{label} learning rate is {group['lr']}, published value is {lr}")
        betas = tuple(float(item) for item in group["betas"])
        if betas != OFFICIAL_BETAS:
            differences.append(f"{label} betas are {betas}, published value is {OFFICIAL_BETAS}")
        if float(group["weight_decay"]) != float(OFFICIAL_ADAMW_WEIGHT_DECAY):
            differences.append(
                f"{label} weight decay is {group['weight_decay']}, published value is {OFFICIAL_ADAMW_WEIGHT_DECAY}"
            )
        if float(group["eps"]) != 1e-8 or bool(group["amsgrad"]):
            differences.append(f"{label} eps is {group['eps']} and amsgrad is {bool(group['amsgrad'])}")
    optimizer.official_differences = tuple(differences)  # type: ignore[attr-defined]
    return tuple(differences)

def require_disjoint(generator: torch.optim.Optimizer, discriminator: torch.optim.Optimizer) -> None:
    left = {id(parameter) for group in generator.param_groups for parameter in group["params"]}
    right = {id(parameter) for group in discriminator.param_groups for parameter in group["params"]}
    if not left or not right:
        raise BuildError("optimizer has no parameters")
    if left & right:
        raise BuildError("generator and discriminator share parameters")

def refuse_amp(runtime: object) -> None:
    if getattr(runtime, "precision", "fp32") == "fp16":
        raise RuntimeFailure(
            "MossFormerGAN refuses fp16. Each successful micro-batch clips, and the "
            "generator step and the discriminator step are not one scaler decision."
        )

def require_group_size(data: object, accumulation: int, component_id: str) -> None:
    group_size = getattr(data, "group_size", None)
    if group_size is None:
        return
    if int(group_size) != accumulation:
        raise RuntimeFailure(
            f"component {component_id}: data group_size {group_size} "
            f"does not match accumulation {accumulation}"
        )


# --- Micro-batch journal. The checkpoint offer stays in ``note_micro_batch``. ---

def note_micro_batch(
    progress: MossProgress,
    runtime: ProcessRuntime,
    journal: StepJournal,
    stepped: dict[str, Any] | None,
    *,
    epoch: int,
    research: Any,
    lifecycle: Any,
    training_control: Any,
    offer: Any,
) -> None:
    if stepped is None:
        offer("partial")
        return
    progress.note_closed_pair()
    progress.step_log.append(stepped)
    journal.write_metrics(
        {
            "discriminator": stepped["discriminator"],
            "generator": stepped["generator"],
            "logical_step": stepped["logical_step"],
            "loss": stepped["loss"],
            "sample_id": stepped["sample_id"],
        }
    )
    journal.write_sample(
        {"logical_step": progress.logical_step, "loss": stepped["loss"], "sample_id": stepped["sample_id"]}
    )
    step_metrics = research.score(
        runtime,
        output_directory=lifecycle.output_directory,
        config_identity=lifecycle.config_identity,
        kind="step",
        epoch=epoch,
        logical_step=progress.logical_step,
        stopping=training_control,
    )
    if step_metrics:
        offer("step", step_metrics)
    else:
        offer("step")

def finish_run(
    reason: str,
    *,
    research: Any,
    runtime: Any,
    lifecycle: Any,
    data: Any,
    progress: MossProgress,
    training_control: Any,
    prepare: Any,
) -> RunResult:
    if reason == "completed":
        epoch = getattr(data, "epoch", None)
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            epoch = None
        extra = research.score(
            runtime, output_directory=lifecycle.output_directory, config_identity=lifecycle.config_identity,
            kind="end", epoch=epoch, logical_step=progress.logical_step, stopping=training_control,
        )
        if extra:
            lifecycle.save_if_needed("final", logical_step=progress.logical_step, metrics=extra, prepare=prepare)
        else:
            lifecycle.save_if_needed("final", logical_step=progress.logical_step, prepare=prepare)
    if progress.halt_reason in {"initialized", "running"}:
        progress.halt_reason = reason
    if reason == "completed" and control_requests_stop(training_control):
        reason = "early_stopping"
    return lifecycle.finish(
        RunResult(
            stop_reason=reason, logical_step=progress.logical_step,
            optimizer_updates=dict(progress.optimizer_updates), output_dir=lifecycle.output_directory,
        )
    )


# --- One micro-batch. PESQ failure skips the optimizer and keeps the cursor. ---

@dataclass
class PairGuard:
    active: bool = False

class MicroBatch:
    """Own the skippable window. The trainer still offers the checkpoint."""

    def __init__(
        self,
        *,
        generator: Any,
        discriminator: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        generator_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        progress: MossProgress,
        guard: PairGuard,
        accumulation: int,
        grad_clip: float,
        n_fft: int,
        hop: int,
        win: int,
        sample_rate: int,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.generator_objective = generator_objective
        self.discriminator_objective = discriminator_objective
        self.progress = progress
        self.guard = guard
        self.accumulation = accumulation
        self.grad_clip = grad_clip
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.sample_rate = sample_rate
        self.average_calls = 0
        self.last_components: dict[str, float] | None = None
        self.loss_scale: list[float] | None = None

    def consume(
        self,
        batch: Batch,
        runtime: ProcessRuntime,
        *,
        update_index: int,
        updates: int | None = None,
        apply_optimizer: bool = True,
        phase_hook: Any,
        step_pair: Any,
        journal: Any,
        skip_path: Any,
    ) -> dict[str, Any] | None:
        if self.guard.active:
            raise RuntimeFailure("a micro-batch cannot start between the two optimizer steps")
        self.generator.train()
        self.discriminator.train()
        inputs = batch.inputs
        if isinstance(inputs, torch.Tensor) and inputs.ndim >= 1 and int(inputs.shape[0]) == 0:
            return self._consume_empty(
                batch, runtime, update_index=update_index, updates=updates,
                apply_optimizer=apply_optimizer, phase_hook=phase_hook, step_pair=step_pair,
                journal=journal, skip_path=skip_path,
            )
        clean, noisy = _waveforms(batch, runtime, self.n_fft)
        progress = self.progress
        progress.counters["micro_batches_processed"] += 1
        progress.counters["official_step"] += 1
        generator_module = self.generator.execution()
        discriminator_module = self.discriminator.execution()
        with runtime.sync_gradients([self.generator, self.discriminator], enabled=False):
            parts = generator_parts(self._terms_view(), generator_module, discriminator_module, clean, noisy)
            result = call_objective(
                self.generator_objective,
                {
                    "audio": parts["audio"],
                    "estimate": parts["audio"],
                    "gan": parts["gan"],
                    "label_mag": parts["label_mag"],
                    "magnitude": parts["magnitude"],
                    "pred_mag": parts["pred_mag"],
                    "ri": parts["ri"],
                    "target": parts["teacher"],
                    "teacher": parts["teacher"],
                    "time": parts["time"],
                },
            )
            generator_loss = result.total
            reason = None
            targets = None
            if not bool(torch.isfinite(generator_loss.detach()).all()):
                reason = "non-finite generator loss"
            else:
                targets = moss_loss.metric_targets(parts["teacher"], parts["audio"], self.sample_rate)
                if targets is None:
                    reason = "pesq"
            if self._skip(runtime, reason, batch, update_index, journal, skip_path):
                return None
            if targets is None:
                raise RuntimeFailure("PESQ targets are missing on a micro-batch that was not skipped")
            ones = torch.ones(clean.shape[0], device=clean.device)
            fake_detached = discriminator_module(parts["label_mag"], parts["pred_mag"].detach())
            real_score = discriminator_module(parts["label_mag"], parts["label_mag"])
            discriminator_loss = call_objective(
                self.discriminator_objective,
                {
                    "fake_score": fake_detached,
                    "ones": ones,
                    "pesq_target": targets,
                    "real_score": real_score,
                },
            ).total
            disc_reason = None if bool(torch.isfinite(discriminator_loss.detach()).all()) else "non-finite discriminator loss"
            if self._skip(runtime, disc_reason, batch, update_index, journal, skip_path):
                return None
            progress.counters["micro_batches_succeeded"] += 1
            start = update_index - int(progress.accu_count)
            remaining = None if updates is None else int(updates) - start
            if remaining is None or remaining >= self.accumulation or remaining < 1:
                planned = self.accumulation
            else:
                planned = remaining
            divisor = window_divisor(self.accumulation, planned)
            progress.accu_count += 1
            self.last_components = {
                "discriminator": float(discriminator_loss.detach()),
                "generator": float(generator_loss.detach()),
            }
            for name, value in result.terms.items():
                self.last_components[name] = float(value.detach())
            phase_hook("before_generator_backward")
            offset = int(progress.accu_count) - 1
            self._backward(runtime, weight_loss(generator_loss, divisor if self.loss_scale is None else self.loss_scale, offset), self.generator)
            phase_hook("after_generator_clip")
            phase_hook("before_discriminator_backward")
            self._backward(
                runtime,
                weight_loss(discriminator_loss, divisor if self.loss_scale is None else self.loss_scale, offset),
                self.discriminator,
            )
            phase_hook("after_discriminator_clip")
        progress.window_generator.append(float(generator_loss.detach()))
        progress.window_discriminator.append(float(discriminator_loss.detach()))
        progress.window_samples.append(str(batch.sample_id))
        if progress.accu_count != planned:
            phase_hook("after_micro_batch")
            return None
        if not apply_optimizer:
            return None
        return step_pair(runtime)

    def _consume_empty(
        self,
        batch: Batch,
        runtime: ProcessRuntime,
        *,
        update_index: int,
        updates: int | None,
        apply_optimizer: bool,
        phase_hook: Any,
        step_pair: Any,
        journal: Any,
        skip_path: Any,
    ) -> dict[str, Any] | None:
        """Backward a connected zero. Do not render or forward the empty batch.

        The two ``agree`` calls and the two backwards stay in the same order as
        a rank that has audio, so a mixed window does not wait forever.
        """

        progress = self.progress
        progress.counters["micro_batches_processed"] += 1
        progress.counters["official_step"] += 1
        with runtime.sync_gradients([self.generator, self.discriminator], enabled=False):
            if self._skip(runtime, None, batch, update_index, journal, skip_path):
                return None
            if self._skip(runtime, None, batch, update_index, journal, skip_path):
                return None
            progress.counters["micro_batches_succeeded"] += 1
            start = update_index - int(progress.accu_count)
            remaining = None if updates is None else int(updates) - start
            if remaining is None or remaining >= self.accumulation or remaining < 1:
                planned = self.accumulation
            else:
                planned = remaining
            divisor = window_divisor(self.accumulation, planned)
            progress.accu_count += 1
            self.last_components = {"discriminator": 0.0, "generator": 0.0}
            phase_hook("before_generator_backward")
            offset = int(progress.accu_count) - 1
            scale = divisor if self.loss_scale is None else self.loss_scale
            # The generator loss scores the fake pair, so its backward reaches
            # the discriminator too. The zero has to reach the same parameters.
            self._backward(
                runtime,
                weight_loss(
                    connected_zero(
                        (
                            *self.generator.unwrapped().parameters(),
                            *self.discriminator.unwrapped().parameters(),
                        ),
                        "trainer",
                    ),
                    scale,
                    offset,
                ),
                self.generator,
            )
            phase_hook("after_generator_clip")
            phase_hook("before_discriminator_backward")
            self._backward(
                runtime,
                weight_loss(
                    connected_zero(self.discriminator.unwrapped().parameters(), "trainer"),
                    scale,
                    offset,
                ),
                self.discriminator,
            )
            phase_hook("after_discriminator_clip")
        progress.window_generator.append(0.0)
        progress.window_discriminator.append(0.0)
        progress.window_samples.append(str(batch.sample_id))
        if progress.accu_count != planned:
            phase_hook("after_micro_batch")
            return None
        if not apply_optimizer:
            return None
        return step_pair(runtime)

    def _step_pair(self, runtime: ProcessRuntime, phase_hook: Any) -> dict[str, Any]:
        progress = self.progress
        self.guard.active = True
        try:
            phase_hook("before_generator_step")
            generator_result = runtime.step(self.optim_g)
            phase_hook("after_generator_step")
            if not generator_result.applied:
                raise RuntimeFailure("generator optimizer did not step")
            phase_hook("before_discriminator_step")
            discriminator_result = runtime.step(self.optim_d)
            phase_hook("after_discriminator_step")
            if not discriminator_result.applied:
                raise RuntimeFailure("discriminator optimizer did not step")
            runtime.end_step()
            self.optim_g.zero_grad(set_to_none=True)
            self.optim_d.zero_grad(set_to_none=True)
            progress.accu_count = 0
            progress.counters["optimizer_updates"] += 1
            for name in progress._optimizer_ids:
                progress.optimizer_updates[name] += 1
            mean_g = sum(progress.window_generator) / len(progress.window_generator)
            mean_d = sum(progress.window_discriminator) / len(progress.window_discriminator)
            sample_id = ",".join(progress.window_samples)
            progress.clear_window()
            local_log = {
                "discriminator": mean_d,
                "generator": mean_g,
                "loss": mean_g,
                "lr_d": float(self.optim_d.param_groups[0]["lr"]),
                "lr_g": float(self.optim_g.param_groups[0]["lr"]),
                "sample_id": sample_id,
            }
        finally:
            self.guard.active = False
        shared = runtime.gather_objects(local_log)[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure("rank 0 did not return a step log")
        shared["logical_step"] = progress.logical_step + 1
        phase_hook("after_micro_batch")
        return shared

    def _backward(self, runtime: ProcessRuntime, loss: torch.Tensor, clip_parameters: Any) -> None:
        modules = (self.generator.unwrapped(), self.discriminator.unwrapped())
        baselines: dict[int, torch.Tensor | None] = {}
        parameters: list[torch.nn.Parameter] = []
        for module in modules:
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                parameters.append(parameter)
                baselines[id(parameter)] = None if parameter.grad is None else parameter.grad.detach().clone()
        runtime.backward(loss)
        runtime.average_new_gradients(parameters, baselines)
        self.average_calls += 1
        runtime.clip_repeated(clip_parameters, self.grad_clip)

    def _skip(
        self,
        runtime: ProcessRuntime,
        reason: str | None,
        batch: Batch,
        update_index: int,
        journal: Any,
        skip_path: Any,
    ) -> bool:
        skipped = runtime.agree(reason is not None, policy="any")
        if not skipped:
            return False
        reasons = runtime.gather_objects(reason)
        case_ids = runtime.gather_objects(_case_ids(batch))
        if not all(item is None or isinstance(item, str) for item in reasons):
            raise RuntimeFailure("skip reason from another rank is not a string")
        agreed = next((item for item in reasons if item not in (None, "peer")), None)
        if agreed is None:
            agreed = "peer" if any(item is not None for item in reasons) else "pesq"
        record = {
            "case_ids_by_rank": list(case_ids),
            "count": 1,
            "official_step": int(self.progress.counters["official_step"]),
            "reason": agreed,
            "reasons_by_rank": list(reasons),
            "update_index": int(update_index),
        }
        self.progress.skip_log.append(record)
        self.progress.last_skip = record
        self.progress.counters["micro_batches_failed"] += 1
        if skip_path is not None and journal is not None:
            journal.write_sidecar(skip_path, record)
        return True

    def _terms_view(self) -> OfficialCall:
        return OfficialCall(
            n_fft=self.n_fft,
            hop=self.hop,
            win=self.win,
            sample_rate=self.sample_rate,
            generator=None,
            discriminator=None,
            device=torch.device("cpu"),
            validation_requires_all_ranks=False,
        )

    def discard_window(self) -> None:
        self.optim_g.zero_grad(set_to_none=True)
        self.optim_d.zero_grad(set_to_none=True)
        self.progress.accu_count = 0
        self.progress.clear_window()

def _waveforms(batch: Batch, runtime: ProcessRuntime, n_fft: int) -> tuple[torch.Tensor, torch.Tensor]:
    moved = runtime.move(batch)
    clean = moved.targets
    noisy = moved.inputs
    if not isinstance(clean, torch.Tensor) or not isinstance(noisy, torch.Tensor):
        raise RuntimeFailure("waveforms must be tensors")
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    if int(clean.shape[-1]) <= n_fft // 2:
        raise RuntimeFailure(
            f"waveform length {int(clean.shape[-1])} is too short for centered STFT "
            f"(need more than {n_fft // 2} samples)"
        )
    return clean, noisy

def share_window(runtime: ProcessRuntime, generator: Any, discriminator: Any, progress: MossProgress) -> None:
    runtime.broadcast_buffers(generator)
    runtime.broadcast_buffers(discriminator)
    if runtime.world_size <= 1:
        return
    gathered = runtime.gather_objects(
        {
            "discriminator": list(progress.window_discriminator),
            "generator": list(progress.window_generator),
            "samples": list(progress.window_samples),
        }
    )
    chosen = gathered[0]
    progress.window_discriminator = [float(item) for item in chosen["discriminator"]]
    progress.window_generator = [float(item) for item in chosen["generator"]]
    progress.window_samples = [str(item) for item in chosen["samples"]]

def open_micro_batch(
    data: Any,
    runtime: ProcessRuntime,
    progress: MossProgress,
    *,
    accumulation: int,
    component_id: str,
    epoch: int,
    index: int,
    open_boundary: Any,
) -> tuple[Any, Batch, int] | None:
    from src.experiments.trainers import _close_epoch_view

    view = data.train_epoch(epoch, start_update=index)
    try:
        updates = int(view.updates)
        counts = runtime.gather_objects(int(updates))
        if any(item != updates for item in counts):
            raise RuntimeFailure(f"component {component_id}: ranks have different micro-batch counts {counts}")
        if updates < 1:
            raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
        if index > updates:
            raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
        if index == updates:
            open_boundary()
            _close_epoch_view(view)
            return None
        if index == 0:
            progress.epoch_success_start = int(progress.counters["micro_batches_succeeded"])
            progress.epoch_update_start = int(progress.counters["optimizer_updates"])
        batches = view.batch(index)
        if len(batches) != 1 or not isinstance(batches[0], Batch):
            raise RuntimeFailure(f"component {component_id}: expected one micro-batch")
    except Exception:
        _close_epoch_view(view)
        raise
    return view, batches[0], updates

def case_ids(batch: Batch) -> list[str]:
    if batch.sample_ids:
        return [str(item) for item in batch.sample_ids]
    return [str(batch.sample_id)]

def _case_ids(batch: Batch) -> list[str]:
    return case_ids(batch)


# --- Epoch boundary: validation, then the plateau action. ---

def open_boundary(progress: MossProgress, *, epoch: int, validate: bool) -> None:
    if progress.phase != "in_epoch":
        raise RuntimeFailure("an epoch boundary is already pending")
    progress.boundary = {
        "closed": False,
        "plateau_done": False,
        "trainer_epoch": int(epoch),
        "validate": validate,
        "validation_done": False,
        "validation_summary": None,
    }
    progress.phase = PENDING_BOUNDARY

def finish_boundary(
    progress: MossProgress,
    runtime: ProcessRuntime,
    data: Any,
    micro: MicroBatch,
    *,
    component_id: str,
    max_logical_steps: int,
    n_fft: int,
    hop: int,
    win: int,
    sample_rate: int,
    offer: Callable[..., None],
    research: Any,
    lifecycle: Any,
    training_control: Any,
) -> None:
    if progress.phase != PENDING_BOUNDARY or progress.boundary is None:
        raise RuntimeFailure(f"component {component_id}: there is no pending MossFormerGAN boundary")
    boundary = progress.boundary
    if boundary["validate"] and not boundary["validation_done"]:
        summary = evaluate(
            runtime, data, micro.generator, micro.discriminator,
            trainer_epoch=int(boundary["trainer_epoch"]), n_fft=n_fft, hop=hop, win=win, sample_rate=sample_rate,
        )
        boundary["validation_summary"] = summary
        boundary["validation_done"] = True
        offer("boundary_pending")
    if boundary["validate"] and not boundary["plateau_done"]:
        summary = boundary.get("validation_summary")
        if not isinstance(summary, dict):
            raise RuntimeFailure(f"component {component_id}: validation summary is missing")
        boundary["plateau_action"] = apply_plateau(
            progress, summary, generator=micro.generator, discriminator=micro.discriminator,
            optim_g=micro.optim_g, optim_d=micro.optim_d,
        )
        boundary["plateau_done"] = True
    micro.discard_window()
    no_success = progress.counters["micro_batches_succeeded"] == progress.epoch_success_start
    no_step = progress.counters["optimizer_updates"] == progress.epoch_update_start
    if no_step:
        progress.epochs_without_step += 1
    else:
        progress.epochs_without_step = 0
    data.epoch = int(boundary["trainer_epoch"]) + 1
    data.update_index = 0
    boundary["closed"] = True
    metrics = scalar_metrics(boundary.get("validation_summary"))
    progress.completed_boundaries.append(dict(boundary))
    progress.boundary = None
    progress.phase = "in_epoch"
    if progress.stop_training:
        if progress.halt_reason == "running":
            progress.halt_reason = "plateau_stop"
    elif no_success:
        progress.halt_reason = "no_successful_micro_batch"
        progress.stop_training = True
    elif progress.epochs_without_step >= max_logical_steps:
        progress.halt_reason = "accumulation_window_never_closed"
        progress.stop_training = True
    if boundary["validate"]:
        observe_validation(training_control, metrics, step=progress.logical_step, scoped=research.attached)
        extra = research.score(
            runtime,
            output_directory=lifecycle.output_directory,
            config_identity=lifecycle.config_identity,
            kind="boundary",
            epoch=int(boundary["trainer_epoch"]),
            logical_step=progress.logical_step,
            official=metrics,
            stopping=training_control,
        )
        if extra:
            metrics = {**metrics, **extra}
    offer("boundary", metrics)


# --- Five stalls restore and halve. Ten stalls stop. The stall count is not cleared. ---

def apply_plateau(
    progress: MossProgress,
    summary: dict[str, Any],
    *,
    generator: Any,
    discriminator: Any,
    optim_g: Any,
    optim_d: Any,
) -> str:
    generator_loss = (summary.get("losses") or {}).get("generator") or {}
    value = generator_loss.get("value")
    if generator_loss.get("status") != "ok" or value is None:
        progress.plateau_history.append({"action": "absent", "logical_step": progress.logical_step})
        return "absent"
    action_loss = float(value)
    progress.best_val, progress.stalls, action = mossformer_validation_decision(
        progress.best_val, progress.stalls, action_loss
    )
    if action == "improve":
        progress.best_snapshot = _capture(generator, discriminator, optim_g, optim_d)
    elif action == "halve":
        _restore_and_halve(progress.best_snapshot, generator, discriminator, optim_g, optim_d)
    elif action == "stop":
        progress.stop_training = True
        progress.halt_reason = "plateau_stop"
    progress.plateau_history.append(
        {
            "action": action,
            "best": progress.best_val,
            "logical_step": progress.logical_step,
            "loss": action_loss,
            "lr_d": float(optim_d.param_groups[0]["lr"]),
            "lr_g": float(optim_g.param_groups[0]["lr"]),
            "stalls": progress.stalls,
        }
    )
    return action

def _capture(generator: Any, discriminator: Any, optim_g: Any, optim_d: Any) -> dict[str, Any]:
    return {
        "discriminator": {
            key: value.detach().cpu().clone() for key, value in discriminator.unwrapped().state_dict().items()
        },
        "generator": {
            key: value.detach().cpu().clone() for key, value in generator.unwrapped().state_dict().items()
        },
        "optim_d": copy.deepcopy(optim_d.state_dict()),
        "optim_g": copy.deepcopy(optim_g.state_dict()),
    }

def _restore_and_halve(snapshot: dict[str, Any] | None, generator: Any, discriminator: Any, optim_g: Any, optim_d: Any) -> None:
    if snapshot is not None:
        generator.unwrapped().load_state_dict(snapshot["generator"], strict=True)
        discriminator.unwrapped().load_state_dict(snapshot["discriminator"], strict=True)
        optim_g.load_state_dict(snapshot["optim_g"])
        optim_d.load_state_dict(snapshot["optim_d"])
    optim_g.param_groups[0]["lr"] *= 0.5
