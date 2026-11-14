"""Official MP-SENet GAN loop for the experiment framework.

``fit`` orders one logical step, the step log, official validation, PESQ
selection, both schedulers, research evaluation, and the checkpoint safe
point. One GAN pair is :class:`~src.experiments.official.mp_senet.training_step.MPSENetUpdate`.
Official validation is :class:`~src.experiments.official.mp_senet.validation.MPSENetOfficialValidator`.
That pair keeps one generator forward, steps the discriminator, then the
generator. ``StandardTrainer`` and the TF-GridNet / TF-Locoformer trainers
are not used for this loop.

``ExponentialLR`` advances once when the epoch's usable micro-batches are
finished. A validation interval does not advance it. When the same update
both validates and ends the epoch, validation and best selection run, then
both schedulers step.

Checkpoint files are written only after both optimizers have stepped and
``end_step`` has returned. ``amp_fp16`` is refused: the two AdamW updates are
not one atomic scaler decision.
"""

from __future__ import annotations

from src.experiments.stopping import observe_validation, should_stop

from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mp_senet.artifacts import write_best
from src.experiments.official.mp_senet.selection import consider_best_pesq
from src.experiments.official.mp_senet.state import (
    IN_EPOCH,
    PENDING_BOUNDARY,
    export_progress,
    open_boundary,
    restore_progress,
)
from src.experiments.official.mp_senet.validation import MPSENetOfficialValidator
from src.experiments.training.control import finish_training_run, open_flat_window, stop_requested
from src.experiments.official.mp_senet.models import (
    MPSENetGANGenerator,
    MPSENetMetricDiscriminator,
    require_official_architecture,
)
from src.experiments.official.mp_senet.training_step import MPSENetUpdate, refuse_gan_amp
from src.experiments.execution.journal import StepJournal
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.selectors import scalar_metrics
from src.experiments.trainers import (
    _close_epoch_view,
    _rank,
)
from src.experiments.training.official_objectives import require_mp_senet

_IN_EPOCH = IN_EPOCH
_PENDING = PENDING_BOUNDARY
OFFICIAL_FAITHFUL = "official_faithful"
LEAKAGE_SAFE = "leakage_safe"
_CLAIMS = frozenset({OFFICIAL_FAITHFUL, LEAKAGE_SAFE, "research"})
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
    """Orders MP-SENet updates, official validation, and checkpoint safe points."""

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
        require_official_architecture(generator.unwrapped().architecture, experiment_claim)
        require_mp_senet(signal_objective, metric_objective, discriminator_objective, experiment_claim)
        optim_g_id = getattr(optim_g, "_experiment_component_id", None)
        optim_d_id = getattr(optim_d, "_experiment_component_id", None)
        if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
            raise BuildError("optimizers have no component id")
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.signal_objective = signal_objective
        self.metric_objective = metric_objective
        self.discriminator_objective = discriminator_objective
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.experiment_claim = experiment_claim
        self.audio_sample_rate = 16000
        self.audio_teachers = ("dry_clean",)
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.validation_interval = _non_negative_int("validation_interval", validation_interval)
        self.best_checkpoint_start_epoch = _non_negative_int(
            "best_checkpoint_start_epoch", best_checkpoint_start_epoch
        )
        self.max_epochs = None if max_epochs is None else _positive_int("max_epochs", max_epochs)
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = _positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self._mp_io: tuple[Any, ...] | None = None
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.scheduler_steps = 0
        self.step_log: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.completed_boundaries: list[dict[str, Any]] = []
        self.boundary: dict[str, Any] | None = None
        self.phase = _IN_EPOCH
        self.best_pesq: float | None = None
        self.best_epoch: int | None = None
        self.best_logical_step: int | None = None
        self.stop_reason = "initialized"
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
        self._average_calls = 0
        self._generator_synced_during_backward = False
        self._update = MPSENetUpdate(
            generator=self.generator,
            discriminator=self.discriminator,
            optim_g=self.optim_g,
            optim_d=self.optim_d,
            signal_objective=self.signal_objective,
            metric_objective=self.metric_objective,
            discriminator_objective=self.discriminator_objective,
            accumulation=self.accumulation,
        )
        self._validator = MPSENetOfficialValidator(
            generator=self.generator,
            discriminator=self.discriminator,
            data=self.data,
            optim_g=self.optim_g,
            scheduler_g=self.scheduler_g,
        )

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or an interrupt."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        if self.phase == _PENDING:
            self._finish_boundary(runtime, component_id)
            if should_stop(self):
                return self._result("completed")
            if self._epoch_cap_reached():
                return self._result("completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                self._drain_mp_io()
                if self.logical_step > 0 and not self._in_official_step:
                    self._offer("stop")
                return self._result("interrupted")
            if self._epoch_cap_reached():
                self._drain_mp_io()
                return self._result("completed")
            epoch, index, view, batches = self._open_window(component_id)
            try:
                local = self.official_step(batches, runtime)
                self.logical_step += 1
                index += self.accumulation
                self.optimizer_updates[self.optim_g_id] += 1
                self.optimizer_updates[self.optim_d_id] += 1
                self.data.epoch = epoch
                self.data.update_index = index
                epoch_done = index == view.updates
                will_validate = self.validation_interval > 0 and self.logical_step % self.validation_interval == 0
                if self._record_step(
                    runtime,
                    local,
                    epoch=epoch,
                    epoch_done=epoch_done,
                    will_validate=will_validate,
                ):
                    continue
                self.phase = _PENDING
                self.boundary = open_boundary(epoch, validate=will_validate, epoch_done=epoch_done)
                self._on_boundary_phase(self.phase)
                self._offer("boundary_pending")
                if self._stop_requested(runtime):
                    return self._result("interrupted")
                self._finish_boundary(runtime, component_id)
                if should_stop(self):
                    return self._result("completed")
                if self._epoch_cap_reached():
                    return self._result("completed")
            finally:
                _close_epoch_view(view)
        self._drain_mp_io()
        return self._result("completed")

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        """Rewind logs. Does not run a discriminator or generator update."""

        self.research.rewind(
            self.lifecycle.output_directory,
            restored=self._restored,
            phase=self.phase,
            progress={"logical_step": int(self.logical_step)},
        )
        self._refuse_amp(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.journal.begin(
            restored=self._restored,
            logical_step=self.logical_step,
            expect_rows=len(self.step_log) if self._restored else None,
        )
        if not self._restored:
            self.logical_step = 0
            self.optimizer_updates = {self.optim_g_id: 0, self.optim_d_id: 0}
            self.scheduler_steps = 0
            self.step_log = []
            self.history = []
            self.completed_boundaries = []
            self.boundary = None
            self.phase = _IN_EPOCH
            self.best_pesq = None
            self.best_epoch = None
            self.best_logical_step = None
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        self._require_group_size(component_id)
        runtime.barrier()
        return str(component_id)

    def _open_window(self, component_id: str) -> tuple[int, int, Any, list[Batch]]:
        """Take the next flat micro-batch window. The epoch length must divide evenly."""

        return open_flat_window(self, component_id)

    def _record_step(
        self,
        runtime: ProcessRuntime,
        local: dict[str, Any],
        *,
        epoch: int,
        epoch_done: bool,
        will_validate: bool,
    ) -> bool:
        """Write or defer the step log. Return True when the epoch continues."""

        shared = _shared_log(runtime, local)
        entry = {
            "applied_d": True,
            "applied_g": True,
            "complex": shared["complex"],
            "consistency": shared["consistency"],
            "discriminator": shared["discriminator"],
            "epoch": epoch,
            "generator": shared["generator"],
            "logical_step": self.logical_step,
            "loss": shared["generator"],
            "lr_d": shared["lr_d"],
            "lr_g": shared["lr_g"],
            "magnitude": shared["magnitude"],
            "metric": shared["metric"],
            "phase": shared["phase"],
            "sample_id": shared["sample_id"],
            "time": shared["time"],
        }
        self.step_log.append(entry)
        offer_step = not will_validate and not epoch_done
        step_metrics: dict[str, float] = {}
        if offer_step:
            step_metrics = self.research.score(
                runtime,
                output_directory=self.lifecycle.output_directory,
                config_identity=self.lifecycle.config_identity,
                kind="step",
                epoch=epoch,
                logical_step=self.logical_step,
                stopping=getattr(self, "training_control", None),
            )
        if runtime.device.type == "cuda" and runtime.world_size == 1:
            self._mp_io = (entry, local, offer_step, step_metrics)
            if offer_step:
                return True
            self._drain_mp_io()
            return False
        self.journal.write_metrics(entry)
        self.journal.write_sample({**local, "logical_step": self.logical_step, "loss": local["generator"]})
        if offer_step:
            if step_metrics:
                self._offer("step", metrics=step_metrics)
            else:
                self._offer("step")
            return True
        return False

    def _drain_mp_io(self) -> None:
        """Write the previous step while the next signal backward is queued."""

        item = self._mp_io
        if item is None:
            return
        self._mp_io = None
        entry, local, offer_step, step_metrics = item
        self.journal.write_metrics(entry)
        self.journal.write_sample(
            {**local, "logical_step": entry["logical_step"], "loss": local["generator"]}
        )
        if offer_step:
            if step_metrics:
                self._offer("step", metrics=step_metrics)
            else:
                self._offer("step")

    def official_step(self, batches: list[Batch], runtime: ProcessRuntime) -> dict[str, float]:
        """Run one official GAN update. Both optimizers step, or neither is checkpointed."""

        self._in_official_step = True
        try:
            return self._update.run(
                batches,
                runtime,
                phase_hook=self._on_update_phase,
                drain_previous=self._drain_mp_io,
                note_generator_sync=self._note_generator_sync,
                note_gradients_averaged=self._note_gradients_averaged,
            )
        finally:
            self._in_official_step = False

    def _note_generator_sync(self, enabled: bool) -> None:
        """Remember the last DDP sync flag seen during generator backward.

        This is a test observation. It is not stored in the checkpoint.
        The CUDA overlap path does not report a flag.
        """

        self._generator_synced_during_backward = bool(enabled)

    def _note_gradients_averaged(self) -> None:
        """Count one gradient reduction, immediately before average_gradients."""

        self._average_calls += 1

    def state_dict(self) -> dict[str, Any]:
        return export_progress(
            best_epoch=self.best_epoch,
            best_logical_step=self.best_logical_step,
            best_pesq=self.best_pesq,
            boundary=self.boundary,
            completed_boundaries=self.completed_boundaries,
            experiment_claim=self.experiment_claim,
            history=self.history,
            logical_step=self.logical_step,
            optimizer_updates=self.optimizer_updates,
            phase=self.phase,
            scheduler_steps=self.scheduler_steps,
            step_log=self.step_log,
            stop_reason=self.stop_reason,
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        restored = restore_progress(state, experiment_claim=self.experiment_claim)
        self.logical_step = restored.logical_step
        self.optimizer_updates = restored.optimizer_updates
        self.phase = restored.phase
        self.history = restored.history
        self.completed_boundaries = restored.completed_boundaries
        self.step_log = restored.step_log
        self.boundary = restored.boundary
        self.scheduler_steps = restored.scheduler_steps
        self.best_pesq = restored.best_pesq
        self.best_epoch = restored.best_epoch
        self.best_logical_step = restored.best_logical_step
        if restored.stop_reason is not None:
            self.stop_reason = restored.stop_reason
        self._validator.invalidate()
        self._restored = True

    def _on_update_phase(self, phase: str, payload: object = None) -> None:
        """Hook for tests. The default records nothing."""

        del phase, payload

    def _on_boundary_phase(self, phase: str) -> None:
        """Hook for tests that record validation and scheduler order."""

        del phase

    def _finish_boundary(
        self,
        runtime: ProcessRuntime,
        component_id: str,
    ) -> None:
        """Validate, select best PESQ, step both schedulers, then run research evaluation."""
        if self.phase != _PENDING or self.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending GAN boundary")
        boundary = self.boundary
        trainer_epoch = int(boundary["trainer_epoch"])
        metrics: dict[str, float] = {}
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self._validate(runtime, trainer_epoch, component_id)
            metrics = scalar_metrics(summary)
            decision = self._consider_best(trainer_epoch, summary)
            self.history.append(decision)
            self._write_best(runtime, self.lifecycle.output_directory, decision)
            boundary["validation_done"] = True
            boundary["selection_done"] = True
            boundary["pesq"] = decision["pesq"]
            boundary["pesq_status"] = decision["pesq_status"]
            boundary["selected"] = decision["selected"]
            if boundary["scheduler"]:
                self.phase = "pending_scheduler"
                self._on_boundary_phase(self.phase)
        if boundary["scheduler"] and not boundary["scheduler_done"]:
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.scheduler_steps += 1
            boundary["scheduler_done"] = True
            boundary["lr_g"] = float(self.optim_g.param_groups[0]["lr"])
            boundary["lr_d"] = float(self.optim_d.param_groups[0]["lr"])
        if boundary["epoch_done"]:
            self.data.epoch = trainer_epoch + 1
            self.data.update_index = 0
        boundary["closed"] = True
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = _IN_EPOCH
        self._on_boundary_phase(self.phase)
        if boundary["validate"]:
            observe_validation(
                getattr(self, "training_control", None),
                metrics,
                step=self.logical_step,
                scoped=self.research.attached,
            )
        extra = self.research.score(
            runtime,
            output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity,
            kind="boundary",
            epoch=trainer_epoch,
            logical_step=self.logical_step,
            official=metrics if boundary["validate"] else {},
            stopping=getattr(self, "training_control", None),
        )
        self._offer("boundary", metrics={**metrics, **extra} if extra else metrics)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        """Run official validation. Scheduling and best selection stay in the boundary."""

        return self._validator.evaluate(runtime, trainer_epoch, component_id)

    def _consider_best(self, epoch: int, summary: dict[str, Any]) -> dict[str, Any]:
        """Apply the official PESQ rule and remember a strict improvement."""

        decision = consider_best_pesq(
            epoch,
            summary,
            start_epoch=self.best_checkpoint_start_epoch,
            best_pesq=self.best_pesq,
            logical_step=self.logical_step,
        )
        if decision["selected"]:
            self.best_pesq = float(decision["pesq"])
            self.best_epoch = int(decision["epoch"])
            self.best_logical_step = int(decision["logical_step"])
        return decision

    def _write_best(self, runtime: ProcessRuntime, directory: Path, decision: dict[str, Any]) -> None:
        error: str | None = None
        if _rank(runtime) == 0 and decision["selected"]:
            try:
                write_best(
                    directory,
                    self.generator.unwrapped().export_for_inference(),
                    decision,
                    self.experiment_claim,
                )
            except (OSError, RuntimeError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        failure = next((item for item in errors if item), None)
        if failure:
            raise RuntimeFailure(f"best checkpoint write failed: {failure}")

    def _refuse_amp(self, runtime: ProcessRuntime) -> None:
        refuse_gan_amp(runtime)

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if int(group_size) != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _offer(
        self,
        event: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Offer this state to the selector. A GAN optimizer pair is not a save point."""

        if self._in_official_step:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")
        self.lifecycle.save_if_needed(event, logical_step=self.logical_step, metrics=metrics)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        """Agree a logical-step cap with a rank interrupt."""

        return stop_requested(self, runtime)

    def _epoch_cap_reached(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

    def _result(self, reason: str) -> RunResult:
        """Score the research end and finish. This path does not broadcast buffers."""

        return finish_training_run(self, reason)


def _shared_log(runtime: ProcessRuntime, local: dict[str, float]) -> dict[str, float | str]:
    """Store rank 0's scalars on every rank so the checkpoint matches."""

    gathered = runtime.gather_objects(dict(local))
    shared = gathered[0]
    if not isinstance(shared, dict):
        raise RuntimeFailure("rank 0 did not return GAN metrics")
    return shared


def _require_disjoint(optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer) -> None:
    """Refuse an optimizer pair that shares a parameter."""
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("MP-SENet GAN optimizers must each own parameters")


def _positive_int(name: str, value: object) -> int:
    """Require a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    """Require an integer that is zero or greater."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value
