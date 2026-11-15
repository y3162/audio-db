"""Official MP-SENet GAN loop.

``fit`` orders start, a resumed pending boundary, one update, the step log,
then either a step save or pending save, validation, best, schedulers,
research, and the boundary save. Counters live on ``MPTrainingProgress``.
``OfficialBoundary.run`` is validation then the schedulers. A step save is
offered after both optimizers, before the next forward. A CUDA log flush is
not a save. ``amp_fp16`` is refused.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.data.pair import bind_declared_audio
from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.mp_senet.boundary import OfficialBoundary
from src.experiments.official.mp_senet.checks import non_negative_int, positive_int, require_group_size, require_mp_gan_setup
from src.experiments.official.mp_senet.recording import StepRecorder
from src.experiments.official.mp_senet.state import MPTrainingProgress
from src.experiments.official.mp_senet.training_step import MPSENetUpdate, refuse_gan_amp
from src.experiments.official.mp_senet.validation import MPSENetOfficialValidator
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.stopping import control_requests_stop, observe_validation
from src.experiments.training.control import close_training_run, open_flat_window_at, stop_after_step
from src.experiments.trainers import _close_epoch_view

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
                index += self.accumulation
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
