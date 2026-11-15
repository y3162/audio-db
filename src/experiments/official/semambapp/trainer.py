"""Official SEMamba++ GAN loop. One micro-batch reuses the generator graph; several reforward."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.semambapp.boundary import SEMambaBoundary, published_metrics
from src.experiments.official.semambapp.checks import SUBSTITUTE_CLAIM, require_group_size, require_semamba_setup
from src.experiments.official.semambapp.recording import StepLog
from src.experiments.official.semambapp.state import (
    PENDING_BOUNDARY,
    SEMambaProgress,
    open_boundary_payload,
)
from src.experiments.official.semambapp.update import SEMambaUpdate, UpdateView
from src.experiments.official.semambapp.validation import SEMambaValidator
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.stopping import should_stop
from src.experiments.trainers import _close_epoch_view
from src.experiments.training.control import (
    finish_training_run,
    open_flat_window,
    stop_after_step,
)

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
                index += self.accumulation
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
