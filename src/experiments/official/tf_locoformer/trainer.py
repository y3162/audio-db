"""TF-Locoformer DNS loop. ``fit`` calls ``step_batch`` after an applied optimizer update.
The boundary calls ``step_validation`` once and does not call ``scheduler.step``.
Counters live on ``LocoformerProgress``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.experiments.calc.optim import WarmupReduceLROnPlateau
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.tf_locoformer.boundary import LocoformerBoundary
from src.experiments.official.tf_locoformer.checks import (
    optional_patience,
    positive_float,
    positive_int,
    require_group_size,
    require_locoformer_setup,
)
from src.experiments.official.tf_locoformer.ranking import select_checkpoints
from src.experiments.official.tf_locoformer.recording import DeferredStepLog
from src.experiments.official.tf_locoformer.state import LocoformerProgress
from src.experiments.official.tf_locoformer.update import LocoformerUpdate
from src.experiments.official.tf_locoformer.validation import LocoformerValidator
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.stopping import observe_validation, should_stop
from src.experiments.training.control import close_training_run, open_grouped_window, stop_after_step
from src.experiments.trainers import _close_epoch_view

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
                index += self.accumulation
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
