"""TF-GridNet MS-SNSD training loop for the experiment framework.

``fit`` orders start, a resumed pending boundary, one accumulation update,
the deferred step log, then either a later step save or a pending save,
validation, ``ReduceLROnPlateau``, n-best, research, and the boundary save.
Counters live on ``GridNetProgress``. ``valid.si_snr`` is not a selection
criterion. ``amp`` may skip the optimizer; the logical step still advances.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.tf_gridnet.boundary import GridNetBoundary
from src.experiments.official.tf_gridnet.checks import positive_float, positive_int, require_gridnet_setup, require_group_size
from src.experiments.official.tf_gridnet.recording import DeferredStepLog
from src.experiments.official.tf_gridnet.state import GridNetProgress
from src.experiments.official.tf_gridnet.update import GridNetUpdate, _forward_loss as _micro_loss
from src.experiments.official.tf_gridnet.validation import GridNetValidator
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.stopping import observe_validation, should_stop
from src.experiments.training.control import close_training_run, open_grouped_window, stop_after_step
from src.experiments.trainers import _close_epoch_view


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
