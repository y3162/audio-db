"""DeepFilterNet3 loop. Cosine runs before the optimizer. Clean is ``Batch.targets``."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.deepfilternet3.boundary import finish as finish_boundary
from src.experiments.official.deepfilternet3.checks import (
    apply_batch_schedule,
    non_negative_int,
    parse_schedule,
    positive_float,
    positive_int,
    require_adamw,
    require_group_size,
)
from src.experiments.official.deepfilternet3.cosine import CosineTable, open_epoch
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.official.deepfilternet3.recording import note_step, record_rfft
from src.experiments.official.deepfilternet3.schedule import schedule_index
from src.experiments.official.deepfilternet3.state import PENDING_VALIDATION, DFProgress
from src.experiments.official.deepfilternet3.training_step import component_losses
from src.experiments.official.deepfilternet3.update import forward_loss, run_accumulation, stream_for
from src.experiments.official.deepfilternet3.validation import evaluate
from src.experiments.official.deepfilternet3.waveforms import clean_noisy as _clean_noisy
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.stopping import observe_validation, should_stop
from src.experiments.trainers import _close_epoch_view
from src.experiments.training.control import finish_training_run, stop_requested
from src.se.models.deepfilternet3.dsp import DFStream

class DeepFilterNet3Trainer:
    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        model: PreparedModel,
        data: Any,
        optimizer: torch.optim.Optimizer,
        *,
        objective: torch.nn.Module,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float = 1.0,
        max_epochs: int = 120,
        early_stopping_patience: int = 25,
        steps_per_epoch: int = 1,
        epochs: int = 120,
        lr_min: float = 1e-6,
        lr_warmup: float = 1e-4,
        warmup_epochs: int = 3,
        weight_decay_end: float = 0.01,
        batch_schedule: list[list[int]] | None = None,
        batch_size_limit: int = 64,
        stop_after_logical_steps: int | None = None,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        require_prepared(model, DeepFilterNet3Model, "model must be a prepared DeepFilterNet3Model")
        if not isinstance(objective, torch.nn.Module):
            raise BuildError("objective must be a loss module")
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        self.optimizer_id = require_adamw(optimizer)
        if model.policy.find_unused_parameters is not True:
            raise BuildError("DeepFilterNet3 requires find_unused_parameters")
        self.objective = objective
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.audio_sample_rate = int(model.unwrapped().generator.params.sr)
        self.audio_teachers = ("official_mixed_target",)
        self.max_logical_steps = positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = positive_int("accumulation", accumulation)
        self.grad_clip = positive_float("grad_clip", grad_clip)
        self.max_epochs = positive_int("max_epochs", max_epochs)
        self.early_stopping_patience = positive_int("early_stopping_patience", early_stopping_patience)
        self.batch_schedule = parse_schedule(batch_schedule)
        self.batch_size_limit = positive_int("batch_size_limit", batch_size_limit)
        self.stop_after_logical_steps = (
            None if stop_after_logical_steps is None else positive_int("stop_after_logical_steps", stop_after_logical_steps)
        )
        steps = positive_int("steps_per_epoch", steps_per_epoch)
        self._cosine = CosineTable(
            optimizer,
            schedule_epochs=positive_int("epochs", epochs),
            lr_min=float(lr_min),
            lr_warmup=float(lr_warmup),
            warmup_epochs=non_negative_int("warmup_epochs", warmup_epochs),
            weight_decay_end=float(weight_decay_end),
            steps_per_epoch=steps,
        )
        self.progress = DFProgress.initial(self.optimizer_id, steps)
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators,
            model_selection,
            data=self.data,
            module=self.model,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection
        self._restored = False
        self._stream: DFStream | None = None

    @property
    def logical_step(self) -> int:
        return self.progress.logical_step

    @property
    def optimizer_updates(self) -> dict[str, int]:
        return self.progress.optimizer_updates

    @property
    def losses(self) -> list[float]:
        return self.progress.losses

    @losses.setter
    def losses(self, value: list[float]) -> None:
        self.progress.losses = value

    @property
    def stop_reason(self) -> str | None:
        return self.progress.stop_reason

    @stop_reason.setter
    def stop_reason(self, value: str) -> None:
        self.progress.stop_reason = value

    def fit(self) -> RunResult:
        """Train until the step budget, the epoch cap, or early stopping. Cosine runs before the optimizer."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        if self.progress.phase == PENDING_VALIDATION:
            self._finish_boundary(runtime, component_id)
            if should_stop(self) or self.progress.early_stopped or self._epoch_cap_reached():
                return self._result("completed")
        while self.progress.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                if self.progress.logical_step > 0:
                    self._offer(runtime, "stop")
                return self._result("interrupted")
            if self._epoch_cap_reached():
                return self._result("completed")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            if index == 0:
                apply_batch_schedule(self.data, epoch, self.batch_schedule, self.batch_size_limit)
            view, steps = open_epoch(
                self.data, self._cosine, self.progress,
                epoch=epoch, index=index, accumulation=self.accumulation, component_id=component_id,
            )
            try:
                position = schedule_index(epoch, index // self.accumulation, steps)
                learning_rate, decay = self._cosine.apply(position)
                local = run_accumulation(
                    self.model, self.optimizer, runtime, view, index,
                    accumulation=self.accumulation, grad_clip=self.grad_clip, component_id=component_id,
                    stream=self._stream, objective=self.objective,
                )
                self.progress.record_step(
                    applied=local.applied, loss=local.loss, learning_rate=learning_rate, decay=decay,
                )
                index += self.accumulation
                self.data.epoch = epoch
                self.data.update_index = index
                note_step(
                    self.journal, self.progress, applied=local.applied, epoch=epoch, loss=local.loss,
                    learning_rate=learning_rate, sample_id=local.sample_id, weight_decay=decay,
                    schedule_index=position,
                )
                if index != view.updates:
                    self._offer_step(runtime, epoch)
                    continue
                self.progress.open_boundary(epoch, steps)
                self._offer(runtime, "boundary_pending")
                if self._stop_requested(runtime):
                    return self._result("interrupted")
                self._finish_boundary(runtime, component_id)
                if should_stop(self) or self.progress.early_stopped or self._epoch_cap_reached():
                    return self._result("completed")
            finally:
                _close_epoch_view(view)
        return self._result("completed")

    def state_dict(self) -> dict[str, Any]:
        return self.progress.export()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        decoded = self.progress.decode(state, self._cosine.tables)
        self._cosine.install(decoded.schedule_steps, decoded.rate_table, decoded.decay_table)
        self.progress.apply(decoded)
        self._restored = True

    def research_predict(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enhanced waveform for research metrics. Official validation does not call this."""

        if self._stream is None:
            raise RuntimeFailure("DeepFilterNet3 research predict has no DF stream")
        parts = component_losses(self._stream, self.model.unwrapped(), targets, inputs, self.objective)
        wave = parts.get("enhanced_wave")
        if not isinstance(wave, torch.Tensor):
            raise RuntimeFailure("DeepFilterNet3 research predict did not return a waveform")
        return wave

    def _finish_boundary(self, runtime: ProcessRuntime, component_id: str) -> None:
        loss = finish_boundary(
            self.progress, runtime, self.data, self.model, self._stream, self.objective,
            patience=self.early_stopping_patience, component_id=component_id,
            output_directory=self.lifecycle.output_directory,
        )
        official = {"loss": float(loss)}
        observe_validation(
            getattr(self, "training_control", None), official,
            step=self.progress.logical_step, scoped=self.research.attached,
        )
        extra = self.research.score(
            runtime, output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity, kind="boundary",
            epoch=int(self.data.epoch) - 1, logical_step=self.progress.logical_step,
            official=official, stopping=getattr(self, "training_control", None),
        )
        self._offer(runtime, "boundary", metrics={**official, **extra} if extra else official)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> float:
        return evaluate(
            runtime, self.data, self.model, self._stream, self.objective, trainer_epoch, component_id,
        )

    def _forward_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        return forward_loss(self._stream, self.model, self.objective, batches, component_id, runtime)

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        self.research.rewind(
            self.lifecycle.output_directory, restored=self._restored, phase=self.progress.phase,
            progress={"logical_step": int(self.progress.logical_step)},
        )
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self.batch_schedule is not None and runtime.world_size != 1:
            raise RuntimeFailure("batch_schedule is implemented for one process")
        self.losses = self.journal.begin(
            restored=self._restored, logical_step=self.progress.logical_step, rewind_samples_on_rank0=False,
        )
        if not self._restored:
            self.progress.reset_fresh()
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        require_group_size(self.data, self.accumulation, str(component_id))
        self._stream = stream_for(self.model.unwrapped(), runtime.device)
        record_rfft(runtime, self.lifecycle.output_directory)
        runtime.barrier()
        return str(component_id)

    def _offer_step(self, runtime: ProcessRuntime, epoch: int) -> None:
        step_metrics = self.research.score(
            runtime, output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity, kind="step", epoch=epoch,
            logical_step=self.progress.logical_step, stopping=getattr(self, "training_control", None),
        )
        if step_metrics:
            self._offer(runtime, "step", metrics=step_metrics)
        else:
            self._offer(runtime, "step")

    def _offer(self, runtime: ProcessRuntime, event: str, metrics: dict[str, float] | None = None) -> None:
        if event == "step":
            self.lifecycle.save_if_needed(
                event, logical_step=self.progress.logical_step, metrics=metrics, prepare=self._sync_ranks,
            )
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.progress.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        runtime.broadcast_buffers(self.model)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        return stop_requested(self, runtime)

    def _epoch_cap_reached(self) -> bool:
        return int(self.data.epoch) >= self.max_epochs

    def _result(self, reason: str) -> RunResult:
        return finish_training_run(self, reason, prepare=self._sync_ranks)
