"""DeepFilterNet3 training loop for the experiment framework.

The loss, cosine table, and batch-size thresholds are the functions already
used by ``DeepFilterNetTask``. This trainer does not call ``train_update``.
``DFStream`` is reset before every waveform analysis, clean first and then
noisy, matching that task. ``Batch.targets`` is clean and ``Batch.inputs`` is
noisy. The cosine index is ``epoch * steps_per_epoch +
index_in_epoch`` after the table is rebuilt for the current epoch length.
It is not the sum of optimizer updates from earlier epochs.

Validation runs at the end of an epoch. Each utterance is scored on its own
and the scores are averaged with equal weight. Best is a strict improvement.
Early stopping resets its counter when validation loss improves and stops
after ``early_stopping_patience`` non-improvements. The shipped patience is 25.

A batch schedule is applied only for world size 1, and only by rebinding the
DeepFilterNet3 sampler. Cached plans for the previous width are discarded.
"""

from __future__ import annotations

from src.experiments.stopping import observe_validation, should_stop

from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.deepfilternet3.artifacts import write_best, write_rfft
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.execution.journal import StepJournal
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.trainers import (
    _close_epoch_view,
    _rank,
)
from src.se.models.deepfilternet3.dsp import DFStream
from src.experiments.calc.batch_schedule import scheduled_batch_size
from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.calc.errors import TrainingError
from src.experiments.official.deepfilternet3.reference import deepfilter_cosine_schedules
from src.experiments.official.deepfilternet3.schedule import schedule_index
from src.experiments.official.deepfilternet3.selection import note_early_stop
from src.experiments.official.deepfilternet3.training_step import component_losses

_PENDING = "pending_validation"
_IN_EPOCH = "in_epoch"


def _clean_noisy(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a micro-batch onto ``component_losses`` argument order.

    Snapshot collation stores the degraded waveform in ``Batch.inputs`` and
    the clean target in ``Batch.targets``. ``_loss_tensor`` takes clean first.
    Passing the fields in declaration order trains and validates on the
    swapped pair, which changes the loss, the local-SNR target, and which
    checkpoint counts as best.
    """

    return batch.targets, batch.inputs


class DeepFilterNet3Trainer:
    """Official DeepFilterNet3 loop on one AdamW and a per-step cosine."""

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
        self.objective = objective
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        _require_adamw(optimizer)
        if model.policy.find_unused_parameters is not True:
            raise BuildError("DeepFilterNet3 requires find_unused_parameters")
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.optimizer_id = str(getattr(optimizer, "_experiment_component_id"))
        self.audio_sample_rate = int(model.unwrapped().generator.params.sr)
        self.audio_teachers = ("official_mixed_target",)
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.grad_clip = _positive_float("grad_clip", grad_clip)
        self.max_epochs = _positive_int("max_epochs", max_epochs)
        self.early_stopping_patience = _positive_int("early_stopping_patience", early_stopping_patience)
        self.schedule_epochs = _positive_int("epochs", epochs)
        self.lr_min = float(lr_min)
        self.lr_warmup = float(lr_warmup)
        self.warmup_epochs = _non_negative_int("warmup_epochs", warmup_epochs)
        self.weight_decay_end = float(weight_decay_end)
        self.batch_schedule = _schedule(batch_schedule)
        self.batch_size_limit = _positive_int("batch_size_limit", batch_size_limit)
        self.stop_after_logical_steps = (
            None if stop_after_logical_steps is None else _positive_int("stop_after_logical_steps", stop_after_logical_steps)
        )
        self._table_steps = _positive_int("steps_per_epoch", steps_per_epoch)
        self._learning_rates, self._decays = self._tables(self._table_steps)
        self.logical_step = 0
        self.optimizer_updates = {self.optimizer_id: 0}
        self.losses: list[float] = []
        self.learning_rates: list[float] = []
        self.weight_decays: list[float] = []
        self.history: list[dict[str, Any]] = []
        self.best_loss: float | None = None
        self.best_epoch: int | None = None
        self.plateau_metric: float | None = None
        self.plateau_count = 0
        self.early_stopped = False
        self.phase = _IN_EPOCH
        self.finished_epoch: int | None = None
        self.boundary: dict[str, Any] | None = None
        self.stop_reason = "initialized"
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

    def fit(self) -> RunResult:

        """Train until the step budget, the epoch cap, or early stopping. The cosine step runs before the optimizer."""
        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        if self.phase == _PENDING:
            self._finish_boundary(runtime, component_id)
            if should_stop(self):
                return self._result("completed")
            if self.early_stopped or self._epoch_cap_reached():
                return self._result("completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                if self.logical_step > 0:
                    self._offer(runtime, "stop")
                return self._result("interrupted")
            if self._epoch_cap_reached():
                return self._result("completed")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            if index == 0:
                self._apply_batch_schedule(epoch)
            epoch, index, view, steps = self._open_window(component_id, epoch, index)
            try:
                self.model.train()
                position = schedule_index(epoch, index // self.accumulation, steps)
                learning_rate, decay = self._apply_cosine(position)
                self.optimizer.zero_grad(set_to_none=True)
                total: torch.Tensor | None = None
                sample_ids: list[str] = []
                for offset in range(self.accumulation):
                    sync = offset == self.accumulation - 1
                    batches = view.batch(index + offset)
                    sample_ids.extend(batch.sample_id for batch in batches)
                    with runtime.sync_gradients([self.model], enabled=sync):
                        with runtime.autocast():
                            micro = self._forward_loss(batches, component_id, runtime)
                        runtime.backward(micro / self.accumulation)
                    detached = (micro / self.accumulation).detach()
                    total = detached if total is None else total + detached
                runtime.clip(self.model, self.grad_clip, optimizer=self.optimizer)
                result = runtime.step(self.optimizer)
                runtime.end_step()
                if total is None or not bool(torch.isfinite(total).all()):
                    raise RuntimeFailure(f"component {component_id}: loss is not finite")
                self.logical_step += 1
                index += self.accumulation
                if result.applied:
                    self.optimizer_updates[self.optimizer_id] += 1
                loss_value = float(total)
                self.losses.append(loss_value)
                self.learning_rates.append(learning_rate)
                self.weight_decays.append(decay)
                self.data.epoch = epoch
                self.data.update_index = index
                self._note_step(
                    applied=result.applied,
                    epoch=epoch,
                    loss=loss_value,
                    learning_rate=learning_rate,
                    sample_id=",".join(sample_ids),
                    weight_decay=decay,
                    schedule_index=position,
                )
                if index != view.updates:
                    step_metrics = self.research.score(
                        runtime,
                        output_directory=self.lifecycle.output_directory,
                        config_identity=self.lifecycle.config_identity,
                        kind="step",
                        epoch=epoch,
                        logical_step=self.logical_step,
                        stopping=getattr(self, "training_control", None),
                    )
                    if step_metrics:
                        self._offer(runtime, "step", metrics=step_metrics)
                    else:
                        self._offer(runtime, "step")
                    continue
                self.phase = _PENDING
                self.finished_epoch = epoch
                self.boundary = {"epoch": epoch, "schedule_steps": steps}
                self._offer(runtime, "boundary_pending")
                if self._stop_requested(runtime):
                    return self._result("interrupted")
                self._finish_boundary(runtime, component_id)
                if should_stop(self):
                    return self._result("completed")
                if self.early_stopped or self._epoch_cap_reached():
                    return self._result("completed")
            finally:
                _close_epoch_view(view)
        return self._result("completed")

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "boundary": self.boundary,
            "early_stopped": self.early_stopped,
            "finished_epoch": self.finished_epoch,
            "history": list(self.history),
            "learning_rates": list(self.learning_rates),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "plateau_count": self.plateau_count,
            "plateau_metric": self.plateau_metric,
            "schedule_steps": self._table_steps,
            "stop_reason": self.stop_reason,
            "weight_decays": list(self.weight_decays),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        logical_step = state.get("logical_step")
        updates = state.get("optimizer_updates")
        phase = state.get("phase")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
            raise RuntimeFailure("trainer state is invalid")
        if phase not in {_IN_EPOCH, _PENDING}:
            raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
        steps = state.get("schedule_steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise RuntimeFailure("schedule_steps is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        self.phase = phase
        self._table_steps = steps
        self._learning_rates, self._decays = self._tables(steps)
        self.history = list(state.get("history") or [])
        self.learning_rates = [float(value) for value in state.get("learning_rates") or []]
        self.weight_decays = [float(value) for value in state.get("weight_decays") or []]
        self.plateau_count = int(state.get("plateau_count") or 0)
        metric = state.get("plateau_metric")
        self.plateau_metric = None if metric is None else float(metric)
        best = state.get("best_loss")
        self.best_loss = None if best is None else float(best)
        best_epoch = state.get("best_epoch")
        self.best_epoch = None if best_epoch is None else int(best_epoch)
        self.early_stopped = bool(state.get("early_stopped", False))
        boundary = state.get("boundary")
        self.boundary = None if boundary is None else dict(boundary)
        finished = state.get("finished_epoch")
        self.finished_epoch = None if finished is None else int(finished)
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
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
        """Validate, record the official best loss, then run research evaluation."""
        if self.phase != _PENDING or self.finished_epoch is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        trainer_epoch = int(self.finished_epoch)
        loss = self._validate(runtime, trainer_epoch, component_id)
        self.plateau_metric, self.plateau_count, stop = note_early_stop(
            self.plateau_metric, self.plateau_count, loss, self.early_stopping_patience
        )
        self.history.append({"epoch": trainer_epoch, "loss": loss})
        improved = self.best_loss is None or loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.best_epoch = trainer_epoch
        self._write_best(runtime, self.lifecycle.output_directory, improved)
        self.phase = _IN_EPOCH
        self.finished_epoch = None
        self.boundary = None
        self.data.epoch = trainer_epoch + 1
        self.data.update_index = 0
        if stop:
            self.early_stopped = True
        official = {"loss": float(loss)}
        observe_validation(
            getattr(self, "training_control", None),
            official,
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
            official=official,
            stopping=getattr(self, "training_control", None),
        )
        self._offer(runtime, "boundary", metrics={**official, **extra} if extra else official)

    def _offer(
        self,
        runtime: ProcessRuntime,
        event: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Offer this state to the selector.

        A boundary and a stop broadcast the BatchNorm buffers whether or not
        the selector saves, as they did before selectors. Inside an epoch the
        broadcast runs only when a file is written.
        """

        if event == "step":
            self.lifecycle.save_if_needed(event, logical_step=self.logical_step, metrics=metrics, prepare=self._sync_ranks)
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        """Broadcast BatchNorm buffers so every rank holds the state that is saved.

        DDP's next forward would copy rank 0's buffers. Ranks see different
        utterances, so the local running stats differ until that copy. This is
        not SyncBatchNorm and does not average the two batches.
        """

        runtime.broadcast_buffers(self.model)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> float:
        """Score each validation utterance with the injected objective and average them with equal weight."""
        module = self.model
        was_training = bool(module.training)
        cursor = (int(self.data.epoch), int(self.data.update_index))
        rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
        values: list[float] = []
        try:
            module.eval()
            if _rank(runtime) == 0:
                with torch.no_grad():
                    for batch in self.data.validation_epoch(trainer_epoch):
                        if not isinstance(batch, Batch):
                            raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                        batch = runtime.move(batch)
                        clean, noisy = _clean_noisy(batch)
                        for row in range(int(clean.shape[0])):
                            value = self._loss_tensor(
                                clean[row : row + 1],
                                noisy[row : row + 1],
                                self.model.unwrapped(),
                            )
                            if not bool(torch.isfinite(value).all()):
                                raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                            values.append(float(value.detach()))
            gathered = runtime.gather_objects(values)
        finally:
            restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
            if was_training:
                module.train()
            moved = (int(self.data.epoch), int(self.data.update_index))
            if moved != cursor:
                raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        scores = next((item for item in gathered if item), None)
        if not scores:
            raise RuntimeFailure(f"component {component_id}: validation produced no utterances")
        return sum(scores) / len(scores)

    def _write_best(self, runtime: ProcessRuntime, directory: Path, improved: bool) -> None:
        if not improved or _rank(runtime) != 0:
            runtime.barrier()
            return
        payload = {
            "epoch": self.best_epoch,
            "generator": self.model.unwrapped().export_for_inference(),
            "logical_step": self.logical_step,
            "loss": self.best_loss,
        }
        write_best(directory, payload)
        runtime.barrier()

    def _forward_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        """Average component_losses over one accumulation window."""
        if not batches:
            raise RuntimeFailure(f"component {component_id}: empty micro-batch")
        total: torch.Tensor | None = None
        for batch in batches:
            batch = runtime.move(batch)
            clean, noisy = _clean_noisy(batch)
            value = self._loss_tensor(clean, noisy, self.model.execution())
            total = value if total is None else total + value
        assert total is not None
        return total / len(batches)

    def _loss_tensor(self, clean: torch.Tensor, noisy: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        """Return the scalar total from one component_losses call."""
        if self._stream is None:
            raise RuntimeFailure("DeepFilterNet3 stream is not ready")
        try:
            parts = component_losses(self._stream, module, clean, noisy, self.objective)
        except TrainingError as exc:
            raise RuntimeFailure(str(exc)) from exc
        total = parts["total"]
        if not bool(torch.isfinite(total).all()):
            raise RuntimeFailure("deepfilternet loss is not finite")
        return total

    def _apply_batch_schedule(self, epoch: int) -> None:
        if self.batch_schedule is None:
            return
        size = scheduled_batch_size(epoch, self.batch_size_limit, self.batch_schedule)
        current = getattr(getattr(self.data, "batcher", None), "micro_batch", None)
        if current == size:
            return
        rebind = getattr(self.data, "rebind_micro_batch", None)
        if not callable(rebind):
            raise RuntimeFailure("data cannot rebind a DeepFilterNet3 micro-batch")
        rebind(size)

    def _retarget(self, steps_per_epoch: int) -> None:
        if steps_per_epoch == self._table_steps and self._learning_rates:
            return
        self._learning_rates, self._decays = self._tables(steps_per_epoch)
        self._table_steps = steps_per_epoch

    def _apply_cosine(self, index: int) -> tuple[float, float]:
        """Copy the cosine-table learning rate and weight decay onto the optimizer before it steps."""
        if index < 0 or index >= len(self._learning_rates):
            raise RuntimeFailure(
                f"deepfilter cosine step {index} is outside the schedule of {len(self._learning_rates)}"
            )
        learning_rate = self._learning_rates[index]
        decay = self._decays[index]
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
            if float(group["weight_decay"]) > 0.0:
                group["weight_decay"] = decay
        return learning_rate, decay

    def _tables(self, steps_per_epoch: int) -> tuple[list[float], list[float]]:
        config = {
            # _require_adamw refuses an optimizer built with any other base values.
            "optimizer": {"lr": 0.001, "weight_decay": 1e-12},
            "scheduler": {
                "epochs": self.schedule_epochs,
                "lr_min": self.lr_min,
                "lr_warmup": self.lr_warmup,
                "name": "deepfilter_cosine",
                "steps_per_epoch": steps_per_epoch,
                "warmup_epochs": self.warmup_epochs,
                "weight_decay_end": self.weight_decay_end,
            },
        }
        try:
            return deepfilter_cosine_schedules(config)
        except TrainingError as exc:
            raise BuildError(str(exc)) from exc

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        """Rewind logs, build the DF stream, and record the FFT kernel."""

        self.research.rewind(
            self.lifecycle.output_directory,
            restored=self._restored,
            phase=self.phase,
            progress={"logical_step": int(self.logical_step)},
        )
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self.batch_schedule is not None and runtime.world_size != 1:
            raise RuntimeFailure("batch_schedule is implemented for one process")
        self.losses = self.journal.begin(
            restored=self._restored,
            logical_step=self.logical_step,
            rewind_samples_on_rank0=False,
        )
        if not self._restored:
            self.logical_step = 0
            self.optimizer_updates = {self.optimizer_id: 0}
            self.losses = []
            self.learning_rates = []
            self.weight_decays = []
            self.history = []
            self.best_loss = None
            self.best_epoch = None
            self.plateau_metric = None
            self.plateau_count = 0
            self.early_stopped = False
            self.phase = _IN_EPOCH
            self.finished_epoch = None
            self.boundary = None
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        self._require_group_size(component_id)
        self._stream = _stream_for(self.model.unwrapped(), runtime.device)
        self._record_rfft(runtime)
        runtime.barrier()
        return str(component_id)

    def _open_window(
        self,
        component_id: str,
        epoch: int,
        index: int,
    ) -> tuple[int, int, Any, int]:
        """Check the cosine epoch and rebuild the table for its length."""

        view = self.data.train_epoch(epoch, start_update=index)
        try:
            if view.updates < 1:
                raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
            if view.updates % self.accumulation != 0:
                raise RuntimeFailure(
                    f"component {component_id}: epoch {epoch} has {view.updates} micro-batches, "
                    f"which is not divisible by accumulation {self.accumulation}"
                )
            steps = view.updates // self.accumulation
            self._retarget(steps)
            if index > view.updates or index % self.accumulation != 0:
                raise RuntimeFailure(
                    f"component {component_id}: update {index} is not an optimizer boundary"
                )
            if index == view.updates:
                raise RuntimeFailure(
                    f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
                )
        except Exception:
            _close_epoch_view(view)
            raise
        return epoch, index, view, steps

    def _note_step(
        self,
        *,
        applied: bool,
        epoch: int,
        loss: float,
        learning_rate: float,
        sample_id: str,
        weight_decay: float,
        schedule_index: int,
    ) -> None:
        self.journal.write_metrics(
            {
                "applied": applied,
                "epoch": epoch,
                "logical_step": self.logical_step,
                "loss": loss,
                "lr": learning_rate,
                "optimizer_updates": dict(self.optimizer_updates),
                "sample_id": sample_id,
                "weight_decay": weight_decay,
            }
        )
        self.journal.write_sample(
            {
                "epoch": epoch,
                "logical_step": self.logical_step,
                "loss": loss,
                "lr": learning_rate,
                "sample_id": sample_id,
                "schedule_index": schedule_index,
                "weight_decay": weight_decay,
            }
        )

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is not None and group_size != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _record_rfft(self, runtime: ProcessRuntime) -> None:
        """Write the kernel actually selected on this host. Rank 0 writes the file."""

        problem = ""
        if _rank(runtime) == 0:
            try:
                from src.se.models.deepfilternet3.dsp import rfft_runtime_record

                write_rfft(Path(self.lifecycle.output_directory), rfft_runtime_record())
            except Exception as exc:
                problem = f"{type(exc).__name__}: {exc}"
        if runtime.world_size > 1:
            reported = runtime.gather_objects(problem)
            problem = next((item for item in reported if item), "")
        if problem:
            raise RuntimeFailure(f"DeepFilterNet3 realfft kernel: {problem}")

    def _stop_requested(self, runtime: object | None = None) -> bool:
        from src.experiments.signals import ranks_should_stop

        capped = self.stop_after_logical_steps is not None and self.logical_step >= self.stop_after_logical_steps
        return ranks_should_stop(runtime, extra=capped)

    def _epoch_cap_reached(self) -> bool:
        return int(self.data.epoch) >= self.max_epochs

    def _result(self, reason: str) -> RunResult:
        if reason == "completed":
            epoch = getattr(self.data, "epoch", None)
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                epoch = None
            extra = self.research.score(
                self.runtime,
                output_directory=self.lifecycle.output_directory,
                config_identity=self.lifecycle.config_identity,
                kind="end",
                epoch=epoch,
                logical_step=self.logical_step,
                stopping=getattr(self, "training_control", None),
            )
            if extra:
                self.lifecycle.save_if_needed("final", logical_step=self.logical_step, metrics=extra, prepare=self._sync_ranks)
            else:
                self.lifecycle.save_if_needed("final", logical_step=self.logical_step, prepare=self._sync_ranks)
        self.stop_reason = reason
        if reason == "completed" and should_stop(self):
            reason = "early_stopping"
        return self.lifecycle.finish(
            RunResult(
                stop_reason=reason,
                logical_step=self.logical_step,
                optimizer_updates=dict(self.optimizer_updates),
                output_dir=self.lifecycle.output_directory,
            )
        )

def _stream_for(model: DeepFilterNet3Model, device: torch.device) -> DFStream:
    """Open a DFStream from the model's own sample rate and FFT sizes."""
    params = model.generator.params
    return DFStream(
        params.sr,
        params.fft_size,
        params.hop_size,
        params.nb_erb,
        params.nb_df,
        params.min_nb_freqs,
        params.norm_tau,
        device,
    )


def _require_adamw(optimizer: torch.optim.Optimizer) -> None:
    """Refuse an optimizer that is not the official AdamW."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError("DeepFilterNet3 optimizer must be AdamW")
    group = optimizer.param_groups[0]
    if float(group["lr"]) != 0.001:
        raise BuildError("DeepFilterNet3 learning rate must be 0.001")
    betas = tuple(group["betas"])
    if betas != (0.9, 0.999):
        raise BuildError("DeepFilterNet3 betas must be (0.9, 0.999)")
    if float(group["weight_decay"]) != 1e-12:
        raise BuildError("DeepFilterNet3 weight decay must be 1e-12")
    if group.get("amsgrad") is not True:
        raise BuildError("DeepFilterNet3 AdamW requires amsgrad")
    if not isinstance(getattr(optimizer, "_experiment_component_id", None), str):
        raise BuildError("optimizer has no component id")


def _schedule(raw: list[list[int]] | None) -> list[tuple[int, int]] | None:
    """Parse the optional world-size-1 batch-width schedule."""
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise BuildError("batch_schedule must be a non-empty list of [epoch, batch_size]")
    schedule = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise BuildError("batch_schedule entries must be [epoch, batch_size]")
        schedule.append((_non_negative_int("batch_schedule epoch", item[0]), _positive_int("batch_schedule batch_size", item[1])))
    if schedule[0][0] != 0:
        raise BuildError("batch_schedule must start at epoch 0")
    return schedule


def _positive_int(name: str, value: object) -> int:
    """Require a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be an integer >= 1")
    return value


def _non_negative_int(name: str, value: object) -> int:
    """Require an integer that is zero or greater."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value


def _positive_float(name: str, value: object) -> float:
    """Require a positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)
