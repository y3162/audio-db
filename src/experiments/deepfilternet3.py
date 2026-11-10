"""DeepFilterNet3 training loop for the experiment framework.

The loss, cosine table, and batch-size thresholds are the functions already
used by ``DeepFilterNetTask``. This trainer does not call ``train_update``.
``DFStream`` is reset before every waveform analysis, clean first and then
noisy, matching that task. The cosine index is ``epoch * steps_per_epoch +
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

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.checkpoint import CheckpointManager
from src.experiments.data import Batch
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.models.deepfilternet3 import DeepFilterNet3Model
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.trainers import (
    _append_jsonl,
    _batch_on_device,
    _close_epoch_view,
    _rank,
    _rewind_metrics,
    _sample_path,
)
from src.se.models.deepfilternet3.dsp import DFStream, vorbis_window
from src.experiments.calc.batch_schedule import scheduled_batch_size
from src.experiments.calc.distributed import unwrap
from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.deepfilternet import (
    _local_snr_db,
    _multi_resolution_spectral,
    _vorbis_istft,
    deepfilter_cosine_schedules,
)

_PENDING = "pending_validation"
_IN_EPOCH = "in_epoch"
_LSNR = 1e-3


def component_losses(
    stream: DFStream,
    module: torch.nn.Module,
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Spectral, local-SNR, and total loss for one batch.

    The call order matches ``DeepFilterNetTask._loss``. ``module`` may be a
    DDP wrapper; parameters are read from the unwrapped generator.
    """

    network = unwrap(module)
    generator = network.generator if isinstance(network, DeepFilterNet3Model) else network
    stream.reset()
    _spec, _erb, _feat, clean_complex = _analyze(stream, generator, clean)
    stream.reset()
    noisy_spec, noisy_erb, noisy_feat, noisy_complex = _analyze(stream, generator, noisy)
    enhanced, _mask, lsnr, _coefs = module(noisy_spec, noisy_erb, noisy_feat)
    enhanced_complex = torch.view_as_complex(enhanced.squeeze(1))
    frames = min(enhanced_complex.shape[-2], clean_complex.shape[-2], noisy_complex.shape[-2])
    bins = min(enhanced_complex.shape[-1], clean_complex.shape[-1], noisy_complex.shape[-1])
    enhanced_complex = enhanced_complex[..., :frames, :bins]
    clean_complex = clean_complex[..., :frames, :bins]
    noisy_complex = noisy_complex[..., :frames, :bins]
    params = generator.params
    window = vorbis_window(params.fft_size).to(device=enhanced_complex.device, dtype=enhanced_complex.real.dtype)
    enhanced_wave = _vorbis_istft(enhanced_complex, params.fft_size, params.hop_size, window)
    clean_wave = _vorbis_istft(clean_complex, params.fft_size, params.hop_size, window)
    spectral = _multi_resolution_spectral(enhanced_wave, clean_wave)
    target = _local_snr_db(
        clean_complex,
        noisy_complex - clean_complex,
        sample_rate=params.sr,
        n_fft=params.fft_size,
        hop=params.hop_size,
        low=params.lsnr_min - 1,
        high=params.lsnr_max + 1,
    )
    predicted = lsnr.reshape(target.shape[0], -1)
    if predicted.shape[-1] != target.shape[-1]:
        raise RuntimeFailure(
            f"local SNR length {predicted.shape[-1]} does not match the 20 ms target {target.shape[-1]}"
        )
    local = F.mse_loss(predicted, target) * _LSNR
    return {"local_snr": local, "spectral": spectral, "total": spectral + local}


def schedule_index(epoch: int, index_in_epoch: int, steps_per_epoch: int) -> int:
    """DeepFilterNet's table index. Past epochs are not summed."""

    if epoch < 0 or index_in_epoch < 0 or steps_per_epoch < 1:
        raise RuntimeFailure("schedule position is invalid")
    if index_in_epoch >= steps_per_epoch:
        raise RuntimeFailure(
            f"index {index_in_epoch} is outside an epoch of {steps_per_epoch} updates"
        )
    return epoch * steps_per_epoch + index_in_epoch


def note_early_stop(metric: float | None, count: int, loss: float, patience: int) -> tuple[float, int, bool]:
    """Reset the counter on a strict improvement. Stop at ``patience`` misses."""

    if patience < 1:
        raise RuntimeFailure("early_stopping_patience must be >= 1")
    if metric is None or loss < metric:
        return loss, 0, False
    count += 1
    return metric, count, count >= patience


class DeepFilterNet3Trainer:
    """Official DeepFilterNet3 loop on one AdamW and a per-step cosine."""

    def __init__(
        self,
        model: PreparedModel,
        data: Any,
        optimizer: torch.optim.Optimizer,
        *,
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
        checkpoint_interval: int = 1,
    ) -> None:
        if not isinstance(model, PreparedModel):
            raise BuildError("model must be a prepared module handle")
        if not isinstance(model.state_module, DeepFilterNet3Model):
            raise BuildError("model must be DeepFilterNet3Model")
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        _require_adamw(optimizer)
        if model.policy.find_unused_parameters is not True:
            raise BuildError("DeepFilterNet3 requires find_unused_parameters")
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.optimizer_id = str(getattr(optimizer, "_experiment_component_id"))
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
        self.checkpoint_interval = _positive_int("checkpoint_interval", checkpoint_interval)
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
        self._restored = False
        self._stream: DFStream | None = None

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        component_id = getattr(self, "_experiment_component_id", "trainer")
        rank = _rank(runtime)
        metrics_path = Path(checkpoint.directory) / "metrics.jsonl"
        sample_path = _sample_path(checkpoint.directory, runtime)
        if self.batch_schedule is not None and runtime.world_size != 1:
            raise RuntimeFailure("batch_schedule is implemented for one process")
        if self._restored:
            if rank == 0:
                self.losses = _rewind_metrics(metrics_path, self.logical_step)
            if sample_path is not None and rank != 0:
                self.losses = _rewind_metrics(sample_path, self.logical_step)
        else:
            if rank == 0:
                metrics_path.write_text("", encoding="utf-8")
            if sample_path is not None:
                sample_path.write_text("", encoding="utf-8")
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
        self._require_group_size(component_id)
        self._stream = _stream_for(self.model.state_module, runtime.device)
        self._record_rfft(runtime, checkpoint)
        runtime.barrier()
        if self.phase == _PENDING:
            self._finish_boundary(runtime, checkpoint, component_id)
            if self.early_stopped or self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                if self.logical_step > 0:
                    checkpoint.save(checkpoint.directory, logical_step=self.logical_step, label="step")
                return self._result(checkpoint, "interrupted")
            if self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            if index == 0:
                self._apply_batch_schedule(epoch)
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
                self.model.module.train()
                position = schedule_index(epoch, index // self.accumulation, steps)
                learning_rate, decay = self._apply_cosine(position)
                self.optimizer.zero_grad(set_to_none=True)
                total: torch.Tensor | None = None
                sample_ids: list[str] = []
                for offset in range(self.accumulation):
                    sync = offset == self.accumulation - 1
                    batches = view.batch(index + offset)
                    sample_ids.extend(batch.sample_id for batch in batches)
                    with runtime.sync_context([self.model], sync=sync):
                        with runtime.autocast():
                            micro = self._forward_loss(batches, component_id, runtime)
                        runtime.backward(micro / self.accumulation)
                    detached = (micro / self.accumulation).detach()
                    total = detached if total is None else total + detached
                runtime.clip(self.model.parameters, self.grad_clip, optimizer=self.optimizer)
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
                sample_id = ",".join(sample_ids)
                if rank == 0:
                    self._append_metrics(
                        metrics_path, loss_value, result.applied, sample_id, learning_rate, decay, epoch
                    )
                if sample_path is not None:
                    _append_jsonl(
                        sample_path,
                        {
                            "epoch": epoch,
                            "logical_step": self.logical_step,
                            "loss": loss_value,
                            "lr": learning_rate,
                            "sample_id": sample_id,
                            "schedule_index": position,
                            "weight_decay": decay,
                        },
                    )
                if index != view.updates:
                    if self.logical_step % self.checkpoint_interval == 0:
                        self._save(runtime, checkpoint, "step")
                    continue
                self.phase = _PENDING
                self.finished_epoch = epoch
                self.boundary = {"epoch": epoch, "schedule_steps": steps}
                self._save(runtime, checkpoint, "validation_pending")
                if self._stop_requested(runtime):
                    return self._result(checkpoint, "interrupted")
                self._finish_boundary(runtime, checkpoint, component_id)
                if self.early_stopped or self._epoch_cap_reached():
                    return self._result(checkpoint, "completed")
            finally:
                _close_epoch_view(view)
        return self._result(checkpoint, "completed")

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

    def _finish_boundary(self, runtime: ProcessRuntime, checkpoint: CheckpointManager, component_id: str) -> None:
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
        self._write_best(runtime, checkpoint.directory, improved)
        self.phase = _IN_EPOCH
        self.finished_epoch = None
        self.boundary = None
        self.data.epoch = trainer_epoch + 1
        self.data.update_index = 0
        if stop:
            self.early_stopped = True
        self._save(runtime, checkpoint, "step")

    def _save(self, runtime: ProcessRuntime, checkpoint: CheckpointManager, label: str) -> None:
        """Broadcast BatchNorm buffers, then write one checkpoint.

        DDP's next forward would copy rank 0's buffers. Ranks see different
        utterances, so the local running stats differ until that copy. This is
        not SyncBatchNorm and does not average the two batches.
        """

        runtime.broadcast_buffers(self.model.state_module)
        checkpoint.save(checkpoint.directory, logical_step=self.logical_step, label=label)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> float:
        module = self.model.module
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
                        batch = _batch_on_device(batch, runtime.device)
                        for row in range(int(batch.inputs.shape[0])):
                            value = self._loss_tensor(
                                batch.inputs[row : row + 1],
                                batch.targets[row : row + 1],
                                self.model.state_module,
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
            "generator": self.model.state_module.export_for_inference(),
            "logical_step": self.logical_step,
            "loss": self.best_loss,
        }
        target = directory / "valid.loss.best.pt"
        temporary = directory / ".valid.loss.best.pt.tmp"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeFailure(f"best checkpoint failed: {exc}") from exc
        runtime.barrier()

    def _forward_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        if not batches:
            raise RuntimeFailure(f"component {component_id}: empty micro-batch")
        total: torch.Tensor | None = None
        for batch in batches:
            batch = _batch_on_device(batch, runtime.device)
            value = self._loss_tensor(batch.inputs, batch.targets, self.model.module)
            total = value if total is None else total + value
        assert total is not None
        return total / len(batches)

    def _loss_tensor(self, clean: torch.Tensor, noisy: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        if self._stream is None:
            raise RuntimeFailure("DeepFilterNet3 stream is not ready")
        try:
            parts = component_losses(self._stream, module, clean, noisy)
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
        group = self.optimizer.param_groups[0]
        config = {
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
        del group
        try:
            return deepfilter_cosine_schedules(config)
        except TrainingError as exc:
            raise BuildError(str(exc)) from exc

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is not None and group_size != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _record_rfft(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> None:
        """Write the kernel actually selected on this host. Rank 0 writes the file."""

        problem = ""
        if _rank(runtime) == 0:
            try:
                from src.se.models.deepfilternet3.dsp import rfft_runtime_record

                record = rfft_runtime_record()
                path = Path(checkpoint.directory) / "rfft.json"
                temporary = path.with_name(".rfft.json.tmp")
                temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, path)
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

    def _result(self, checkpoint: CheckpointManager, reason: str) -> RunResult:
        self.stop_reason = reason
        return RunResult(
            stop_reason=reason,
            logical_step=self.logical_step,
            optimizer_updates=dict(self.optimizer_updates),
            output_dir=checkpoint.directory,
        )

    def _append_metrics(
        self,
        path: Path,
        loss_value: float,
        applied: bool,
        sample_id: str,
        learning_rate: float,
        decay: float,
        epoch: int,
    ) -> None:
        record = {
            "applied": applied,
            "epoch": epoch,
            "logical_step": self.logical_step,
            "loss": loss_value,
            "lr": learning_rate,
            "optimizer_updates": dict(self.optimizer_updates),
            "sample_id": sample_id,
            "weight_decay": decay,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _analyze(stream: DFStream, generator: torch.nn.Module, waveform: torch.Tensor):
    return _analyze_features(stream, generator, waveform)


def _analyze_features(stream: DFStream, generator: torch.nn.Module, waveform: torch.Tensor):
    params = generator.params
    padded = F.pad(waveform, (0, params.fft_size))
    stream.reset()
    complex_spec = stream.analyze(padded)
    spec, erb, feat = stream.features(complex_spec)
    return spec, erb, feat, complex_spec


def _stream_for(model: DeepFilterNet3Model, device: torch.device) -> DFStream:
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
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be an integer >= 1")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)
