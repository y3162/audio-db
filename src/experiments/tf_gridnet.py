"""TF-GridNet MS-SNSD training loop for the experiment framework.

``StandardTrainer`` steps a scheduler after every optimizer update. The
MS-SNSD recipe steps ``ReduceLROnPlateau`` once per finished chunk epoch,
using the weighted validation loss. This trainer owns that boundary.
``ProcessRuntime`` still owns backward, gradient clipping, and ``optimizer.step``.

One epoch is the micro-batch sequence ``plan_chunk_epoch`` already exposed
through ``SnapshotDataModule.train_epoch``. A ``max_logical_steps`` stop in
the middle of that sequence does not validate, step the scheduler, or select
a checkpoint. The version-1 checkpoint file is one per logical step, so the
committed boundary has two states:

- ``validation_pending``: the epoch's optimizer updates are done, and
  validation, the scheduler, and n-best selection have not run.
- the replacement ``step`` file: those three have all run once.

A crash between them resumes from ``validation_pending`` and runs the three
once. They are not split into more files, because a scheduler step that was
saved without the matching selection would run again on resume.
``valid.si_snr`` is not a selection criterion. The old trainer's SI-SNR
ranking stays in ``src/training`` and is not copied here.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from src.experiments.checkpoint import CheckpointManager
from src.experiments.data import Batch
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.losses import MultiResolutionL1Loss
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
from src.training.core.average import average_checkpoint_files, replace_symlink
from src.training.core.rng import capture_rank_rng, restore_rank_rng

_EPOCH_FILE = re.compile(r"^(\d+)epoch\.pt$")
_PENDING = "pending_validation"
_IN_EPOCH = "in_epoch"


def weighted_validation_loss(parts: list[tuple[float, int]]) -> tuple[float, int]:
    """Mean of batch-mean losses weighted by sample count.

    A rank mean of means is a different number when the ranks do not hold
    the same number of samples. The weight is the number of waveforms in
    the batch, which is the count ``_reduce_weighted_reports`` multiplies
    by ``len(case_id)``.
    """

    if not parts:
        raise RuntimeFailure("validation produced no batches")
    total = 0.0
    count = 0
    for loss, weight in parts:
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise RuntimeFailure("validation weight must be a positive integer")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise RuntimeFailure("validation loss must be a number")
        total += float(loss) * weight
        count += weight
    if count < 1:
        raise RuntimeFailure("validation produced no batches")
    return total / count, count


def rank_validation_loss(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Lowest ``valid.loss`` first. Equal losses keep the earlier epoch."""

    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise RuntimeFailure("keep_nbest must be a positive integer")
    ordered = sorted(history, key=lambda row: (float(row["loss"]), int(row["epoch"])))
    return ordered[:keep]


def early_stop(history: list[dict[str, Any]], espnet_epoch: int, patience: int) -> bool:
    """Stop when the best epoch is more than ``patience`` epochs behind.

    The best row is the lowest loss, then the earlier epoch. This is the
    chunk early-stopping rule. It does not read ``valid.si_snr``.
    """

    if not history:
        return False
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise RuntimeFailure("early_stopping_patience must be an integer >= 0")
    best = min(history, key=lambda row: (float(row["loss"]), int(row["epoch"])))
    return int(espnet_epoch) - int(best["epoch"]) > patience


class TFGridNetTrainer:
    """Single-optimizer MS-SNSD loop. The scheduler steps at epoch end."""

    def __init__(
        self,
        model: PreparedModel,
        loss: MultiResolutionL1Loss,
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
        checkpoint_interval: int = 1,
    ) -> None:
        if not isinstance(model, PreparedModel):
            raise BuildError("model must be a prepared module handle")
        if not isinstance(loss, MultiResolutionL1Loss):
            raise BuildError("loss must be MultiResolutionL1Loss")
        if loss.normalize_variance:
            raise BuildError("MS-SNSD loss cannot normalize variance")
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise BuildError("scheduler must be ReduceLROnPlateau")
        if scheduler.mode != "min":
            raise BuildError("MS-SNSD scheduler mode must be min")
        optimizer_id = getattr(optimizer, "_experiment_component_id", None)
        if not isinstance(optimizer_id, str):
            raise BuildError("optimizer has no component id")
        self.model = model
        self.loss = loss
        self.data = data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_id = optimizer_id
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.grad_clip = _positive_float("grad_clip", grad_clip)
        self.keep_nbest = _positive_int("keep_nbest", keep_nbest)
        self.max_epochs = None if max_epochs is None else _positive_int("max_epochs", max_epochs)
        if early_stopping_patience is not None and (
            isinstance(early_stopping_patience, bool)
            or not isinstance(early_stopping_patience, int)
            or early_stopping_patience < 0
        ):
            raise BuildError("early_stopping_patience must be an integer >= 0")
        self.early_stopping_patience = early_stopping_patience
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = _positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self.checkpoint_interval = _positive_int("checkpoint_interval", checkpoint_interval)
        self.logical_step = 0
        self.optimizer_updates = {optimizer_id: 0}
        self.losses: list[float] = []
        self.learning_rates: list[float] = []
        self.history: list[dict[str, Any]] = []
        self.completed_boundaries: list[dict[str, Any]] = []
        self.boundary: dict[str, Any] | None = None
        self.phase = _IN_EPOCH
        self.finished_epoch: int | None = None
        self.early_stopped = False
        self.stop_reason = "initialized"
        self._restored = False

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        """Train until the step budget, an epoch cap, or early stopping."""

        component_id = getattr(self, "_experiment_component_id", "trainer")
        rank = _rank(runtime)
        metrics_path = Path(checkpoint.directory) / "metrics.jsonl"
        sample_path = _sample_path(checkpoint.directory, runtime)
        if self._restored:
            if not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index"):
                raise RuntimeFailure("resumed data has no epoch position")
            if rank == 0:
                self.losses = _rewind_metrics(metrics_path, self.logical_step)
            if sample_path is not None:
                sample_losses = _rewind_metrics(sample_path, self.logical_step)
                if rank != 0:
                    self.losses = sample_losses
        else:
            if rank == 0:
                metrics_path.write_text("", encoding="utf-8")
            if sample_path is not None:
                sample_path.write_text("", encoding="utf-8")
            self.logical_step = 0
            self.optimizer_updates = {self.optimizer_id: 0}
            self.losses = []
            self.learning_rates = []
            self.history = []
            self.completed_boundaries = []
            self.boundary = None
            self.phase = _IN_EPOCH
            self.finished_epoch = None
            self.early_stopped = False
            self.data.epoch = 0
            self.data.update_index = 0
        self._require_group_size(component_id)
        runtime.barrier()
        if self._restored and len(self.learning_rates) != self.logical_step:
            raise RuntimeFailure(
                f"component {component_id}: learning-rate history does not match logical step"
            )
        if self.phase == _PENDING:
            self._finish_boundary(runtime, checkpoint, component_id)
            if self.early_stopped or self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                return self._result(checkpoint, "interrupted")
            if self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            view = self.data.train_epoch(epoch, start_update=index)
            try:
                if view.updates < 1:
                    raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
                if index > view.updates:
                    raise RuntimeFailure(
                        f"component {component_id}: update {index} is past epoch {epoch}"
                    )
                if index == view.updates:
                    raise RuntimeFailure(
                        f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
                    )
                if index + self.accumulation > view.updates:
                    raise RuntimeFailure(
                        f"component {component_id}: accumulation {self.accumulation} "
                        f"does not fit at update {index}"
                    )
                self.model.module.train()
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
                        micro = micro / self.accumulation
                        runtime.backward(micro)
                    detached = micro.detach()
                    total = detached if total is None else total + detached
                if self.grad_clip is not None:
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
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                self.losses.append(loss_value)
                self.learning_rates.append(learning_rate)
                self.data.epoch = epoch
                self.data.update_index = index
                sample_id = ",".join(sample_ids)
                if rank == 0:
                    self._append_metrics(metrics_path, loss_value, result.applied, sample_id, learning_rate, epoch)
                if sample_path is not None:
                    _append_jsonl(
                        sample_path,
                        {
                            "epoch": epoch,
                            "logical_step": self.logical_step,
                            "loss": loss_value,
                            "lr": learning_rate,
                            "sample_id": sample_id,
                        },
                    )
                epoch_done = index == view.updates
                if not epoch_done:
                    if self.logical_step % self.checkpoint_interval == 0:
                        checkpoint.save(checkpoint.directory, logical_step=self.logical_step, label="step")
                    continue
                self.phase = _PENDING
                self.finished_epoch = epoch
                self.boundary = _open_boundary(epoch)
                self._on_boundary_phase(_PENDING)
                checkpoint.save(
                    checkpoint.directory,
                    logical_step=self.logical_step,
                    label="validation_pending",
                )
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
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "early_stopped": self.early_stopped,
            "finished_epoch": self.finished_epoch,
            "history": list(self.history),
            "learning_rates": list(self.learning_rates),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        logical_step = state.get("logical_step")
        updates = state.get("optimizer_updates")
        phase = state.get("phase")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
            raise RuntimeFailure("trainer state is invalid")
        if phase not in {_IN_EPOCH, _PENDING}:
            raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
        history = state.get("history")
        rates = state.get("learning_rates")
        completed = state.get("completed_boundaries")
        if not isinstance(history, list) or not isinstance(rates, list) or not isinstance(completed, list):
            raise RuntimeFailure("trainer history is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        self.phase = phase
        self.history = list(history)
        self.learning_rates = [float(value) for value in rates]
        self.completed_boundaries = list(completed)
        boundary = state.get("boundary")
        self.boundary = None if boundary is None else dict(boundary)
        finished = state.get("finished_epoch")
        if finished is not None and (isinstance(finished, bool) or not isinstance(finished, int)):
            raise RuntimeFailure("finished_epoch is invalid")
        self.finished_epoch = finished
        self.early_stopped = bool(state.get("early_stopped", False))
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
        self._restored = True

    def _on_boundary_phase(self, phase: str) -> None:
        """Hook for tests that record validation, scheduler, and selection order."""

        del phase

    def _finish_boundary(
        self,
        runtime: ProcessRuntime,
        checkpoint: CheckpointManager,
        component_id: str,
    ) -> None:
        if self.phase != _PENDING or self.finished_epoch is None or self.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        trainer_epoch = int(self.finished_epoch)
        espnet_epoch = trainer_epoch + 1
        summary = self._validate(runtime, trainer_epoch, component_id)
        self.boundary["validation_done"] = True
        self.boundary["validation_loss"] = summary["loss"]
        self.boundary["validation_weight"] = summary["weight"]
        self.phase = "pending_scheduler"
        self._on_boundary_phase(self.phase)
        self.scheduler.step(summary["loss"])
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.boundary["scheduler_updated"] = True
        self.boundary["learning_rate"] = learning_rate
        self.phase = "pending_selection"
        self._on_boundary_phase(self.phase)
        self._select(runtime, checkpoint.directory, espnet_epoch, summary["loss"])
        self.boundary["selection_done"] = True
        self.completed_boundaries.append(dict(self.boundary))
        self.boundary = None
        self.phase = _IN_EPOCH
        self.finished_epoch = None
        self.data.epoch = trainer_epoch + 1
        self.data.update_index = 0
        self._on_boundary_phase(self.phase)
        if (
            self.early_stopping_patience is not None
            and early_stop(self.history, espnet_epoch, self.early_stopping_patience)
        ):
            self.early_stopped = True
        checkpoint.save(checkpoint.directory, logical_step=self.logical_step, label="step")

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, float | int]:
        module = self.model.module
        was_training = bool(module.training)
        cursor = (int(self.data.epoch), int(self.data.update_index))
        rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
        local_sum = 0.0
        local_count = 0
        try:
            module.eval()
            with torch.no_grad():
                for batch in self.data.validation_epoch(trainer_epoch):
                    if not isinstance(batch, Batch):
                        raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                    batch = _batch_on_device(batch, runtime.device)
                    prediction = self.model.state_module(batch.inputs)
                    value = self.loss(prediction, batch.targets)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise RuntimeFailure(f"component {component_id}: validation loss is not a scalar")
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                    weight = int(batch.inputs.shape[0])
                    local_sum += float(value.detach()) * weight
                    local_count += weight
            gathered = runtime.gather_objects({"count": local_count, "sum": local_sum})
            total = 0.0
            count = 0
            for item in gathered:
                total += float(item["sum"])
                count += int(item["count"])
            if count < 1:
                raise RuntimeFailure(f"component {component_id}: validation produced no batches")
            loss = total / count
        finally:
            restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
            if was_training:
                module.train()
            moved = (int(self.data.epoch), int(self.data.update_index))
            if moved != cursor:
                raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return {"loss": loss, "weight": count}

    def _select(
        self,
        runtime: ProcessRuntime,
        directory: Path,
        espnet_epoch: int,
        loss: float,
    ) -> None:
        self.history.append({"epoch": int(espnet_epoch), "loss": float(loss)})
        ranked = rank_validation_loss(self.history, self.keep_nbest)
        error: str | None = None
        if _rank(runtime) == 0:
            try:
                _write_nbest(
                    directory,
                    self.model.state_module.export_for_inference(),
                    espnet_epoch,
                    ranked,
                    self.keep_nbest,
                    self.logical_step,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        failure = next((item for item in errors if item), None)
        if failure:
            raise RuntimeFailure(f"n-best selection failed: {failure}")
        runtime.barrier()

    def _forward_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        if not batches:
            raise RuntimeFailure(f"component {component_id}: empty micro-batch")
        total: torch.Tensor | None = None
        for batch in batches:
            batch = _batch_on_device(batch, runtime.device)
            try:
                prediction = self.model.module(batch.inputs)
                value = self.loss(prediction, batch.targets)
            except ValueError as exc:
                raise RuntimeFailure(str(exc)) from exc
            if not isinstance(value, torch.Tensor):
                raise RuntimeFailure(f"component {component_id}: loss did not return a tensor")
            total = value if total is None else total + value
        assert total is not None
        return total / len(batches)

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if group_size != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _stop_requested(self, runtime: object | None = None) -> bool:
        from src.experiments.signals import ranks_should_stop

        capped = self.stop_after_logical_steps is not None and self.logical_step >= self.stop_after_logical_steps
        return ranks_should_stop(runtime, extra=capped)

    def _epoch_cap_reached(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

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
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _open_boundary(trainer_epoch: int) -> dict[str, Any]:
    return {
        "espnet_epoch": trainer_epoch + 1,
        "learning_rate": None,
        "optimizer_updated": True,
        "scheduler_updated": False,
        "selection_done": False,
        "trainer_epoch": trainer_epoch,
        "validation_done": False,
        "validation_loss": None,
        "validation_weight": None,
    }


def _write_nbest(
    directory: Path,
    exported: dict[str, torch.Tensor],
    espnet_epoch: int,
    ranked: list[dict[str, Any]],
    keep: int,
    logical_step: int,
) -> None:
    """Write the epoch export and the valid.loss n-best files. Rank 0 only."""

    if not ranked:
        raise RuntimeFailure("n-best selection received an empty history")
    _save_state(directory / f"{espnet_epoch}epoch.pt", exported)
    best = ranked[0]
    if int(best["epoch"]) == int(espnet_epoch):
        _save_state(directory / "valid.loss.best.pt", exported)
        _write_json(
            directory / "valid.loss.best.pt.json",
            {
                "epoch": int(espnet_epoch),
                "global_step": int(logical_step),
                "metric": "loss",
                "value": float(best["loss"]),
            },
        )
    kept = {int(row["epoch"]) for row in ranked}
    for path in directory.glob("*epoch.pt"):
        match = _EPOCH_FILE.match(path.name)
        if match and int(match.group(1)) not in kept:
            path.unlink()
    best_name = f"{int(best['epoch'])}epoch.pt"
    if keep > 1 and len(ranked) >= keep:
        epochs = [int(row["epoch"]) for row in ranked]
        averaged = average_checkpoint_files([directory / f"{epoch}epoch.pt" for epoch in epochs])
        average_name = f"valid.loss.ave_{keep}best.pt"
        average_path = directory / average_name
        _save_state(average_path, averaged)
        _write_json(
            directory / f"{average_name}.json",
            {
                "average": "espnet_nbest_float_mean_int_sum",
                "criterion": "valid.loss.min",
                "epochs": epochs,
                "files": [f"{epoch}epoch.pt" for epoch in epochs],
                "global_step": int(logical_step),
                "metric": "loss",
                "values": [float(row["loss"]) for row in ranked],
            },
        )
        replace_symlink(directory / "valid.loss.ave.pt", average_name)
        return
    replace_symlink(directory / "valid.loss.ave_1best.pt", best_name)
    replace_symlink(directory / "valid.loss.ave.pt", "valid.loss.ave_1best.pt")


def _save_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)
