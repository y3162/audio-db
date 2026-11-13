"""Trainers constructed by the component factory.

``StandardTrainer`` performs forward, loss, backward, and one optimizer
update per logical step. ``GanTrainer`` and ``MixtureTrainer`` only hold the
objects required to build the Phase 1 graphs. Their training loops are later
phases.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.errors import BuildError, CheckpointError, ExperimentError, RuntimeFailure
from src.experiments.execution.checkpoint import CheckpointManager
from src.experiments.execution.results import RunResult
from src.experiments.execution.runtime import PreparedModel, ProcessRuntime
from src.experiments.stopping import MetricEvent, should_stop


class StandardTrainer:
    """Single-model, single-optimizer loop on micro-batches."""

    supports_training_control = True
    training_metric_sources = ("train",)
    training_metric_names = ("train.mean_rank_loss",)

    def __init__(
        self,
        model: PreparedModel,
        loss: torch.nn.Module,
        data: Any,
        optimizer: torch.optim.Optimizer,
        *,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float | None = None,
        scheduler: Any | None = None,
        stop_after_logical_steps: int | None = None,
    ) -> None:
        if not isinstance(model, PreparedModel):
            raise BuildError("model must be a prepared module handle")
        if not callable(loss):
            raise BuildError("loss must be callable")
        if not hasattr(data, "train_epoch"):
            raise BuildError("data must provide train_epoch")
        optimizer_id = getattr(optimizer, "_experiment_component_id", None)
        if not isinstance(optimizer_id, str):
            raise BuildError("optimizer has no component id")
        if isinstance(max_logical_steps, bool) or not isinstance(max_logical_steps, int) or max_logical_steps < 1:
            raise BuildError("max_logical_steps must be a positive integer")
        if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
            raise BuildError("accumulation must be a positive integer")
        if grad_clip is not None and (
            isinstance(grad_clip, bool) or not isinstance(grad_clip, (int, float)) or grad_clip <= 0
        ):
            raise BuildError("grad_clip must be a positive number")
        if stop_after_logical_steps is not None and (
            isinstance(stop_after_logical_steps, bool)
            or not isinstance(stop_after_logical_steps, int)
            or stop_after_logical_steps < 1
        ):
            raise BuildError("stop_after_logical_steps must be a positive integer")
        self.model = model
        self.loss = loss
        self.data = data
        self.optimizer = optimizer
        self.optimizer_id = optimizer_id
        self.max_logical_steps = max_logical_steps
        self.accumulation = accumulation
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.scheduler = scheduler
        self.stop_after_logical_steps = stop_after_logical_steps
        self.logical_step = 0
        self.optimizer_updates = {optimizer_id: 0}
        self.losses: list[float] = []
        self.stop_reason = "initialized"
        self._restored = False

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        """Run until ``max_logical_steps``. A restored trainer continues.

        The loss is divided by accumulation. A new run clears ``metrics.jsonl``.
        Resume keeps rows at or before the restored logical step and drops
        anything written past that commit.
        """

        component_id = getattr(self, "_experiment_component_id", "trainer")
        rank = _rank(runtime)
        metrics_path = Path(checkpoint.directory) / "metrics.jsonl"
        sample_path = _sample_path(checkpoint.directory, runtime)
        if self._restored:
            if not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index"):
                raise CheckpointError("resumed data has no epoch position")
            if rank == 0:
                self.losses = _rewind_metrics(metrics_path, self.logical_step)
            if sample_path is not None:
                sample_losses = _rewind_metrics(sample_path, self.logical_step)
                if rank != 0:
                    self.losses = sample_losses
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
        else:
            if rank == 0:
                metrics_path.write_text("", encoding="utf-8")
            if sample_path is not None:
                sample_path.write_text("", encoding="utf-8")
            epoch = 0
            index = 0
            self.logical_step = 0
            self.optimizer_updates = {self.optimizer_id: 0}
            self.losses = []
        self._require_group_size(component_id)
        runtime.barrier()
        while self.logical_step < self.max_logical_steps and not should_stop(self):
            from src.experiments.execution.signals import ranks_should_stop

            capped = (
                self.stop_after_logical_steps is not None
                and self.logical_step >= self.stop_after_logical_steps
            )
            if ranks_should_stop(runtime, extra=capped):
                if self.logical_step > 0:
                    self._offer(checkpoint, "stop")
                self.stop_reason = "interrupted"
                return RunResult(
                    stop_reason="interrupted",
                    logical_step=self.logical_step,
                    optimizer_updates=dict(self.optimizer_updates),
                    output_dir=checkpoint.directory,
                )
            self._before_logical_step(runtime)
            view = self.data.train_epoch(epoch, start_update=index)
            try:
                if index >= view.updates:
                    epoch += 1
                    index = 0
                    if epoch > self.max_logical_steps + 1:
                        raise RuntimeFailure(f"component {component_id}: data has no updates")
                    continue
                if index + self.accumulation > view.updates:
                    raise RuntimeFailure(
                        f"component {component_id}: accumulation {self.accumulation} "
                        f"does not fit at update {index}"
                    )
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
                self._before_optimizer_step(runtime)
                if self.grad_clip is not None:
                    runtime.clip(self.model.parameters, self.grad_clip, optimizer=self.optimizer)
                result = runtime.step(self.optimizer)
                runtime.end_step()
                if result.applied:
                    self.optimizer_updates[self.optimizer_id] += 1
                    if self.scheduler is not None:
                        self.scheduler.step()
                self.logical_step += 1
                index += self.accumulation
                if hasattr(self.data, "epoch") and hasattr(self.data, "update_index"):
                    self.data.epoch = epoch
                    self.data.update_index = index
                if total is None or not bool(torch.isfinite(total).all()):
                    raise RuntimeFailure(f"component {component_id}: loss is not finite")
                loss_value = float(total)
                self.losses.append(loss_value)
                sample_id = ",".join(sample_ids)
                if rank == 0:
                    self._append_metrics(metrics_path, loss_value, result.applied, sample_id)
                if sample_path is not None:
                    _append_jsonl(
                        sample_path,
                        {
                            "logical_step": self.logical_step,
                            "loss": loss_value,
                            "sample_id": sample_id,
                        },
                    )
                control = getattr(self, "training_control", None)
                if control is not None:
                    rank_losses = runtime.gather_objects(loss_value)
                    mean = sum(rank_losses) / len(rank_losses)
                    control.observe(MetricEvent("train", self.logical_step, {"mean_rank_loss": mean}))
                self._offer(checkpoint, "step")
            finally:
                _close_epoch_view(view)
        self._offer(checkpoint, "final")
        self.stop_reason = "early_stopping" if should_stop(self) else "completed"
        return RunResult(
            stop_reason=self.stop_reason,
            logical_step=self.logical_step,
            optimizer_updates=dict(self.optimizer_updates),
            output_dir=checkpoint.directory,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        logical_step = state.get("logical_step")
        updates = state.get("optimizer_updates")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
            raise RuntimeFailure("trainer state is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
        self._restored = True

    def _before_logical_step(self, runtime: ProcessRuntime) -> None:
        """Hook for tests that change the grad scale before the next backward."""

        del runtime

    def _before_optimizer_step(self, runtime: ProcessRuntime) -> None:
        """Hook for tests that edit gradients before unscale, clip, and step."""

        del runtime

    def _before_checkpoint(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> None:
        """Hook for tests that change state after a step and before publish."""

        del runtime, checkpoint

    def _offer(self, checkpoint: CheckpointManager, event: str) -> None:
        """Offer this state to the selector. ``_before_checkpoint`` runs only when a file is written.

        ``stop`` and ``final`` follow the offer of the last finished step
        without a state change, so the selector can see that Latest already
        holds this state. Saving it again would run ``_before_checkpoint``
        twice and a resumed run would no longer match a continuous one.
        """

        checkpoint.offer(
            event,
            logical_step=self.logical_step,
            prepare=lambda runtime: self._before_checkpoint(runtime, checkpoint),
        )

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if group_size != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _forward_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        if not batches:
            raise RuntimeFailure(f"component {component_id}: empty micro-batch")
        total: torch.Tensor | None = None
        for batch in batches:
            batch = _batch_on_device(batch, runtime.device)
            prediction = self.model.module(batch.inputs)
            value = self.loss(prediction, batch.targets)
            if not isinstance(value, torch.Tensor):
                raise RuntimeFailure(f"component {component_id}: loss did not return a tensor")
            total = value if total is None else total + value
        assert total is not None
        return total / len(batches)

    def _append_metrics(self, path: Path, loss_value: float, applied: bool, sample_id: str) -> None:
        record = {
            "applied": applied,
            "logical_step": self.logical_step,
            "loss": loss_value,
            "optimizer_updates": dict(self.optimizer_updates),
            "sample_id": sample_id,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _close_epoch_view(view: object) -> None:
    """Release an epoch view. Data modules without ``close`` stay unchanged."""

    close = getattr(view, "close", None)
    if callable(close):
        close()


def _batch_on_device(batch: Batch, device: torch.device) -> Batch:
    """Move model inputs to ``device``. Leave them in place when they already match."""

    inputs = batch.inputs
    targets = batch.targets
    lengths = batch.lengths
    non_blocking = device.type == "cuda"
    changed = False
    if inputs.device != device:
        inputs = inputs.to(device, non_blocking=non_blocking and inputs.device.type == "cpu")
        changed = True
    if targets.device != device:
        targets = targets.to(device, non_blocking=non_blocking and targets.device.type == "cpu")
        changed = True
    if lengths is not None and lengths.device != device:
        lengths = lengths.to(device, non_blocking=non_blocking and lengths.device.type == "cpu")
        changed = True
    if not changed:
        return batch
    return replace(batch, inputs=inputs, targets=targets, lengths=lengths)


def _rank(runtime: ProcessRuntime) -> int:
    return int(getattr(runtime, "rank", 0))


def _sample_path(directory: Path, runtime: ProcessRuntime) -> Path | None:
    """Per-rank sample trace. World size 1 keeps sample ids in ``metrics.jsonl``."""

    if int(getattr(runtime, "world_size", 1)) == 1:
        return None
    return directory / f"samples-rank{_rank(runtime)}.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _rewind_metrics(path: Path, logical_step: int) -> list[float]:
    """Drop metric rows committed past ``logical_step`` and return kept losses."""

    if not path.is_file():
        return []
    kept: list[dict[str, Any]] = []
    losses: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"{path.name} line {line_number} is not JSON") from exc
        step = record.get("logical_step") if isinstance(record, dict) else None
        loss = record.get("loss") if isinstance(record, dict) else None
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(loss, (int, float)):
            raise CheckpointError(f"{path.name} line {line_number} has no logical step")
        if step > logical_step:
            continue
        kept.append(record)
        losses.append(float(loss))
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)
    return losses


class GanTrainer:
    """Holds a two-optimizer graph. The GAN update order is not implemented."""

    def __init__(
        self,
        generator: PreparedModel,
        discriminator: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        max_logical_steps: int = 1,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.max_logical_steps = max_logical_steps

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        del runtime, checkpoint
        raise ExperimentError("GanTrainer training is not implemented in Phase 1")

    def state_dict(self) -> dict[str, Any]:
        return {"max_logical_steps": self.max_logical_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state
        raise ExperimentError("GanTrainer restore is not implemented in Phase 1")


class MixtureTrainer:
    """Holds expert modules in the inject list order. Mixture training is not implemented."""

    def __init__(
        self,
        experts: list[PreparedModel],
        router: PreparedModel,
        data: Any,
        optim_experts: torch.optim.Optimizer,
        optim_router: torch.optim.Optimizer,
        max_logical_steps: int = 1,
    ) -> None:
        self.experts = list(experts)
        self.router = router
        self.data = data
        self.optim_experts = optim_experts
        self.optim_router = optim_router
        self.max_logical_steps = max_logical_steps

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        del runtime, checkpoint
        raise ExperimentError("MixtureTrainer training is not implemented in Phase 1")

    def state_dict(self) -> dict[str, Any]:
        return {"max_logical_steps": self.max_logical_steps, "experts": len(self.experts)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state
        raise ExperimentError("MixtureTrainer restore is not implemented in Phase 1")
