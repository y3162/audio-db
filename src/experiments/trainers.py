"""Trainers constructed by the component factory.

``StandardTrainer`` performs forward, loss, backward, and one optimizer
update per logical step. ``GanTrainer`` and ``MixtureTrainer`` only hold the
objects required to build the Phase 1 graphs. Their training loops are later
phases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.experiments.checkpoint import CheckpointManager
from src.experiments.data import Batch
from src.experiments.errors import BuildError, CheckpointError, ExperimentError, RuntimeFailure
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime


class StandardTrainer:
    """Single-model, single-optimizer loop on micro-batches."""

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
        checkpoint_interval: int = 1,
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
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval < 1
        ):
            raise BuildError("checkpoint_interval must be a positive integer")
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
        self.checkpoint_interval = checkpoint_interval
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
        metrics_path = Path(checkpoint.directory) / "metrics.jsonl"
        if self._restored:
            if not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index"):
                raise CheckpointError("resumed data has no epoch position")
            self.losses = _rewind_metrics(metrics_path, self.logical_step)
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
        else:
            metrics_path.write_text("", encoding="utf-8")
            epoch = 0
            index = 0
            self.logical_step = 0
            self.optimizer_updates = {self.optimizer_id: 0}
            self.losses = []
        while self.logical_step < self.max_logical_steps:
            if (
                self.stop_after_logical_steps is not None
                and self.logical_step >= self.stop_after_logical_steps
            ):
                self.stop_reason = "interrupted"
                return RunResult(
                    stop_reason="interrupted",
                    logical_step=self.logical_step,
                    optimizer_updates=dict(self.optimizer_updates),
                    output_dir=checkpoint.directory,
                )
            view = self.data.train_epoch(epoch, start_update=index)
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
                        micro = self._forward_loss(batches, component_id)
                    micro = micro / self.accumulation
                    runtime.backward(micro)
                detached = micro.detach()
                total = detached if total is None else total + detached
            if self.grad_clip is not None:
                runtime.clip(self.model.parameters, self.grad_clip)
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
            self._append_metrics(metrics_path, loss_value, result.applied, ",".join(sample_ids))
            if self.logical_step % self.checkpoint_interval == 0:
                checkpoint.save(checkpoint.directory, logical_step=self.logical_step, label="step")
        self.stop_reason = "completed"
        return RunResult(
            stop_reason="completed",
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

    def _forward_loss(self, batches: list[Batch], component_id: str) -> torch.Tensor:
        if not batches:
            raise RuntimeFailure(f"component {component_id}: empty micro-batch")
        total: torch.Tensor | None = None
        for batch in batches:
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
            raise CheckpointError(f"metrics.jsonl line {line_number} is not JSON") from exc
        step = record.get("logical_step") if isinstance(record, dict) else None
        loss = record.get("loss") if isinstance(record, dict) else None
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(loss, (int, float)):
            raise CheckpointError(f"metrics.jsonl line {line_number} has no logical step")
        if step > logical_step:
            continue
        kept.append(record)
        losses.append(float(loss))
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept)
    temporary = path.with_name(".metrics.jsonl.tmp")
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
