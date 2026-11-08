"""Trainers constructed by the component factory.

``StandardTrainer`` performs forward, loss, backward, and one optimizer
update per logical step. ``GanTrainer`` and ``MixtureTrainer`` only hold the
objects required to build the Phase 1 graphs. Their training loops are later
phases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.experiments.checkpoint import CheckpointManager
from src.experiments.data import Batch
from src.experiments.errors import BuildError, ExperimentError, RuntimeFailure
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
        self.model = model
        self.loss = loss
        self.data = data
        self.optimizer = optimizer
        self.optimizer_id = optimizer_id
        self.max_logical_steps = max_logical_steps
        self.accumulation = accumulation
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.scheduler = scheduler
        self.logical_step = 0
        self.optimizer_updates = {optimizer_id: 0}
        self.losses: list[float] = []
        self.stop_reason = "initialized"

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        """Run ``max_logical_steps`` updates. The loss is divided by accumulation."""

        component_id = getattr(self, "_experiment_component_id", "trainer")
        metrics_path = Path(checkpoint.directory) / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")
        epoch = 0
        index = 0
        self.logical_step = 0
        self.optimizer_updates = {self.optimizer_id: 0}
        self.losses = []
        while self.logical_step < self.max_logical_steps:
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
            for offset in range(self.accumulation):
                sync = offset == self.accumulation - 1
                batches = view.batch(index + offset)
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
            self._append_metrics(metrics_path, loss_value, result.applied)
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

    def _append_metrics(self, path: Path, loss_value: float, applied: bool) -> None:
        record = {
            "applied": applied,
            "logical_step": self.logical_step,
            "loss": loss_value,
            "optimizer_updates": dict(self.optimizer_updates),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


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
