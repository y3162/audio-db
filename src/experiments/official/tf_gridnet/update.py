"""One TF-GridNet accumulation update.

The injected spectral objective is averaged inside each micro-batch, then
divided by the accumulation count. SI-SDR is not computed. This module does
not move the data cursor, count a logical step, or save.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.training.objectives import call_objective, term_floats


@dataclass(frozen=True)
class GridNetStep:
    """Scalars from one finished accumulation window."""

    loss: float
    applied: bool
    learning_rate: float
    sample_id: str
    terms: dict[str, float]


class GridNetUpdate:
    """Forward, backward, clip, and one Adam step for a grouped window."""

    def __init__(
        self,
        model: PreparedModel,
        objective: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        accumulation: int,
        grad_clip: float,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.accumulation = accumulation
        self.grad_clip = grad_clip

    def run(self, planned: list[list[Batch]], runtime: ProcessRuntime, component_id: str) -> GridNetStep:
        """Run one window. The last micro-batch is the one that synchronizes."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total: torch.Tensor | None = None
        sample_ids: list[str] = []
        term_acc: dict[str, float] = {}
        for offset, batches in enumerate(planned):
            sync = offset == self.accumulation - 1
            sample_ids.extend(batch.sample_id for batch in batches)
            with runtime.sync_gradients([self.model], enabled=sync):
                with runtime.autocast():
                    micro, terms = _forward_loss(self.model, self.objective, batches, component_id, runtime)
                micro = micro / self.accumulation
                runtime.backward(micro)
            detached = micro.detach()
            total = detached if total is None else total + detached
            for name, value in terms.items():
                term_acc[name] = term_acc.get(name, 0.0) + value / self.accumulation
        if self.grad_clip is not None:
            runtime.clip(self.model, self.grad_clip, optimizer=self.optimizer)
        if total is None:
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        host_loss = runtime.host_scalar(total)
        result = runtime.step(self.optimizer)
        runtime.end_step()
        loss_value = float(host_loss)
        if not math.isfinite(loss_value):
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        return GridNetStep(
            loss=loss_value,
            applied=bool(result.applied),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            sample_id=",".join(sample_ids),
            terms=term_acc,
        )


def _forward_loss(
    model: PreparedModel,
    objective: torch.nn.Module,
    batches: list[Batch],
    component_id: str,
    runtime: ProcessRuntime,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average the injected spectral objective over one micro-batch group."""

    if not batches:
        raise RuntimeFailure(f"component {component_id}: empty micro-batch")
    total: torch.Tensor | None = None
    term_sums: dict[str, float] = {}
    for batch in batches:
        batch = runtime.move(batch)
        try:
            prediction = model(batch.inputs)
            result = call_objective(objective, {"estimate": prediction, "target": batch.targets})
        except ValueError as exc:
            raise RuntimeFailure(str(exc)) from exc
        value = result.total
        if not isinstance(value, torch.Tensor):
            raise RuntimeFailure(f"component {component_id}: loss did not return a tensor")
        for name, amount in term_floats(result).items():
            term_sums[name] = term_sums.get(name, 0.0) + amount
        total = value if total is None else total + value
    assert total is not None
    count = len(batches)
    return total / count, {name: amount / count for name, amount in term_sums.items()}
