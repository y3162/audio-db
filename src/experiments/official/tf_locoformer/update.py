"""One TF-Locoformer accumulation update.

The injected spectral objective is trained. SI-SDR is a detached monitor and
is not added to the backward. This module does not call ``step_batch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.experiments.calc.spectral import si_sdr_loss
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.training.objectives import call_objective, term_floats


@dataclass(frozen=True)
class LocoformerStep:
    """Scalars from one finished accumulation window, before ``step_batch``."""

    loss: float
    applied: bool
    used_lr: float
    si_sdr: float
    sample_id: str
    terms: dict[str, float]


class LocoformerUpdate:
    """Forward, backward, clip, and one AdamW step for a grouped window."""

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

    def run(self, planned: list[list[Batch]], runtime: ProcessRuntime, component_id: str) -> LocoformerStep:
        """Run one window. The learning rate is the one used for this step."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total: torch.Tensor | None = None
        sample_ids: list[str] = []
        monitor_sum: torch.Tensor | None = None
        monitor_count = 0
        term_acc: dict[str, float] = {}
        for offset, batches in enumerate(planned):
            sync = offset == self.accumulation - 1
            sample_ids.extend(batch.sample_id for batch in batches)
            with runtime.sync_gradients([self.model], enabled=sync):
                with runtime.autocast():
                    micro, monitor, count, terms = _forward_loss(
                        self.model, self.objective, batches, component_id, runtime
                    )
                micro = micro / self.accumulation
                runtime.backward(micro)
            detached = micro.detach()
            total = detached if total is None else total + detached
            monitor_sum = monitor if monitor_sum is None else monitor_sum + monitor
            monitor_count += count
            for name, value in terms.items():
                term_acc[name] = term_acc.get(name, 0.0) + value / self.accumulation
        if self.grad_clip is not None:
            runtime.clip(self.model, self.grad_clip, optimizer=self.optimizer)
        if total is None:
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        if monitor_sum is None or monitor_count < 1:
            raise RuntimeFailure(f"component {component_id}: SI-SDR monitor has no batches")
        used_lr = float(self.optimizer.param_groups[0]["lr"])
        host_loss = runtime.host_scalar(total)
        host_monitor = runtime.host_scalar(monitor_sum / monitor_count)
        result = runtime.step(self.optimizer)
        runtime.end_step()
        loss_value = float(host_loss)
        if not math.isfinite(loss_value):
            raise RuntimeFailure(f"component {component_id}: loss is not finite")
        return LocoformerStep(
            loss=loss_value,
            applied=bool(result.applied),
            used_lr=used_lr,
            si_sdr=float(host_monitor),
            sample_id=",".join(sample_ids),
            terms=term_acc,
        )


def _forward_loss(
    model: PreparedModel,
    objective: torch.nn.Module,
    batches: list[Batch],
    component_id: str,
    runtime: ProcessRuntime,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float]]:
    """Average the injected objective and keep SI-SDR as a detached monitor."""

    if not batches:
        raise RuntimeFailure(f"component {component_id}: empty micro-batch")
    total: torch.Tensor | None = None
    monitor: torch.Tensor | None = None
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
        with torch.no_grad():
            term = si_sdr_loss(prediction, batch.targets).detach()
            monitor = term if monitor is None else monitor + term
    assert total is not None and monitor is not None
    count = len(batches)
    return total / count, monitor, count, {name: amount / count for name, amount in term_sums.items()}
