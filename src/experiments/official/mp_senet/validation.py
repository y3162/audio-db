"""Official MP-SENet validation for one epoch.

Rank 0 scores the validation split with a shadow reference task and
``metrics.aggregate_reports``. Every rank then receives that one summary.
The training cursor and the RNG captured around the call are restored.

The shadow task is a cache of architecture, seed, and the optimizer group
seen on the first call. Each call copies the live generator and discriminator
weights into it. ``invalidate`` drops the cache so a resumed process does not
keep a task built before ``load_state_dict``.

This module does not choose when validation runs. It does not step a
scheduler, select a best PESQ, write a checkpoint, or call the research
evaluator.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.metrics import aggregate_reports
from src.experiments.official.mp_senet.reference import validation_model
from src.experiments.runtime import PreparedModel, ProcessRuntime


class MPSENetOfficialValidator:
    """Rank-0 official report. The caller decides the epoch and what to do next."""

    def __init__(
        self,
        *,
        generator: PreparedModel,
        discriminator: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.optim_g = optim_g
        self.scheduler_g = scheduler_g
        self._shadow = None

    def invalidate(self) -> None:
        """Drop the cached reference task. The next call builds a new one."""

        self._shadow = None

    def evaluate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        """Run the official validation report on rank 0 and share that summary."""

        cursor = (int(self.data.epoch), int(self.data.update_index))
        rng = runtime.capture_rng()
        error: str | None = None
        summary: dict[str, Any] | None = None
        try:
            if runtime.rank == 0:
                try:
                    summary = self._rank0(trainer_epoch, runtime, component_id)
                except (TrainingError, RuntimeFailure, RuntimeError, ValueError, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
        finally:
            runtime.restore_rng(rng)
        errors = runtime.gather_objects(error)
        failure = next((item for item in errors if item), None)
        if failure:
            raise RuntimeFailure(f"validation failed: {failure}")
        gathered = runtime.gather_objects(summary)
        shared = gathered[0]
        if not isinstance(shared, dict):
            raise RuntimeFailure(f"component {component_id}: rank 0 did not return a validation summary")
        moved = (int(self.data.epoch), int(self.data.update_index))
        if moved != cursor:
            raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return shared

    def _rank0(self, trainer_epoch: int, runtime: ProcessRuntime, component_id: str) -> dict[str, Any]:
        task = self._task(runtime)
        reports: list[dict[str, Any]] = []
        for batch in self.data.validation_epoch(trainer_epoch):
            if not isinstance(batch, Batch):
                raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
            batch = runtime.move(batch)
            report = task.validation_report({"clean": batch.targets, "noisy": batch.inputs})
            reports.append(report)
        if not reports:
            raise RuntimeFailure(f"component {component_id}: rank 0 validation produced no batches")
        return aggregate_reports(reports)

    def _task(self, runtime: ProcessRuntime):
        if self._shadow is None:
            group = self.optim_g.param_groups[0]
            betas = group["betas"]
            self._shadow = validation_model(
                {
                    "loss": "official",
                    "model": dict(self.generator.unwrapped().architecture),
                    "optimizer": {
                        "betas": [float(betas[0]), float(betas[1])],
                        "lr": float(group["lr"]),
                        "weight_decay": float(group["weight_decay"]),
                    },
                    "scheduler": {"gamma": float(self.scheduler_g.gamma)},
                    "seed": int(self.generator.unwrapped().seed),
                },
                runtime.device,
            )
        task = self._shadow
        if task.discriminator is None:
            raise RuntimeFailure("shadow MP-SENet task has no discriminator")
        task.generator.load_state_dict(self.generator.unwrapped().generator.state_dict(), strict=True)
        task.discriminator.load_state_dict(self.discriminator.unwrapped().discriminator.state_dict(), strict=True)
        task.eval_mode()
        return task
