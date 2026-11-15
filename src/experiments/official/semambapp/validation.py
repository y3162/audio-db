"""Rank-0 SEMamba++ validation through the shadow task. The shadow is not saved."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.calc.metrics import _aggregate_validation
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.semambapp.reference import validation_model
from src.experiments.runtime import ProcessRuntime


class SEMambaValidator:
    """Copy live weights into one shadow task and return the shared summary."""

    def __init__(self, *, generator: Any, mrd: Any, cqt: Any, optim_g: torch.optim.Optimizer, scheduler_g: Any, grad_clip: float | None) -> None:
        self.generator = generator
        self.mrd = mrd
        self.cqt = cqt
        self.optim_g = optim_g
        self.scheduler_g = scheduler_g
        self.grad_clip = grad_clip
        self._shadow: Any = None

    def invalidate(self) -> None:
        self._shadow = None

    def evaluate(self, runtime: ProcessRuntime, data: Any, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        cursor = (int(data.epoch), int(data.update_index))
        rng = runtime.capture_rng()
        error: str | None = None
        summary: dict[str, Any] | None = None
        try:
            if runtime.rank == 0:
                try:
                    summary = self._rank0(runtime, data, trainer_epoch, component_id)
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
        moved = (int(data.epoch), int(data.update_index))
        if moved != cursor:
            raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return shared

    def _rank0(self, runtime: ProcessRuntime, data: Any, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        task = self._shadow_task(runtime)
        reports: list[dict[str, Any]] = []
        for batch in data.validation_epoch(trainer_epoch):
            if not isinstance(batch, Batch):
                raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
            batch = runtime.move(batch)
            report = task.validation_report({"clean": batch.targets, "noisy": batch.inputs})
            reports.append(report)
        if not reports:
            raise RuntimeFailure(f"component {component_id}: rank 0 validation produced no batches")
        summary = _aggregate_validation(reports)
        summary["cases"] = [
            {
                "lengths": report["lengths"],
                "losses": report["losses"],
                "pesq": report["pesq"],
                "sample_rate": report["sample_rate"],
                "si_sdr": report["si_sdr"],
                "stoi": report["stoi"],
            }
            for report in reports
        ]
        return summary

    def _shadow_task(self, runtime: ProcessRuntime) -> Any:
        if self._shadow is None:
            architecture = dict(self.generator.unwrapped().architecture)
            hop = int(architecture.pop("hop"))
            win = int(architecture.pop("win"))
            sampling_rate = int(architecture.pop("sampling_rate"))
            group = self.optim_g.param_groups[0]
            betas = group["betas"]
            self._shadow = validation_model(
                {
                    "grad_clip": self.grad_clip,
                    "loss": "official",
                    "model": {
                        "architecture": architecture,
                        "name": "semambapp",
                        "sampling_rate": sampling_rate,
                        "stft": {"hop": hop, "win": win},
                        "variant": "semambapp",
                    },
                    "optimizer": {
                        "betas": [float(betas[0]), float(betas[1])],
                        "lr": float(group["lr"]),
                        "name": "adamw",
                        "weight_decay": float(group["weight_decay"]),
                    },
                    "scheduler": {"gamma": float(self.scheduler_g.gamma), "name": "exponential"},
                    "seed": int(self.generator.unwrapped().seed),
                },
                runtime.device,
            )
        task = self._shadow
        task.generator.load_state_dict(self.generator.unwrapped().generator.state_dict(), strict=True)
        task.discriminator_mrd.load_state_dict(self.mrd.unwrapped().discriminator.state_dict(), strict=True)
        task.discriminator_band.load_state_dict(self.cqt.unwrapped().discriminator.state_dict(), strict=True)
        task.eval_mode()
        return task
