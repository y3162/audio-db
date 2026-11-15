"""Official MP-SENet validation for one epoch.

Rank 0 scores the validation split with a shadow generator and discriminator
and ``metrics.aggregate_reports``. Every rank then receives that one summary.
The training cursor and the RNG captured around the call are restored.

The shadow is a cache of architecture and seed. It has no optimizer and no
scheduler. Each call copies the live generator and discriminator weights into
it. ``invalidate`` drops the cache so a resumed process does not keep a
shadow built before ``load_state_dict``.

This module does not choose when validation runs. It does not step a
scheduler, select a best PESQ, write a checkpoint, or call the research
evaluator. It does not construct the reference learning task.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.calc.rng import seed_model_init
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.discriminator import MetricDiscriminator
from src.experiments.official.mp_senet.metrics import aggregate_reports
from src.experiments.official.mp_senet.models import network_config
from src.experiments.official.mp_senet.validation_math import report_batch
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.se import build_model


class _Shadow:
    """Eval-mode weights for one report. Not a training task."""

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        architecture: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.spec = architecture
        self.device = device

    def eval_mode(self) -> None:
        self.generator.eval()
        self.discriminator.eval()

    def validation_report(self, batch: dict[str, Any]) -> dict[str, Any]:
        return report_batch(
            generator=self.generator,
            discriminator=self.discriminator,
            spec=self.spec,
            device=self.device,
            batch=batch,
            official=True,
        )


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
        # Kept so a caller can see which optimizer group the live model uses.
        # The shadow does not construct another optimizer from them.
        self.optim_g = optim_g
        self.scheduler_g = scheduler_g
        self._shadow: _Shadow | None = None

    def invalidate(self) -> None:
        """Drop the cached shadow. The next call builds a new one."""

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
        shadow = self._shadow_for(runtime)
        reports: list[dict[str, Any]] = []
        for batch in self.data.validation_epoch(trainer_epoch):
            if not isinstance(batch, Batch):
                raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
            batch = runtime.move(batch)
            report = shadow.validation_report({"clean": batch.targets, "noisy": batch.inputs})
            reports.append(report)
        if not reports:
            raise RuntimeFailure(f"component {component_id}: rank 0 validation produced no batches")
        return aggregate_reports(reports)

    def _shadow_for(self, runtime: ProcessRuntime) -> _Shadow:
        if self._shadow is None:
            architecture = dict(self.generator.unwrapped().architecture)
            seed = int(self.generator.unwrapped().seed)
            self._shadow = _build_shadow(architecture, seed, runtime.device)
        shadow = self._shadow
        shadow.generator.load_state_dict(self.generator.unwrapped().generator.state_dict(), strict=True)
        shadow.discriminator.load_state_dict(
            self.discriminator.unwrapped().discriminator.state_dict(), strict=True
        )
        shadow.eval_mode()
        return shadow


def _build_shadow(architecture: dict[str, Any], seed: int, device: torch.device) -> _Shadow:
    """Build eval modules under the same seed as the reference task, without an optimizer.

    The discarded generator consumes the same initialization draws the metric
    discriminator expects. Its weights are replaced before the report.
    """

    with seed_model_init(seed):
        generator = build_model("mp_senet", network_config(architecture)).to(device)
        if "disc_dim" in architecture:
            discriminator = MetricDiscriminator(dim=int(architecture["disc_dim"])).to(device)
        else:
            discriminator = MetricDiscriminator().to(device)
    return _Shadow(generator, discriminator, architecture, device)
