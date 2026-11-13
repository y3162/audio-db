"""Official MP-SENet GAN loop for the experiment framework.

The update order follows ``MPSENetTask._official``. One generator forward is
kept until the discriminator has stepped. The generator backward then uses
that graph and the updated discriminator. ``StandardTrainer`` and the
TF-GridNet / TF-Locoformer trainers are not used for this loop.

``ExponentialLR`` advances once when the epoch's usable micro-batches are
finished. A validation interval does not advance it. When the same update
both validates and ends the epoch, validation and best selection run, then
both schedulers step.

Checkpoint files are written only after both optimizers have stepped and
``end_step`` has returned. ``amp_fp16`` is refused: the two AdamW updates are
not one atomic scaler decision.
"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

import torch

from src.experiments.checkpoint import CheckpointManager
from src.experiments.data import Batch
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mp_senet.models import (
    MPSENetGANGenerator,
    MPSENetMetricDiscriminator,
    pack_forward,
)
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.selectors import scalar_metrics
from src.experiments.trainers import (
    _append_jsonl,
    _batch_on_device,
    _close_epoch_view,
    _rank,
    _rewind_metrics,
    _sample_path,
)
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.losses import METRIC_WEIGHT, discriminator_loss, generator_losses, generator_signal_losses
from src.experiments.official.mp_senet.reference import _require_discriminator_resolution, validation_model

_IN_EPOCH = "in_epoch"
_PENDING = "pending_boundary"
OFFICIAL_FAITHFUL = "official_faithful"
LEAKAGE_SAFE = "leakage_safe"
_CLAIMS = frozenset({OFFICIAL_FAITHFUL, LEAKAGE_SAFE})
OFFICIAL_VALIDATION_INTERVAL = 5000
LEAKAGE_SAFE_VALIDATION_INTERVAL = 2675
OFFICIAL_CHECKPOINT_INTERVAL = 5000
LEAKAGE_SAFE_CHECKPOINT_INTERVAL = 2675
BEST_CHECKPOINT_START_EPOCH = 40
OFFICIAL_LEARNING_RATE = 0.0005
OFFICIAL_BETAS = (0.8, 0.99)
OFFICIAL_WEIGHT_DECAY = 0.01
OFFICIAL_GAMMA = 0.99
_LOSS_NAMES = ("total", "magnitude", "phase", "complex", "consistency", "metric", "time")


class MPSENetGANTrainer:
    """Discriminator-then-generator update for the official MP-SENet loss."""

    def __init__(
        self,
        generator: PreparedModel,
        discriminator: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
        scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
        *,
        max_logical_steps: int,
        experiment_claim: str,
        accumulation: int = 1,
        validation_interval: int = OFFICIAL_VALIDATION_INTERVAL,
        best_checkpoint_start_epoch: int = BEST_CHECKPOINT_START_EPOCH,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
    ) -> None:
        if not isinstance(generator, PreparedModel) or not isinstance(generator.state_module, MPSENetGANGenerator):
            raise BuildError("generator must be a prepared MPSENetGANGenerator")
        if not isinstance(discriminator, PreparedModel) or not isinstance(
            discriminator.state_module, MPSENetMetricDiscriminator
        ):
            raise BuildError("discriminator must be a prepared MPSENetMetricDiscriminator")
        if generator.state_module.architecture != discriminator.state_module.architecture:
            raise BuildError("discriminator initialization architecture does not match the generator")
        if generator.state_module.seed != discriminator.state_module.seed:
            raise BuildError("generator and discriminator seeds differ")
        if not isinstance(optim_g, torch.optim.AdamW) or not isinstance(optim_d, torch.optim.AdamW):
            raise BuildError("MP-SENet GAN optimizers must be AdamW")
        if not isinstance(scheduler_g, torch.optim.lr_scheduler.ExponentialLR):
            raise BuildError("scheduler_g must be ExponentialLR")
        if not isinstance(scheduler_d, torch.optim.lr_scheduler.ExponentialLR):
            raise BuildError("scheduler_d must be ExponentialLR")
        _require_disjoint(optim_g, optim_d)
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        if experiment_claim not in _CLAIMS:
            raise BuildError("experiment_claim must be official_faithful or leakage_safe")
        optim_g_id = getattr(optim_g, "_experiment_component_id", None)
        optim_d_id = getattr(optim_d, "_experiment_component_id", None)
        if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
            raise BuildError("optimizers have no component id")
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.experiment_claim = experiment_claim
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.validation_interval = _non_negative_int("validation_interval", validation_interval)
        self.best_checkpoint_start_epoch = _non_negative_int(
            "best_checkpoint_start_epoch", best_checkpoint_start_epoch
        )
        self.max_epochs = None if max_epochs is None else _positive_int("max_epochs", max_epochs)
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = _positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self._mp_io: tuple[Any, ...] | None = None
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.scheduler_steps = 0
        self.step_log: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.completed_boundaries: list[dict[str, Any]] = []
        self.boundary: dict[str, Any] | None = None
        self.phase = _IN_EPOCH
        self.best_pesq: float | None = None
        self.best_epoch: int | None = None
        self.best_logical_step: int | None = None
        self.stop_reason = "initialized"
        self._restored = False
        self._in_official_step = False
        self._shadow = None
        self._average_calls = 0
        self._generator_synced_during_backward = False

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        """Train until the step budget, an epoch cap, or an interrupt."""

        self._refuse_amp(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        rank = _rank(runtime)
        metrics_path = Path(checkpoint.directory) / "metrics.jsonl"
        sample_path = _sample_path(checkpoint.directory, runtime)
        if self._restored:
            if not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index"):
                raise RuntimeFailure("resumed data has no epoch position")
            if rank == 0:
                kept = _rewind_metrics(metrics_path, self.logical_step)
                if len(kept) != len(self.step_log):
                    raise RuntimeFailure("metrics.jsonl does not match the trainer log")
            if sample_path is not None:
                kept = _rewind_metrics(sample_path, self.logical_step)
                if len(kept) != len(self.step_log):
                    raise RuntimeFailure("sample log does not match the trainer log")
        else:
            if rank == 0:
                metrics_path.write_text("", encoding="utf-8")
            if sample_path is not None:
                sample_path.write_text("", encoding="utf-8")
            self.logical_step = 0
            self.optimizer_updates = {self.optim_g_id: 0, self.optim_d_id: 0}
            self.scheduler_steps = 0
            self.step_log = []
            self.history = []
            self.completed_boundaries = []
            self.boundary = None
            self.phase = _IN_EPOCH
            self.best_pesq = None
            self.best_epoch = None
            self.best_logical_step = None
            self.data.epoch = 0
            self.data.update_index = 0
        self._check_claim()
        self._require_group_size(component_id)
        runtime.barrier()
        if self.phase == _PENDING:
            self._finish_boundary(runtime, checkpoint, component_id)
            if self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                self._drain_mp_io()
                if self.logical_step > 0 and not self._in_official_step:
                    self._offer(checkpoint, "stop")
                return self._result(checkpoint, "interrupted")
            if self._epoch_cap_reached():
                self._drain_mp_io()
                return self._result(checkpoint, "completed")
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
            view = self.data.train_epoch(epoch, start_update=index)
            try:
                if view.updates < 1:
                    raise RuntimeFailure(f"component {component_id}: epoch {epoch} has no micro-batches")
                if view.updates % self.accumulation != 0:
                    raise RuntimeFailure(
                        f"component {component_id}: epoch micro-batches {view.updates} "
                        f"are not divisible by accumulation {self.accumulation}"
                    )
                if index > view.updates:
                    raise RuntimeFailure(f"component {component_id}: update {index} is past epoch {epoch}")
                if index == view.updates:
                    raise RuntimeFailure(
                        f"component {component_id}: epoch {epoch} is exhausted without a pending boundary"
                    )
                if index + self.accumulation > view.updates:
                    raise RuntimeFailure(
                        f"component {component_id}: accumulation {self.accumulation} does not fit at update {index}"
                    )
                batches: list[Batch] = []
                for offset in range(self.accumulation):
                    batches.extend(view.batch(index + offset))
                local = self.official_step(batches, runtime)
                shared = _shared_log(runtime, local)
                self.logical_step += 1
                index += self.accumulation
                self.optimizer_updates[self.optim_g_id] += 1
                self.optimizer_updates[self.optim_d_id] += 1
                self.data.epoch = epoch
                self.data.update_index = index
                entry = {
                    "applied_d": True,
                    "applied_g": True,
                    "complex": shared["complex"],
                    "consistency": shared["consistency"],
                    "discriminator": shared["discriminator"],
                    "epoch": epoch,
                    "generator": shared["generator"],
                    "logical_step": self.logical_step,
                    "loss": shared["generator"],
                    "lr_d": shared["lr_d"],
                    "lr_g": shared["lr_g"],
                    "magnitude": shared["magnitude"],
                    "metric": shared["metric"],
                    "phase": shared["phase"],
                    "sample_id": shared["sample_id"],
                    "time": shared["time"],
                }
                self.step_log.append(entry)
                epoch_done = index == view.updates
                will_validate = self.validation_interval > 0 and self.logical_step % self.validation_interval == 0
                offer_step = not will_validate and not epoch_done
                if runtime.device.type == "cuda" and runtime.world_size == 1:
                    self._mp_io = (entry, local, metrics_path, sample_path, rank, offer_step, checkpoint)
                    if offer_step:
                        continue
                    self._drain_mp_io()
                else:
                    if rank == 0:
                        _append_jsonl(metrics_path, entry)
                    if sample_path is not None:
                        _append_jsonl(sample_path, {**local, "logical_step": self.logical_step, "loss": local["generator"]})
                    if offer_step:
                        self._offer(checkpoint, "step")
                        continue
                self.phase = _PENDING
                self.boundary = _open_boundary(epoch, validate=will_validate, epoch_done=epoch_done)
                self._on_boundary_phase(self.phase)
                self._offer(checkpoint, "boundary_pending")
                if self._stop_requested(runtime):
                    return self._result(checkpoint, "interrupted")
                self._finish_boundary(runtime, checkpoint, component_id)
                if self._epoch_cap_reached():
                    return self._result(checkpoint, "completed")
            finally:
                _close_epoch_view(view)
        self._drain_mp_io()
        return self._result(checkpoint, "completed")

    def _drain_mp_io(self) -> None:
        """Write the previous step while the next signal backward is queued."""

        item = self._mp_io
        if item is None:
            return
        self._mp_io = None
        entry, local, metrics_path, sample_path, rank, offer_step, checkpoint = item
        if rank == 0:
            _append_jsonl(metrics_path, entry)
        if sample_path is not None:
            _append_jsonl(sample_path, {**local, "logical_step": entry["logical_step"], "loss": local["generator"]})
        if offer_step:
            self._offer(checkpoint, "step")

    def _cuda_overlapped_step(self, batch: Batch, runtime: ProcessRuntime) -> dict[str, float]:
        """PESQ runs while the signal backward uses the same generator graph.

        The discriminator optimizer still steps before the metric term and the
        generator optimizer. CPU training keeps the original order so the
        reference comparison stays exact.
        """

        self._on_update_phase("zero_grad")
        self.optim_d.zero_grad(set_to_none=True)
        self.optim_g.zero_grad(set_to_none=True)
        clean, noisy = self._waveforms(batch, runtime)
        if not _inputs_ready(clean, noisy, self.generator.state_module.n_fft):
            raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
        self._on_update_phase("before_generator_forward")
        try:
            parts = pack_forward(self.generator.module(noisy, clean))
            _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"generator forward failed: {type(exc).__name__}: {exc}") from exc
        pending = _schedule_metric_targets(
            clean,
            parts["audio"],
            int(self.generator.state_module.architecture["sampling_rate"]),
        )
        ones = torch.ones(clean.shape[0], device=clean.device)
        signal = generator_signal_losses(
            parts["clean_mag"],
            parts["mag"],
            parts["clean_phase"],
            parts["phase"],
            parts["clean_complex"],
            parts["complex"],
            parts["recomputed_complex"],
            clean,
            parts["audio"],
        )
        self._backward_if_finite(runtime, signal["signal"] / self.accumulation, "generator signal loss", retain_graph=True)
        self._drain_mp_io()
        try:
            targets = pending.result()
            real = self.discriminator.module(parts["clean_mag"], parts["clean_mag"])
            fake = self.discriminator.module(parts["clean_mag"], parts["mag_hat"].detach())
            loss_d = discriminator_loss(real, fake, ones, targets) / self.accumulation
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"discriminator forward failed: {type(exc).__name__}: {exc}") from exc
        self._backward_if_finite(runtime, loss_d, "discriminator loss")
        discriminator_result = runtime.step(self.optim_d)
        self._on_update_phase("after_discriminator_step", [(clean, parts)])
        raw = self.discriminator.state_module.discriminator
        frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            try:
                metric = raw(parts["clean_mag"], parts["mag_hat"])
                metric_loss = torch.nn.functional.mse_loss(metric.flatten(), ones)
            except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
                raise RuntimeFailure(f"generator loss failed: {type(exc).__name__}: {exc}") from exc
            self._backward_if_finite(runtime, metric_loss * METRIC_WEIGHT / self.accumulation, "generator metric loss")
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        self._average_calls += 1
        runtime.average_gradients(self.generator.parameters)
        total = signal["signal"] + metric_loss * METRIC_WEIGHT
        logged = torch.stack(
            [
                signal["complex"].detach(),
                signal["consistency"].detach(),
                loss_d.detach() * self.accumulation,
                total.detach(),
                signal["magnitude"].detach(),
                metric_loss.detach(),
                signal["phase"].detach(),
                signal["time"].detach(),
            ]
        )
        host = torch.empty(logged.shape, dtype=torch.float32, pin_memory=True)
        host.copy_(logged.detach().float(), non_blocking=True)
        generator_result = runtime.step(self.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(self.generator.state_module)
        runtime.broadcast_buffers(self.discriminator.state_module)
        if not discriminator_result.applied or not generator_result.applied:
            raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
        complex_v, consistency_v, disc_v, total_v, magnitude_v, metric_v, phase_v, time_v = (
            float(item) for item in host.tolist()
        )
        self._on_update_phase("end_step")
        return {
            "complex": complex_v,
            "consistency": consistency_v,
            "discriminator": disc_v,
            "generator": total_v,
            "lr_d": float(self.optim_d.param_groups[0]["lr"]),
            "lr_g": float(self.optim_g.param_groups[0]["lr"]),
            "magnitude": magnitude_v,
            "metric": metric_v,
            "phase": phase_v,
            "sample_id": batch.sample_id,
            "time": time_v,
        }

    def official_step(self, batches: list[Batch], runtime: ProcessRuntime) -> dict[str, float]:
        """Run one official GAN update. Both optimizers step, or neither is checkpointed."""

        self._refuse_amp(runtime)
        if len(batches) != self.accumulation:
            raise RuntimeFailure(
                f"official step received {len(batches)} micro-batches, accumulation is {self.accumulation}"
            )
        counts = runtime.gather_objects(len(batches))
        if any(item != len(batches) for item in counts):
            raise RuntimeFailure("ranks disagree on the number of micro-batches")
        self.generator.module.train()
        self.discriminator.module.train()
        self._in_official_step = True
        try:
            if runtime.device.type == "cuda" and runtime.world_size == 1 and len(batches) == 1:
                return self._cuda_overlapped_step(batches[0], runtime)
            self._on_update_phase("zero_grad")
            self.optim_d.zero_grad(set_to_none=True)
            self.optim_g.zero_grad(set_to_none=True)
            stored: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
            discriminator_total = 0.0
            sample_ids: list[str] = []
            with runtime.sync_context([self.generator], sync=False):
                for index, batch in enumerate(batches):
                    last = index == len(batches) - 1
                    clean, noisy = self._waveforms(batch, runtime)
                    sample_ids.append(batch.sample_id)
                    ready = _inputs_ready(clean, noisy, self.generator.state_module.n_fft)
                    if not runtime.agree(ready, policy="all"):
                        raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
                    self._on_update_phase("before_generator_forward")
                    error: str | None = None
                    parts: dict[str, torch.Tensor] | None = None
                    try:
                        parts = pack_forward(self.generator.module(noisy, clean))
                        _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
                    except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    errors = runtime.gather_objects(error)
                    if any(errors):
                        raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
                    if parts is None:
                        raise RuntimeFailure("generator forward produced no tensors")
                    stored.append((clean, parts))
                    ones = torch.ones(clean.shape[0], device=clean.device)
                    with runtime.sync_context([self.discriminator], sync=last):
                        disc_error: str | None = None
                        loss: torch.Tensor | None = None
                        try:
                            real = self.discriminator.module(parts["clean_mag"], parts["clean_mag"])
                            fake = self.discriminator.module(parts["clean_mag"], parts["mag_hat"].detach())
                            targets = _metric_targets(clean, parts["audio"], int(self.generator.state_module.architecture["sampling_rate"]))
                            loss = discriminator_loss(real, fake, ones, targets) / self.accumulation
                        except (TrainingError, RuntimeError, ValueError) as exc:
                            disc_error = f"{type(exc).__name__}: {exc}"
                        disc_errors = runtime.gather_objects(disc_error)
                        if any(disc_errors):
                            raise RuntimeFailure(
                                "discriminator forward failed: " + "; ".join(item for item in disc_errors if item)
                            )
                        if loss is None:
                            raise RuntimeFailure("discriminator loss is missing")
                        self._backward_if_finite(runtime, loss, "discriminator loss")
                        discriminator_total += float(loss.detach()) * self.accumulation
                discriminator_result = runtime.step(self.optim_d)
                self._on_update_phase("after_discriminator_step", stored)
                raw = self.discriminator.state_module.discriminator
                frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
                component_totals = {name: 0.0 for name in _LOSS_NAMES}
                try:
                    for parameter in frozen:
                        parameter.requires_grad_(False)
                    for clean, parts in stored:
                        ones = torch.ones(clean.shape[0], device=clean.device)
                        gen_error: str | None = None
                        total: torch.Tensor | None = None
                        try:
                            metric = raw(parts["clean_mag"], parts["mag_hat"])
                            losses = generator_losses(
                                parts["clean_mag"],
                                parts["mag"],
                                parts["clean_phase"],
                                parts["phase"],
                                parts["clean_complex"],
                                parts["complex"],
                                parts["recomputed_complex"],
                                clean,
                                parts["audio"],
                                metric,
                                ones,
                            )
                            total = losses["total"] / self.accumulation
                        except (TrainingError, RuntimeError, ValueError) as exc:
                            gen_error = f"{type(exc).__name__}: {exc}"
                        gen_errors = runtime.gather_objects(gen_error)
                        if any(gen_errors):
                            raise RuntimeFailure(
                                "generator loss failed: " + "; ".join(item for item in gen_errors if item)
                            )
                        if total is None:
                            raise RuntimeFailure("generator loss is missing")
                        module = self.generator.module
                        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
                            self._generator_synced_during_backward = bool(module.require_backward_grad_sync)
                        self._backward_if_finite(runtime, total, "generator loss")
                        for name in _LOSS_NAMES:
                            component_totals[name] += float(losses[name].detach())
                finally:
                    for parameter in frozen:
                        parameter.requires_grad_(True)
                self._average_calls += 1
                runtime.average_gradients(self.generator.parameters)
            generator_result = runtime.step(self.optim_g)
            runtime.end_step()
            runtime.broadcast_buffers(self.generator.state_module)
            runtime.broadcast_buffers(self.discriminator.state_module)
            if not discriminator_result.applied or not generator_result.applied:
                raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
            count = len(batches)
            self._on_update_phase("end_step")
            return {
                "complex": component_totals["complex"] / count,
                "consistency": component_totals["consistency"] / count,
                "discriminator": discriminator_total / count,
                "generator": component_totals["total"] / count,
                "lr_d": float(self.optim_d.param_groups[0]["lr"]),
                "lr_g": float(self.optim_g.param_groups[0]["lr"]),
                "magnitude": component_totals["magnitude"] / count,
                "metric": component_totals["metric"] / count,
                "phase": component_totals["phase"] / count,
                "sample_id": ",".join(sample_ids),
                "time": component_totals["time"] / count,
            }
        finally:
            self._in_official_step = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_logical_step": self.best_logical_step,
            "best_pesq": self.best_pesq,
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "experiment_claim": self.experiment_claim,
            "history": list(self.history),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "scheduler_steps": self.scheduler_steps,
            "step_log": list(self.step_log),
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
        if state.get("experiment_claim") != self.experiment_claim:
            raise RuntimeFailure("experiment claim does not match this trainer")
        history = state.get("history")
        completed = state.get("completed_boundaries")
        step_log = state.get("step_log")
        if not isinstance(history, list) or not isinstance(completed, list) or not isinstance(step_log, list):
            raise RuntimeFailure("trainer history is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        self.phase = phase
        self.history = list(history)
        self.completed_boundaries = list(completed)
        self.step_log = list(step_log)
        boundary = state.get("boundary")
        self.boundary = None if boundary is None else dict(boundary)
        self.scheduler_steps = int(state.get("scheduler_steps", 0))
        best = state.get("best_pesq")
        self.best_pesq = None if best is None else float(best)
        best_epoch = state.get("best_epoch")
        best_step = state.get("best_logical_step")
        if best_epoch is not None and (isinstance(best_epoch, bool) or not isinstance(best_epoch, int)):
            raise RuntimeFailure("best_epoch is invalid")
        if best_step is not None and (isinstance(best_step, bool) or not isinstance(best_step, int)):
            raise RuntimeFailure("best_logical_step is invalid")
        self.best_epoch = best_epoch
        self.best_logical_step = best_step
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
        self._shadow = None
        self._restored = True

    def _on_update_phase(self, phase: str, payload: object = None) -> None:
        """Hook for tests. The default records nothing."""

        del phase, payload

    def _on_boundary_phase(self, phase: str) -> None:
        """Hook for tests that record validation and scheduler order."""

        del phase

    def _finish_boundary(
        self,
        runtime: ProcessRuntime,
        checkpoint: CheckpointManager,
        component_id: str,
    ) -> None:
        if self.phase != _PENDING or self.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending GAN boundary")
        boundary = self.boundary
        trainer_epoch = int(boundary["trainer_epoch"])
        metrics: dict[str, float] = {}
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self._validate(runtime, trainer_epoch, component_id)
            metrics = scalar_metrics(summary)
            decision = self._consider_best(trainer_epoch, summary)
            self.history.append(decision)
            self._write_best(runtime, checkpoint.directory, decision)
            boundary["validation_done"] = True
            boundary["selection_done"] = True
            boundary["pesq"] = decision["pesq"]
            boundary["pesq_status"] = decision["pesq_status"]
            boundary["selected"] = decision["selected"]
            if boundary["scheduler"]:
                self.phase = "pending_scheduler"
                self._on_boundary_phase(self.phase)
        if boundary["scheduler"] and not boundary["scheduler_done"]:
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.scheduler_steps += 1
            boundary["scheduler_done"] = True
            boundary["lr_g"] = float(self.optim_g.param_groups[0]["lr"])
            boundary["lr_d"] = float(self.optim_d.param_groups[0]["lr"])
        if boundary["epoch_done"]:
            self.data.epoch = trainer_epoch + 1
            self.data.update_index = 0
        boundary["closed"] = True
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = _IN_EPOCH
        self._on_boundary_phase(self.phase)
        self._offer(checkpoint, "boundary", metrics=metrics)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        cursor = (int(self.data.epoch), int(self.data.update_index))
        rng = runtime.capture_rng()
        error: str | None = None
        summary: dict[str, Any] | None = None
        try:
            if runtime.rank == 0:
                try:
                    summary = self._validate_rank0(trainer_epoch, runtime, component_id)
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

    def _validate_rank0(self, trainer_epoch: int, runtime: ProcessRuntime, component_id: str) -> dict[str, Any]:
        task = self._shadow_task(runtime)
        reports: list[dict[str, Any]] = []
        for batch in self.data.validation_epoch(trainer_epoch):
            if not isinstance(batch, Batch):
                raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
            batch = _batch_on_device(batch, runtime.device)
            report = task.validation_report({"clean": batch.targets, "noisy": batch.inputs})
            reports.append(report)
        if not reports:
            raise RuntimeFailure(f"component {component_id}: rank 0 validation produced no batches")
        return _aggregate_reports(reports)

    def _shadow_task(self, runtime: ProcessRuntime):
        if self._shadow is None:
            group = self.optim_g.param_groups[0]
            betas = group["betas"]
            self._shadow = validation_model(
                {
                    "loss": "official",
                    "model": dict(self.generator.state_module.architecture),
                    "optimizer": {
                        "betas": [float(betas[0]), float(betas[1])],
                        "lr": float(group["lr"]),
                        "weight_decay": float(group["weight_decay"]),
                    },
                    "scheduler": {"gamma": float(self.scheduler_g.gamma)},
                    "seed": int(self.generator.state_module.seed),
                },
                runtime.device,
            )
        task = self._shadow
        if task.discriminator is None:
            raise RuntimeFailure("shadow MP-SENet task has no discriminator")
        task.generator.load_state_dict(self.generator.state_module.generator.state_dict(), strict=True)
        task.discriminator.load_state_dict(self.discriminator.state_module.discriminator.state_dict(), strict=True)
        task.eval_mode()
        return task

    def _consider_best(self, epoch: int, summary: dict[str, Any]) -> dict[str, Any]:
        """Keep the earlier checkpoint unless PESQ is strictly higher after the start epoch."""

        pesq = summary.get("pesq")
        if not isinstance(pesq, dict):
            pesq = {"status": "na", "value": None, "reason": "missing pesq"}
        status = pesq.get("status")
        value = pesq.get("value")
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "logical_step": int(self.logical_step),
            "pesq": None if value is None else float(value),
            "pesq_status": status,
            "reason": None,
            "selected": False,
        }
        if epoch < self.best_checkpoint_start_epoch:
            record["reason"] = "before_start_epoch"
            return record
        if status != "ok" or value is None:
            record["reason"] = "unavailable" if status == "unavailable" else "not_ok"
            return record
        score = float(value)
        if not math.isfinite(score):
            record["reason"] = "non_finite"
            return record
        if self.best_pesq is not None and score <= float(self.best_pesq):
            record["reason"] = "not_higher"
            return record
        self.best_pesq = score
        self.best_epoch = int(epoch)
        self.best_logical_step = int(self.logical_step)
        record["selected"] = True
        record["reason"] = "improved"
        return record

    def _write_best(self, runtime: ProcessRuntime, directory: Path, decision: dict[str, Any]) -> None:
        error: str | None = None
        if _rank(runtime) == 0 and decision["selected"]:
            try:
                _write_best_files(
                    directory,
                    self.generator.state_module.export_for_inference(),
                    decision,
                    self.experiment_claim,
                )
            except (OSError, RuntimeError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        failure = next((item for item in errors if item), None)
        if failure:
            raise RuntimeFailure(f"best checkpoint write failed: {failure}")

    def _waveforms(self, batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        batch = _batch_on_device(batch, runtime.device)
        clean = batch.targets
        noisy = batch.inputs
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        return clean, noisy

    def _backward_if_finite(
        self,
        runtime: ProcessRuntime,
        loss: torch.Tensor,
        what: str,
        *,
        retain_graph: bool = False,
    ) -> None:
        if runtime.device.type != "cuda":
            finite = bool(torch.isfinite(loss.detach()).all())
            if not runtime.agree(finite, policy="all"):
                raise RuntimeFailure(f"{what} is not finite on every rank")
        runtime.backward(loss, retain_graph=retain_graph)

    def _refuse_amp(self, runtime: ProcessRuntime) -> None:
        if getattr(runtime, "precision", "fp32") == "amp_fp16":
            raise RuntimeFailure(
                "MP-SENet official GAN refuses amp_fp16. "
                "The current scaler decides overflow per optimizer, so the discriminator "
                "step and the generator step are not one atomic update. FP32 is unchanged."
            )

    def _check_claim(self) -> None:
        identity_fn = getattr(self.data, "identity", None)
        identity = identity_fn() if callable(identity_fn) else {}
        if not isinstance(identity, dict):
            raise RuntimeFailure("data identity is not a mapping")
        algorithm = identity.get("algorithm")
        if algorithm is not None and algorithm != "fixed_pair.mp_senet":
            raise RuntimeFailure(f"MP-SENet GAN requires fixed_pair.mp_senet data, got {algorithm}")
        profile = identity.get("profile")
        uses_test = identity.get("selection_uses_test")
        if self.experiment_claim == OFFICIAL_FAITHFUL:
            if profile not in (None, OFFICIAL_FAITHFUL):
                raise RuntimeFailure("official_faithful data profile must be official_faithful")
            if uses_test is False:
                raise RuntimeFailure("official_faithful selection uses the dataset test split as validation")
            return
        if profile not in (None, LEAKAGE_SAFE):
            raise RuntimeFailure("leakage_safe data profile must be leakage_safe")
        if uses_test is True:
            raise RuntimeFailure("leakage_safe selection keeps the test split out of best-checkpoint decisions")

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if int(group_size) != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _offer(
        self,
        checkpoint: CheckpointManager,
        event: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Offer this state to the selector. A GAN optimizer pair is not a save point."""

        if self._in_official_step:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")
        checkpoint.offer(event, logical_step=self.logical_step, metrics=metrics)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        from src.experiments.signals import ranks_should_stop

        capped = self.stop_after_logical_steps is not None and self.logical_step >= self.stop_after_logical_steps
        return ranks_should_stop(runtime, extra=capped)

    def _epoch_cap_reached(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

    def _result(self, checkpoint: CheckpointManager, reason: str) -> RunResult:
        if reason == "completed":
            checkpoint.offer("final", logical_step=self.logical_step)
        self.stop_reason = reason
        return RunResult(
            stop_reason=reason,
            logical_step=self.logical_step,
            optimizer_updates=dict(self.optimizer_updates),
            output_dir=checkpoint.directory,
        )


def _schedule_metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int):
    """Copy waveforms without blocking, then score PESQ on a side thread."""

    from src.experiments.calc.losses import metric_targets

    clean_cpu = torch.empty(clean.shape, dtype=clean.dtype, pin_memory=True)
    enhanced_cpu = torch.empty(enhanced.shape, dtype=enhanced.dtype, pin_memory=True)
    clean_cpu.copy_(clean.detach(), non_blocking=True)
    enhanced_cpu.copy_(enhanced.detach(), non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    box: dict[str, object] = {}
    done = threading.Event()

    def run() -> None:
        try:
            event.synchronize()
            box["value"] = metric_targets(clean_cpu, enhanced_cpu, sample_rate)
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run, name="mp-pesq", daemon=True)
    thread.start()

    class _Pending:
        def result(self):
            done.wait()
            if "error" in box:
                raise box["error"]
            return box.get("value")

    return _Pending()


def _metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    """Look up PESQ targets on the losses module so tests can patch that function."""

    from src.experiments.calc import losses as loss_lib

    return loss_lib.metric_targets(clean, enhanced, sample_rate)


def _inputs_ready(clean: torch.Tensor, noisy: torch.Tensor, n_fft: int) -> bool:
    if clean.ndim != 2 or clean.shape != noisy.shape:
        return False
    if int(clean.shape[-1]) <= n_fft // 2:
        return False
    if clean.is_cuda:
        # A non-finite waveform makes the loss non-finite, which rejects the step.
        return True
    return bool(torch.isfinite(clean).all()) and bool(torch.isfinite(noisy).all())


def _shared_log(runtime: ProcessRuntime, local: dict[str, float]) -> dict[str, float | str]:
    """Store rank 0's scalars on every rank so the checkpoint matches."""

    gathered = runtime.gather_objects(dict(local))
    shared = gathered[0]
    if not isinstance(shared, dict):
        raise RuntimeFailure("rank 0 did not return GAN metrics")
    return shared


def _require_disjoint(optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer) -> None:
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("MP-SENet GAN optimizers must each own parameters")


def _open_boundary(trainer_epoch: int, *, validate: bool, epoch_done: bool) -> dict[str, Any]:
    return {
        "closed": False,
        "epoch_done": bool(epoch_done),
        "lr_d": None,
        "lr_g": None,
        "pesq": None,
        "pesq_status": None,
        "scheduler": bool(epoch_done),
        "scheduler_done": False,
        "selected": False,
        "selection_done": False,
        "trainer_epoch": int(trainer_epoch),
        "validate": bool(validate),
        "validation_done": False,
    }


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of batch-level official reports. Unavailable stays unavailable."""

    names: list[str] = []
    for report in reports:
        for name in report["losses"]:
            if name not in names:
                names.append(name)
    losses = {
        name: _combine_metric([report["losses"][name] for report in reports if name in report["losses"]])
        for name in names
    }
    pesq = _combine_metric([report["pesq"] for report in reports])
    policy = next((report["pesq"].get("pesq_policy") for report in reports if isinstance(report.get("pesq"), dict)), None)
    if policy is not None:
        pesq["pesq_policy"] = policy
    lengths: list[dict[str, int]] = []
    for report in reports:
        rows = report.get("audio_lengths")
        if isinstance(rows, list):
            lengths.extend(rows)
    return {
        "audio_lengths": lengths,
        "losses": losses,
        "pesq": pesq,
        "stoi": _combine_metric([report["stoi"] for report in reports]),
    }


def _combine_metric(items: list[dict[str, Any]]) -> dict[str, Any]:
    if any(item.get("status") == "unavailable" for item in items):
        reason = next(item.get("reason") for item in items if item.get("status") == "unavailable")
        return {"status": "unavailable", "value": None, "reason": reason, "count": 0}
    values = [float(item["value"]) for item in items if item.get("status") == "ok" and item.get("value") is not None]
    if values:
        return {"status": "ok", "value": sum(values) / len(values), "reason": None, "count": len(values)}
    reason = next((item.get("reason") for item in items if item.get("reason")), "no score")
    return {"status": "na", "value": None, "reason": reason, "count": 0}


def _write_best_files(
    directory: Path,
    exported: dict[str, torch.Tensor],
    decision: dict[str, Any],
    experiment_claim: str,
) -> None:
    payload_path = directory / "best_generator.pt"
    meta_path = directory / "best_generator.json"
    temporary = directory / ".best_generator.pt.tmp"
    torch.save(exported, temporary)
    os.replace(temporary, payload_path)
    meta = {
        "checkpoint": payload_path.name,
        "epoch": decision["epoch"],
        "experiment_claim": experiment_claim,
        "logical_step": decision["logical_step"],
        "metric": "pesq",
        "pesq": decision["pesq"],
        "pesq_policy": "official_include_negative_one",
    }
    meta_temporary = directory / ".best_generator.json.tmp"
    meta_temporary.write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(meta_temporary, meta_path)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value
