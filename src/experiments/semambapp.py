"""Official SEMamba++ GAN loop for the experiment framework.

The update order is ``SEMambaTask.train_update``. One micro-batch keeps the
generator graph, steps the discriminators, and reuses that graph. Two or more
micro-batches preview the generator without a graph, step the discriminators,
then forward the generator again. Losses are means, so each backward divides
by the accumulation count.

``ExponentialLR`` advances once when the data epoch is finished. A validation
interval does not advance it. Checkpoint files are written only after both
optimizers have stepped. ``best_generator.pt`` is not written.

``amp_fp16`` is refused. The two AdamW updates are not one atomic scaler
decision, and changing the update order to make them atomic is out of scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.experiments.checkpoint import CheckpointManager
from src.experiments.data import Batch
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.models.semambapp import (
    SEMambaPPCQTDiscriminator,
    SEMambaPPGenerator,
    SEMambaPPMRD,
    pack_forward,
    scan_execution,
)
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
from src.experiments.calc.metrics import _aggregate_validation
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.semambapp import validation_model
from src.experiments.calc.spectral import (
    compressed_complex_losses,
    feature_matching,
    least_squares_discriminator,
    least_squares_generator,
    summed_slaney_log_mel,
)

_IN_EPOCH = "in_epoch"
_PENDING = "pending_boundary"
_LOSS_NAMES = ("adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")
_ADV = 1.0
_FM = 1.0
_MEL = 0.1
_MAG = 0.9
_PHASE = 0.3
_COMPLEX = 0.1
_CONSISTENCY = 0.1
_OFFICIAL_LR = 0.0002
_OFFICIAL_BETAS = (0.8, 0.99)
_OFFICIAL_WEIGHT_DECAY = 0.01
_OFFICIAL_GAMMA = 0.99
_SUBSTITUTE_CLAIM = "protocol_conformant_substitute_materials"


class SEMambaPPTrainer:
    """Discriminator-then-generator update for the official SEMamba++ loss."""

    def __init__(
        self,
        generator: PreparedModel,
        mrd: PreparedModel,
        cqt: PreparedModel,
        data: Any,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.ExponentialLR,
        scheduler_d: torch.optim.lr_scheduler.ExponentialLR,
        *,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float | None = None,
        validation_interval: int = 0,
        checkpoint_interval: int = 1,
        best_checkpoint_start_epoch: int = 40,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
        claim_label: str = _SUBSTITUTE_CLAIM,
    ) -> None:
        if not isinstance(generator, PreparedModel) or not isinstance(generator.state_module, SEMambaPPGenerator):
            raise BuildError("generator must be a prepared SEMambaPPGenerator")
        if not isinstance(mrd, PreparedModel) or not isinstance(mrd.state_module, SEMambaPPMRD):
            raise BuildError("mrd must be a prepared SEMambaPPMRD")
        if not isinstance(cqt, PreparedModel) or not isinstance(cqt.state_module, SEMambaPPCQTDiscriminator):
            raise BuildError("cqt must be a prepared SEMambaPPCQTDiscriminator")
        if generator.state_module.architecture != mrd.state_module.architecture:
            raise BuildError("MRD initialization architecture does not match the generator")
        if generator.state_module.architecture != cqt.state_module.architecture:
            raise BuildError("CQT initialization architecture does not match the generator")
        if generator.state_module.seed != mrd.state_module.seed or generator.state_module.seed != cqt.state_module.seed:
            raise BuildError("generator, MRD, and CQT seeds differ")
        _require_adamw(optim_g, "optim_g")
        _require_adamw(optim_d, "optim_d")
        _require_scheduler(scheduler_g, "scheduler_g")
        _require_scheduler(scheduler_d, "scheduler_d")
        _require_disjoint(optim_g, optim_d)
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        optim_g_id = getattr(optim_g, "_experiment_component_id", None)
        optim_d_id = getattr(optim_d, "_experiment_component_id", None)
        if not isinstance(optim_g_id, str) or not isinstance(optim_d_id, str):
            raise BuildError("optimizers have no component id")
        if not isinstance(claim_label, str) or not claim_label:
            raise BuildError("claim_label must be a non-empty string")
        if claim_label == "official_reproduction":
            raise BuildError("semambapp.gsr.official is blocked")
        self.generator = generator
        self.mrd = mrd
        self.cqt = cqt
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.claim_label = claim_label
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.grad_clip = None if grad_clip is None else _positive_float("grad_clip", grad_clip)
        self.validation_interval = _non_negative_int("validation_interval", validation_interval)
        self.checkpoint_interval = _positive_int("checkpoint_interval", checkpoint_interval)
        self.best_checkpoint_start_epoch = _non_negative_int(
            "best_checkpoint_start_epoch", best_checkpoint_start_epoch
        )
        self.max_epochs = None if max_epochs is None else _positive_int("max_epochs", max_epochs)
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = _positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.scheduler_steps = 0
        self.step_log: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.completed_boundaries: list[dict[str, Any]] = []
        self.boundary: dict[str, Any] | None = None
        self.phase = _IN_EPOCH
        self.stop_reason = "initialized"
        self.capture_gradients = False
        self.last_generator_grads: dict[str, torch.Tensor] | None = None
        self.last_discriminator_grads: dict[str, torch.Tensor] | None = None
        self.generator_graph_forwards = 0
        self.generator_preview_forwards = 0
        self.generator_reforward_forwards = 0
        self.mrd_forwards = 0
        self.cqt_forwards = 0
        self.sync_trace: list[str] = []
        self.frozen_during_generator = False
        self.scan_report: dict[str, str] | None = None
        self._restored = False
        self._in_official_step = False
        self._shadow = None

    def run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> RunResult:
        """Train until the step budget, an epoch cap, or an interrupt."""

        self._refuse_amp(runtime)
        self._require_scan(runtime)
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
            self.data.epoch = 0
            self.data.update_index = 0
        self._check_data()
        self._require_group_size(component_id)
        runtime.barrier()
        if self.phase == _PENDING:
            self._finish_boundary(runtime, checkpoint, component_id)
            if self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                if self.logical_step > 0 and not self._in_official_step:
                    self._checkpoint(runtime, checkpoint, logical_step=self.logical_step, label="step")
                return self._result(checkpoint, "interrupted")
            if self._epoch_cap_reached():
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
                    "adversarial": shared["adversarial"],
                    "applied_d": True,
                    "applied_g": True,
                    "complex": shared["complex"],
                    "consistency": shared["consistency"],
                    "discriminator": shared["discriminator"],
                    "epoch": epoch,
                    "generator": shared["generator"],
                    "generator_forwards": shared["generator_forwards"],
                    "generator_previews": shared["generator_previews"],
                    "generator_reforwards": shared["generator_reforwards"],
                    "logical_step": self.logical_step,
                    "loss": shared["generator"],
                    "lr_d": shared["lr_d"],
                    "lr_g": shared["lr_g"],
                    "magnitude": shared["magnitude"],
                    "matching": shared["matching"],
                    "mel": shared["mel"],
                    "phase": shared["phase"],
                    "sample_id": shared["sample_id"],
                    "scan_configured": shared["scan_configured"],
                    "scan_executed": shared["scan_executed"],
                }
                self.step_log.append(entry)
                if rank == 0:
                    _append_jsonl(metrics_path, entry)
                if sample_path is not None:
                    _append_jsonl(
                        sample_path,
                        {
                            "logical_step": self.logical_step,
                            "loss": local["generator"],
                            "realization_id": local["realization_id"],
                            "sample_id": local["sample_id"],
                        },
                    )
                epoch_done = index == view.updates
                will_validate = self.validation_interval > 0 and self.logical_step % self.validation_interval == 0
                if not will_validate and not epoch_done:
                    if self.logical_step % self.checkpoint_interval == 0:
                        self._checkpoint(runtime, checkpoint, logical_step=self.logical_step, label="step")
                    continue
                self.phase = _PENDING
                self.boundary = _open_boundary(epoch, validate=will_validate, epoch_done=epoch_done)
                self._on_boundary_phase(self.phase)
                self._checkpoint(runtime, checkpoint, logical_step=self.logical_step, label="validation_pending")
                if self._stop_requested(runtime):
                    return self._result(checkpoint, "interrupted")
                self._finish_boundary(runtime, checkpoint, component_id)
                if self._epoch_cap_reached():
                    return self._result(checkpoint, "completed")
            finally:
                _close_epoch_view(view)
        return self._result(checkpoint, "completed")

    def official_step(self, batches: list[Batch], runtime: ProcessRuntime) -> dict[str, float | str]:
        """Run one official GAN update. Both optimizers step, or neither is checkpointed."""

        self._refuse_amp(runtime)
        self._require_scan(runtime)
        if len(batches) != self.accumulation:
            raise RuntimeFailure(
                f"official step received {len(batches)} micro-batches, accumulation is {self.accumulation}"
            )
        counts = runtime.gather_objects(len(batches))
        if any(item != len(batches) for item in counts):
            raise RuntimeFailure("ranks disagree on the number of micro-batches")
        self.generator.module.train()
        self.mrd.module.train()
        self.cqt.module.train()
        self._in_official_step = True
        graph_before = self.generator_graph_forwards
        preview_before = self.generator_preview_forwards
        reforward_before = self.generator_reforward_forwards
        try:
            self._on_update_phase("zero_grad")
            self.optim_d.zero_grad(set_to_none=True)
            reuse_graph = len(batches) == 1
            cached: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]] = []
            discriminator_total = 0.0
            sample_ids: list[str] = []
            realization_ids: list[str] = []
            for index, batch in enumerate(batches):
                last = index == len(batches) - 1
                clean, noisy = self._waveforms(batch, runtime)
                sample_ids.append(batch.sample_id)
                realization_ids.append(batch.realization_id)
                ready = _inputs_ready(clean, noisy)
                if not runtime.agree(ready, policy="all"):
                    raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
                if reuse_graph:
                    self.sync_trace.append("generator_forward_sync")
                    with runtime.sync_context([self.generator], sync=True):
                        parts = self._generator_forward(runtime, clean, noisy, preview=False)
                    self.generator_graph_forwards += 1
                    audio = parts["audio"]
                    cached.append((clean, noisy, parts))
                else:
                    self.sync_trace.append("generator_preview_nograd")
                    with torch.no_grad():
                        preview = self._generator_forward(runtime, clean, noisy, preview=True)
                    self.generator_preview_forwards += 1
                    audio = preview["audio"]
                    cached.append((clean, noisy, None))
                self.sync_trace.append("discriminator_sync" if last else "discriminator_nosync")
                with runtime.sync_context([self.mrd, self.cqt], sync=last):
                    loss = self._discriminator_loss(runtime, clean, audio.detach())
                    self._backward_if_finite(runtime, loss / self.accumulation, "discriminator loss")
                discriminator_total += float(loss.detach())
                self.mrd_forwards += 1
                self.cqt_forwards += 1
            if self.grad_clip is not None:
                runtime.clip(self.mrd.parameters, self.grad_clip)
                runtime.clip(self.cqt.parameters, self.grad_clip)
            disc_norm = _grad_norm(self.mrd.state_module.discriminator, self.cqt.state_module.discriminator)
            if self.capture_gradients:
                self.last_discriminator_grads = _clone_grads(
                    self.mrd.state_module.discriminator, self.cqt.state_module.discriminator
                )
            self._on_update_phase("before_discriminator_step")
            discriminator_result = runtime.step(self.optim_d)
            self._on_update_phase("after_discriminator_step")
            frozen = [
                parameter
                for module in (self.mrd.state_module, self.cqt.state_module)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.optim_g.zero_grad(set_to_none=True)
            component_totals = {name: 0.0 for name in _LOSS_NAMES}
            generator_total = 0.0
            try:
                for parameter in frozen:
                    parameter.requires_grad_(False)
                self.frozen_during_generator = True
                self._on_update_phase("before_generator_backward")
                for index, (clean, _noisy, parts) in enumerate(cached):
                    last = index == len(cached) - 1
                    if parts is None:
                        self.sync_trace.append("generator_reforward_sync" if last else "generator_reforward_nosync")
                        with runtime.sync_context([self.generator], sync=last):
                            parts = self._generator_forward(runtime, clean, _noisy, preview=False)
                            self.generator_reforward_forwards += 1
                            generator_total += self._generator_backward(runtime, clean, parts, component_totals)
                    else:
                        self.sync_trace.append("generator_backward_retained")
                        generator_total += self._generator_backward(runtime, clean, parts, component_totals)
            finally:
                for parameter in frozen:
                    parameter.requires_grad_(True)
                self.frozen_during_generator = False
            if self.grad_clip is not None:
                runtime.clip(self.generator.parameters, self.grad_clip)
            gen_norm = _grad_norm(self.generator.state_module.generator)
            if self.capture_gradients:
                self.last_generator_grads = _clone_grads(self.generator.state_module.generator)
            self._on_update_phase("before_generator_step")
            generator_result = runtime.step(self.optim_g)
            runtime.end_step()
            runtime.broadcast_buffers(self.generator.state_module)
            runtime.broadcast_buffers(self.mrd.state_module)
            runtime.broadcast_buffers(self.cqt.state_module)
            if not discriminator_result.applied or not generator_result.applied:
                raise RuntimeFailure("SEMamba++ requires both optimizers to apply in one logical step")
            count = len(batches)
            report = self.scan_report or {"configured": "", "executed": ""}
            self._on_update_phase("end_step")
            metrics: dict[str, float | str] = {
                "adversarial": component_totals["adversarial"] / count,
                "complex": component_totals["complex"] / count,
                "consistency": component_totals["consistency"] / count,
                "discriminator": discriminator_total / count,
                "discriminator_grad_norm": disc_norm,
                "generator": generator_total / count,
                "generator_forwards": float(self.generator_graph_forwards - graph_before),
                "generator_grad_norm": gen_norm,
                "generator_previews": float(self.generator_preview_forwards - preview_before),
                "generator_reforwards": float(self.generator_reforward_forwards - reforward_before),
                "lr_d": float(self.optim_d.param_groups[0]["lr"]),
                "lr_g": float(self.optim_g.param_groups[0]["lr"]),
                "magnitude": component_totals["magnitude"] / count,
                "matching": component_totals["matching"] / count,
                "mel": component_totals["mel"] / count,
                "phase": component_totals["phase"] / count,
                "realization_id": ",".join(realization_ids),
                "sample_id": ",".join(sample_ids),
                "scan_configured": report["configured"],
                "scan_executed": report["executed"],
            }
            return metrics
        finally:
            self._in_official_step = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "claim_label": self.claim_label,
            "completed_boundaries": list(self.completed_boundaries),
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
        if state.get("claim_label") != self.claim_label:
            raise RuntimeFailure("claim_label does not match this trainer")
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
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self._validate(runtime, trainer_epoch, component_id)
            self.history.append(
                {
                    "epoch": trainer_epoch,
                    "logical_step": int(self.logical_step),
                    "reason": "semambapp_does_not_select_best_generator",
                    "selected": False,
                    "summary": summary,
                }
            )
            boundary["validation_done"] = True
            boundary["selection_done"] = True
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
        self._checkpoint(runtime, checkpoint, logical_step=self.logical_step, label="step")

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

    def _shadow_task(self, runtime: ProcessRuntime):
        if self._shadow is None:
            architecture = dict(self.generator.state_module.architecture)
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
                    "seed": int(self.generator.state_module.seed),
                },
                runtime.device,
            )
        task = self._shadow
        task.generator.load_state_dict(self.generator.state_module.generator.state_dict(), strict=True)
        task.discriminator_mrd.load_state_dict(self.mrd.state_module.discriminator.state_dict(), strict=True)
        task.discriminator_band.load_state_dict(self.cqt.state_module.discriminator.state_dict(), strict=True)
        task.eval_mode()
        return task

    def _generator_forward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        *,
        preview: bool,
    ) -> dict[str, torch.Tensor]:
        module = self.generator.state_module if preview else self.generator.module
        error: str | None = None
        parts: dict[str, torch.Tensor] | None = None
        try:
            parts = pack_forward(module(clean, noisy))
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
        if parts is None:
            raise RuntimeFailure("generator forward produced no tensors")
        return parts

    def _discriminator_loss(self, runtime: ProcessRuntime, clean: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        error: str | None = None
        loss: torch.Tensor | None = None
        try:
            real_q, fake_q, _real_qf, _fake_qf = self.cqt.module(clean, audio)
            real_r, fake_r, _real_rf, _fake_rf = self.mrd.module(clean, audio)
            loss_q = sum(least_squares_discriminator(real, fake) for real, fake in zip(real_q, fake_q, strict=True))
            loss_r = sum(least_squares_discriminator(real, fake) for real, fake in zip(real_r, fake_r, strict=True))
            loss = loss_q + loss_r
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("discriminator forward failed: " + "; ".join(item for item in errors if item))
        if loss is None:
            raise RuntimeFailure("discriminator loss is missing")
        return loss

    def _generator_backward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        parts: dict[str, torch.Tensor],
        components: dict[str, float],
    ) -> float:
        error: str | None = None
        total: torch.Tensor | None = None
        terms: dict[str, torch.Tensor] | None = None
        try:
            band = self.cqt.state_module.discriminator
            mrd = self.mrd.state_module.discriminator
            _real_q, fake_q, real_qf, fake_qf = band(clean, parts["audio"])
            _real_r, fake_r, real_rf, fake_rf = mrd(clean, parts["audio"])
            adversarial = sum(least_squares_generator(score) for score in fake_q) + sum(
                least_squares_generator(score) for score in fake_r
            )
            matching = feature_matching(real_qf, fake_qf, scale=2.0, reduction="sum") + feature_matching(
                real_rf, fake_rf, scale=2.0, reduction="sum"
            )
            spectral = compressed_complex_losses(
                parts["clean_mag"],
                parts["mag"],
                parts["clean_phase"],
                parts["phase"],
                parts["clean_complex"],
                parts["complex"],
                parts["recomputed"],
                phase_mode="matrix",
            )
            mel = summed_slaney_log_mel(clean, parts["audio"], self.generator.state_module.sampling_rate)
            total = (
                adversarial * _ADV
                + matching * _FM
                + mel * _MEL
                + spectral["magnitude"] * _MAG
                + spectral["phase"] * _PHASE
                + spectral["complex"] * _COMPLEX
                + spectral["consistency"] * _CONSISTENCY
            )
            terms = {
                "adversarial": adversarial,
                "complex": spectral["complex"],
                "consistency": spectral["consistency"],
                "magnitude": spectral["magnitude"],
                "matching": matching,
                "mel": mel,
                "phase": spectral["phase"],
            }
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator loss failed: " + "; ".join(item for item in errors if item))
        if total is None or terms is None:
            raise RuntimeFailure("generator loss is missing")
        self._backward_if_finite(runtime, total / self.accumulation, "generator loss")
        for name, value in terms.items():
            components[name] += float(value.detach())
        return float(total.detach())

    def _waveforms(self, batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        batch = _batch_on_device(batch, runtime.device)
        clean = batch.targets
        noisy = batch.inputs
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        return clean, noisy

    def _backward_if_finite(self, runtime: ProcessRuntime, loss: torch.Tensor, what: str) -> None:
        finite = bool(torch.isfinite(loss.detach()).all())
        if not runtime.agree(finite, policy="all"):
            raise RuntimeFailure(f"{what} is not finite on every rank")
        runtime.backward(loss)

    def _refuse_amp(self, runtime: ProcessRuntime) -> None:
        if getattr(runtime, "precision", "fp32") == "amp_fp16":
            raise RuntimeFailure(
                "SEMamba++ official GAN refuses amp_fp16. "
                "The current scaler decides overflow per optimizer, so the discriminator "
                "step and the generator step are not one atomic update. FP32 is unchanged."
            )

    def _require_scan(self, runtime: ProcessRuntime) -> None:
        if self.scan_report is None:
            self.scan_report = scan_execution(self.generator.state_module.generator, runtime.device)
        report = self.scan_report
        if report["configured"] == "fused" and report["executed"] == "unavailable":
            raise RuntimeFailure(
                "scan_backend fused was requested but mamba-ssm is not importable. "
                "Refusing to run the reference kernel instead."
            )
        if report["configured"] == "fused" and runtime.device.type == "cuda" and report["executed"] != "fused":
            raise RuntimeFailure(
                f"scan_backend fused on CUDA resolved to {report['executed']}, not the fused kernel"
            )

    def _check_data(self) -> None:
        identity_fn = getattr(self.data, "identity", None)
        if not callable(identity_fn):
            return
        identity = identity_fn()
        if not isinstance(identity, dict):
            raise RuntimeFailure("data identity is not a mapping")
        algorithm = identity.get("algorithm")
        if algorithm is not None and algorithm != "online.semambapp_gsr":
            raise RuntimeFailure(f"SEMamba++ requires online.semambapp_gsr data, got {algorithm}")
        claim = identity.get("claim_label")
        if claim == "official_reproduction":
            raise RuntimeFailure("semambapp.gsr.official is blocked; refusing official_reproduction")
        if claim is not None and claim != self.claim_label:
            raise RuntimeFailure(f"data claim_label {claim} does not match trainer claim_label {self.claim_label}")

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if int(group_size) != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _checkpoint(
        self,
        runtime: ProcessRuntime,
        checkpoint: CheckpointManager,
        *,
        logical_step: int,
        label: str,
    ) -> None:
        if self._in_official_step:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")
        runtime.broadcast_buffers(self.generator.state_module)
        runtime.broadcast_buffers(self.mrd.state_module)
        runtime.broadcast_buffers(self.cqt.state_module)
        checkpoint.save(checkpoint.directory, logical_step=logical_step, label=label)

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


def _grad_norm(*modules: torch.nn.Module) -> float:
    total = 0.0
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            total += float(parameter.grad.detach().square().sum())
    return total**0.5


def _clone_grads(*modules: torch.nn.Module) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for module in modules:
        prefix = module.__class__.__name__
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            cloned[f"{prefix}.{name}"] = parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
    return cloned


def _inputs_ready(clean: torch.Tensor, noisy: torch.Tensor) -> bool:
    if clean.ndim != 2 or clean.shape != noisy.shape:
        return False
    if int(clean.shape[-1]) <= 1024:
        return False
    return bool(torch.isfinite(clean).all()) and bool(torch.isfinite(noisy).all())


def _shared_log(runtime: ProcessRuntime, local: dict[str, float | str]) -> dict[str, float | str]:
    """Store rank 0's scalars on every rank so the checkpoint matches."""

    gathered = runtime.gather_objects(dict(local))
    shared = gathered[0]
    if not isinstance(shared, dict):
        raise RuntimeFailure("rank 0 did not return GAN metrics")
    return shared


def _require_adamw(optimizer: torch.optim.Optimizer, name: str) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise BuildError(f"{name} must be AdamW")
    if not optimizer.param_groups:
        raise BuildError(f"{name} has no parameters")
    for group in optimizer.param_groups:
        if float(group["lr"]) != _OFFICIAL_LR:
            raise BuildError(f"{name} learning rate must be {_OFFICIAL_LR}")
        betas = tuple(float(value) for value in group["betas"])
        if betas != _OFFICIAL_BETAS:
            raise BuildError(f"{name} betas must be {_OFFICIAL_BETAS}")
        if float(group["weight_decay"]) != _OFFICIAL_WEIGHT_DECAY:
            raise BuildError(f"{name} weight_decay must be {_OFFICIAL_WEIGHT_DECAY}")
        if float(group["eps"]) != 1e-8:
            raise BuildError(f"{name} eps must be 1e-8")
        if bool(group.get("amsgrad", False)):
            raise BuildError(f"{name} amsgrad must stay off")


def _require_scheduler(scheduler: torch.optim.lr_scheduler.ExponentialLR, name: str) -> None:
    if not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
        raise BuildError(f"{name} must be ExponentialLR")
    if float(scheduler.gamma) != _OFFICIAL_GAMMA:
        raise BuildError(f"{name} gamma must be {_OFFICIAL_GAMMA}")


def _require_disjoint(optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer) -> None:
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("SEMamba++ optimizers must each own parameters")


def _open_boundary(trainer_epoch: int, *, validate: bool, epoch_done: bool) -> dict[str, Any]:
    return {
        "closed": False,
        "epoch_done": bool(epoch_done),
        "lr_d": None,
        "lr_g": None,
        "scheduler": bool(epoch_done),
        "scheduler_done": False,
        "selected": False,
        "selection_done": False,
        "trainer_epoch": int(trainer_epoch),
        "validate": bool(validate),
        "validation_done": False,
    }


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be a non-negative integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{name} must be a positive number")
    number = float(value)
    if number <= 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise BuildError(f"{name} must be a positive finite number")
    return number
