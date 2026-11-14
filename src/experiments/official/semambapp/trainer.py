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

from src.experiments.stopping import observe_validation, should_stop

from typing import Any

import torch

from src.experiments.checkpoint import CheckpointManager
from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.semambapp.models import (
    SEMambaPPCQTDiscriminator,
    SEMambaPPGenerator,
    SEMambaPPMRD,
    pack_forward,
    scan_execution,
)
from src.experiments.execution.journal import StepJournal
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.selectors import scalar_metrics
from src.experiments.trainers import _close_epoch_view
from src.experiments.calc.metrics import _aggregate_validation
from src.experiments.calc.errors import TrainingError
from src.experiments.official.semambapp.reference import validation_model
from src.experiments.training.objectives import call_objective

_IN_EPOCH = "in_epoch"
_PENDING = "pending_boundary"
_LOSS_NAMES = ("adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")
_OFFICIAL_LR = 0.0002
_OFFICIAL_BETAS = (0.8, 0.99)
_OFFICIAL_WEIGHT_DECAY = 0.01
_OFFICIAL_GAMMA = 0.99
_SUBSTITUTE_CLAIM = "protocol_conformant_substitute_materials"


class SEMambaPPTrainer:
    """Discriminator-then-generator update for the official SEMamba++ loss."""

    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

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
        generator_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float | None = None,
        validation_interval: int = 0,
        best_checkpoint_start_epoch: int = 40,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
        claim_label: str = _SUBSTITUTE_CLAIM,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        require_prepared(generator, SEMambaPPGenerator, "generator must be a prepared SEMambaPPGenerator")
        require_prepared(mrd, SEMambaPPMRD, "mrd must be a prepared SEMambaPPMRD")
        require_prepared(cqt, SEMambaPPCQTDiscriminator, "cqt must be a prepared SEMambaPPCQTDiscriminator")
        if generator.unwrapped().architecture != mrd.unwrapped().architecture:
            raise BuildError("MRD initialization architecture does not match the generator")
        if generator.unwrapped().architecture != cqt.unwrapped().architecture:
            raise BuildError("CQT initialization architecture does not match the generator")
        if generator.unwrapped().seed != mrd.unwrapped().seed or generator.unwrapped().seed != cqt.unwrapped().seed:
            raise BuildError("generator, MRD, and CQT seeds differ")
        _require_adamw(optim_g, "optim_g")
        _require_adamw(optim_d, "optim_d")
        _require_scheduler(scheduler_g, "scheduler_g")
        _require_scheduler(scheduler_d, "scheduler_d")
        if not isinstance(generator_objective, torch.nn.Module) or not isinstance(discriminator_objective, torch.nn.Module):
            raise BuildError("SEMamba++ objectives must be loss modules")
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
        self.generator_objective = generator_objective
        self.discriminator_objective = discriminator_objective
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.claim_label = claim_label
        self.audio_sample_rate = int(generator.unwrapped().sampling_rate)
        self.audio_teachers = ("dry_clean",)
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.grad_clip = None if grad_clip is None else _positive_float("grad_clip", grad_clip)
        self.validation_interval = _non_negative_int("validation_interval", validation_interval)
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
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators,
            model_selection,
            data=self.data,
            module=self.generator,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection
        self._restored = False
        self._in_official_step = False
        self._shadow = None

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or an interrupt."""

        self.lifecycle.start()
        runtime = self.runtime
        checkpoint = self.lifecycle.checkpoint
        component_id = self._prepare_run(runtime, checkpoint)
        if should_stop(self):
            return self._result(checkpoint, "completed")
        if self.phase == _PENDING:
            self._finish_boundary(runtime, checkpoint, component_id)
            if should_stop(self):
                return self._result(checkpoint, "completed")
            if self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                if self.logical_step > 0 and not self._in_official_step:
                    self._offer(runtime, checkpoint, "stop")
                return self._result(checkpoint, "interrupted")
            if self._epoch_cap_reached():
                return self._result(checkpoint, "completed")
            epoch, index, view, batches = self._open_window(component_id)
            try:
                local = self.official_step(batches, runtime)
                self.logical_step += 1
                index += self.accumulation
                self.optimizer_updates[self.optim_g_id] += 1
                self.optimizer_updates[self.optim_d_id] += 1
                self.data.epoch = epoch
                self.data.update_index = index
                epoch_done = index == view.updates
                will_validate = self.validation_interval > 0 and self.logical_step % self.validation_interval == 0
                if self._record_step(
                    runtime,
                    checkpoint,
                    local,
                    epoch=epoch,
                    epoch_done=epoch_done,
                    will_validate=will_validate,
                ):
                    continue
                self.phase = _PENDING
                self.boundary = _open_boundary(epoch, validate=will_validate, epoch_done=epoch_done)
                self._on_boundary_phase(self.phase)
                self._offer(runtime, checkpoint, "boundary_pending")
                if self._stop_requested(runtime):
                    return self._result(checkpoint, "interrupted")
                self._finish_boundary(runtime, checkpoint, component_id)
                if should_stop(self):
                    return self._result(checkpoint, "completed")
                if self._epoch_cap_reached():
                    return self._result(checkpoint, "completed")
            finally:
                _close_epoch_view(view)
        return self._result(checkpoint, "completed")

    def _prepare_run(self, runtime: ProcessRuntime, checkpoint: CheckpointManager) -> str:
        """Rewind logs. Does not run a discriminator or generator update."""

        self.research.rewind(
            checkpoint.directory,
            restored=self._restored,
            phase=self.phase,
            progress={"logical_step": int(self.logical_step)},
        )
        self._refuse_amp(runtime)
        self._require_scan(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(checkpoint.directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.journal.begin(
            restored=self._restored,
            logical_step=self.logical_step,
            expect_rows=len(self.step_log) if self._restored else None,
        )
        if not self._restored:
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
        bind_trainer_data(self)
        self._require_group_size(component_id)
        runtime.barrier()
        return str(component_id)

    def _open_window(self, component_id: str) -> tuple[int, int, Any, list[Batch]]:
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
        except Exception:
            _close_epoch_view(view)
            raise
        return epoch, index, view, batches

    def _record_step(
        self,
        runtime: ProcessRuntime,
        checkpoint: CheckpointManager,
        local: dict[str, Any],
        *,
        epoch: int,
        epoch_done: bool,
        will_validate: bool,
    ) -> bool:
        """Write the step log. Return True when the epoch continues."""

        shared = _shared_log(runtime, local)
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
        self.journal.write_metrics(entry)
        self.journal.write_sample(
            {
                "logical_step": self.logical_step,
                "loss": local["generator"],
                "realization_id": local["realization_id"],
                "sample_id": local["sample_id"],
            }
        )
        if will_validate or epoch_done:
            return False
        step_metrics = self.research.score(
            runtime,
            checkpoint,
            kind="step",
            epoch=epoch,
            logical_step=self.logical_step,
            stopping=getattr(self, "training_control", None),
        )
        if step_metrics:
            self._offer(runtime, checkpoint, "step", metrics=step_metrics)
        else:
            self._offer(runtime, checkpoint, "step")
        return True

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
        self.generator.train()
        self.mrd.train()
        self.cqt.train()
        self._in_official_step = True
        graph_before = self.generator_graph_forwards
        preview_before = self.generator_preview_forwards
        reforward_before = self.generator_reforward_forwards
        try:
            self._on_update_phase("zero_grad")
            self.optim_d.zero_grad(set_to_none=True)
            reuse_graph = len(batches) == 1
            cached: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]] = []
            discriminator_total = None
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
                    with runtime.sync_gradients([self.generator], enabled=True):
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
                with runtime.sync_gradients([self.mrd, self.cqt], enabled=last):
                    loss = self._discriminator_loss(runtime, clean, audio.detach())
                    self._backward_if_finite(runtime, loss / self.accumulation, "discriminator loss")
                term = loss.detach()
                discriminator_total = term if discriminator_total is None else discriminator_total + term
                self.mrd_forwards += 1
                self.cqt_forwards += 1
            if self.grad_clip is not None:
                runtime.clip(self.mrd, self.grad_clip)
                runtime.clip(self.cqt, self.grad_clip)
            disc_norm = _grad_norm(self.mrd.unwrapped().discriminator, self.cqt.unwrapped().discriminator)
            if self.capture_gradients:
                self.last_discriminator_grads = _clone_grads(
                    self.mrd.unwrapped().discriminator, self.cqt.unwrapped().discriminator
                )
            self._on_update_phase("before_discriminator_step")
            discriminator_result = runtime.step(self.optim_d)
            self._on_update_phase("after_discriminator_step")
            frozen = [
                parameter
                for module in (self.mrd.unwrapped(), self.cqt.unwrapped())
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.optim_g.zero_grad(set_to_none=True)
            component_totals: dict[str, Any] = {}
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
                        with runtime.sync_gradients([self.generator], enabled=last):
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
                runtime.clip(self.generator, self.grad_clip)
            log_terms = torch.stack(
                [value.detach() if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=runtime.device) for value in component_totals.values()]
            )
            if discriminator_total is None or not isinstance(generator_total, torch.Tensor):
                raise RuntimeFailure("generator loss is missing")
            log_pair = torch.stack([discriminator_total.detach(), generator_total.detach()])
            if runtime.device.type == "cuda":
                host_terms = torch.empty(log_terms.shape, dtype=torch.float32, pin_memory=True)
                host_pair = torch.empty(log_pair.shape, dtype=torch.float32, pin_memory=True)
                host_terms.copy_(log_terms.detach().float(), non_blocking=True)
                host_pair.copy_(log_pair.detach().float(), non_blocking=True)
            else:
                host_terms = log_terms.detach().float().cpu()
                host_pair = log_pair.detach().float().cpu()
            gen_norm = _grad_norm(self.generator.unwrapped().generator)
            if self.capture_gradients:
                self.last_generator_grads = _clone_grads(self.generator.unwrapped().generator)
            self._on_update_phase("before_generator_step")
            generator_result = runtime.step(self.optim_g)
            runtime.end_step()
            runtime.broadcast_buffers(self.generator)
            runtime.broadcast_buffers(self.mrd)
            runtime.broadcast_buffers(self.cqt)
            if not discriminator_result.applied or not generator_result.applied:
                raise RuntimeFailure("SEMamba++ requires both optimizers to apply in one logical step")
            count = len(batches)
            report = self.scan_report or {"configured": "", "executed": ""}
            self._on_update_phase("end_step")
            component_totals = {
                name: float(number) for name, number in zip(component_totals, host_terms.tolist(), strict=True)
            }
            discriminator_total = float(host_pair[0])
            generator_total = float(host_pair[1])
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
        metrics: dict[str, float] = {}
        if boundary["validate"] and not boundary["validation_done"]:
            summary = self._validate(runtime, trainer_epoch, component_id)
            metrics = scalar_metrics(summary)
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
        if boundary["validate"]:
            observe_validation(
                getattr(self, "training_control", None),
                metrics,
                step=self.logical_step,
                scoped=self.research.attached,
            )
        extra = self.research.score(
            runtime,
            checkpoint,
            kind="boundary",
            epoch=trainer_epoch,
            logical_step=self.logical_step,
            official=metrics if boundary["validate"] else {},
            stopping=getattr(self, "training_control", None),
        )
        self._offer(runtime, checkpoint, "boundary", metrics={**metrics, **extra} if extra else metrics)

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

    def _shadow_task(self, runtime: ProcessRuntime):
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

    def _generator_forward(
        self,
        runtime: ProcessRuntime,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        *,
        preview: bool,
    ) -> dict[str, torch.Tensor]:
        module = self.generator.unwrapped() if preview else self.generator.execution()
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
            real_q, fake_q, _real_qf, _fake_qf = self.cqt(clean, audio)
            real_r, fake_r, _real_rf, _fake_rf = self.mrd(clean, audio)
            loss = call_objective(
                self.discriminator_objective,
                {
                    "fake_cqt": fake_q,
                    "fake_mrd": fake_r,
                    "real_cqt": real_q,
                    "real_mrd": real_r,
                },
            ).total
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
            band = self.cqt.unwrapped().discriminator
            mrd = self.mrd.unwrapped().discriminator
            _real_q, fake_q, real_qf, fake_qf = band(clean, parts["audio"])
            _real_r, fake_r, real_rf, fake_rf = mrd(clean, parts["audio"])
            result = call_objective(
                self.generator_objective,
                {
                    "clean_complex": parts["clean_complex"],
                    "clean_mag": parts["clean_mag"],
                    "clean_phase": parts["clean_phase"],
                    "estimate": parts["audio"],
                    "fake_cqt": fake_q,
                    "fake_cqt_features": fake_qf,
                    "fake_mrd": fake_r,
                    "fake_mrd_features": fake_rf,
                    "generated_complex": parts["complex"],
                    "generated_mag": parts["mag"],
                    "generated_phase": parts["phase"],
                    "real_cqt_features": real_qf,
                    "real_mrd_features": real_rf,
                    "recomputed_complex": parts["recomputed"],
                    "reference": clean,
                    "sample_rate": int(self.generator.unwrapped().sampling_rate),
                },
            )
            total = result.total
            terms = result.terms
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeFailure("generator loss failed: " + "; ".join(item for item in errors if item))
        if total is None or terms is None:
            raise RuntimeFailure("generator loss is missing")
        self._backward_if_finite(runtime, total / self.accumulation, "generator loss")
        for name, value in terms.items():
            current = components.get(name, 0.0)
            components[name] = current + value.detach()
        return total.detach()

    def _waveforms(self, batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        batch = runtime.move(batch)
        clean = batch.targets
        noisy = batch.inputs
        if clean.ndim != 2 or clean.shape != noisy.shape:
            raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
        return clean, noisy

    def _backward_if_finite(self, runtime: ProcessRuntime, loss: torch.Tensor, what: str) -> None:
        if runtime.device.type != "cuda":
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
            self.scan_report = scan_execution(self.generator.unwrapped().generator, runtime.device)
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
        runtime: ProcessRuntime,
        checkpoint: CheckpointManager,
        event: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Offer this state to the selector. A GAN optimizer pair is not a save point.

        A boundary and a stop broadcast the buffers whether or not the
        selector saves, as they did before selectors. Inside an epoch the
        broadcast runs only when a file is written.
        """

        if self._in_official_step:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")
        if event == "step":
            self.lifecycle.save_if_needed(event, logical_step=self.logical_step, metrics=metrics, prepare=self._sync_ranks)
            return
        self._sync_ranks(runtime)
        self.lifecycle.save_if_needed(event, logical_step=self.logical_step, metrics=metrics)

    def _sync_ranks(self, runtime: ProcessRuntime) -> None:
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.mrd)
        runtime.broadcast_buffers(self.cqt)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        from src.experiments.signals import ranks_should_stop

        capped = self.stop_after_logical_steps is not None and self.logical_step >= self.stop_after_logical_steps
        return ranks_should_stop(runtime, extra=capped)

    def _epoch_cap_reached(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

    def _result(self, checkpoint: CheckpointManager, reason: str) -> RunResult:
        if reason == "completed":
            epoch = getattr(self.data, "epoch", None)
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                epoch = None
            extra = self.research.score(
                self.runtime,
                checkpoint,
                kind="end",
                epoch=epoch,
                logical_step=self.logical_step,
                stopping=getattr(self, "training_control", None),
            )
            if extra:
                self.lifecycle.save_if_needed("final", logical_step=self.logical_step, metrics=extra, prepare=self._sync_ranks)
            else:
                self.lifecycle.save_if_needed("final", logical_step=self.logical_step, prepare=self._sync_ranks)
        self.stop_reason = reason
        if reason == "completed" and should_stop(self):
            reason = "early_stopping"
        return self.lifecycle.finish(
            RunResult(
                stop_reason=reason,
                logical_step=self.logical_step,
                optimizer_updates=dict(self.optimizer_updates),
                output_dir=checkpoint.directory,
            )
        )


def _grad_norm(*modules: torch.nn.Module) -> float:
    pieces = [
        parameter.grad.detach().square().sum()
        for module in modules
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not pieces:
        return 0.0
    values = torch.stack(pieces).detach().cpu().tolist()
    return sum(float(value) for value in values) ** 0.5


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
    if clean.is_cuda:
        return True
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
