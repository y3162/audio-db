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

from src.experiments.stopping import observe_validation, should_stop

import threading
from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mp_senet.artifacts import write_best
from src.experiments.official.mp_senet.metrics import aggregate_reports
from src.experiments.official.mp_senet.selection import consider_best_pesq
from src.experiments.training.control import finish_training_run, open_flat_window, stop_requested
from src.experiments.official.mp_senet.models import (
    MPSENetGANGenerator,
    MPSENetMetricDiscriminator,
    pack_forward,
    require_official_architecture,
)
from src.experiments.execution.journal import StepJournal
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.selectors import scalar_metrics
from src.experiments.trainers import (
    _close_epoch_view,
    _rank,
)
from src.experiments.calc.errors import TrainingError
from src.experiments.training.objectives import call_objective
from src.experiments.training.official_objectives import require_mp_senet
from src.experiments.official.mp_senet.reference import _require_discriminator_resolution, validation_model

_IN_EPOCH = "in_epoch"
_PENDING = "pending_boundary"
OFFICIAL_FAITHFUL = "official_faithful"
LEAKAGE_SAFE = "leakage_safe"
_CLAIMS = frozenset({OFFICIAL_FAITHFUL, LEAKAGE_SAFE, "research"})
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

    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

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
        signal_objective: torch.nn.Module,
        metric_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        max_logical_steps: int,
        experiment_claim: str,
        accumulation: int = 1,
        validation_interval: int = OFFICIAL_VALIDATION_INTERVAL,
        best_checkpoint_start_epoch: int = BEST_CHECKPOINT_START_EPOCH,
        max_epochs: int | None = None,
        stop_after_logical_steps: int | None = None,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        require_prepared(generator, MPSENetGANGenerator, "generator must be a prepared MPSENetGANGenerator")
        require_prepared(
            discriminator,
            MPSENetMetricDiscriminator,
            "discriminator must be a prepared MPSENetMetricDiscriminator",
        )
        if generator.unwrapped().architecture != discriminator.unwrapped().architecture:
            raise BuildError("discriminator initialization architecture does not match the generator")
        if generator.unwrapped().seed != discriminator.unwrapped().seed:
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
            raise BuildError("experiment_claim must be official_faithful, leakage_safe, or research")
        require_official_architecture(generator.unwrapped().architecture, experiment_claim)
        require_mp_senet(signal_objective, metric_objective, discriminator_objective, experiment_claim)
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
        self.signal_objective = signal_objective
        self.metric_objective = metric_objective
        self.discriminator_objective = discriminator_objective
        self.optim_g_id = optim_g_id
        self.optim_d_id = optim_d_id
        self.experiment_claim = experiment_claim
        self.audio_sample_rate = 16000
        self.audio_teachers = ("dry_clean",)
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
        self._average_calls = 0
        self._generator_synced_during_backward = False

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or an interrupt."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        if self.phase == _PENDING:
            self._finish_boundary(runtime, component_id)
            if should_stop(self):
                return self._result("completed")
            if self._epoch_cap_reached():
                return self._result("completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                self._drain_mp_io()
                if self.logical_step > 0 and not self._in_official_step:
                    self._offer("stop")
                return self._result("interrupted")
            if self._epoch_cap_reached():
                self._drain_mp_io()
                return self._result("completed")
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
                    local,
                    epoch=epoch,
                    epoch_done=epoch_done,
                    will_validate=will_validate,
                ):
                    continue
                self.phase = _PENDING
                self.boundary = _open_boundary(epoch, validate=will_validate, epoch_done=epoch_done)
                self._on_boundary_phase(self.phase)
                self._offer("boundary_pending")
                if self._stop_requested(runtime):
                    return self._result("interrupted")
                self._finish_boundary(runtime, component_id)
                if should_stop(self):
                    return self._result("completed")
                if self._epoch_cap_reached():
                    return self._result("completed")
            finally:
                _close_epoch_view(view)
        self._drain_mp_io()
        return self._result("completed")

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        """Rewind logs. Does not run a discriminator or generator update."""

        self.research.rewind(
            self.lifecycle.output_directory,
            restored=self._restored,
            phase=self.phase,
            progress={"logical_step": int(self.logical_step)},
        )
        self._refuse_amp(runtime)
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
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
            self.best_pesq = None
            self.best_epoch = None
            self.best_logical_step = None
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        self._require_group_size(component_id)
        runtime.barrier()
        return str(component_id)

    def _open_window(self, component_id: str) -> tuple[int, int, Any, list[Batch]]:
        """Take the next flat micro-batch window. The epoch length must divide evenly."""

        return open_flat_window(self, component_id)

    def _record_step(
        self,
        runtime: ProcessRuntime,
        local: dict[str, Any],
        *,
        epoch: int,
        epoch_done: bool,
        will_validate: bool,
    ) -> bool:
        """Write or defer the step log. Return True when the epoch continues."""

        shared = _shared_log(runtime, local)
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
        offer_step = not will_validate and not epoch_done
        step_metrics: dict[str, float] = {}
        if offer_step:
            step_metrics = self.research.score(
                runtime,
                output_directory=self.lifecycle.output_directory,
                config_identity=self.lifecycle.config_identity,
                kind="step",
                epoch=epoch,
                logical_step=self.logical_step,
                stopping=getattr(self, "training_control", None),
            )
        if runtime.device.type == "cuda" and runtime.world_size == 1:
            self._mp_io = (entry, local, offer_step, step_metrics)
            if offer_step:
                return True
            self._drain_mp_io()
            return False
        self.journal.write_metrics(entry)
        self.journal.write_sample({**local, "logical_step": self.logical_step, "loss": local["generator"]})
        if offer_step:
            if step_metrics:
                self._offer("step", metrics=step_metrics)
            else:
                self._offer("step")
            return True
        return False

    def _drain_mp_io(self) -> None:
        """Write the previous step while the next signal backward is queued."""

        item = self._mp_io
        if item is None:
            return
        self._mp_io = None
        entry, local, offer_step, step_metrics = item
        self.journal.write_metrics(entry)
        self.journal.write_sample(
            {**local, "logical_step": entry["logical_step"], "loss": local["generator"]}
        )
        if offer_step:
            if step_metrics:
                self._offer("step", metrics=step_metrics)
            else:
                self._offer("step")

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
        if not _inputs_ready(clean, noisy, self.generator.unwrapped().n_fft):
            raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
        self._on_update_phase("before_generator_forward")
        try:
            parts = pack_forward(self.generator(noisy, clean))
            _require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"generator forward failed: {type(exc).__name__}: {exc}") from exc
        pending = _schedule_metric_targets(
            clean,
            parts["audio"],
            int(self.generator.unwrapped().architecture["sampling_rate"]),
        )
        ones = torch.ones(clean.shape[0], device=clean.device)
        signal = call_objective(self.signal_objective, _signal_context(parts, clean))
        self._backward_if_finite(runtime, signal.total / self.accumulation, "generator signal loss", retain_graph=True)
        self._drain_mp_io()
        try:
            targets = pending.result()
            real = self.discriminator(parts["clean_mag"], parts["clean_mag"])
            fake = self.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
            loss_d = call_objective(
                self.discriminator_objective,
                {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
            ).total / self.accumulation
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"discriminator forward failed: {type(exc).__name__}: {exc}") from exc
        self._backward_if_finite(runtime, loss_d, "discriminator loss")
        discriminator_result = runtime.step(self.optim_d)
        self._on_update_phase("after_discriminator_step", [(clean, parts)])
        raw = self.discriminator.unwrapped().discriminator
        frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            try:
                metric = raw(parts["clean_mag"], parts["mag_hat"])
                metric_out = call_objective(self.metric_objective, {"metric": metric, "ones": ones})
            except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
                raise RuntimeFailure(f"generator loss failed: {type(exc).__name__}: {exc}") from exc
            self._backward_if_finite(runtime, metric_out.total / self.accumulation, "generator metric loss")
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        self._average_calls += 1
        runtime.average_gradients(self.generator)
        total = signal.total + metric_out.total
        logged = torch.stack(
            [
                signal.terms["complex"].detach(),
                signal.terms["consistency"].detach(),
                loss_d.detach() * self.accumulation,
                total.detach(),
                signal.terms["magnitude"].detach(),
                metric_out.terms["metric"].detach(),
                signal.terms["phase"].detach(),
                signal.terms["time"].detach(),
            ]
        )
        host = torch.empty(logged.shape, dtype=torch.float32, pin_memory=True)
        host.copy_(logged.detach().float(), non_blocking=True)
        generator_result = runtime.step(self.optim_g)
        runtime.end_step()
        runtime.broadcast_buffers(self.generator)
        runtime.broadcast_buffers(self.discriminator)
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
        self.generator.train()
        self.discriminator.train()
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
            with runtime.sync_gradients([self.generator], enabled=False):
                for index, batch in enumerate(batches):
                    last = index == len(batches) - 1
                    clean, noisy = self._waveforms(batch, runtime)
                    sample_ids.append(batch.sample_id)
                    ready = _inputs_ready(clean, noisy, self.generator.unwrapped().n_fft)
                    if not runtime.agree(ready, policy="all"):
                        raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
                    self._on_update_phase("before_generator_forward")
                    error: str | None = None
                    parts: dict[str, torch.Tensor] | None = None
                    try:
                        parts = pack_forward(self.generator(noisy, clean))
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
                    with runtime.sync_gradients([self.discriminator], enabled=last):
                        disc_error: str | None = None
                        loss: torch.Tensor | None = None
                        try:
                            real = self.discriminator(parts["clean_mag"], parts["clean_mag"])
                            fake = self.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
                            targets = _metric_targets(clean, parts["audio"], int(self.generator.unwrapped().architecture["sampling_rate"]))
                            loss = call_objective(
                                self.discriminator_objective,
                                {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
                            ).total / self.accumulation
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
                raw = self.discriminator.unwrapped().discriminator
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
                            signal = call_objective(self.signal_objective, _signal_context(parts, clean))
                            metric_out = call_objective(self.metric_objective, {"metric": metric, "ones": ones})
                            total_tensor = signal.total + metric_out.total
                            losses = {
                                "total": total_tensor,
                                "magnitude": signal.terms["magnitude"],
                                "phase": signal.terms["phase"],
                                "complex": signal.terms["complex"],
                                "consistency": signal.terms["consistency"],
                                "metric": metric_out.terms["metric"],
                                "time": signal.terms["time"],
                            }
                            total = total_tensor / self.accumulation
                        except (TrainingError, RuntimeError, ValueError) as exc:
                            gen_error = f"{type(exc).__name__}: {exc}"
                        gen_errors = runtime.gather_objects(gen_error)
                        if any(gen_errors):
                            raise RuntimeFailure(
                                "generator loss failed: " + "; ".join(item for item in gen_errors if item)
                            )
                        if total is None:
                            raise RuntimeFailure("generator loss is missing")
                        synced = self.generator.gradient_sync_enabled()
                        if synced is not None:
                            self._generator_synced_during_backward = synced
                        self._backward_if_finite(runtime, total, "generator loss")
                        for name in _LOSS_NAMES:
                            component_totals[name] += float(losses[name].detach())
                finally:
                    for parameter in frozen:
                        parameter.requires_grad_(True)
                self._average_calls += 1
                runtime.average_gradients(self.generator)
            generator_result = runtime.step(self.optim_g)
            runtime.end_step()
            runtime.broadcast_buffers(self.generator)
            runtime.broadcast_buffers(self.discriminator)
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
        component_id: str,
    ) -> None:
        """Validate, select best PESQ, step both schedulers, then run research evaluation."""
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
            self._write_best(runtime, self.lifecycle.output_directory, decision)
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
        if boundary["validate"]:
            observe_validation(
                getattr(self, "training_control", None),
                metrics,
                step=self.logical_step,
                scoped=self.research.attached,
            )
        extra = self.research.score(
            runtime,
            output_directory=self.lifecycle.output_directory,
            config_identity=self.lifecycle.config_identity,
            kind="boundary",
            epoch=trainer_epoch,
            logical_step=self.logical_step,
            official=metrics if boundary["validate"] else {},
            stopping=getattr(self, "training_control", None),
        )
        self._offer("boundary", metrics={**metrics, **extra} if extra else metrics)

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, Any]:
        """Run the official validation report on rank 0 and share that summary."""
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
        return aggregate_reports(reports)

    def _shadow_task(self, runtime: ProcessRuntime):
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

    def _consider_best(self, epoch: int, summary: dict[str, Any]) -> dict[str, Any]:
        """Apply the official PESQ rule and remember a strict improvement."""

        decision = consider_best_pesq(
            epoch,
            summary,
            start_epoch=self.best_checkpoint_start_epoch,
            best_pesq=self.best_pesq,
            logical_step=self.logical_step,
        )
        if decision["selected"]:
            self.best_pesq = float(decision["pesq"])
            self.best_epoch = int(decision["epoch"])
            self.best_logical_step = int(decision["logical_step"])
        return decision

    def _write_best(self, runtime: ProcessRuntime, directory: Path, decision: dict[str, Any]) -> None:
        error: str | None = None
        if _rank(runtime) == 0 and decision["selected"]:
            try:
                write_best(
                    directory,
                    self.generator.unwrapped().export_for_inference(),
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
        batch = runtime.move(batch)
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
        event: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Offer this state to the selector. A GAN optimizer pair is not a save point."""

        if self._in_official_step:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")
        self.lifecycle.save_if_needed(event, logical_step=self.logical_step, metrics=metrics)

    def _stop_requested(self, runtime: object | None = None) -> bool:
        """Agree a logical-step cap with a rank interrupt."""

        return stop_requested(self, runtime)

    def _epoch_cap_reached(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

    def _result(self, reason: str) -> RunResult:
        """Score the research end and finish. This path does not broadcast buffers."""

        return finish_training_run(self, reason)


def _signal_context(parts: dict[str, torch.Tensor], clean: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the signal-term context from one generator forward."""
    return {
        "clean_audio": clean,
        "clean_complex": parts["clean_complex"],
        "clean_mag": parts["clean_mag"],
        "clean_phase": parts["clean_phase"],
        "generated_audio": parts["audio"],
        "generated_complex": parts["complex"],
        "generated_mag": parts["mag"],
        "generated_phase": parts["phase"],
        "recomputed_complex": parts["recomputed_complex"],
    }


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
    """Accept a matching pair long enough for the STFT. CPU also requires finite values."""
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
    """Refuse an optimizer pair that shares a parameter."""
    generator_ids = {id(parameter) for group in optim_g.param_groups for parameter in group["params"]}
    discriminator_ids = {id(parameter) for group in optim_d.param_groups for parameter in group["params"]}
    if generator_ids & discriminator_ids:
        raise BuildError("generator and discriminator optimizers share a parameter")
    if not generator_ids or not discriminator_ids:
        raise BuildError("MP-SENet GAN optimizers must each own parameters")


def _open_boundary(trainer_epoch: int, *, validate: bool, epoch_done: bool) -> dict[str, Any]:
    """Record whether this safe point still owes validation or a scheduler step."""
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


def _positive_int(name: str, value: object) -> int:
    """Require a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    """Require an integer that is zero or greater."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BuildError(f"{name} must be an integer >= 0")
    return value
