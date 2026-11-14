"""TF-Locoformer DNS training loop for the experiment framework.

``TFGridNetTrainer`` steps ``ReduceLROnPlateau`` once per finished chunk
epoch. This trainer does not reuse that schedule. It calls the existing
``WarmupReduceLROnPlateau``: ``step_batch`` after an optimizer update that
actually ran, and ``step_validation`` once after the epoch's validation,
including epochs that are still inside warmup.

The first optimizer update uses the learning rate from the optimizer
constructor. ``step_batch`` then sets the rate for the next update. An AMP
update that returns ``applied=False`` does not advance ``step_num``.

``validation_pending`` has the same meaning as in the TF-GridNet trainer.
The pending file is written after ``step_batch`` and before validation.
Resume runs validation, the plateau update, and checkpoint selection once.

Training logs ``si_sdr`` as the negative SI-SDR loss. Validation stores
``si_snr`` as SI-SDR in dB, the negation of that loss. Checkpoint selection
minimizes ``valid.loss`` and maximizes ``valid.si_snr``. Those are the
research rules of the removed ``src/training`` trainer. The DNS recipe's
``best_model_criterion`` names ``si_snr`` while the loss writes
``si_snr_loss``. This trainer follows the research ranker, and the declared
validation profile is the research set
``tf_locoformer.dns.accept_recipe.research_valid_fixed@1``, which is not
``cv_synthetic_small``.
"""

from __future__ import annotations

from src.experiments.stopping import observe_validation, should_stop

import math
from pathlib import Path
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.data.pair import bind_trainer_data
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.training.objectives import call_objective, coerce_spectral_objective, term_floats
from src.experiments.execution.journal import StepJournal
from src.experiments.results import RunResult
from src.experiments.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.official.tf_gridnet.selection import early_stop, rank_validation_loss
from src.experiments.official.tf_locoformer.metrics import per_sample_si_snr
from src.experiments.official.tf_locoformer.selection import rank_validation_si_snr
from src.experiments.training.control import finish_training_run, open_grouped_window, stop_requested
from src.experiments.official.tf_locoformer.artifacts import write_selection
from src.experiments.trainers import (
    _close_epoch_view,
    _rank,
)
from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.calc.optim import WarmupReduceLROnPlateau
from src.experiments.calc.spectral import si_sdr_loss

_PENDING = "pending_validation"
_IN_EPOCH = "in_epoch"
RESEARCH_VALIDATION_PROFILE = "tf_locoformer.dns.accept_recipe.research_valid_fixed@1"


class TFLocoformerTrainer:
    """Single-optimizer DNS loop with warmup per step and plateau per epoch."""

    supports_training_control = True
    progress_keys = ("logical_step", "epoch")

    def __init__(
        self,
        model: PreparedModel,
        objective: torch.nn.Module,
        data: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupReduceLROnPlateau,
        *,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float = 5.0,
        keep_nbest: int = 5,
        max_epochs: int | None = None,
        early_stopping_patience: int | None = None,
        stop_after_logical_steps: int | None = None,
        validation_profile: str = RESEARCH_VALIDATION_PROFILE,
        evaluators: Any | None = None,
        model_selection: Any | None = None,
    ) -> None:
        require_prepared(model)
        objective = coerce_spectral_objective(objective, label="DNS")
        if not hasattr(data, "train_epoch") or not hasattr(data, "validation_epoch"):
            raise BuildError("data must provide train_epoch and validation_epoch")
        if not isinstance(scheduler, WarmupReduceLROnPlateau):
            raise BuildError("scheduler must be WarmupReduceLROnPlateau")
        if scheduler.mode != "min":
            raise BuildError("DNS scheduler mode must be min")
        if not isinstance(validation_profile, str) or not validation_profile:
            raise BuildError("validation_profile must be a non-empty string")
        optimizer_id = getattr(optimizer, "_experiment_component_id", None)
        if not isinstance(optimizer_id, str):
            raise BuildError("optimizer has no component id")
        self.model = model
        self.objective = objective
        self.data = data
        self.audio_sample_rate = 16000
        self.audio_teachers = ("dry_clean",)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_id = optimizer_id
        self.validation_profile = validation_profile
        self.max_logical_steps = _positive_int("max_logical_steps", max_logical_steps)
        self.accumulation = _positive_int("accumulation", accumulation)
        self.grad_clip = _positive_float("grad_clip", grad_clip)
        self.keep_nbest = _positive_int("keep_nbest", keep_nbest)
        self.max_epochs = None if max_epochs is None else _positive_int("max_epochs", max_epochs)
        if early_stopping_patience is not None and (
            isinstance(early_stopping_patience, bool)
            or not isinstance(early_stopping_patience, int)
            or early_stopping_patience < 0
        ):
            raise BuildError("early_stopping_patience must be an integer >= 0")
        self.early_stopping_patience = early_stopping_patience
        if stop_after_logical_steps is not None:
            stop_after_logical_steps = _positive_int("stop_after_logical_steps", stop_after_logical_steps)
        self.stop_after_logical_steps = stop_after_logical_steps
        self.logical_step = 0
        self.optimizer_updates = {optimizer_id: 0}
        self.losses: list[float] = []
        self.learning_rates: list[float] = []
        self.scheduled_rates: list[float] = []
        self.history: list[dict[str, Any]] = []
        self.completed_boundaries: list[dict[str, Any]] = []
        self.boundary: dict[str, Any] | None = None
        self.phase = _IN_EPOCH
        self.finished_epoch: int | None = None
        self.early_stopped = False
        self.stop_reason = "initialized"
        from src.experiments.evaluation.boundary import ResearchSession

        self.research = ResearchSession(
            evaluators,
            model_selection,
            data=self.data,
            module=self.model,
            predict=getattr(self, "research_predict", None),
        )
        self.evaluators = self.research.evaluators
        self.model_selection = self.research.selection
        self._restored = False
        self._step_io: dict[str, Any] | None = None

    def fit(self) -> RunResult:
        """Train until the step budget, an epoch cap, or early stopping."""

        self.lifecycle.start()
        runtime = self.runtime
        component_id = self._prepare_run(runtime)
        if should_stop(self):
            return self._result("completed")
        if self._restored and (
            len(self.learning_rates) != self.logical_step or len(self.scheduled_rates) != self.logical_step
        ):
            raise RuntimeFailure(
                f"component {component_id}: learning-rate history does not match logical step"
            )
        if self.phase == _PENDING:
            self._finish_boundary(runtime, component_id)
            if should_stop(self):
                return self._result("completed")
            if self.early_stopped or self._epoch_cap_reached():
                return self._result("completed")
        while self.logical_step < self.max_logical_steps:
            if self._stop_requested(runtime):
                self._drain_step_io()
                if self.logical_step > 0:
                    self.lifecycle.save_if_needed("stop", logical_step=self.logical_step)
                return self._result("interrupted")
            if self._epoch_cap_reached():
                self._drain_step_io()
                return self._result("completed")
            epoch, index, view, planned = self._open_window(component_id)
            try:
                self._drain_step_io()
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
                            micro, monitor, count, terms = self._forward_loss(batches, component_id, runtime)
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
                si_sdr = float(host_monitor)
                scheduled_lr = used_lr
                if result.applied:
                    self.optimizer_updates[self.optimizer_id] += 1
                    self.scheduler.step_batch()
                    scheduled_lr = float(self.optimizer.param_groups[0]["lr"])
                self.logical_step += 1
                index += self.accumulation
                self.losses.append(loss_value)
                self.learning_rates.append(used_lr)
                self.scheduled_rates.append(scheduled_lr)
                self.data.epoch = epoch
                self.data.update_index = index
                sample_id = ",".join(sample_ids)
                epoch_done = index == view.updates
                step_metrics: dict[str, float] = {}
                if not epoch_done:
                    step_metrics = self.research.score(
                        runtime,
                        output_directory=self.lifecycle.output_directory,
                        config_identity=self.lifecycle.config_identity,
                        kind="step",
                        epoch=epoch,
                        logical_step=self.logical_step,
                        stopping=getattr(self, "training_control", None),
                    )
                self._queue_step(
                    loss=loss_value,
                    applied=result.applied,
                    sample_id=sample_id,
                    used_lr=used_lr,
                    scheduled_lr=scheduled_lr,
                    si_sdr=si_sdr,
                    epoch=epoch,
                    logical_step=self.logical_step,
                    offer_step=not epoch_done,
                    terms=term_acc,
                    step_metrics=step_metrics,
                )
                if not epoch_done:
                    continue
                self._drain_step_io()
                self.phase = _PENDING
                self.finished_epoch = epoch
                self.boundary = _open_boundary(epoch, warmup_updated=bool(result.applied))
                self._on_boundary_phase(_PENDING)
                self.lifecycle.save_if_needed("boundary_pending", logical_step=self.logical_step)
                if self._stop_requested(runtime):
                    return self._result("interrupted")
                self._finish_boundary(runtime, component_id)
                if should_stop(self):
                    return self._result("completed")
                if self.early_stopped or self._epoch_cap_reached():
                    return self._result("completed")
            finally:
                _close_epoch_view(view)
        self._drain_step_io()
        return self._result("completed")

    def state_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "early_stopped": self.early_stopped,
            "finished_epoch": self.finished_epoch,
            "history": list(self.history),
            "learning_rates": list(self.learning_rates),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "scheduled_rates": list(self.scheduled_rates),
            "stop_reason": self.stop_reason,
            "validation_profile": self.validation_profile,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        logical_step = state.get("logical_step")
        updates = state.get("optimizer_updates")
        phase = state.get("phase")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
            raise RuntimeFailure("trainer state is invalid")
        if phase not in {_IN_EPOCH, _PENDING}:
            raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
        if state.get("validation_profile") != self.validation_profile:
            raise RuntimeFailure("validation profile does not match this trainer")
        history = state.get("history")
        rates = state.get("learning_rates")
        scheduled = state.get("scheduled_rates")
        completed = state.get("completed_boundaries")
        if (
            not isinstance(history, list)
            or not isinstance(rates, list)
            or not isinstance(scheduled, list)
            or not isinstance(completed, list)
        ):
            raise RuntimeFailure("trainer history is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        self.phase = phase
        self.history = list(history)
        self.learning_rates = [float(value) for value in rates]
        self.scheduled_rates = [float(value) for value in scheduled]
        self.completed_boundaries = list(completed)
        boundary = state.get("boundary")
        self.boundary = None if boundary is None else dict(boundary)
        finished = state.get("finished_epoch")
        if finished is not None and (isinstance(finished, bool) or not isinstance(finished, int)):
            raise RuntimeFailure("finished_epoch is invalid")
        self.finished_epoch = finished
        self.early_stopped = bool(state.get("early_stopped", False))
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
        self._restored = True

    def _on_boundary_phase(self, phase: str) -> None:
        """Hook for tests that record validation, scheduler, and selection order."""

        del phase

    def _finish_boundary(
        self,
        runtime: ProcessRuntime,
        component_id: str,
    ) -> None:
        """Validate, step the plateau side of the warmup scheduler, write both rankings, then run research evaluation."""
        if self.phase != _PENDING or self.finished_epoch is None or self.boundary is None:
            raise RuntimeFailure(f"component {component_id}: there is no pending epoch boundary")
        trainer_epoch = int(self.finished_epoch)
        espnet_epoch = trainer_epoch + 1
        summary = self._validate(runtime, trainer_epoch, component_id)
        self.boundary["validation_done"] = True
        self.boundary["validation_loss"] = summary["loss"]
        self.boundary["validation_si_snr"] = summary["si_snr"]
        self.boundary["validation_weight"] = summary["weight"]
        self.phase = "pending_scheduler"
        self._on_boundary_phase(self.phase)
        step_num_before = int(self.scheduler.step_num)
        self.scheduler.step_validation(float(summary["loss"]))
        if int(self.scheduler.step_num) != step_num_before:
            raise RuntimeFailure(f"component {component_id}: validation advanced the warmup step")
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.boundary["scheduler_updated"] = True
        self.boundary["learning_rate"] = learning_rate
        self.phase = "pending_selection"
        self._on_boundary_phase(self.phase)
        self._select(
            runtime,
            self.lifecycle.output_directory,
            espnet_epoch,
            float(summary["loss"]),
            float(summary["si_snr"]),
        )
        self.boundary["selection_done"] = True
        self.completed_boundaries.append(dict(self.boundary))
        self.boundary = None
        self.phase = _IN_EPOCH
        self.finished_epoch = None
        self.data.epoch = trainer_epoch + 1
        self.data.update_index = 0
        self._on_boundary_phase(self.phase)
        if (
            self.early_stopping_patience is not None
            and early_stop(self.history, espnet_epoch, self.early_stopping_patience)
        ):
            self.early_stopped = True
        official = {"loss": float(summary["loss"]), "si_snr": float(summary["si_snr"])}
        observe_validation(
            getattr(self, "training_control", None),
            official,
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
            official=official,
            stopping=getattr(self, "training_control", None),
        )
        self.lifecycle.save_if_needed(
            "boundary",
            logical_step=self.logical_step,
            metrics={**official, **extra} if extra else official,
        )

    def _validate(self, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, float | int]:
        """Evaluate the injected objective and the SI-SDR monitor without moving the training cursor."""
        module = self.model
        was_training = bool(module.training)
        cursor = (int(self.data.epoch), int(self.data.update_index))
        rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
        local_loss_sum = 0.0
        local_loss_count = 0
        local_si_sum = 0.0
        local_si_count = 0
        try:
            module.eval()
            with torch.no_grad():
                for batch in self.data.validation_epoch(trainer_epoch):
                    if not isinstance(batch, Batch):
                        raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                    batch = runtime.move(batch)
                    try:
                        prediction = self.model.inference(batch.inputs)
                    except ValueError as exc:
                        raise RuntimeFailure(str(exc)) from exc
                    value = self._objective_total(prediction, batch.targets)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise RuntimeFailure(f"component {component_id}: validation loss is not a scalar")
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                    scores = per_sample_si_snr(prediction, batch.targets)
                    weight = int(batch.inputs.shape[0])
                    if int(scores.numel()) != weight:
                        raise RuntimeFailure(f"component {component_id}: SI-SNR count does not match the batch")
                    if not bool(torch.isfinite(scores).all()):
                        raise RuntimeFailure(f"component {component_id}: validation SI-SNR is not finite")
                    local_loss_sum += float(value.detach()) * weight
                    local_loss_count += weight
                    local_si_sum += float(scores.detach().sum())
                    local_si_count += weight
            gathered = runtime.gather_objects(
                {
                    "loss_count": local_loss_count,
                    "loss_sum": local_loss_sum,
                    "si_count": local_si_count,
                    "si_sum": local_si_sum,
                }
            )
            loss_sum = 0.0
            loss_count = 0
            si_sum = 0.0
            si_count = 0
            for item in gathered:
                loss_sum += float(item["loss_sum"])
                loss_count += int(item["loss_count"])
                si_sum += float(item["si_sum"])
                si_count += int(item["si_count"])
            if loss_count < 1 or si_count != loss_count:
                raise RuntimeFailure(f"component {component_id}: validation produced no batches")
            loss = loss_sum / loss_count
            si_snr = si_sum / si_count
        finally:
            restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
            if was_training:
                module.train()
            moved = (int(self.data.epoch), int(self.data.update_index))
            if moved != cursor:
                raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return {"loss": loss, "si_snr": si_snr, "weight": loss_count}

    def _select(
        self,
        runtime: ProcessRuntime,
        directory: Path,
        espnet_epoch: int,
        loss: float,
        si_snr: float,
    ) -> None:
        """Rank valid.loss and valid.si_snr, then write the official selection files."""
        self.history.append({"epoch": int(espnet_epoch), "loss": float(loss), "si_snr": float(si_snr)})
        by_loss = rank_validation_loss(self.history, self.keep_nbest)
        by_si = rank_validation_si_snr(self.history, self.keep_nbest)
        error: str | None = None
        if _rank(runtime) == 0:
            try:
                write_selection(
                    directory,
                    self.model.unwrapped().export_for_inference(),
                    espnet_epoch,
                    by_loss,
                    by_si,
                    self.keep_nbest,
                    self.logical_step,
                    self.validation_profile,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = runtime.gather_objects(error)
        failure = next((item for item in errors if item), None)
        if failure:
            raise RuntimeFailure(f"n-best selection failed: {failure}")
        runtime.barrier()

    def _objective_total(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the total of the injected objective. SI-SDR is not added here."""

        try:
            result = call_objective(self.objective, {"estimate": prediction, "target": target})
        except ValueError as exc:
            raise RuntimeFailure(str(exc)) from exc
        return result.total

    def _forward_loss(
        self,
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
                prediction = self.model(batch.inputs)
                result = call_objective(self.objective, {"estimate": prediction, "target": batch.targets})
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

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if group_size != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _stop_requested(self, runtime: object | None = None) -> bool:
        """Agree a logical-step cap with a rank interrupt."""

        return stop_requested(self, runtime)

    def _epoch_cap_reached(self) -> bool:
        return self.max_epochs is not None and int(self.data.epoch) >= self.max_epochs

    def _result(self, reason: str) -> RunResult:
        """Score the research end and finish. This path does not broadcast buffers."""

        return finish_training_run(self, reason)

    def _prepare_run(self, runtime: ProcessRuntime) -> str:
        """Rewind logs and counters. Does not run an optimizer update."""

        self.research.rewind(
            self.lifecycle.output_directory,
            restored=self._restored,
            phase=self.phase,
            progress={"logical_step": int(self.logical_step)},
        )
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(self.lifecycle.output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise RuntimeFailure("resumed data has no epoch position")
        self.losses = self.journal.begin(restored=self._restored, logical_step=self.logical_step)
        if not self._restored:
            self.logical_step = 0
            self.optimizer_updates = {self.optimizer_id: 0}
            self.losses = []
            self.learning_rates = []
            self.scheduled_rates = []
            self.history = []
            self.completed_boundaries = []
            self.boundary = None
            self.phase = _IN_EPOCH
            self.finished_epoch = None
            self.early_stopped = False
            self.data.epoch = 0
            self.data.update_index = 0
        bind_trainer_data(self)
        self._require_group_size(component_id)
        runtime.barrier()
        return str(component_id)

    def _open_window(
        self,
        component_id: str,
    ) -> tuple[int, int, Any, list[list[Batch]]]:
        """Check the epoch cursor and take the next accumulation window."""

        return open_grouped_window(self, component_id)

    def _queue_step(self, **record: Any) -> None:
        self._step_io = record

    def _drain_step_io(self) -> None:
        """Write the previous step while the next forward is already queued."""

        item = self._step_io
        if item is None:
            return
        self._step_io = None
        loss_value = item["loss"]
        applied = item["applied"]
        sample_id = item["sample_id"]
        used_lr = item["used_lr"]
        scheduled_lr = item["scheduled_lr"]
        si_sdr = item["si_sdr"]
        epoch = item["epoch"]
        logical_step = item["logical_step"]
        offer_step = item["offer_step"]
        terms = item["terms"]
        step_metrics = item["step_metrics"]
        self.journal.write_metrics(
            {
                "applied": applied,
                "epoch": epoch,
                "logical_step": logical_step,
                "loss": loss_value,
                "lr": used_lr,
                "optimizer_updates": dict(self.optimizer_updates),
                "sample_id": sample_id,
                "scheduled_lr": scheduled_lr,
                "si_sdr": si_sdr,
                "step_num": int(self.scheduler.step_num),
                "terms": terms,
            }
        )
        self.journal.write_sample(
            {
                "epoch": epoch,
                "logical_step": logical_step,
                "loss": loss_value,
                "lr": used_lr,
                "sample_id": sample_id,
                "scheduled_lr": scheduled_lr,
                "si_sdr": si_sdr,
            }
        )
        if offer_step:
            if step_metrics:
                self.lifecycle.save_if_needed("step", logical_step=logical_step, metrics=step_metrics)
            else:
                self.lifecycle.save_if_needed("step", logical_step=logical_step)


def _open_boundary(trainer_epoch: int, *, warmup_updated: bool) -> dict[str, Any]:
    """Record that validation, the plateau update, and selection have not run yet."""
    return {
        "espnet_epoch": trainer_epoch + 1,
        "learning_rate": None,
        "optimizer_updated": True,
        "scheduler_updated": False,
        "selection_done": False,
        "trainer_epoch": trainer_epoch,
        "validation_done": False,
        "validation_loss": None,
        "validation_si_snr": None,
        "validation_weight": None,
        "warmup_updated": bool(warmup_updated),
    }


def _positive_int(name: str, value: object) -> int:
    """Require a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    """Require a positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)
