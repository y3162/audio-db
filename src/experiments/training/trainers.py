"""Trainers constructed by the component factory.

``StandardTrainer`` performs forward, loss, backward, and one optimizer
update per logical step. Official speech-enhancement loops live under
``src/models/``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from src.experiments.data import Batch
from src.experiments.errors import BuildError, CheckpointError, RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.execution.results import RunResult
from src.experiments.execution.runtime import PreparedModel, ProcessRuntime, require_prepared
from src.experiments.stopping import MetricEvent, should_stop
from src.experiments.training.sample_weight import sample_scales, window_is_uniform


class StandardTrainer:
    """Single-model, single-optimizer loop on micro-batches."""

    supports_training_control = True
    progress_keys = ("logical_step", "epoch")
    training_metric_sources = ("train",)
    training_metric_names = ("train.mean_rank_loss",)

    def __init__(
        self,
        model: PreparedModel,
        loss: torch.nn.Module,
        data: Any,
        optimizer: torch.optim.Optimizer,
        *,
        max_logical_steps: int,
        accumulation: int = 1,
        grad_clip: float | None = None,
        scheduler: Any | None = None,
        stop_after_logical_steps: int | None = None,
    ) -> None:
        require_prepared(model)
        if not callable(loss):
            raise BuildError("loss must be callable")
        if not hasattr(data, "train_epoch"):
            raise BuildError("data must provide train_epoch")
        optimizer_id = getattr(optimizer, "_experiment_component_id", None)
        if not isinstance(optimizer_id, str):
            raise BuildError("optimizer has no component id")
        if isinstance(max_logical_steps, bool) or not isinstance(max_logical_steps, int) or max_logical_steps < 1:
            raise BuildError("max_logical_steps must be a positive integer")
        if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
            raise BuildError("accumulation must be a positive integer")
        if grad_clip is not None and (
            isinstance(grad_clip, bool) or not isinstance(grad_clip, (int, float)) or grad_clip <= 0
        ):
            raise BuildError("grad_clip must be a positive number")
        if stop_after_logical_steps is not None and (
            isinstance(stop_after_logical_steps, bool)
            or not isinstance(stop_after_logical_steps, int)
            or stop_after_logical_steps < 1
        ):
            raise BuildError("stop_after_logical_steps must be a positive integer")
        self.model = model
        self.loss = loss
        self.data = data
        self.optimizer = optimizer
        self.optimizer_id = optimizer_id
        self.max_logical_steps = max_logical_steps
        self.accumulation = accumulation
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.scheduler = scheduler
        self.stop_after_logical_steps = stop_after_logical_steps
        self.logical_step = 0
        self.optimizer_updates = {optimizer_id: 0}
        self.losses: list[float] = []
        self.stop_reason = "initialized"
        self._restored = False

    def preflight_probe(self) -> dict[str, Any]:
        """Run one accumulation of forward, loss, and backward. Do not update.

        The caller snapshots and restores RNG, parameters, buffers, gradients,
        optimizer state, and the data cursor. This method does not advance
        ``logical_step`` and does not call the optimizer.
        """

        runtime = getattr(self, "runtime", None)
        if not isinstance(runtime, ProcessRuntime):
            raise BuildError("preflight has no runtime")
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self._require_group_size(component_id)
        was_training = bool(self.model.training)
        self.model.train()
        view = self.data.train_epoch(0, start_update=0)
        try:
            if view.updates < 1:
                raise RuntimeFailure(
                    f"component {component_id}: preflight accumulation {self.accumulation} "
                    f"does not fit in {view.updates} updates"
                )
            self.optimizer.zero_grad(set_to_none=True)
            total, sample_ids, _window = self._backward_window(view, 0, component_id, runtime)
            if total is None or total.numel() != 1 or not bool(torch.isfinite(total).all()):
                raise RuntimeFailure(f"component {component_id}: preflight loss is not finite")
            missing = 0
            present = 0
            for parameter in self.model.parameters:
                if not parameter.requires_grad:
                    continue
                if parameter.grad is None:
                    missing += 1
                else:
                    present += 1
            if missing:
                raise RuntimeFailure(
                    f"component {component_id}: preflight left {missing} trainable parameters without gradients"
                )
            return {
                "finite": True,
                "loss": float(total),
                "parameters_missing_grad": missing,
                "parameters_with_grad": present,
                "sample_id": ",".join(sample_ids),
                "status": "ok",
            }
        finally:
            if not was_training:
                self.model.eval()
            _close_epoch_view(view)

    def fit(self) -> RunResult:
        """Run until ``max_logical_steps``. A restored trainer continues.

        ``lifecycle.start`` restores when Experiment asked for a resume. The
        loss is divided by accumulation. A new run clears ``metrics.jsonl``.
        Resume keeps rows at or before the restored logical step and drops
        anything written past that commit. ``lifecycle.finish`` publishes the
        outcome once.
        """

        self.lifecycle.start()
        runtime = self.runtime
        output_directory = self.lifecycle.output_directory
        component_id = getattr(self, "_experiment_component_id", "trainer")
        self.journal = StepJournal(output_directory, runtime)
        if self._restored and (not hasattr(self.data, "epoch") or not hasattr(self.data, "update_index")):
            raise CheckpointError("resumed data has no epoch position")
        self.losses = self.journal.begin(restored=self._restored, logical_step=self.logical_step)
        if self._restored:
            epoch = int(self.data.epoch)
            index = int(self.data.update_index)
        else:
            epoch = 0
            index = 0
            self.logical_step = 0
            self.optimizer_updates = {self.optimizer_id: 0}
        self._require_group_size(component_id)
        runtime.barrier()
        while self.logical_step < self.max_logical_steps and not should_stop(self):
            from src.experiments.execution.signals import ranks_should_stop

            capped = (
                self.stop_after_logical_steps is not None
                and self.logical_step >= self.stop_after_logical_steps
            )
            if ranks_should_stop(runtime, extra=capped):
                if self.logical_step > 0:
                    self._offer("stop")
                self.stop_reason = "interrupted"
                return self.lifecycle.finish(
                    RunResult(
                        stop_reason="interrupted",
                        logical_step=self.logical_step,
                        optimizer_updates=dict(self.optimizer_updates),
                        output_dir=output_directory,
                    )
                )
            self._before_logical_step(runtime)
            view = self.data.train_epoch(epoch, start_update=index)
            try:
                if index >= view.updates:
                    epoch += 1
                    index = 0
                    if epoch > self.max_logical_steps + 1:
                        raise RuntimeFailure(f"component {component_id}: data has no updates")
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                total, sample_ids, window = self._backward_window(view, index, component_id, runtime)
                self._before_optimizer_step(runtime)
                if self.grad_clip is not None:
                    runtime.clip(self.model, self.grad_clip, optimizer=self.optimizer)
                result = runtime.step(self.optimizer)
                runtime.end_step()
                if result.applied:
                    self.optimizer_updates[self.optimizer_id] += 1
                    if self.scheduler is not None:
                        self.scheduler.step()
                self.logical_step += 1
                index += window
                if hasattr(self.data, "epoch") and hasattr(self.data, "update_index"):
                    self.data.epoch = epoch
                    self.data.update_index = index
                if total is None or not bool(torch.isfinite(total).all()):
                    raise RuntimeFailure(f"component {component_id}: loss is not finite")
                loss_value = float(total)
                self.losses.append(loss_value)
                sample_id = ",".join(sample_ids)
                self.journal.write_metrics(
                    {
                        "applied": result.applied,
                        "logical_step": self.logical_step,
                        "loss": loss_value,
                        "optimizer_updates": dict(self.optimizer_updates),
                        "sample_id": sample_id,
                    }
                )
                self.journal.write_sample(
                    {
                        "logical_step": self.logical_step,
                        "loss": loss_value,
                        "sample_id": sample_id,
                    }
                )
                control = getattr(self, "training_control", None)
                if control is not None:
                    rank_losses = runtime.gather_objects(loss_value)
                    mean = sum(rank_losses) / len(rank_losses)
                    control.observe(MetricEvent("train", self.logical_step, {"mean_rank_loss": mean}))
                self._offer("step")
            finally:
                _close_epoch_view(view)
        self._offer("final")
        self.stop_reason = "early_stopping" if should_stop(self) else "completed"
        return self.lifecycle.finish(
            RunResult(
                stop_reason=self.stop_reason,
                logical_step=self.logical_step,
                optimizer_updates=dict(self.optimizer_updates),
                output_dir=output_directory,
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        logical_step = state.get("logical_step")
        updates = state.get("optimizer_updates")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
            raise RuntimeFailure("trainer state is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason
        self._restored = True

    def _before_logical_step(self, runtime: ProcessRuntime) -> None:
        """Hook for tests that change the grad scale before the next backward."""

        del runtime

    def _before_optimizer_step(self, runtime: ProcessRuntime) -> None:
        """Hook for tests that edit gradients before unscale, clip, and step."""

        del runtime

    def _before_checkpoint(self, runtime: ProcessRuntime) -> None:
        """Hook for tests that change state after a step and before publish."""

        del runtime

    def _offer(self, event: str) -> None:
        """Offer this state to the selector. ``_before_checkpoint`` runs only when a file is written.

        ``stop`` and ``final`` follow the offer of the last finished step
        without a state change, so the selector can see that Latest already
        holds this state. Saving it again would run ``_before_checkpoint``
        twice and a resumed run would no longer match a continuous one.
        """

        self.lifecycle.save_if_needed(
            event,
            logical_step=self.logical_step,
            progress={"logical_step": int(self.logical_step)},
            prepare=self._before_checkpoint,
        )

    def _backward_window(
        self,
        view: Any,
        index: int,
        component_id: str,
        runtime: ProcessRuntime,
    ) -> tuple[torch.Tensor | None, list[str], int]:
        """Backward one optimizer window.

        A full window with equal micro-batch sizes on every rank divides by
        ``accumulation``. A short tail or an uneven rank scales by the real
        sample count. An empty rank backpropagates a zero that stays connected
        to the parameters so the all-reduce still runs.
        """

        remaining = int(view.updates) - index
        if remaining < 1:
            raise RuntimeFailure(
                f"component {component_id}: accumulation {self.accumulation} "
                f"does not fit at update {index}"
            )
        window = self.accumulation if remaining >= self.accumulation else remaining
        pieces: list[tuple[list[Batch], int]] = []
        for offset in range(window):
            batches = view.batch(index + offset)
            count = sum(int(batch.inputs.shape[0]) for batch in batches)
            pieces.append((batches, count))
        counts = [count for _batches, count in pieces]
        if int(runtime.world_size) == 1:
            rank_counts = [counts]
        else:
            rank_counts = [list(row) for row in runtime.gather_objects(counts)]
        uniform = window_is_uniform(rank_counts, self.accumulation)
        n_global = sum(sum(row) for row in rank_counts)
        scales = (
            None
            if uniform
            else sample_scales(counts, world_size=int(runtime.world_size), n_global=n_global)
        )
        total: torch.Tensor | None = None
        sample_ids: list[str] = []
        for offset, (batches, count) in enumerate(pieces):
            sync = offset == window - 1
            sample_ids.extend(batch.sample_id for batch in batches if batch.sample_id)
            with runtime.sync_gradients([self.model], enabled=sync):
                with runtime.autocast():
                    if count == 0:
                        micro = self._empty_loss(batches, component_id, runtime)
                    else:
                        micro = self._forward_loss(batches, component_id, runtime)
                    if uniform:
                        micro = micro / self.accumulation
                    else:
                        assert scales is not None
                        micro = micro * scales[offset]
                    runtime.backward(micro)
            detached = micro.detach()
            total = detached if total is None else total + detached
        return total, sample_ids, window

    def _empty_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        """A finite zero that still runs the module forward.

        DDP decides which parameters join the all-reduce during forward.
        A zero built outside the module does not enter that collective, so
        the rank that still has audio waits forever. The task loss is not
        used: mean reduction of an empty tensor is NaN.
        """

        total: torch.Tensor | None = None
        for batch in batches:
            batch = _batch_on_device(batch, runtime.device)
            prediction = self.model(batch.inputs)
            if not isinstance(prediction, torch.Tensor):
                raise RuntimeFailure(f"component {component_id}: model did not return a tensor")
            value = prediction.reshape(-1).sum() * 0
            total = value if total is None else total + value
        if total is None:
            return self._zero_loss(component_id)
        return total

    def _zero_loss(self, component_id: str) -> torch.Tensor:
        total: torch.Tensor | None = None
        for parameter in self.model.parameters:
            if not parameter.requires_grad:
                continue
            term = parameter.reshape(-1)[:1].sum() * 0
            total = term if total is None else total + term
        if total is None:
            raise RuntimeFailure(f"component {component_id}: empty rank has no trainable parameters")
        return total

    def _require_group_size(self, component_id: str) -> None:
        group_size = getattr(self.data, "group_size", None)
        if group_size is None:
            return
        if group_size != self.accumulation:
            raise RuntimeFailure(
                f"component {component_id}: data group_size {group_size} "
                f"does not match accumulation {self.accumulation}"
            )

    def _forward_loss(self, batches: list[Batch], component_id: str, runtime: ProcessRuntime) -> torch.Tensor:
        if not batches:
            raise RuntimeFailure(f"component {component_id}: empty micro-batch")
        total: torch.Tensor | None = None
        for batch in batches:
            batch = _batch_on_device(batch, runtime.device)
            prediction = self.model(batch.inputs)
            value = self.loss(prediction, batch.targets)
            if not isinstance(value, torch.Tensor):
                raise RuntimeFailure(f"component {component_id}: loss did not return a tensor")
            total = value if total is None else total + value
        assert total is not None
        return total / len(batches)

def _close_epoch_view(view: object) -> None:
    """Release an epoch view. Data modules without ``close`` stay unchanged."""

    close = getattr(view, "close", None)
    if callable(close):
        close()


def _batch_on_device(batch: Batch, device: torch.device) -> Batch:
    """Move model inputs to ``device``. Leave them in place when they already match."""

    inputs = batch.inputs
    targets = batch.targets
    lengths = batch.lengths
    non_blocking = device.type == "cuda"
    changed = False
    if inputs.device != device:
        inputs = inputs.to(device, non_blocking=non_blocking and inputs.device.type == "cpu")
        changed = True
    if targets.device != device:
        targets = targets.to(device, non_blocking=non_blocking and targets.device.type == "cpu")
        changed = True
    if lengths is not None and lengths.device != device:
        lengths = lengths.to(device, non_blocking=non_blocking and lengths.device.type == "cpu")
        changed = True
    if not changed:
        return batch
    return replace(batch, inputs=inputs, targets=targets, lengths=lengths)


def _rank(runtime: ProcessRuntime) -> int:
    return int(getattr(runtime, "rank", 0))
