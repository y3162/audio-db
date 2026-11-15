"""Checkpoint payload for the TF-Locoformer trainer.

``LocoformerProgress`` is the only object that stores the live counters.
``losses`` is rebuilt from the step journal and is not part of the payload.
``learning_rates`` are the rates that were used. ``scheduled_rates`` are the
rates ``step_batch`` left for the next update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.experiments.errors import RuntimeFailure

IN_EPOCH = "in_epoch"
PENDING_VALIDATION = "pending_validation"
_PHASES = frozenset({IN_EPOCH, PENDING_VALIDATION})


def open_boundary(trainer_epoch: int, *, warmup_updated: bool) -> dict[str, Any]:
    """Record that validation, the plateau update, and selection have not run."""

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


@dataclass(frozen=True)
class _DecodedProgress:
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[Any]
    learning_rates: list[float]
    scheduled_rates: list[float]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str | None


def _decode_progress(state: dict[str, Any], *, optimizer_id: str, validation_profile: str) -> _DecodedProgress:
    """Validate and copy a payload. This does not touch the live progress."""

    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    if state.get("validation_profile") != validation_profile:
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
    converted = {str(key): int(value) for key, value in updates.items()}
    copied_rates = [float(value) for value in rates]
    copied_scheduled = [float(value) for value in scheduled]
    boundary = state.get("boundary")
    copied_boundary = None if boundary is None else dict(boundary)
    finished = state.get("finished_epoch")
    if finished is not None and (isinstance(finished, bool) or not isinstance(finished, int)):
        raise RuntimeFailure("finished_epoch is invalid")
    if optimizer_id not in converted:
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer id")
    stop_reason = state.get("stop_reason")
    return _DecodedProgress(
        logical_step=logical_step,
        optimizer_updates=converted,
        phase=phase,
        history=list(history),
        learning_rates=copied_rates,
        scheduled_rates=copied_scheduled,
        completed_boundaries=list(completed),
        boundary=copied_boundary,
        finished_epoch=finished,
        early_stopped=bool(state.get("early_stopped", False)),
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )


@dataclass
class LocoformerProgress:
    """Live TF-Locoformer counters. Checkpoint code sees them only through ``export``."""

    logical_step: int
    optimizer_updates: dict[str, int]
    learning_rates: list[float]
    scheduled_rates: list[float]
    history: list[Any]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    phase: str
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str
    validation_profile: str
    losses: list[float]
    _optimizer_id: str = field(repr=False)

    @classmethod
    def initial(cls, optimizer_id: str, validation_profile: str) -> LocoformerProgress:
        return cls(
            logical_step=0,
            optimizer_updates={optimizer_id: 0},
            learning_rates=[],
            scheduled_rates=[],
            history=[],
            completed_boundaries=[],
            boundary=None,
            phase=IN_EPOCH,
            finished_epoch=None,
            early_stopped=False,
            stop_reason="initialized",
            validation_profile=validation_profile,
            losses=[],
            _optimizer_id=optimizer_id,
        )

    @property
    def pending(self) -> bool:
        return self.phase == PENDING_VALIDATION

    def reset_fresh(self) -> None:
        """Zero a run that is not resuming. ``stop_reason`` and the profile stay."""

        self.logical_step = 0
        self.optimizer_updates = {self._optimizer_id: 0}
        self.losses = []
        self.learning_rates = []
        self.scheduled_rates = []
        self.history = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None
        self.early_stopped = False

    def count_applied_optimizer(self) -> None:
        """Count one optimizer step that ``runtime.step`` applied. Does not move the logical step."""

        self.optimizer_updates[self._optimizer_id] += 1

    def record_step(self, *, loss: float, used_lr: float, scheduled_lr: float) -> None:
        """Count the logical step and both learning-rate histories."""

        self.logical_step += 1
        self.losses.append(loss)
        self.learning_rates.append(used_lr)
        self.scheduled_rates.append(scheduled_lr)

    def place_cursor(self, data: Any, epoch: int, index: int) -> None:
        data.epoch = int(epoch)
        data.update_index = int(index)

    def reset_cursor(self, data: Any) -> None:
        data.epoch = 0
        data.update_index = 0

    def open_pending(self, trainer_epoch: int, *, warmup_updated: bool) -> None:
        self.phase = PENDING_VALIDATION
        self.finished_epoch = int(trainer_epoch)
        self.boundary = open_boundary(trainer_epoch, warmup_updated=warmup_updated)

    def note_validation(self, loss: float, si_snr: float, weight: int) -> None:
        boundary = self._open_boundary()
        boundary["validation_done"] = True
        boundary["validation_loss"] = loss
        boundary["validation_si_snr"] = si_snr
        boundary["validation_weight"] = weight

    def note_scheduler(self, learning_rate: float) -> None:
        boundary = self._open_boundary()
        boundary["scheduler_updated"] = True
        boundary["learning_rate"] = float(learning_rate)

    def append_history(self, row: dict[str, Any]) -> None:
        self.history.append(row)

    def close_boundary(self) -> None:
        boundary = self._open_boundary()
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None

    def roll_epoch(self, data: Any, trainer_epoch: int) -> None:
        data.epoch = int(trainer_epoch) + 1
        data.update_index = 0

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def export(self) -> dict[str, Any]:
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

    def restore(self, state: dict[str, Any]) -> None:
        """Replace the counters after the payload has been checked. ``losses`` stays."""

        decoded = _decode_progress(
            state,
            optimizer_id=self._optimizer_id,
            validation_profile=self.validation_profile,
        )
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.history = decoded.history
        self.learning_rates = decoded.learning_rates
        self.scheduled_rates = decoded.scheduled_rates
        self.completed_boundaries = decoded.completed_boundaries
        self.boundary = decoded.boundary
        self.finished_epoch = decoded.finished_epoch
        self.early_stopped = decoded.early_stopped
        if decoded.stop_reason is not None:
            self.stop_reason = decoded.stop_reason

    def _open_boundary(self) -> dict[str, Any]:
        boundary = self.boundary
        if boundary is None or self.finished_epoch is None:
            raise RuntimeFailure("there is no pending epoch boundary")
        return boundary
