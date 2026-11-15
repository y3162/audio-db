"""Checkpoint payload for the TF-GridNet trainer.

``GridNetProgress`` is the only object that stores the live counters.
``losses`` is rebuilt from the step journal and is not part of the payload.
``export`` writes the existing ``states["trainer"]`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.experiments.errors import RuntimeFailure

IN_EPOCH = "in_epoch"
PENDING_VALIDATION = "pending_validation"
_PHASES = frozenset({IN_EPOCH, PENDING_VALIDATION})


def open_boundary(trainer_epoch: int) -> dict[str, Any]:
    """Record that validation, the scheduler, and n-best selection have not run."""

    return {
        "espnet_epoch": trainer_epoch + 1,
        "learning_rate": None,
        "optimizer_updated": True,
        "scheduler_updated": False,
        "selection_done": False,
        "trainer_epoch": trainer_epoch,
        "validation_done": False,
        "validation_loss": None,
        "validation_weight": None,
    }


@dataclass(frozen=True)
class _DecodedProgress:
    """Temporary decode of one payload. It is not stored on the live progress."""

    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[Any]
    learning_rates: list[float]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str | None


def _decode_progress(
    state: dict[str, Any],
    *,
    optimizer_id: str,
) -> _DecodedProgress:
    """Validate and copy a payload. This does not touch the live progress."""

    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    history = state.get("history")
    rates = state.get("learning_rates")
    completed = state.get("completed_boundaries")
    if not isinstance(history, list) or not isinstance(rates, list) or not isinstance(completed, list):
        raise RuntimeFailure("trainer history is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    copied_rates = [float(value) for value in rates]
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
        completed_boundaries=list(completed),
        boundary=copied_boundary,
        finished_epoch=finished,
        early_stopped=bool(state.get("early_stopped", False)),
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )


@dataclass
class GridNetProgress:
    """Live TF-GridNet counters. Checkpoint code sees them only through ``export``."""

    logical_step: int
    optimizer_updates: dict[str, int]
    learning_rates: list[float]
    history: list[Any]
    completed_boundaries: list[Any]
    boundary: dict[str, Any] | None
    phase: str
    finished_epoch: int | None
    early_stopped: bool
    stop_reason: str
    losses: list[float]
    _optimizer_id: str = field(repr=False)

    @classmethod
    def initial(cls, optimizer_id: str) -> GridNetProgress:
        """Zero counters for a trainer that has not restored a checkpoint."""

        return cls(
            logical_step=0,
            optimizer_updates={optimizer_id: 0},
            learning_rates=[],
            history=[],
            completed_boundaries=[],
            boundary=None,
            phase=IN_EPOCH,
            finished_epoch=None,
            early_stopped=False,
            stop_reason="initialized",
            losses=[],
            _optimizer_id=optimizer_id,
        )

    @property
    def pending(self) -> bool:
        """True after a pending save and before that boundary has been closed."""

        return self.phase == PENDING_VALIDATION

    def reset_fresh(self) -> None:
        """Zero a run that is not resuming. ``stop_reason`` stays as it is."""

        self.logical_step = 0
        self.optimizer_updates = {self._optimizer_id: 0}
        self.losses = []
        self.learning_rates = []
        self.history = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None
        self.early_stopped = False

    def record_step(self, *, loss: float, applied: bool, learning_rate: float) -> None:
        """Count one logical step. Only an applied optimizer step increments its id."""

        self.logical_step += 1
        if applied:
            self.optimizer_updates[self._optimizer_id] += 1
        self.losses.append(loss)
        self.learning_rates.append(learning_rate)

    def place_cursor(self, data: Any, epoch: int, index: int) -> None:
        """Write the data cursor after a finished update. Data owns the fields."""

        data.epoch = int(epoch)
        data.update_index = int(index)

    def reset_cursor(self, data: Any) -> None:
        """Put a fresh run at the start of epoch 0."""

        data.epoch = 0
        data.update_index = 0

    def open_pending(self, trainer_epoch: int) -> None:
        """Open the boundary that a pending checkpoint is about to store."""

        self.phase = PENDING_VALIDATION
        self.finished_epoch = int(trainer_epoch)
        self.boundary = open_boundary(trainer_epoch)

    def note_validation(self, loss: float, weight: int) -> None:
        """Record the validation totals on the open boundary."""

        boundary = self._open_boundary()
        boundary["validation_done"] = True
        boundary["validation_loss"] = loss
        boundary["validation_weight"] = weight

    def note_scheduler(self, learning_rate: float) -> None:
        """Record that ReduceLROnPlateau has stepped for this boundary."""

        boundary = self._open_boundary()
        boundary["scheduler_updated"] = True
        boundary["learning_rate"] = float(learning_rate)

    def append_history(self, row: dict[str, Any]) -> None:
        self.history.append(row)

    def close_boundary(self) -> None:
        """Append the open boundary and return to the epoch."""

        boundary = self._open_boundary()
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = IN_EPOCH
        self.finished_epoch = None

    def roll_epoch(self, data: Any, trainer_epoch: int) -> None:
        """Advance past an epoch that the closed boundary has finished."""

        data.epoch = int(trainer_epoch) + 1
        data.update_index = 0

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def export(self) -> dict[str, Any]:
        """Copy the counters into the existing ``states["trainer"]`` dict."""

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
            "stop_reason": self.stop_reason,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Replace the counters from a stored trainer dict.

        Decoding finishes before any checkpoint field is written. ``losses``
        is not in the payload and is left untouched.
        """

        decoded = _decode_progress(state, optimizer_id=self._optimizer_id)
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.history = decoded.history
        self.learning_rates = decoded.learning_rates
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
