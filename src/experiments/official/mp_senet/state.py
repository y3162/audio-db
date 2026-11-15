"""Checkpoint payload for the MP-SENet trainer.

``MPTrainingProgress`` is the only object that stores the live counters.
``export`` writes the existing ``states["trainer"]`` keys. This module does
not train, validate, score, or write a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.experiments.errors import RuntimeFailure

IN_EPOCH = "in_epoch"
PENDING_BOUNDARY = "pending_boundary"
_PHASES = frozenset({IN_EPOCH, PENDING_BOUNDARY})


def open_boundary(trainer_epoch: int, *, validate: bool, epoch_done: bool) -> dict[str, Any]:
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


@dataclass
class MPTrainingProgress:
    """Live MP-SENet counters. Checkpoint code sees them only through ``export``."""

    logical_step: int
    optimizer_updates: dict[str, int]
    scheduler_steps: int
    phase: str
    boundary: dict[str, Any] | None
    history: list[Any]
    completed_boundaries: list[Any]
    step_log: list[Any]
    best_pesq: float | None
    best_epoch: int | None
    best_logical_step: int | None
    stop_reason: str
    _optimizer_ids: tuple[str, str] = field(repr=False)

    @classmethod
    def initial(cls, optim_g_id: str, optim_d_id: str) -> MPTrainingProgress:
        """Zero counters for a trainer that has not restored a checkpoint."""

        return cls(
            logical_step=0,
            optimizer_updates={optim_g_id: 0, optim_d_id: 0},
            scheduler_steps=0,
            phase=IN_EPOCH,
            boundary=None,
            history=[],
            completed_boundaries=[],
            step_log=[],
            best_pesq=None,
            best_epoch=None,
            best_logical_step=None,
            stop_reason="initialized",
            _optimizer_ids=(optim_g_id, optim_d_id),
        )

    @property
    def pending(self) -> bool:
        """True after a pending save and before that boundary has been closed."""

        return self.phase == PENDING_BOUNDARY

    def reset_fresh(self) -> None:
        """Zero a run that is not resuming. ``stop_reason`` stays as it is."""

        optim_g_id, optim_d_id = self._optimizer_ids
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.scheduler_steps = 0
        self.phase = IN_EPOCH
        self.boundary = None
        self.history = []
        self.completed_boundaries = []
        self.step_log = []
        self.best_pesq = None
        self.best_epoch = None
        self.best_logical_step = None

    def complete_update(self) -> None:
        """Count one finished GAN pair on the logical step and both optimizers."""

        self.logical_step += 1
        for key in self.optimizer_updates:
            self.optimizer_updates[key] += 1

    def append_step(self, entry: dict[str, Any]) -> None:
        self.step_log.append(entry)

    def append_history(self, decision: dict[str, Any]) -> None:
        self.history.append(decision)

    def open_pending(self, trainer_epoch: int, *, validate: bool, epoch_done: bool) -> None:
        """Open the boundary that a pending checkpoint is about to store."""

        self.phase = PENDING_BOUNDARY
        self.boundary = open_boundary(trainer_epoch, validate=validate, epoch_done=epoch_done)

    def remember_best(self, decision: dict[str, Any]) -> None:
        """Store a strict PESQ improvement. A rejection leaves the current best."""

        if not decision["selected"]:
            return
        self.best_pesq = float(decision["pesq"])
        self.best_epoch = int(decision["epoch"])
        self.best_logical_step = int(decision["logical_step"])

    def mark_validation(self, decision: dict[str, Any]) -> None:
        """Record that this open boundary has finished validation and selection."""

        boundary = self._open_boundary()
        boundary["validation_done"] = True
        boundary["selection_done"] = True
        boundary["pesq"] = decision["pesq"]
        boundary["pesq_status"] = decision["pesq_status"]
        boundary["selected"] = decision["selected"]

    def mark_phase(self, phase: str) -> None:
        """Set the in-memory phase. ``pending_scheduler`` is not a checkpoint phase."""

        self.phase = phase

    def mark_scheduler_done(self, lr_g: float, lr_d: float) -> None:
        """Record one epoch-end step of both schedulers."""

        boundary = self._open_boundary()
        boundary["scheduler_done"] = True
        boundary["lr_g"] = float(lr_g)
        boundary["lr_d"] = float(lr_d)
        self.scheduler_steps += 1

    def close_boundary(self) -> None:
        """Append the open boundary and return to the epoch."""

        boundary = self._open_boundary()
        boundary["closed"] = True
        self.completed_boundaries.append(dict(boundary))
        self.boundary = None
        self.phase = IN_EPOCH

    def place_cursor(self, data: Any, epoch: int, index: int) -> None:
        """Write the data cursor after a finished update. Data owns the fields."""

        data.epoch = int(epoch)
        data.update_index = int(index)

    def reset_cursor(self, data: Any) -> None:
        """Put a fresh run at the start of epoch 0."""

        data.epoch = 0
        data.update_index = 0

    def roll_epoch(self, data: Any, trainer_epoch: int) -> None:
        """Advance past an epoch that the closed boundary has finished."""

        data.epoch = int(trainer_epoch) + 1
        data.update_index = 0

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def export(self, *, experiment_claim: str) -> dict[str, Any]:
        """Copy the counters into the existing ``states["trainer"]`` dict."""

        return {
            "best_epoch": self.best_epoch,
            "best_logical_step": self.best_logical_step,
            "best_pesq": self.best_pesq,
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "experiment_claim": experiment_claim,
            "history": list(self.history),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "scheduler_steps": self.scheduler_steps,
            "step_log": list(self.step_log),
            "stop_reason": self.stop_reason,
        }

    def restore(self, state: dict[str, Any], *, experiment_claim: str) -> None:
        """Replace the counters from a stored trainer dict. Invalid payloads raise."""

        logical_step = state.get("logical_step")
        updates = state.get("optimizer_updates")
        phase = state.get("phase")
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
            raise RuntimeFailure("trainer state is invalid")
        if phase not in _PHASES:
            raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
        if state.get("experiment_claim") != experiment_claim:
            raise RuntimeFailure("experiment claim does not match this trainer")
        history = state.get("history")
        completed = state.get("completed_boundaries")
        step_log = state.get("step_log")
        if not isinstance(history, list) or not isinstance(completed, list) or not isinstance(step_log, list):
            raise RuntimeFailure("trainer history is invalid")
        boundary = state.get("boundary")
        best = state.get("best_pesq")
        best_epoch = state.get("best_epoch")
        best_step = state.get("best_logical_step")
        if best_epoch is not None and (isinstance(best_epoch, bool) or not isinstance(best_epoch, int)):
            raise RuntimeFailure("best_epoch is invalid")
        if best_step is not None and (isinstance(best_step, bool) or not isinstance(best_step, int)):
            raise RuntimeFailure("best_logical_step is invalid")
        self.logical_step = logical_step
        self.optimizer_updates = {str(key): int(value) for key, value in updates.items()}
        self.phase = phase
        self.history = list(history)
        self.completed_boundaries = list(completed)
        self.step_log = list(step_log)
        self.boundary = None if boundary is None else dict(boundary)
        self.scheduler_steps = int(state.get("scheduler_steps", 0))
        self.best_pesq = None if best is None else float(best)
        self.best_epoch = best_epoch
        self.best_logical_step = best_step
        stop_reason = state.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason

    def _open_boundary(self) -> dict[str, Any]:
        boundary = self.boundary
        if boundary is None:
            raise RuntimeFailure("there is no pending GAN boundary")
        return boundary
