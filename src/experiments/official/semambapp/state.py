"""Persistent SEMamba++ progress. Observation counters stay on the update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.experiments.errors import RuntimeFailure

IN_EPOCH = "in_epoch"
PENDING_BOUNDARY = "pending_boundary"
_PHASES = frozenset({IN_EPOCH, PENDING_BOUNDARY})


@dataclass
class _DecodedProgress:
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[dict[str, Any]]
    completed_boundaries: list[dict[str, Any]]
    step_log: list[dict[str, Any]]
    boundary: dict[str, Any] | None
    scheduler_steps: int
    stop_reason: str | None


@dataclass
class SEMambaProgress:
    """The checkpoint fields for one SEMamba++ run."""

    boundary: dict[str, Any] | None
    claim_label: str
    completed_boundaries: list[dict[str, Any]]
    history: list[dict[str, Any]]
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    scheduler_steps: int
    step_log: list[dict[str, Any]]
    stop_reason: str | None
    _optimizer_ids: tuple[str, str]

    @classmethod
    def initial(cls, optim_g_id: str, optim_d_id: str, claim_label: str) -> SEMambaProgress:
        return cls(
            boundary=None,
            claim_label=claim_label,
            completed_boundaries=[],
            history=[],
            logical_step=0,
            optimizer_updates={optim_g_id: 0, optim_d_id: 0},
            phase=IN_EPOCH,
            scheduler_steps=0,
            step_log=[],
            stop_reason="initialized",
            _optimizer_ids=(optim_g_id, optim_d_id),
        )

    def export(self) -> dict[str, Any]:
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

    def restore(self, state: dict[str, Any]) -> None:
        decoded = _decode_progress(state, self.claim_label, self._optimizer_ids)
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.history = decoded.history
        self.completed_boundaries = decoded.completed_boundaries
        self.step_log = decoded.step_log
        self.boundary = decoded.boundary
        self.scheduler_steps = decoded.scheduler_steps
        if decoded.stop_reason is not None:
            self.stop_reason = decoded.stop_reason

    def reset_fresh(self) -> None:
        optim_g_id, optim_d_id = self._optimizer_ids
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.scheduler_steps = 0
        self.step_log = []
        self.history = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH

    def complete_update(self) -> None:
        self.logical_step += 1
        for name in self._optimizer_ids:
            self.optimizer_updates[name] += 1

    def open_boundary(self, payload: dict[str, Any]) -> None:
        self.boundary = payload
        self.phase = PENDING_BOUNDARY

    def close_boundary(self, payload: dict[str, Any]) -> None:
        self.completed_boundaries.append(dict(payload))
        self.boundary = None
        self.phase = IN_EPOCH


def open_boundary_payload(trainer_epoch: int, *, validate: bool, epoch_done: bool) -> dict[str, Any]:
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


def _decode_progress(
    state: dict[str, Any],
    claim_label: str,
    optimizer_ids: tuple[str, str],
) -> _DecodedProgress:
    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    if state.get("claim_label") != claim_label:
        raise RuntimeFailure("claim_label does not match this trainer")
    history = state.get("history")
    completed = state.get("completed_boundaries")
    step_log = state.get("step_log")
    if not isinstance(history, list) or not isinstance(completed, list) or not isinstance(step_log, list):
        raise RuntimeFailure("trainer history is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    missing = [name for name in optimizer_ids if name not in converted]
    if missing:
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer ids")
    boundary = state.get("boundary")
    decoded_boundary = None if boundary is None else dict(boundary)
    scheduler_steps = int(state.get("scheduler_steps", 0))
    stop_reason = state.get("stop_reason")
    kept_reason = stop_reason if isinstance(stop_reason, str) else None
    return _DecodedProgress(
        logical_step=logical_step,
        optimizer_updates=converted,
        phase=str(phase),
        history=list(history),
        completed_boundaries=list(completed),
        step_log=list(step_log),
        boundary=decoded_boundary,
        scheduler_steps=scheduler_steps,
        stop_reason=kept_reason,
    )
