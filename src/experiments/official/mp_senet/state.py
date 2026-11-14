"""Checkpoint payload for the MP-SENet trainer.

The trainer owns the live fields. This module copies them out and checks a
loaded dict. It does not train, validate, score, or write a file. A second
copy of the progress is not kept here after ``restore_progress`` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def export_progress(
    *,
    best_epoch: int | None,
    best_logical_step: int | None,
    best_pesq: float | None,
    boundary: dict[str, Any] | None,
    completed_boundaries: list[dict[str, Any]],
    experiment_claim: str,
    history: list[dict[str, Any]],
    logical_step: int,
    optimizer_updates: dict[str, int],
    phase: str,
    scheduler_steps: int,
    step_log: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    """Copy the trainer fields into the existing ``states["trainer"]`` dict."""

    return {
        "best_epoch": best_epoch,
        "best_logical_step": best_logical_step,
        "best_pesq": best_pesq,
        "boundary": boundary,
        "completed_boundaries": list(completed_boundaries),
        "experiment_claim": experiment_claim,
        "history": list(history),
        "logical_step": logical_step,
        "optimizer_updates": dict(optimizer_updates),
        "phase": phase,
        "scheduler_steps": scheduler_steps,
        "step_log": list(step_log),
        "stop_reason": stop_reason,
    }


@dataclass(frozen=True)
class RestoredProgress:
    """Values to assign onto the trainer. ``stop_reason`` None keeps the current one."""

    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    history: list[Any]
    completed_boundaries: list[Any]
    step_log: list[Any]
    boundary: dict[str, Any] | None
    scheduler_steps: int
    best_pesq: float | None
    best_epoch: int | None
    best_logical_step: int | None
    stop_reason: str | None


def restore_progress(state: dict[str, Any], *, experiment_claim: str) -> RestoredProgress:
    """Check a stored trainer dict and return the fields to assign."""

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
    stop_reason = state.get("stop_reason")
    return RestoredProgress(
        logical_step=logical_step,
        optimizer_updates={str(key): int(value) for key, value in updates.items()},
        phase=phase,
        history=list(history),
        completed_boundaries=list(completed),
        step_log=list(step_log),
        boundary=None if boundary is None else dict(boundary),
        scheduler_steps=int(state.get("scheduler_steps", 0)),
        best_pesq=None if best is None else float(best),
        best_epoch=best_epoch,
        best_logical_step=best_step,
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )
