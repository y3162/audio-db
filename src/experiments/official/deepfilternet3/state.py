"""Persistent DeepFilterNet3 progress. The loss list is journal-backed and not saved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.experiments.errors import RuntimeFailure

IN_EPOCH = "in_epoch"
PENDING_VALIDATION = "pending_validation"
_PHASES = frozenset({IN_EPOCH, PENDING_VALIDATION})


@dataclass
class _DecodedProgress:
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    schedule_steps: int
    rate_table: list[float]
    decay_table: list[float]
    history: list[Any]
    learning_rates: list[float]
    weight_decays: list[float]
    plateau_count: int
    plateau_metric: float | None
    best_loss: float | None
    best_epoch: int | None
    early_stopped: bool
    boundary: dict[str, Any] | None
    finished_epoch: int | None
    stop_reason: str | None


@dataclass
class DFProgress:
    """Checkpoint fields for one DeepFilterNet3 run, plus the unsaved loss list."""

    best_epoch: int | None
    best_loss: float | None
    boundary: dict[str, Any] | None
    early_stopped: bool
    finished_epoch: int | None
    history: list[dict[str, Any]]
    learning_rates: list[float]
    logical_step: int
    losses: list[float]
    optimizer_updates: dict[str, int]
    phase: str
    plateau_count: int
    plateau_metric: float | None
    schedule_steps: int
    stop_reason: str | None
    weight_decays: list[float]
    _optimizer_id: str

    @classmethod
    def initial(cls, optimizer_id: str, schedule_steps: int) -> DFProgress:
        return cls(
            best_epoch=None,
            best_loss=None,
            boundary=None,
            early_stopped=False,
            finished_epoch=None,
            history=[],
            learning_rates=[],
            logical_step=0,
            losses=[],
            optimizer_updates={optimizer_id: 0},
            phase=IN_EPOCH,
            plateau_count=0,
            plateau_metric=None,
            schedule_steps=schedule_steps,
            stop_reason="initialized",
            weight_decays=[],
            _optimizer_id=optimizer_id,
        )

    def export(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "boundary": self.boundary,
            "early_stopped": self.early_stopped,
            "finished_epoch": self.finished_epoch,
            "history": list(self.history),
            "learning_rates": list(self.learning_rates),
            "logical_step": self.logical_step,
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "plateau_count": self.plateau_count,
            "plateau_metric": self.plateau_metric,
            "schedule_steps": self.schedule_steps,
            "stop_reason": self.stop_reason,
            "weight_decays": list(self.weight_decays),
        }

    def decode(self, state: dict[str, Any], tables: Callable[[int], tuple[list[float], list[float]]]) -> _DecodedProgress:
        return _decode_progress(state, self._optimizer_id, tables)

    def apply(self, decoded: _DecodedProgress) -> None:
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.phase = decoded.phase
        self.schedule_steps = decoded.schedule_steps
        self.history = decoded.history
        self.learning_rates = decoded.learning_rates
        self.weight_decays = decoded.weight_decays
        self.plateau_count = decoded.plateau_count
        self.plateau_metric = decoded.plateau_metric
        self.best_loss = decoded.best_loss
        self.best_epoch = decoded.best_epoch
        self.early_stopped = decoded.early_stopped
        self.boundary = decoded.boundary
        self.finished_epoch = decoded.finished_epoch
        if decoded.stop_reason is not None:
            self.stop_reason = decoded.stop_reason

    def reset_fresh(self) -> None:
        self.logical_step = 0
        self.optimizer_updates = {self._optimizer_id: 0}
        self.losses = []
        self.learning_rates = []
        self.weight_decays = []
        self.history = []
        self.best_loss = None
        self.best_epoch = None
        self.plateau_metric = None
        self.plateau_count = 0
        self.early_stopped = False
        self.phase = IN_EPOCH
        self.finished_epoch = None
        self.boundary = None

    def record_step(self, *, applied: bool, loss: float, learning_rate: float, decay: float) -> None:
        self.logical_step += 1
        if applied:
            self.optimizer_updates[self._optimizer_id] += 1
        self.losses.append(loss)
        self.learning_rates.append(learning_rate)
        self.weight_decays.append(decay)

    def open_boundary(self, epoch: int, schedule_steps: int) -> None:
        self.phase = PENDING_VALIDATION
        self.finished_epoch = epoch
        self.boundary = {"epoch": epoch, "schedule_steps": schedule_steps}


def _decode_progress(
    state: dict[str, Any],
    optimizer_id: str,
    tables: Callable[[int], tuple[list[float], list[float]]],
) -> _DecodedProgress:
    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if phase not in _PHASES:
        raise RuntimeFailure(f"trainer phase {phase!r} is invalid")
    steps = state.get("schedule_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise RuntimeFailure("schedule_steps is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    if optimizer_id not in converted:
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer ids")
    rate_table, decay_table = tables(steps)
    history = list(state.get("history") or [])
    learning_rates = [float(value) for value in state.get("learning_rates") or []]
    weight_decays = [float(value) for value in state.get("weight_decays") or []]
    plateau_count = int(state.get("plateau_count") or 0)
    metric = state.get("plateau_metric")
    plateau_metric = None if metric is None else float(metric)
    best = state.get("best_loss")
    best_loss = None if best is None else float(best)
    best_epoch = state.get("best_epoch")
    decoded_epoch = None if best_epoch is None else int(best_epoch)
    boundary = state.get("boundary")
    decoded_boundary = None if boundary is None else dict(boundary)
    finished = state.get("finished_epoch")
    finished_epoch = None if finished is None else int(finished)
    stop_reason = state.get("stop_reason")
    return _DecodedProgress(
        logical_step=logical_step,
        optimizer_updates=converted,
        phase=str(phase),
        schedule_steps=steps,
        rate_table=rate_table,
        decay_table=decay_table,
        history=history,
        learning_rates=learning_rates,
        weight_decays=weight_decays,
        plateau_count=plateau_count,
        plateau_metric=plateau_metric,
        best_loss=best_loss,
        best_epoch=decoded_epoch,
        early_stopped=bool(state.get("early_stopped", False)),
        boundary=decoded_boundary,
        finished_epoch=finished_epoch,
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
    )
