"""Persistent MossFormerGAN progress, including a partial accumulation window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.experiments.errors import RuntimeFailure
from src.experiments.official.mossformergan.checks import COUNTER_KEYS, empty_counters
from src.experiments.runtime import PreparedModel

IN_EPOCH = "in_epoch"
PENDING_BOUNDARY = "pending_boundary"
_PHASES = frozenset({IN_EPOCH, PENDING_BOUNDARY})


@dataclass
class _Decoded:
    logical_step: int
    optimizer_updates: dict[str, int]
    counters: dict[str, int]
    accu_count: int
    phase: str
    boundary: dict[str, Any] | None
    plateau_history: list[Any]
    completed_boundaries: list[Any]
    step_log: list[Any]
    skip_log: list[Any]
    best_val: float
    stalls: int
    best_snapshot: Any
    stop_training: bool
    halt_reason: str
    epoch_success_start: int
    epoch_update_start: int
    epochs_without_step: int
    window_generator: list[float]
    window_discriminator: list[float]
    window_samples: list[str]
    generator_grads: dict[str, Any]
    discriminator_grads: dict[str, Any]


class ProgressView:
    """Read the persistent counters from the progress object that owns them."""

    @property
    def logical_step(self) -> int:
        return self.progress.logical_step

    @property
    def optimizer_updates(self) -> dict[str, int]:
        return self.progress.optimizer_updates

    @property
    def phase(self) -> str:
        return self.progress.phase

    @property
    def halt_reason(self) -> str:
        return self.progress.halt_reason

    @property
    def stop_training(self) -> bool:
        return self.progress.stop_training

    @property
    def counters(self) -> dict[str, int]:
        return self.progress.counters

    @property
    def _accu_count(self) -> int:
        return self.progress.accu_count

    @property
    def _average_calls(self) -> int:
        return self._micro.average_calls

    @property
    def _stalls(self) -> int:
        return self.progress.stalls

    @property
    def _in_optimizer_pair(self) -> bool:
        return self._guard.active

    @_in_optimizer_pair.setter
    def _in_optimizer_pair(self, value: bool) -> None:
        self._guard.active = bool(value)


@dataclass
class MossProgress:
    """The checkpoint fields for one MossFormerGAN run. Partial gradients stay on the parameters."""

    accu_count: int
    best_snapshot: dict[str, Any] | None
    best_val: float
    boundary: dict[str, Any] | None
    completed_boundaries: list[dict[str, Any]]
    counters: dict[str, int]
    epoch_success_start: int
    epoch_update_start: int
    epochs_without_step: int
    experiment_claim: str
    halt_reason: str
    logical_step: int
    optimizer_updates: dict[str, int]
    phase: str
    plateau_history: list[dict[str, Any]]
    skip_log: list[dict[str, Any]]
    stalls: int
    step_log: list[dict[str, Any]]
    stop_training: bool
    window_discriminator: list[float]
    window_generator: list[float]
    window_samples: list[str]
    _optimizer_ids: tuple[str, str]
    last_skip: dict[str, Any] | None = None

    @classmethod
    def initial(cls, optim_g_id: str, optim_d_id: str, experiment_claim: str) -> MossProgress:
        return cls(
            accu_count=0,
            best_snapshot=None,
            best_val=float("inf"),
            boundary=None,
            completed_boundaries=[],
            counters=empty_counters(),
            epoch_success_start=0,
            epoch_update_start=0,
            epochs_without_step=0,
            experiment_claim=experiment_claim,
            halt_reason="initialized",
            logical_step=0,
            optimizer_updates={optim_g_id: 0, optim_d_id: 0},
            phase=IN_EPOCH,
            plateau_history=[],
            skip_log=[],
            stalls=0,
            step_log=[],
            stop_training=False,
            window_discriminator=[],
            window_generator=[],
            window_samples=[],
            _optimizer_ids=(optim_g_id, optim_d_id),
        )

    def export(self, generator_grads: dict[str, torch.Tensor], discriminator_grads: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            "accu_count": int(self.accu_count),
            "best_snapshot": self.best_snapshot,
            "best_val": self.best_val,
            "boundary": self.boundary,
            "completed_boundaries": list(self.completed_boundaries),
            "counters": dict(self.counters),
            "discriminator_grads": discriminator_grads if self.accu_count else {},
            "epoch_success_start": int(self.epoch_success_start),
            "epoch_update_start": int(self.epoch_update_start),
            "epochs_without_step": int(self.epochs_without_step),
            "experiment_claim": self.experiment_claim,
            "generator_grads": generator_grads if self.accu_count else {},
            "halt_reason": self.halt_reason,
            "logical_step": int(self.logical_step),
            "optimizer_updates": dict(self.optimizer_updates),
            "phase": self.phase,
            "plateau_history": list(self.plateau_history),
            "skip_log": list(self.skip_log),
            "stalls": int(self.stalls),
            "step_log": list(self.step_log),
            "stop_training": bool(self.stop_training),
            "window_discriminator": list(self.window_discriminator),
            "window_generator": list(self.window_generator),
            "window_samples": list(self.window_samples),
        }

    def decode(self, state: dict[str, Any], accumulation: int) -> _Decoded:
        return _decode(state, self.experiment_claim, self._optimizer_ids, accumulation)

    def apply(self, decoded: _Decoded) -> None:
        self.logical_step = decoded.logical_step
        self.optimizer_updates = decoded.optimizer_updates
        self.counters = decoded.counters
        self.accu_count = decoded.accu_count
        self.phase = decoded.phase
        self.boundary = decoded.boundary
        self.plateau_history = decoded.plateau_history
        self.completed_boundaries = decoded.completed_boundaries
        self.step_log = decoded.step_log
        self.skip_log = decoded.skip_log
        self.last_skip = None if not decoded.skip_log else dict(decoded.skip_log[-1])
        self.best_val = decoded.best_val
        self.stalls = decoded.stalls
        self.best_snapshot = decoded.best_snapshot
        self.stop_training = decoded.stop_training
        self.halt_reason = decoded.halt_reason
        self.epoch_success_start = decoded.epoch_success_start
        self.epoch_update_start = decoded.epoch_update_start
        self.epochs_without_step = decoded.epochs_without_step
        self.window_generator = decoded.window_generator
        self.window_discriminator = decoded.window_discriminator
        self.window_samples = decoded.window_samples

    def reset_fresh(self) -> None:
        optim_g_id, optim_d_id = self._optimizer_ids
        self.logical_step = 0
        self.optimizer_updates = {optim_g_id: 0, optim_d_id: 0}
        self.counters = empty_counters()
        self.accu_count = 0
        self.skip_log = []
        self.last_skip = None
        self.step_log = []
        self.completed_boundaries = []
        self.boundary = None
        self.phase = IN_EPOCH
        self.best_val = float("inf")
        self.stalls = 0
        self.best_snapshot = None
        self.stop_training = False
        self.halt_reason = "running"
        self.plateau_history = []
        self.epoch_success_start = 0
        self.epoch_update_start = 0
        self.epochs_without_step = 0
        self.clear_window()

    def clear_window(self) -> None:
        self.window_generator = []
        self.window_discriminator = []
        self.window_samples = []

    def note_closed_pair(self) -> None:
        self.logical_step += 1


def grad_state(prepared: PreparedModel) -> dict[str, torch.Tensor]:
    grads = {}
    for name, parameter in prepared.unwrapped().named_parameters():
        if parameter.grad is not None:
            grads[name] = parameter.grad.detach().cpu().clone()
    return grads


def check_grads(prepared: PreparedModel, grads: object) -> dict[str, torch.Tensor]:
    if not isinstance(grads, dict):
        raise RuntimeFailure("checkpoint gradients are not a mapping")
    named = dict(prepared.unwrapped().named_parameters())
    checked: dict[str, torch.Tensor] = {}
    for name, value in grads.items():
        if name not in named or not isinstance(value, torch.Tensor):
            raise RuntimeFailure(f"checkpoint gradient {name!r} does not match a parameter")
        parameter = named[name]
        if tuple(value.shape) != tuple(parameter.shape):
            raise RuntimeFailure(f"checkpoint gradient {name} has the wrong shape")
        checked[name] = value
    return checked


def write_grads(prepared: PreparedModel, grads: dict[str, torch.Tensor]) -> None:
    named = dict(prepared.unwrapped().named_parameters())
    for parameter in named.values():
        parameter.grad = None
    for name, value in grads.items():
        parameter = named[name]
        parameter.grad = value.detach().to(device=parameter.device, dtype=parameter.dtype)


def _decode(
    state: dict[str, Any],
    experiment_claim: str,
    optimizer_ids: tuple[str, str],
    accumulation: int,
) -> _Decoded:
    logical_step = state.get("logical_step")
    updates = state.get("optimizer_updates")
    counters = state.get("counters")
    phase = state.get("phase")
    if isinstance(logical_step, bool) or not isinstance(logical_step, int) or not isinstance(updates, dict):
        raise RuntimeFailure("trainer state is invalid")
    if not isinstance(counters, dict) or phase not in _PHASES:
        raise RuntimeFailure("trainer counters or phase are invalid")
    if state.get("experiment_claim") != experiment_claim:
        raise RuntimeFailure("experiment claim does not match this trainer")
    if state.get("stop_training") not in {True, False}:
        raise RuntimeFailure("trainer stop flag is invalid")
    converted = {str(key): int(value) for key, value in updates.items()}
    if any(name not in converted for name in optimizer_ids):
        raise RuntimeFailure("trainer optimizer_updates does not include the registered optimizer ids")
    restored_counters = empty_counters()
    for key in COUNTER_KEYS:
        restored_counters[key] = int(counters.get(key) or 0)
    accu = state.get("accu_count")
    if isinstance(accu, bool) or not isinstance(accu, int) or accu < 0 or accu >= accumulation:
        raise RuntimeFailure("accumulator count is invalid")
    generator_grads = state.get("generator_grads") or {}
    discriminator_grads = state.get("discriminator_grads") or {}
    if accu == 0 and (generator_grads or discriminator_grads):
        raise RuntimeFailure("a closed window cannot carry gradients")
    boundary = state.get("boundary")
    history = state.get("plateau_history")
    completed = state.get("completed_boundaries")
    step_log = state.get("step_log")
    skip_log = state.get("skip_log")
    if not all(isinstance(item, list) for item in (history, completed, step_log, skip_log)):
        raise RuntimeFailure("trainer history is invalid")
    window_generator = [float(item) for item in state.get("window_generator") or []]
    window_discriminator = [float(item) for item in state.get("window_discriminator") or []]
    samples = state.get("window_samples") or []
    if not isinstance(samples, list) or not all(isinstance(item, str) for item in samples):
        raise RuntimeFailure("window sample ids are invalid")
    if accu != len(window_generator) or accu != len(window_discriminator):
        raise RuntimeFailure("window losses do not match the accumulator")
    if len(samples) != accu:
        raise RuntimeFailure("window sample ids do not match the accumulator")
    halt = state.get("halt_reason")
    return _Decoded(
        logical_step=logical_step,
        optimizer_updates=converted,
        counters=restored_counters,
        accu_count=accu,
        phase=str(phase),
        boundary=None if boundary is None else dict(boundary),
        plateau_history=list(history),
        completed_boundaries=list(completed),
        step_log=list(step_log),
        skip_log=list(skip_log),
        best_val=float(state.get("best_val", float("inf"))),
        stalls=int(state.get("stalls") or 0),
        best_snapshot=state.get("best_snapshot"),
        stop_training=bool(state.get("stop_training")),
        halt_reason=halt if isinstance(halt, str) else "resumed",
        epoch_success_start=int(state.get("epoch_success_start") or 0),
        epoch_update_start=int(state.get("epoch_update_start") or 0),
        epochs_without_step=int(state.get("epochs_without_step") or 0),
        window_generator=window_generator,
        window_discriminator=window_discriminator,
        window_samples=list(samples),
        generator_grads=generator_grads,
        discriminator_grads=discriminator_grads,
    )
