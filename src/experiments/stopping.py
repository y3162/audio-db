"""Metric events and independent stopping policies.

Events carry unqualified metric names; their source provides the namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Mapping, Protocol


@dataclass(frozen=True)
class MetricEvent:
    source: str
    logical_step: int
    metrics: Mapping[str, float]
    event_id: str | None = None
    expected_metrics: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.source not in {"train", "train_eval", "validation"}:
            raise ValueError("invalid metric source")
        if isinstance(self.logical_step, bool) or not isinstance(self.logical_step, int) or self.logical_step < 0:
            raise ValueError("logical_step must be a non-negative integer")
        if self.expected_metrics is not None and (
            not isinstance(self.expected_metrics, frozenset)
            or not all(isinstance(name, str) and name for name in self.expected_metrics)
        ):
            raise ValueError("expected_metrics must be a frozenset of names")


@dataclass(frozen=True)
class StopDecision:
    stop: bool


class StopPolicy(Protocol):
    def observe(self, event: MetricEvent) -> StopDecision: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


class PatienceStopPolicy:
    """Stop after consecutive non-improving observations of a named metric."""

    def __init__(self, metric: str, mode: str, patience: int, min_delta: float = 0.0, start_after: int = 0):
        if not isinstance(metric, str) or metric.count(".") != 1:
            raise ValueError("metric must be source.name")
        source, name = metric.split(".")
        if source not in {"train", "train_eval", "validation"} or not name:
            raise ValueError("metric must be source.name")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if isinstance(start_after, bool) or not isinstance(start_after, int) or start_after < 0:
            raise ValueError("start_after must be a non-negative integer")
        if isinstance(min_delta, bool) or not isinstance(min_delta, Real) or not isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        self.metric, self.source, self.name = metric, source, name
        self.mode, self.patience, self.min_delta, self.start_after = mode, patience, float(min_delta), start_after
        self.best = None
        self.best_step = None
        self.observations = 0
        self.bad_observations = 0
        self.stopped = False

    def observe(self, event: MetricEvent) -> StopDecision:
        if event.source != self.source:
            return StopDecision(self.stopped)
        if event.expected_metrics is not None and self.name not in event.expected_metrics:
            return StopDecision(self.stopped)
        if self.name not in event.metrics:
            raise ValueError(f"missing metric {self.metric}")
        value = event.metrics[self.name]
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"metric {self.metric} must be finite")
        if self.stopped:
            return StopDecision(True)
        value = float(value)
        self.observations += 1
        if self.observations <= self.start_after:
            return StopDecision(False)
        improved = self.best is None or (self.best - value > self.min_delta if self.mode == "min" else value - self.best > self.min_delta)
        if improved:
            self.best, self.best_step, self.bad_observations = value, event.logical_step, 0
        else:
            self.bad_observations += 1
            self.stopped = self.bad_observations >= self.patience
        return StopDecision(self.stopped)

    def state_dict(self) -> dict:
        return {"best": self.best, "best_step": self.best_step, "observations": self.observations,
                "bad_observations": self.bad_observations, "stopped": self.stopped}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {"best", "best_step", "observations", "bad_observations", "stopped"}:
            raise ValueError("invalid stopping state")
        best, step, count, bad, stopped = (state[k] for k in ("best", "best_step", "observations", "bad_observations", "stopped"))
        if (best is None) != (step is None) or (best is not None and (isinstance(best, bool) or not isinstance(best, Real) or not isfinite(best))):
            raise ValueError("invalid stopping best")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("invalid stopping best step")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (count, bad)) or bad > count:
            raise ValueError("invalid stopping counts")
        if not isinstance(stopped, bool) or stopped != (bad >= self.patience) or (best is None and bad != 0):
            raise ValueError("invalid stopping decision")
        if best is None and count > self.start_after:
            raise ValueError("invalid stopping observation count")
        if best is not None and (count <= self.start_after or bad >= count - self.start_after):
            raise ValueError("invalid stopping observation count")
        self.best, self.best_step, self.observations, self.bad_observations, self.stopped = best, step, count, bad, stopped


class TrainingControl:
    """Deliver events at trainer-declared safe boundaries and synchronize ranks."""

    def __init__(self, policy: StopPolicy, runtime: object, *, supported_sources: tuple[str, ...] = ("validation",)):
        for method in ("observe", "state_dict", "load_state_dict"):
            if not callable(getattr(policy, method, None)):
                raise ValueError(f"stopping policy has no {method}")
        metric = getattr(policy, "metric", None)
        if isinstance(metric, str) and metric.split(".", 1)[0] not in supported_sources:
            raise ValueError(f"this trainer emits only {', '.join(supported_sources)} metrics")
        self.policy = policy
        self.runtime = runtime
        self.stopped = False
        self.last_event = None

    def synchronize(self) -> None:
        """Check restored policy state on every rank before training starts."""
        error = None
        try:
            local = ("ok", self.state_dict())
        except Exception as exc:
            error = exc
            local = ("error", f"{type(exc).__name__}: {exc}")
        gathered = self.runtime.gather_objects(local) if self.runtime.world_size > 1 else [local]
        if error is not None:
            raise error
        if any(item[0] == "error" for item in gathered):
            raise RuntimeError(f"stopping state failed on another rank: {gathered}")
        if any(item != local for item in gathered):
            raise RuntimeError("ranks disagree on stopping policy state")

    def observe(self, event: MetricEvent) -> bool:
        error = None
        try:
            decision = self.policy.observe(event)
            if not isinstance(decision, StopDecision) or not isinstance(decision.stop, bool):
                raise ValueError("stopping policy returned an invalid decision")
            self.stopped = decision.stop
            self.last_event = {"source": event.source, "logical_step": event.logical_step,
                               "metrics": dict(event.metrics), "event_id": event.event_id}
            local = ("ok", self.state_dict())
        except Exception as exc:
            error = exc
            local = ("error", f"{type(exc).__name__}: {exc}")
        gathered = self.runtime.gather_objects(local) if self.runtime.world_size > 1 else [local]
        if error is not None:
            raise error
        if any(item[0] == "error" for item in gathered):
            raise RuntimeError(f"stopping policy failed on another rank: {gathered}")
        if any(item != local for item in gathered):
            raise RuntimeError("ranks disagree on stopping policy state")
        return self.stopped

    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "stopped": self.stopped, "last_event": self.last_event}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {"policy", "stopped", "last_event"}:
            raise ValueError("invalid training control state")
        if not isinstance(state["policy"], dict) or not isinstance(state["stopped"], bool):
            raise ValueError("invalid training control state")
        event = state["last_event"]
        if event is not None and (not isinstance(event, dict) or set(event) != {"source", "logical_step", "metrics", "event_id"}):
            raise ValueError("invalid last metric event")
        self.policy.load_state_dict(state["policy"])
        if hasattr(self.policy, "stopped") and self.policy.stopped != state["stopped"]:
            raise ValueError("stopping policy and control disagree")
        self.stopped, self.last_event = state["stopped"], event


def observe_validation(trainer: object, runtime: object, metrics: Mapping[str, float]) -> None:
    control = getattr(trainer, "training_control", None)
    if control is not None:
        control.observe(MetricEvent("validation", trainer.logical_step, metrics))


def should_stop(trainer: object) -> bool:
    control = getattr(trainer, "training_control", None)
    return control is not None and control.stopped
