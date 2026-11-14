"""Checkpoint save policy.

A trainer offers every point where its state is consistent. A selector
answers each offer with a :class:`Decision`: replace the single Latest, write
named Archives, delete named Archives. The checkpoint manager owns the Latest
file name, serialization, verification, and atomic replacement, so a selector
never touches a file.

Every rank builds the same selector and receives the same offers. A selector
must therefore be a deterministic function of its constructor arguments, its
restored state, and the offered points. The manager compares the decisions
and the selector state across ranks before it writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.experiments.errors import BuildError, CheckpointError

# ``step``: one logical step finished inside an epoch.
# ``partial``: a safe point inside an accumulation window; no logical step finished.
# ``boundary_pending``: an epoch or validation boundary opened; its work has not run.
# ``boundary``: the boundary work finished. ``metrics`` holds validation results when it validated.
# ``stop``: a safe stop was requested. The state is the state of the previous offer.
# ``final``: training completed. The state is the state of the previous offer.
EVENTS = ("step", "partial", "boundary_pending", "boundary", "stop", "final")
# Logical steps between Latest replacements when a config does not set
# ``latest_every``. Long runs still save at boundaries, a safe stop, and
# completion. ``latest_every: 1`` is the explicit every-step policy.
# PeriodicArchiveSelector and TopKSelector keep their own default of 1.
DEFAULT_LATEST_EVERY = 1000


@dataclass(frozen=True)
class SavePoint:
    """One point where the trainer state may be saved.

    ``progress`` holds the counters the trainer actually has. ``logical_step``
    stays for selectors and checkpoints that name that counter. A point whose
    ``progress`` omits it is an event-based point.
    """

    event: str
    logical_step: int | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    # True on ``stop`` and ``final`` when Latest already holds exactly this state.
    latest_is_current: bool = False
    progress: Mapping[str, int] = field(default_factory=dict)
    progress_explicit: bool = False

    def __post_init__(self) -> None:
        if self.logical_step is not None and (
            isinstance(self.logical_step, bool)
            or not isinstance(self.logical_step, int)
            or self.logical_step < 0
        ):
            raise CheckpointError("logical_step must be an integer >= 0")
        if not self.progress and not self.progress_explicit:
            if self.logical_step is None:
                raise CheckpointError("save point has no progress")
            progress = {"logical_step": int(self.logical_step)}
        else:
            progress = dict(self.progress)
        cleaned: dict[str, int] = {}
        for key, value in progress.items():
            if not isinstance(key, str) or not key:
                raise CheckpointError("progress keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CheckpointError(f"progress {key} must be an integer >= 0")
            cleaned[key] = int(value)
        if "logical_step" in cleaned:
            if self.logical_step is None:
                object.__setattr__(self, "logical_step", cleaned["logical_step"])
            elif cleaned["logical_step"] != int(self.logical_step):
                raise CheckpointError("logical_step does not match progress")
        object.__setattr__(self, "progress", cleaned)


@dataclass(frozen=True)
class Decision:
    """What to write and delete at one save point.

    ``archive`` and ``delete`` are file names inside the experiment's
    ``archive/`` directory. Latest is not addressable by name.
    """

    latest: bool = False
    archive: tuple[str, ...] = ()
    delete: tuple[str, ...] = ()


class Selector:
    """Base class. Subclasses override :meth:`decide`.

    ``decide`` may update the selector's own state. The state returned by
    ``state_dict`` after ``decide`` is stored inside the checkpoints that the
    decision writes, so a resume continues the same retention table.
    """

    def decide(self, point: SavePoint) -> Decision:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise CheckpointError(f"{type(self).__name__} has no state to restore")


class StandardSelector(Selector):
    """Keep Latest only.

    Latest is replaced every ``latest_every`` logical steps, at each boundary,
    on a safe stop, and when training completes. The default interval is
    :data:`DEFAULT_LATEST_EVERY`, so omitting it does not save every step.
    ``latest_every: 1`` does. ``partial`` also replaces Latest at safe points
    inside an accumulation window.
    """

    checkpoint_class_id = "src.experiments.selectors.StandardSelector"

    def __init__(
        self,
        latest_every: int = DEFAULT_LATEST_EVERY,
        partial: bool = False,
        progress_key: str = "logical_step",
    ) -> None:
        self.latest_every = _positive_int("latest_every", latest_every)
        if not isinstance(partial, bool):
            raise BuildError("partial must be a bool")
        if not isinstance(progress_key, str) or not progress_key:
            raise BuildError("progress_key must be a non-empty string")
        self.partial = partial
        self.progress_key = progress_key

    def decide(self, point: SavePoint) -> Decision:
        return Decision(latest=self.wants_latest(point))

    def counter(self, point: SavePoint) -> int:
        """Read the configured progress key. A missing key is an error."""

        if self.progress_key not in point.progress:
            raise CheckpointError(f"save point is missing progress {self.progress_key}")
        return int(point.progress[self.progress_key])

    def wants_latest(self, point: SavePoint) -> bool:
        if point.event == "step":
            return self.counter(point) % self.latest_every == 0
        if point.event in ("boundary_pending", "boundary"):
            return True
        if point.event in ("stop", "final"):
            return not point.latest_is_current
        if point.event == "partial":
            return self.partial
        raise CheckpointError(f"unknown save point event {point.event!r}")


class PeriodicArchiveSelector(StandardSelector):
    """Latest as :class:`StandardSelector`, plus an Archive every ``archive_every`` steps.

    The Archive of a step that ends an epoch is written when the boundary
    opens and replaced when the boundary finishes. ``keep_last`` bounds the
    number of Archives; ``None`` keeps all of them. ``name`` is formatted with
    ``step``.
    """

    checkpoint_class_id = "src.experiments.selectors.PeriodicArchiveSelector"

    def __init__(
        self,
        latest_every: int = 1,
        archive_every: int = 1,
        keep_last: int | None = None,
        name: str = "step_{step:08d}.pt",
        partial: bool = False,
        progress_key: str = "logical_step",
    ) -> None:
        # ``latest_every`` stays 1 when omitted. This class is not the standard
        # policy, and existing configs rely on that interval.
        super().__init__(latest_every=latest_every, partial=partial, progress_key=progress_key)
        self.archive_every = _positive_int("archive_every", archive_every)
        self.keep_last = None if keep_last is None else _positive_int("keep_last", keep_last)
        self.name = _name_pattern(name)
        self._kept: list[str] = []

    def decide(self, point: SavePoint) -> Decision:
        latest = self.wants_latest(point)
        archived = point.event in ("step", "boundary_pending", "boundary") or (
            point.event == "partial" and self.partial
        )
        if not archived:
            return Decision(latest=latest)
        step = self.counter(point)
        if step % self.archive_every != 0:
            return Decision(latest=latest)
        name = self.name.format(step=step)
        if name in self._kept:
            self._kept.remove(name)
        self._kept.append(name)
        dropped: list[str] = []
        if self.keep_last is not None:
            while len(self._kept) > self.keep_last:
                dropped.append(self._kept.pop(0))
        return Decision(latest=latest, archive=(name,), delete=tuple(dropped))

    def state_dict(self) -> dict[str, Any]:
        return {"kept": list(self._kept)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        kept = state.get("kept")
        if not isinstance(kept, list) or not all(isinstance(item, str) for item in kept):
            raise CheckpointError("PeriodicArchiveSelector state is invalid")
        self._kept = list(kept)


class TopKSelector(StandardSelector):
    """Latest as :class:`StandardSelector`, plus the ``k`` best boundaries as Archives.

    A boundary is ranked by ``metrics[metric]``. A boundary without that
    metric, or with a non-finite value, is not ranked. On a tie the earlier
    step stays ahead. ``name`` is formatted with ``step``.
    """

    checkpoint_class_id = "src.experiments.selectors.TopKSelector"

    def __init__(
        self,
        metric: str,
        mode: str = "min",
        k: int = 1,
        latest_every: int = 1,
        name: str = "best_{step:08d}.pt",
        partial: bool = False,
        progress_key: str = "logical_step",
    ) -> None:
        # Same as PeriodicArchiveSelector: an omitted interval stays every step.
        super().__init__(latest_every=latest_every, partial=partial, progress_key=progress_key)
        if not isinstance(metric, str) or not metric:
            raise BuildError("metric must be a non-empty string")
        if mode not in {"min", "max"}:
            raise BuildError("mode must be min or max")
        self.metric = metric
        self.mode = mode
        self.k = _positive_int("k", k)
        self.name = _name_pattern(name)
        self._entries: list[dict[str, Any]] = []

    def decide(self, point: SavePoint) -> Decision:
        latest = self.wants_latest(point)
        # A bare name such as ``loss`` stays on the official boundary. A
        # namespaced name such as ``validation.wer`` is also ranked when the
        # research evaluator attached it to a finished step or to completion.
        ranked = point.event == "boundary" or ("." in self.metric and point.event in {"step", "final"})
        if not ranked or self.metric not in point.metrics:
            return Decision(latest=latest)
        value = float(point.metrics[self.metric])
        if not math.isfinite(value):
            return Decision(latest=latest)
        step = self.counter(point)
        name = self.name.format(step=step)
        entries = [entry for entry in self._entries if entry["name"] != name]
        entries.append({"name": name, "step": int(step), "value": value})
        sign = 1.0 if self.mode == "min" else -1.0
        entries.sort(key=lambda entry: (sign * entry["value"], entry["step"]))
        kept, dropped = entries[: self.k], entries[self.k :]
        self._entries = kept
        selected = any(entry["name"] == name for entry in kept)
        return Decision(
            latest=latest,
            archive=(name,) if selected else (),
            delete=tuple(entry["name"] for entry in dropped if entry["name"] != name),
        )

    def state_dict(self) -> dict[str, Any]:
        return {"entries": [dict(entry) for entry in self._entries]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        entries = state.get("entries")
        if not isinstance(entries, list):
            raise CheckpointError("TopKSelector state is invalid")
        restored: list[dict[str, Any]] = []
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("name"), str)
                or isinstance(entry.get("step"), bool)
                or not isinstance(entry.get("step"), int)
                or not isinstance(entry.get("value"), (int, float))
            ):
                raise CheckpointError("TopKSelector state is invalid")
            restored.append({"name": entry["name"], "step": entry["step"], "value": float(entry["value"])})
        self._entries = restored


def scalar_metrics(summary: Mapping[str, Any] | None) -> dict[str, float]:
    """Finite numbers of a validation summary, for a ``boundary`` save point.

    A top-level number is kept under its key. ``{"status": "ok", "value": x}``
    is kept as ``x``. Anything else, including ``na`` and ``unavailable``, is
    left out; a missing metric is not a zero.
    """

    found: dict[str, float] = {}
    for key, value in (summary or {}).items():
        if isinstance(value, Mapping):
            if value.get("status") != "ok":
                continue
            value = value.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        found[str(key)] = float(value)
    return found


def build_selector(class_path: str, kwargs: Mapping[str, Any]) -> Selector:
    """Construct the selector named in the experiment YAML."""

    from src.experiments.core.factory import ComponentFactory, ComponentSpec

    try:
        selector = ComponentFactory().create(ComponentSpec(class_path, dict(kwargs)))
    except BuildError as exc:
        raise BuildError(f"checkpoint.selector: {exc}") from exc
    for method in ("decide", "state_dict", "load_state_dict"):
        if not callable(getattr(selector, method, None)):
            raise BuildError(f"checkpoint.selector {class_path} does not implement {method}")
    return selector  # type: ignore[return-value]


def selector_identity(selector: object) -> str:
    """Class identity stored in a checkpoint. It does not depend on the import path used in YAML."""

    from src.experiments.core.identity import policy_identity

    return policy_identity(selector)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _name_pattern(pattern: object) -> str:
    if not isinstance(pattern, str) or "{step" not in pattern:
        raise BuildError("name must be a format string that contains {step}")
    try:
        pattern.format(step=0)
    except (IndexError, KeyError, ValueError) as exc:
        raise BuildError(f"name is not a valid format string: {exc}") from exc
    return pattern
