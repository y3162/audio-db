"""Choose a best checkpoint from metric events. This policy does not write files."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.metrics.base import finite_number

_SOURCES = frozenset({"train", "train_eval", "validation"})
_HISTORY_LIMIT = 32


class BestMetricSelection:
    """Keep the best finite observation of ``source.name``.

    Ties keep the earlier logical step unless ``tie_break`` is ``later``.
    Missing, non-finite, and incomplete values are counted as rejections and
    never become the best. ``k`` is the length of the ranking metadata.
    Archive files for that ranking stay on ``TopKSelector``.
    """

    checkpoint_class_id = "src.experiments.selection.BestMetricSelection"

    def __init__(self, metric: str, mode: str, tie_break: str = "earlier", k: int = 1) -> None:
        if not isinstance(metric, str) or metric.count(".") != 1:
            raise BuildError("metric must be source.name")
        source, name = metric.split(".")
        if source not in _SOURCES or not name:
            raise BuildError("metric must be source.name")
        if mode not in {"min", "max"}:
            raise BuildError("mode must be min or max")
        if tie_break not in {"earlier", "later"}:
            raise BuildError("tie_break must be earlier or later")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise BuildError("k must be a positive integer")
        self.metric = metric
        self.source = source
        self.name = name
        self.mode = mode
        self.tie_break = tie_break
        self.k = k
        self.best: float | None = None
        self.best_step: int | None = None
        self.best_epoch: int | None = None
        self.observations = 0
        self.rejected = 0
        self.history: list[dict[str, Any]] = []
        self.ranking: list[dict[str, Any]] = []
        self.last_event: dict[str, Any] | None = None
        self.provenance: dict[str, Any] = {}
        self.best_observation: int | None = None
        self._improved = False

    def observe(
        self,
        *,
        source: str,
        name: str,
        logical_step: int | None,
        epoch: int | None,
        status: str,
        value: object,
        provenance: Mapping[str, Any] | None = None,
    ) -> bool:
        """Record one result. Return True when this call changes the selected best.

        ``logical_step`` is the trainer's counter when it has one. Without it,
        ties use this policy's own observation count. That count is not written
        into the checkpoint as a training step.
        """

        if source != self.source or name != self.name:
            return False
        self._improved = False
        if logical_step is not None and (
            isinstance(logical_step, bool) or not isinstance(logical_step, int) or logical_step < 0
        ):
            raise RuntimeFailure("selection logical_step must be a non-negative integer")
        number = finite_number(value)
        if status != "ok" or number is None:
            self.rejected += 1
            self.last_event = {
                "epoch": epoch,
                "logical_step": logical_step,
                "name": name,
                "source": source,
                "status": status,
            }
            return False
        self.observations += 1
        row: dict[str, Any] = {"epoch": epoch, "logical_step": logical_step, "value": number}
        if logical_step is None:
            row["observation"] = self.observations
        self.history.append(row)
        if len(self.history) > _HISTORY_LIMIT:
            self.history = self.history[-_HISTORY_LIMIT:]
        self.ranking = _insert(self.ranking, row, mode=self.mode, tie_break=self.tie_break, k=self.k)
        self.last_event = {**row, "name": name, "source": source, "status": "ok"}
        leader = self.ranking[0]
        leader_step = leader["logical_step"]
        leader_observation = leader.get("observation")
        changed = (
            self.best != leader["value"]
            or self.best_step != leader_step
            or self.best_observation != leader_observation
        )
        if changed:
            self.best = float(leader["value"])
            self.best_step = None if leader_step is None else int(leader_step)
            self.best_epoch = None if leader["epoch"] is None else int(leader["epoch"])
            self.best_observation = None if leader_observation is None else int(leader_observation)
            self.provenance = dict(provenance or {})
            self._improved = True
        return self._improved

    def state_dict(self) -> dict[str, Any]:
        payload = {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "class_id": self.checkpoint_class_id,
            "history": [dict(row) for row in self.history],
            "k": self.k,
            "last_event": None if self.last_event is None else dict(self.last_event),
            "metric": self.metric,
            "mode": self.mode,
            "observations": self.observations,
            "provenance": dict(self.provenance),
            "ranking": [dict(row) for row in self.ranking],
            "rejected": self.rejected,
            "tie_break": self.tie_break,
        }
        if self.best_observation is not None:
            payload["best_observation"] = self.best_observation
        return payload

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise RuntimeFailure("model selection state is not a mapping")
        expected = {
            "best",
            "best_epoch",
            "best_step",
            "class_id",
            "history",
            "k",
            "last_event",
            "metric",
            "mode",
            "observations",
            "provenance",
            "ranking",
            "rejected",
            "tie_break",
        }
        extra = set(state) - expected
        if extra - {"best_observation"}:
            raise RuntimeFailure("model selection state keys do not match")
        if state["class_id"] != self.checkpoint_class_id or state["metric"] != self.metric:
            raise RuntimeFailure("model selection state does not match this component")
        if state["mode"] != self.mode or state["tie_break"] != self.tie_break or state["k"] != self.k:
            raise RuntimeFailure("model selection settings do not match this component")
        best = state["best"]
        step = state["best_step"]
        epoch = state["best_epoch"]
        observation = state.get("best_observation")
        if best is None and (step is not None or observation is not None):
            raise RuntimeFailure("model selection best and step disagree")
        if best is not None and step is None and observation is None:
            raise RuntimeFailure("model selection best and step disagree")
        if best is not None and step is not None and (
            finite_number(best) is None or isinstance(step, bool) or not isinstance(step, int)
        ):
            raise RuntimeFailure("model selection best is invalid")
        if best is not None and step is None and finite_number(best) is None:
            raise RuntimeFailure("model selection best is invalid")
        if observation is not None and (isinstance(observation, bool) or not isinstance(observation, int) or observation < 1):
            raise RuntimeFailure("model selection observation is invalid")
        if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int)):
            raise RuntimeFailure("model selection epoch is invalid")
        observations = state["observations"]
        rejected = state["rejected"]
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (observations, rejected)):
            raise RuntimeFailure("model selection counts are invalid")
        ranking = _rows(state["ranking"], "ranking")
        history = _rows(state["history"], "history")
        if not isinstance(state["provenance"], dict) or (
            state["last_event"] is not None and not isinstance(state["last_event"], dict)
        ):
            raise RuntimeFailure("model selection provenance is invalid")
        if ranking and float(ranking[0]["value"]) != float(best):
            raise RuntimeFailure("model selection ranking does not match the best")
        if ranking and ranking[0]["logical_step"] != step:
            raise RuntimeFailure("model selection ranking does not match the best")
        if ranking and ranking[0].get("observation") != observation:
            raise RuntimeFailure("model selection ranking does not match the best")
        self.best = None if best is None else float(best)
        self.best_step = None if step is None else int(step)
        self.best_observation = None if observation is None else int(observation)
        self.best_epoch = None if epoch is None else int(epoch)
        self.observations = int(observations)
        self.rejected = int(rejected)
        self.history = history
        self.ranking = ranking
        self.last_event = None if state["last_event"] is None else dict(state["last_event"])
        self.provenance = dict(state["provenance"])
        self._improved = False


def _better(value: float, step: int, other: float, other_step: int, *, mode: str, tie_break: str) -> bool:
    if mode == "min":
        if value < other:
            return True
        if value > other:
            return False
    else:
        if value > other:
            return True
        if value < other:
            return False
    if tie_break == "later":
        return step > other_step
    return step < other_step


def _order_key(row: Mapping[str, Any]) -> int:
    if row["logical_step"] is None:
        return int(row["observation"])
    return int(row["logical_step"])


def _same_point(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["logical_step"] is None or right["logical_step"] is None:
        return left.get("observation") == right.get("observation") and left["logical_step"] is right["logical_step"]
    return left["logical_step"] == right["logical_step"]


def _insert(ranking: list[dict[str, Any]], row: dict[str, Any], *, mode: str, tie_break: str, k: int) -> list[dict[str, Any]]:
    kept = [item for item in ranking if not _same_point(item, row)]
    kept.append(dict(row))
    kept.sort(key=_order_key)
    ordered: list[dict[str, Any]] = []
    for item in kept:
        placed = False
        for index, current in enumerate(ordered):
            if _better(
                float(item["value"]),
                _order_key(item),
                float(current["value"]),
                _order_key(current),
                mode=mode,
                tie_break=tie_break,
            ):
                ordered.insert(index, item)
                placed = True
                break
        if not placed:
            ordered.append(item)
    return ordered[:k]


def _rows(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RuntimeFailure(f"model selection {label} is invalid")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or finite_number(item.get("value")) is None:
            raise RuntimeFailure(f"model selection {label} is invalid")
        step = item.get("logical_step")
        epoch = item.get("epoch")
        observation = item.get("observation")
        if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int)):
            raise RuntimeFailure(f"model selection {label} is invalid")
        if step is None:
            if isinstance(observation, bool) or not isinstance(observation, int) or observation < 1:
                raise RuntimeFailure(f"model selection {label} is invalid")
            rows.append(
                {
                    "epoch": epoch,
                    "logical_step": None,
                    "observation": int(observation),
                    "value": float(item["value"]),
                }
            )
            continue
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise RuntimeFailure(f"model selection {label} is invalid")
        rows.append({"epoch": epoch, "logical_step": int(step), "value": float(item["value"])})
    return rows


def selection_is_finite(value: object) -> bool:
    return finite_number(value) is not None and math.isfinite(float(value))  # type: ignore[arg-type]
