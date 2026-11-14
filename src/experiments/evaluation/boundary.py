"""Connect research evaluators and model selection at a safe trainer boundary.

Official validation, schedulers, and official best files run before this
module. A session with neither evaluators nor a model selection returns
without touching RNG, weights, or the checkpoint metrics.

The session does not read the trainer. The trainer passes the data, the
inference module, the predict function, the restored flag, the phase name,
and the progress it actually publishes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.experiments.errors import RuntimeFailure
from src.experiments.evaluation.evaluator import AudioPairEvaluator
from src.experiments.metrics.base import MetricResult, finite_number
from src.experiments.selection.export import export_best
from src.experiments.stopping import MetricEvent

LOG_NAME = "research_metrics.jsonl"


class ResearchSession:
    """Research evaluators and selection for one trainer.

    ``module`` is the inference module the trainer chooses. ``predict`` is an
    optional waveform function. ``data`` supplies lineage and is not advanced
    by this session.
    """

    def __init__(
        self,
        evaluators: Any,
        selection: Any,
        *,
        data: Any,
        module: Any,
        predict: Any = None,
    ) -> None:
        if evaluators is None:
            stored: tuple[Any, ...] = ()
        elif isinstance(evaluators, (list, tuple)):
            stored = tuple(evaluators)
        else:
            stored = (evaluators,)
        for evaluator in stored:
            if not isinstance(evaluator, AudioPairEvaluator):
                raise RuntimeFailure("evaluators must be AudioPairEvaluator instances")
        self.evaluators = stored
        self.selection = selection
        self.data = data
        self.module = module
        self.predict = predict if callable(predict) else None

    @property
    def attached(self) -> bool:
        return bool(self.evaluators) or self.selection is not None

    def rewind(self, directory: Path, *, restored: bool, phase: str, progress: Mapping[str, int]) -> None:
        """Drop research rows past a restored counter. No counter means no rewind."""

        if not self.attached or not restored:
            return
        step = progress.get("logical_step")
        if step is None:
            return
        drop_current = str(phase).startswith("pending")
        _rewind(Path(directory) / LOG_NAME, int(step), drop_current=drop_current)

    def metrics(
        self,
        runtime: Any,
        checkpoint: Any,
        *,
        kind: str,
        epoch: int | None,
        progress: Mapping[str, int] | None = None,
        official: Mapping[str, float] | None = None,
        stopping: Any = None,
    ) -> dict[str, float]:
        """Run due research evaluators and return finite namespaced metrics.

        The returned mapping is merged into the save point. Official keys are
        not replaced. An empty mapping means the caller can keep its original
        offer. A missing ``logical_step`` stays missing; this method does not
        invent 0.
        """

        if not self.attached:
            return {}
        checked = _checked_progress(progress, epoch)
        logical_step = checked.get("logical_step")
        due = [
            evaluator
            for evaluator in self.evaluators
            if evaluator.due(logical_step=logical_step, epoch=epoch, kind=kind, progress=checked)
        ]
        if kind != "boundary" and not due:
            return {}
        directory = Path(checkpoint.directory)
        provenance = _provenance(self.data, kind)
        bridged: dict[str, float] = {}
        if kind == "boundary":
            for key, value in (official or {}).items():
                number = finite_number(value)
                if number is None:
                    continue
                bridged[f"validation.{key}"] = number
                _selection_official(self, key, number, logical_step, epoch, provenance)
        results: list[tuple[AudioPairEvaluator, list[MetricResult]]] = []
        predict = _predict(self)
        for evaluator in due:
            outcome = evaluator.evaluate(
                runtime,
                logical_step=logical_step,
                epoch=epoch,
                predict=predict,
                progress=checked,
            )
            results.append((evaluator, outcome))
            for result in outcome:
                namespaced = f"{evaluator.source}.{result.name}"
                if namespaced in bridged:
                    raise RuntimeFailure(
                        f"research metric {namespaced} collides with an official validation metric"
                    )
                number = finite_number(result.value) if result.status == "ok" else None
                if number is not None:
                    bridged[namespaced] = number
                _selection_result(self, evaluator.source, result, logical_step, epoch, provenance)
                _deliver_stopping(stopping, evaluator.source, result, logical_step)
        if results or (kind == "boundary" and bridged):
            _write_log(runtime, directory, checkpoint, kind, epoch, logical_step, provenance, results, bridged)
            _export_if_needed(self, runtime, directory, checkpoint)
            _agree(runtime, self)
        return bridged

    def score(
        self,
        runtime: Any,
        checkpoint: Any,
        *,
        kind: str,
        epoch: int | None,
        logical_step: int,
        official: Mapping[str, float] | None = None,
        stopping: Any = None,
    ) -> dict[str, float]:
        """Score with the trainer's ``logical_step``. Callers that have no counter use :meth:`metrics`."""

        return self.metrics(
            runtime,
            checkpoint,
            kind=kind,
            epoch=epoch,
            progress={"logical_step": int(logical_step)},
            official=official,
            stopping=stopping,
        )


def _checked_progress(progress: Mapping[str, int] | None, epoch: int | None) -> dict[str, int]:
    checked: dict[str, int] = {}
    if progress is not None:
        if not isinstance(progress, Mapping):
            raise RuntimeFailure("progress must be a mapping")
        for key, value in progress.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeFailure(f"progress {key} must be an integer >= 0")
            checked[str(key)] = int(value)
    if epoch is not None and "epoch" not in checked:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise RuntimeFailure("epoch must be an integer")
        checked["epoch"] = int(epoch)
    return checked


def _predict(session: ResearchSession):
    method = session.predict
    if not callable(method):
        return None

    def predict(inputs: Any, targets: Any) -> Any:
        return method(inputs, targets)

    return predict


def _selection_official(
    session: ResearchSession,
    key: str,
    value: float,
    logical_step: int | None,
    epoch: int | None,
    provenance: Mapping[str, Any],
) -> None:
    selection = session.selection
    if selection is None:
        return
    selection.observe(
        source="validation",
        name=str(key),
        logical_step=logical_step,
        epoch=epoch,
        status="ok",
        value=value,
        provenance=provenance,
    )


def _selection_result(
    session: ResearchSession,
    source: str,
    result: MetricResult,
    logical_step: int | None,
    epoch: int | None,
    provenance: Mapping[str, Any],
) -> None:
    selection = session.selection
    if selection is None or selection.source != source or selection.name != result.name:
        return
    selection.observe(
        source=source,
        name=result.name,
        logical_step=logical_step,
        epoch=epoch,
        status=result.status,
        value=result.value,
        provenance=provenance,
    )


def _deliver_stopping(stopping: Any, source: str, result: MetricResult, logical_step: int | None) -> None:
    number = finite_number(result.value) if result.status == "ok" else None
    if stopping is None or number is None:
        return
    stopping.observe(
        MetricEvent(
            source,
            logical_step,
            {result.name: number},
            expected_metrics=frozenset({result.name}),
        )
    )


def _export_if_needed(session: ResearchSession, runtime: Any, directory: Path, checkpoint: Any) -> None:
    selection = session.selection
    if selection is None or not getattr(selection, "_improved", False):
        return
    export_best(runtime, directory, session.module, selection, config_sha256=_config_hash(checkpoint))


def _agree(runtime: Any, session: ResearchSession) -> None:
    if int(getattr(runtime, "world_size", 1)) <= 1:
        return
    payload = {
        "evaluators": [evaluator.state_dict() for evaluator in session.evaluators],
        "selection": None if session.selection is None else session.selection.state_dict(),
    }
    gathered = runtime.gather_objects(payload)
    if any(item != payload for item in gathered):
        raise RuntimeFailure("ranks disagree on research evaluation or model selection")


def _provenance(data: Any, kind: str) -> dict[str, Any]:
    recorded: dict[str, Any] = {"independent": "UNKNOWN", "kind": kind, "selection_uses_test": None}
    method = getattr(data, "lineage", None)
    if not callable(method):
        recorded["status"] = "missing"
        return recorded
    try:
        details = method()
    except Exception as exc:
        recorded["status"] = "error"
        recorded["reason"] = f"{type(exc).__name__}: {exc}"
        return recorded
    if not isinstance(details, dict):
        recorded["status"] = "missing"
        return recorded
    for key in ("claim_label", "dataset_id", "profile", "protocol_id", "selection_uses_test", "snapshot_hash"):
        if key in details:
            recorded[key] = details[key]
    uses_test = details.get("selection_uses_test")
    if isinstance(uses_test, bool):
        recorded["selection_uses_test"] = uses_test
    recorded["status"] = "recorded"
    recorded["independent"] = False if recorded["selection_uses_test"] is True else "UNKNOWN"
    return recorded


def _write_log(
    runtime: Any,
    directory: Path,
    checkpoint: Any,
    kind: str,
    epoch: int | None,
    logical_step: int | None,
    provenance: Mapping[str, Any],
    results: list[tuple[Any, list[MetricResult]]],
    bridged: Mapping[str, float],
) -> None:
    if int(getattr(runtime, "rank", 0)) != 0:
        _barrier(runtime)
        return
    digest = _config_hash(checkpoint)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for evaluator, outcome in results:
        for result in outcome:
            key = f"{evaluator.source}.{result.name}"
            seen.add(key)
            row = result.as_record()
            row.update(
                {
                    "config_sha256": digest,
                    "epoch": epoch,
                    "kind": kind,
                    "provenance": dict(provenance),
                    "record": "metric",
                    "source": evaluator.source,
                }
            )
            if logical_step is not None:
                row["logical_step"] = logical_step
            rows.append(row)
    for key, value in bridged.items():
        if key in seen or "." not in key:
            continue
        source, name = key.split(".", 1)
        official = {
            "config_sha256": digest,
            "count": None,
            "epoch": epoch,
            "excluded": 0,
            "failures": 0,
            "kind": kind,
            "metric": name,
            "provenance": dict(provenance),
            "record": "official",
            "source": source,
            "status": "ok",
            "value": value,
        }
        if logical_step is not None:
            official["logical_step"] = logical_step
        rows.append(official)
    path = directory / LOG_NAME
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _barrier(runtime)


def _config_hash(checkpoint: Any) -> str | None:
    config = getattr(checkpoint, "_config", None)
    if config is None:
        return None
    from src.experiments.config import config_sha256

    return config_sha256(config)


def _rewind(path: Path, logical_step: int, *, drop_current: bool) -> None:
    if not path.is_file():
        return
    kept: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeFailure(f"{path.name} line {line_number} is not JSON") from exc
        step = record.get("logical_step") if isinstance(record, dict) else None
        if isinstance(step, bool) or not isinstance(step, int):
            raise RuntimeFailure(f"{path.name} line {line_number} has no logical step")
        if step > logical_step or (drop_current and step == logical_step):
            continue
        kept.append(record)
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def _barrier(runtime: Any) -> None:
    barrier = getattr(runtime, "barrier", None)
    if callable(barrier):
        barrier()
