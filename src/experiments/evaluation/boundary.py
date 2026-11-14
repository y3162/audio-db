"""Connect research evaluators and model selection at a safe trainer boundary.

Official validation, schedulers, and official best files run before this
module. A trainer with neither evaluators nor a model selection returns
without touching RNG, weights, or the checkpoint metrics.
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


def attach_research(trainer: Any, evaluators: Any, model_selection: Any) -> None:
    """Store optional components. Existing official constructors omit both."""

    if evaluators is None:
        stored: tuple[Any, ...] = ()
    elif isinstance(evaluators, (list, tuple)):
        stored = tuple(evaluators)
    else:
        stored = (evaluators,)
    for evaluator in stored:
        if not isinstance(evaluator, AudioPairEvaluator):
            raise RuntimeFailure("evaluators must be AudioPairEvaluator instances")
    trainer.evaluators = stored
    trainer.model_selection = model_selection


def research_attached(trainer: Any) -> bool:
    return bool(getattr(trainer, "evaluators", None)) or getattr(trainer, "model_selection", None) is not None


def rewind_on_resume(trainer: Any, directory: Path) -> None:
    """Drop research log rows that belong to a step the checkpoint did not commit."""

    if not research_attached(trainer) or not getattr(trainer, "_restored", False):
        return
    phase = str(getattr(trainer, "phase", ""))
    drop_current = phase.startswith("pending")
    _rewind(directory / LOG_NAME, int(trainer.logical_step), drop_current=drop_current)


def final_research_metrics(trainer: Any, checkpoint: Any) -> dict[str, float]:
    """Score ``every.unit: end`` once training has reached a completed result."""

    runtime = getattr(trainer, "_research_runtime", None)
    if runtime is None or not research_attached(trainer):
        return {}
    epoch = getattr(getattr(trainer, "data", None), "epoch", None)
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        epoch = None
    return research_metrics(trainer, runtime, checkpoint, kind="end", epoch=epoch)


def research_metrics(
    trainer: Any,
    runtime: Any,
    checkpoint: Any,
    *,
    kind: str,
    epoch: int | None,
    official: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Run due research evaluators and return finite namespaced metrics.

    The returned mapping is merged into the save point. Official keys are not
    replaced. An empty mapping means the caller can keep its original offer.
    """

    if not research_attached(trainer):
        return {}
    logical_step = int(trainer.logical_step)
    due = [
        evaluator
        for evaluator in trainer.evaluators
        if evaluator.due(logical_step=logical_step, epoch=epoch, kind=kind)
    ]
    if kind != "boundary" and not due:
        return {}
    directory = Path(checkpoint.directory)
    provenance = _provenance(trainer.data, kind)
    bridged: dict[str, float] = {}
    if kind == "boundary":
        for key, value in (official or {}).items():
            number = finite_number(value)
            if number is None:
                continue
            bridged[f"validation.{key}"] = number
            _selection_official(trainer, runtime, key, number, logical_step, epoch, provenance)
    results: list[tuple[AudioPairEvaluator, list[MetricResult]]] = []
    predict = _predict(trainer)
    for evaluator in due:
        outcome = evaluator.evaluate(runtime, logical_step=logical_step, epoch=epoch, predict=predict)
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
            _selection_result(trainer, runtime, evaluator.source, result, logical_step, epoch, provenance)
            _deliver_stopping(trainer, evaluator.source, result, logical_step)
    if results or (kind == "boundary" and bridged):
        _write_log(runtime, directory, checkpoint, kind, epoch, logical_step, provenance, results, bridged)
        _export_if_needed(trainer, runtime, directory, checkpoint)
        _agree(runtime, trainer)
    return bridged


def _selection_official(
    trainer: Any,
    runtime: Any,
    key: str,
    value: float,
    logical_step: int,
    epoch: int | None,
    provenance: Mapping[str, Any],
) -> None:
    selection = getattr(trainer, "model_selection", None)
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
    del runtime


def _selection_result(
    trainer: Any,
    runtime: Any,
    source: str,
    result: MetricResult,
    logical_step: int,
    epoch: int | None,
    provenance: Mapping[str, Any],
) -> None:
    selection = getattr(trainer, "model_selection", None)
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
    del runtime


def _deliver_stopping(trainer: Any, source: str, result: MetricResult, logical_step: int) -> None:
    control = getattr(trainer, "training_control", None)
    number = finite_number(result.value) if result.status == "ok" else None
    if control is None or number is None:
        return
    control.observe(
        MetricEvent(
            source,
            logical_step,
            {result.name: number},
            expected_metrics=frozenset({result.name}),
        )
    )


def _predict(trainer: Any):
    method = getattr(trainer, "research_predict", None)
    if not callable(method):
        return None

    def predict(inputs: Any, targets: Any) -> Any:
        return method(inputs, targets)

    return predict


def _export_if_needed(trainer: Any, runtime: Any, directory: Path, checkpoint: Any) -> None:
    selection = getattr(trainer, "model_selection", None)
    if selection is None or not getattr(selection, "_improved", False):
        return
    export_best(runtime, directory, trainer, selection, config_sha256=_config_hash(checkpoint))


def _agree(runtime: Any, trainer: Any) -> None:
    if int(getattr(runtime, "world_size", 1)) <= 1:
        return
    payload = {
        "evaluators": [evaluator.state_dict() for evaluator in getattr(trainer, "evaluators", ())],
        "selection": None
        if getattr(trainer, "model_selection", None) is None
        else trainer.model_selection.state_dict(),
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
    logical_step: int,
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
                    "logical_step": logical_step,
                    "provenance": dict(provenance),
                    "record": "metric",
                    "source": evaluator.source,
                }
            )
            rows.append(row)
    for key, value in bridged.items():
        if key in seen or "." not in key:
            continue
        source, name = key.split(".", 1)
        rows.append(
            {
                "config_sha256": digest,
                "count": None,
                "epoch": epoch,
                "excluded": 0,
                "failures": 0,
                "kind": kind,
                "logical_step": logical_step,
                "metric": name,
                "provenance": dict(provenance),
                "record": "official",
                "source": source,
                "status": "ok",
                "value": value,
            }
        )
    selection = None
    path = directory / LOG_NAME
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        del selection
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
