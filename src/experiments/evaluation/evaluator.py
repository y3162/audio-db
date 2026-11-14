"""Run research metrics on a data split without changing the training step."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.evaluation.forward import estimate_waveform
from src.experiments.evaluation.schedule import EvaluationSchedule
from src.experiments.execution.runtime import PreparedModel
from src.experiments.metrics.base import MetricResult

_SOURCES = frozenset({"validation", "train_eval"})
_LOG_NAME = "research_metrics.jsonl"


class AudioPairEvaluator:
    """Evaluate waveform pairs on ``validation`` or ``train_eval``.

    The evaluator reads a split iterator, scores the injected metrics, and
    restores mode, RNG, and the training cursor. It does not step an optimizer
    or write an official inference file.
    """

    checkpoint_class_id = "src.experiments.evaluation.AudioPairEvaluator"

    def __init__(
        self,
        *,
        source: str,
        metrics: Sequence[Any],
        model: Any,
        data: Any,
        every: Mapping[str, object] | None = None,
        objective: Any | None = None,
        sample_rate: int = 16000,
    ) -> None:
        if source not in _SOURCES:
            raise BuildError("evaluator source must be validation or train_eval")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise BuildError("sample_rate must be a positive integer")
        if not isinstance(metrics, (list, tuple)) or not metrics:
            raise BuildError("evaluator metrics must be a non-empty list")
        names = [getattr(metric, "name", None) for metric in metrics]
        if any(not isinstance(name, str) or not name for name in names):
            raise BuildError("each metric needs a name")
        if len(set(names)) != len(names):
            raise BuildError("evaluator metrics repeat a name")
        for metric in metrics:
            if not callable(getattr(metric, "reset", None)) or not callable(getattr(metric, "local_stats", None)):
                raise BuildError(f"metric {getattr(metric, 'name', metric)!r} cannot accumulate")
            if not callable(getattr(metric, "result_from", None)):
                raise BuildError(f"metric {metric.name} cannot reduce")
        self.source = source
        self.metrics = tuple(metrics)
        self.model = model
        self.data = data
        self.objective = objective
        self.sample_rate = sample_rate
        self.schedule = EvaluationSchedule(every)
        self._completed: list[tuple[str, str, int]] = []

    def due(
        self,
        *,
        logical_step: int | None = None,
        epoch: int | None = None,
        kind: str,
        progress: Mapping[str, int] | None = None,
    ) -> bool:
        token = self.schedule.token(
            logical_step=logical_step,
            epoch=epoch,
            source=self.source,
            progress=progress,
        )
        if token in self._completed:
            return False
        return self.schedule.due(logical_step=logical_step, epoch=epoch, kind=kind, progress=progress)

    def state_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.checkpoint_class_id,
            "completed": [list(token) for token in self._completed],
            "source": self.source,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise RuntimeFailure("evaluator state is not a mapping")
        if state.get("class_id") != self.checkpoint_class_id or state.get("source") != self.source:
            raise RuntimeFailure("evaluator state does not match this component")
        completed = state.get("completed")
        if not isinstance(completed, list):
            raise RuntimeFailure("evaluator completed list is invalid")
        restored: list[tuple[str, str, int]] = []
        for item in completed:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 3
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                or isinstance(item[2], bool)
                or not isinstance(item[2], int)
            ):
                raise RuntimeFailure("evaluator completed token is invalid")
            restored.append((item[0], item[1], int(item[2])))
        self._completed = restored

    def mark(
        self,
        *,
        logical_step: int | None = None,
        epoch: int | None = None,
        progress: Mapping[str, int] | None = None,
    ) -> None:
        token = self.schedule.token(
            logical_step=logical_step,
            epoch=epoch,
            source=self.source,
            progress=progress,
        )
        if token not in self._completed:
            self._completed.append(token)

    def evaluate(
        self,
        runtime: Any,
        *,
        logical_step: int | None = None,
        epoch: int | None = None,
        predict: Any | None = None,
        progress: Mapping[str, int] | None = None,
    ) -> list[MetricResult]:
        """Score the split. Rank-local statistics are summed before the result is built."""

        module = self.model.state_module if isinstance(self.model, PreparedModel) else self.model
        if not isinstance(module, torch.nn.Module):
            raise RuntimeFailure("evaluator model is not a module")
        cursor = _cursor(self.data)
        modes = _modes(module)
        rng = runtime.capture_rng()
        error: str | None = None
        stats: dict[str, float] | None = None
        try:
            module.eval()
            with torch.inference_mode():
                stats = self._local_stats(runtime, module, epoch, predict)
        except (RuntimeFailure, RuntimeError, ValueError, TypeError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            stats = _zero_stats(self.metrics)
        finally:
            _restore_modes(modes)
            runtime.restore_rng(rng)
            moved = _cursor(self.data)
            if moved != cursor:
                error = error or "research evaluation moved the training cursor"
        payload = {"error": error, "stats": stats}
        gathered = runtime.gather_objects(payload) if int(getattr(runtime, "world_size", 1)) > 1 else [payload]
        failure = next((item["error"] for item in gathered if item.get("error")), None)
        if failure:
            raise RuntimeFailure(f"research evaluation failed: {failure}")
        merged = _sum_stats([item["stats"] for item in gathered])
        results = [metric.result_from(_slice_stats(merged, metric.name)) for metric in self.metrics]
        self.mark(logical_step=logical_step, epoch=epoch, progress=progress)
        return results

    def _local_stats(self, runtime: Any, module: torch.nn.Module, epoch: int | None, predict: Any) -> dict[str, float]:
        for metric in self.metrics:
            metric.reset()
        iterator = _iterate(self.data, self.source, epoch)
        device = runtime.device
        for batch in iterator:
            inputs = batch.inputs.to(device)
            targets = batch.targets.to(device)
            if inputs.ndim != 2 or targets.shape != inputs.shape:
                raise RuntimeFailure("research evaluation expected matching [batch, time] waveforms")
            estimate = _predict(predict, module, inputs, targets)
            if not isinstance(estimate, torch.Tensor) or estimate.ndim != 2 or estimate.shape[0] != inputs.shape[0]:
                raise RuntimeFailure("research estimate is not a [batch, time] waveform")
            rate = _sample_rate(module, batch, self.sample_rate)
            weight = int(inputs.shape[0])
            if _needs_loss(self.metrics):
                loss_value = float(_loss_value(self.objective, estimate, targets).detach())
                for metric in self.metrics:
                    if type(metric).__name__ == "LossMetric":
                        metric.update(loss_value, weight)
            ids = _sample_ids(batch, weight)
            for index in range(weight):
                length = _length(batch, index, int(inputs.shape[-1]))
                reference = targets[index, :length]
                heard = estimate[index, : min(length, int(estimate.shape[-1]))]
                self._score_row(reference, heard, rate, ids[index])
        return _prefixed(self.metrics)

    def _score_row(self, reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int, sample_id: str) -> None:
        for metric in self.metrics:
            kind = type(metric).__name__
            if kind == "LossMetric":
                continue
            if kind == "WerMetric":
                if getattr(metric, "recognizer", None) is None:
                    raise RuntimeFailure("WerMetric requires an injected recognizer")
                metric.update(_transcript(self.data, sample_id), _recognize(metric, estimate, sample_rate))
                continue
            if kind == "SISDRMetric":
                metric.score(reference, estimate)
                continue
            if kind in {"PESQMetric", "STOIMetric", "ESTOIMetric"}:
                metric.score(reference, estimate, sample_rate)
                continue
            raise RuntimeFailure(f"metric {metric.name} has no research scoring path")


def _needs_loss(metrics: Sequence[Any]) -> bool:
    return any(type(metric).__name__ == "LossMetric" for metric in metrics)


def _predict(predict: Any, module: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if callable(predict):
        return predict(inputs, targets)
    return estimate_waveform(module, inputs, targets)


def _loss_value(objective: Any, estimate: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if objective is None:
        raise RuntimeFailure("LossMetric requires an injected objective")
    from src.experiments.training.objectives import call_objective

    result = call_objective(objective, {"estimate": estimate, "target": targets})
    value = result.total
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeFailure("research loss is not a scalar")
    return value


def _iterate(data: Any, source: str, epoch: int | None):
    if epoch is None:
        raise RuntimeFailure(f"{source} evaluation requires an epoch")
    if source == "validation":
        method = getattr(data, "validation_epoch", None)
        if not callable(method):
            raise RuntimeFailure("data has no validation_epoch")
        return method(int(epoch))
    method = getattr(data, "train_eval_epoch", None)
    if not callable(method):
        raise RuntimeFailure(
            "train_eval requires data.train_eval_epoch(epoch). "
            "That iterator must not move the training cursor."
        )
    return method(int(epoch))


def _transcript(data: Any, sample_id: str) -> str | None:
    lookup = getattr(data, "reference_transcript", None)
    if not callable(lookup):
        return None
    value = lookup(sample_id)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeFailure(f"reference transcript for {sample_id} is not a string")
    return value


def _recognize(metric: Any, estimate: torch.Tensor, sample_rate: int) -> str | None:
    recognizer = getattr(metric, "recognizer", None)
    if recognizer is None:
        return None
    try:
        text = recognizer.transcribe(estimate.detach(), sample_rate)
    except Exception:
        return None
    if not isinstance(text, str):
        return None
    return text


def _sample_ids(batch: Any, count: int) -> tuple[str, ...]:
    ids = getattr(batch, "sample_ids", ()) or ()
    if len(ids) == count:
        return tuple(str(item) for item in ids)
    base = str(getattr(batch, "sample_id", "sample"))
    return tuple(f"{base}-{index}" for index in range(count))


def _length(batch: Any, index: int, width: int) -> int:
    lengths = getattr(batch, "lengths", None)
    if lengths is None:
        return width
    value = int(lengths[index])
    if value < 1 or value > width:
        raise RuntimeFailure("research evaluation length is outside the waveform")
    return value


def _sample_rate(module: torch.nn.Module, batch: Any, configured: int) -> int:
    batch_rate = getattr(batch, "sample_rate", None)
    if isinstance(batch_rate, int) and not isinstance(batch_rate, bool) and batch_rate > 0:
        return batch_rate
    module_rate = getattr(module, "sampling_rate", None)
    if module_rate is None:
        architecture = getattr(module, "architecture", None)
        if isinstance(architecture, dict):
            module_rate = architecture.get("sampling_rate", architecture.get("sr"))
    if isinstance(module_rate, int) and not isinstance(module_rate, bool) and module_rate > 0:
        return int(module_rate)
    return configured


def _cursor(data: Any) -> tuple[object, object]:
    return (getattr(data, "epoch", None), getattr(data, "update_index", None))


def _modes(module: torch.nn.Module) -> list[tuple[torch.nn.Module, bool]]:
    return [(child, child.training) for child in module.modules()]


def _restore_modes(modes: list[tuple[torch.nn.Module, bool]]) -> None:
    for child, training in modes:
        child.train(training)


def _zero_stats(metrics: Sequence[Any]) -> dict[str, float]:
    for metric in metrics:
        metric.reset()
    return _prefixed(metrics)


def _prefixed(metrics: Sequence[Any]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for metric in metrics:
        for key, value in metric.local_stats().items():
            merged[f"{metric.name}/{key}"] = float(value)
    return merged


def _slice_stats(merged: Mapping[str, float], name: str) -> dict[str, float]:
    prefix = f"{name}/"
    return {key[len(prefix) :]: value for key, value in merged.items() if key.startswith(prefix)}


def _sum_stats(parts: list[Mapping[str, float]]) -> dict[str, float]:
    if not parts or any(part is None for part in parts):
        raise RuntimeFailure("research evaluation produced no statistics")
    keys = list(parts[0])
    for part in parts[1:]:
        if list(part) != keys:
            raise RuntimeFailure("ranks gathered different research statistics")
    return {key: sum(float(part[key]) for part in parts) for key in keys}
