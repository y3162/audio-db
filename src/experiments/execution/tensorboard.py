"""TensorBoard events derived from ``metrics.jsonl``.

The JSONL file is the log that resume rewinds. This module only mirrors the
finite scalars in that file. It is not read when training state is restored.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from src.experiments.errors import CheckpointError

EVENTS_DIR = "tensorboard"
_SKIP = frozenset({"logical_step", "sample_id"})


def events_directory(directory: Path) -> Path:
    return Path(directory) / EVENTS_DIR


def scalar_series(record: dict[str, Any]) -> tuple[int, list[tuple[str, float]]]:
    """Return ``logical_step`` and the finite numeric fields under it.

    Booleans, strings, and non-finite numbers are not series. Nested mappings
    become ``parent/child`` tags. ``logical_step`` is the step axis, and
    ``sample_id`` is not a scalar.
    """

    step = record.get("logical_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CheckpointError("metrics record has no logical step")
    points: list[tuple[str, float]] = []
    _collect(record, "", points)
    return int(step), points


def _collect(value: Any, prefix: str, points: list[tuple[str, float]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key in _SKIP:
                continue
            tag = f"{prefix}/{key}" if prefix else key
            _collect(item, tag, points)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    number = float(value)
    if math.isfinite(number):
        points.append((prefix, number))


def read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"{path.name} line {line_number} is not JSON") from exc
        if not isinstance(record, dict):
            raise CheckpointError(f"{path.name} line {line_number} has no logical step")
        rows.append(record)
    return rows


def rebuild_tensorboard(directory: Path) -> None:
    """Replace ``tensorboard/`` with one event file for the current JSONL."""

    directory = Path(directory)
    target = events_directory(directory)
    if target.exists():
        shutil.rmtree(target)
    rows = read_metric_rows(directory / "metrics.jsonl")
    if not rows:
        return
    _write_events(target, rows)


def _write_events(target: Path, rows: list[dict[str, Any]]) -> None:
    from torch.utils.tensorboard import SummaryWriter

    target.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(target))
    try:
        for record in rows:
            _add(writer, record)
        writer.flush()
    finally:
        writer.close()


def _add(writer: Any, record: dict[str, Any]) -> None:
    step, points = scalar_series(record)
    for tag, value in points:
        writer.add_scalar(tag, value, step)


def append_tensorboard(directory: Path, record: dict[str, Any], writer: Any | None) -> Any:
    """Append one metrics row. The returned writer stays open for the next row."""

    if writer is None:
        from torch.utils.tensorboard import SummaryWriter

        target = events_directory(directory)
        target.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(target))
    _add(writer, record)
    writer.flush()
    return writer
