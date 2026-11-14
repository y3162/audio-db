"""Rank-aware JSONL for training logs.

Trainers choose the record fields and when a line is written. This module
opens, appends, truncates, and rewinds the files. It does not decide a
training step or a checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.experiments.errors import CheckpointError, RuntimeFailure


def rank_of(runtime: Any) -> int:
    return int(getattr(runtime, "rank", 0))


def sample_path(directory: Path, runtime: Any) -> Path | None:
    """Per-rank sample trace. World size 1 keeps sample ids in ``metrics.jsonl``."""

    if int(getattr(runtime, "world_size", 1)) == 1:
        return None
    return Path(directory) / f"samples-rank{rank_of(runtime)}.jsonl"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def rewind_metrics(path: Path, logical_step: int) -> list[float]:
    """Drop metric rows committed past ``logical_step`` and return kept losses."""

    if not path.is_file():
        return []
    kept: list[dict[str, Any]] = []
    losses: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"{path.name} line {line_number} is not JSON") from exc
        step = record.get("logical_step") if isinstance(record, dict) else None
        loss = record.get("loss") if isinstance(record, dict) else None
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(loss, (int, float)):
            raise CheckpointError(f"{path.name} line {line_number} has no logical step")
        if step > logical_step:
            continue
        kept.append(record)
        losses.append(float(loss))
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)
    return losses


def rewind_jsonl_count(path: Path, count: int) -> None:
    """Keep the first ``count`` non-empty lines. Used by the MossFormerGAN skip log."""

    if not path.is_file():
        if count:
            raise RuntimeFailure("skip log is missing")
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) < count:
        raise RuntimeFailure("skip log is shorter than the trainer")
    body = "".join(line + "\n" for line in lines[:count])
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


class StepJournal:
    """``metrics.jsonl``, optional per-rank sample logs, and named sidecars."""

    def __init__(self, directory: Path, runtime: Any) -> None:
        self.directory = Path(directory)
        self.runtime = runtime
        self.rank = rank_of(runtime)
        self.world_size = int(getattr(runtime, "world_size", 1))

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.jsonl"

    @property
    def sample_path(self) -> Path | None:
        if self.world_size == 1:
            return None
        return self.directory / f"samples-rank{self.rank}.jsonl"

    def sidecar(self, name: str) -> Path:
        return self.directory / name

    def begin(
        self,
        *,
        restored: bool,
        logical_step: int,
        samples: bool = True,
        rewind_samples_on_rank0: bool = True,
        expect_rows: int | None = None,
    ) -> list[float]:
        """Truncate on a new run. Rewind on resume and return the kept losses.

        Rank 0 reads ``metrics.jsonl``. Another rank reads its sample log when
        that log is rewound. ``expect_rows`` checks the trainer's own step log
        after a resume. A fresh run returns an empty list.
        """

        losses: list[float] = []
        if restored:
            if self.rank == 0:
                losses = rewind_metrics(self.metrics_path, logical_step)
                if expect_rows is not None and len(losses) != expect_rows:
                    raise RuntimeFailure("metrics.jsonl does not match the trainer log")
            if samples and self.sample_path is not None and (rewind_samples_on_rank0 or self.rank != 0):
                sample_losses = rewind_metrics(self.sample_path, logical_step)
                if expect_rows is not None and len(sample_losses) != expect_rows:
                    raise RuntimeFailure("sample log does not match the trainer log")
                if self.rank != 0:
                    losses = sample_losses
            return losses
        if self.rank == 0:
            self.metrics_path.write_text("", encoding="utf-8")
        if samples and self.sample_path is not None:
            self.sample_path.write_text("", encoding="utf-8")
        return []

    def begin_sidecar(self, name: str, *, restored: bool, keep_lines: int | None = None) -> Path:
        """Truncate or rewind one rank-0 sidecar. Other ranks only learn the path."""

        path = self.sidecar(name)
        if self.rank != 0:
            return path
        if restored:
            if keep_lines is not None:
                rewind_jsonl_count(path, keep_lines)
            return path
        path.write_text("", encoding="utf-8")
        return path

    def write_metrics(self, record: dict[str, Any]) -> None:
        if self.rank == 0:
            append_jsonl(self.metrics_path, record)

    def write_sample(self, record: dict[str, Any]) -> None:
        if self.sample_path is not None:
            append_jsonl(self.sample_path, record)

    def write_sidecar(self, path: Path, record: dict[str, Any]) -> None:
        if self.rank == 0:
            append_jsonl(path, record)
