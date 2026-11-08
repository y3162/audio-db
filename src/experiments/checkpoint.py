"""Checkpoint boundary for Phase 1.

Saving and restoring ``audio-db-experiment-v1`` files is Phase 2. This module
only keeps the output directory, refuses protected paths, and locks one run.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Literal

from src.experiments.errors import CheckpointError, ExperimentError

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROTECTED = (
    "data/checkpoints",
    "data/corpora",
    "data/db",
    "data/staging",
)


def assert_output_allowed(path: Path) -> None:
    """Refuse experiment output under published data directories."""

    resolved = path.resolve()
    for relative in _PROTECTED:
        protected = (REPO_ROOT / relative).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise CheckpointError(f"refusing to write experiment output under {relative}: {path}")


class DirectoryLock:
    """Non-blocking exclusive lock on ``directory/.lock``."""

    def __init__(self, directory: Path) -> None:
        self._handle = (directory / ".lock").open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise ExperimentError(f"experiment is already running: {directory}") from exc

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()

    def __del__(self) -> None:
        self.release()


class CheckpointManager:
    """Phase 1 checkpoint handle. File restore and save are refused."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, directory: Path, *, logical_step: int, label: str) -> Path:
        del directory, logical_step, label
        raise CheckpointError("checkpoint save is not implemented")

    def restore(self, path: Path, *, mode: Literal["resume", "initialize"]) -> dict[str, Any]:
        del path, mode
        raise CheckpointError("checkpoint restore is not implemented")


def write_status(directory: Path, status: str) -> None:
    """Replace ``status.json`` in ``directory``."""

    payload = json.dumps({"status": status}) + "\n"
    target = directory / "status.json"
    temporary = directory / ".status.json.tmp"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)
