"""Atomic training checkpoints.

A checkpoint is written to a temporary file in the destination directory and
then renamed. A killed writer leaves the previous checkpoint in place.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch

from src.evaluation.envinfo import REPO_ROOT
from src.training.errors import TrainingError

_PROTECTED = (
    "data/checkpoints",
    "data/corpora",
    "data/db",
    "data/staging",
)


def assert_output_allowed(path: Path) -> None:
    """Refuse writes into published checkpoints, corpora, the database, or staging."""
    resolved = path.resolve()
    for relative in _PROTECTED:
        protected = (REPO_ROOT / relative).resolve()
        if resolved == protected or _is_relative_to(resolved, protected):
            raise TrainingError(f"refusing to write training output under {relative}: {path}")


def save_atomic(path: Path, payload: dict[str, Any], *, verify_training: bool = False) -> None:
    assert_output_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_output_allowed(path.parent)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if verify_training:
            restored = load_checkpoint(temporary, torch.device("cpu"))
            for key in ("global_step", "epoch", "micro_batches_consumed"):
                if key not in restored or restored[key] != payload[key]:
                    raise TrainingError(f"checkpoint verification failed: {key}")
            for key in ("task", "rng", "dataset", "config", "batch_plan"):
                if key not in restored:
                    raise TrainingError(f"checkpoint verification failed: missing {key}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise TrainingError(f"checkpoint {path} is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("format") != "audio-db-training-v1":
        raise TrainingError(f"checkpoint {path} is not an audio-db training checkpoint")
    return payload


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
