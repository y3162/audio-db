"""Atomic training checkpoints.

A checkpoint is written to a temporary file in the destination directory and
then renamed. A killed writer leaves the previous checkpoint in place.
"""

from __future__ import annotations

import os
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


def save_atomic(path: Path, payload: dict[str, Any]) -> None:
    assert_output_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_output_allowed(path.parent)
    temporary = path.parent / f".{path.name}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "audio-db-training-v1":
        raise TrainingError(f"checkpoint {path} is not an audio-db training checkpoint")
    return payload


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
