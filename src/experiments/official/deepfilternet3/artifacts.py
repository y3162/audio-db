"""DeepFilterNet3 inference files. These are not resume checkpoints.

``valid.loss.best.pt`` stores the exported generator when validation improves.
``rfft.json`` records the real-FFT kernel selected on this host. The trainer
decides whether the loss improved and which kernel record to write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.experiments.errors import RuntimeFailure


def write_best(directory: Path, payload: dict[str, Any]) -> None:
    target = directory / "valid.loss.best.pt"
    temporary = directory / ".valid.loss.best.pt.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeFailure(f"best checkpoint failed: {exc}") from exc


def write_rfft(directory: Path, record: dict[str, Any]) -> None:
    path = Path(directory) / "rfft.json"
    temporary = path.with_name(".rfft.json.tmp")
    try:
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
