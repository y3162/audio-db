"""Write the research best weights. The selection policy does not touch files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from src.experiments.errors import RuntimeFailure
from src.experiments.execution.runtime import PreparedModel

DIRECTORY = "research_selection"
WEIGHTS = "best.pt"
METADATA = "best.json"
FORMAT = "audio-db-research-selection-v1"


def export_best(
    runtime: Any,
    directory: Path,
    trainer: Any,
    selection: Any,
    *,
    config_sha256: str | None,
) -> None:
    """Replace the research inference weights when the selected step changed.

    Rank 0 writes ``research_selection/best.pt`` and ``best.json``. Official
    filenames are not used. Other ranks wait at the barrier.
    """

    if selection.best is None or selection.best_step is None:
        _barrier(runtime)
        return
    error: str | None = None
    if int(getattr(runtime, "rank", 0)) == 0:
        try:
            module = _module(trainer)
            payload = {
                "config_sha256": config_sha256,
                "epoch": selection.best_epoch,
                "format": FORMAT,
                "logical_step": int(selection.best_step),
                "metric": selection.metric,
                "mode": selection.mode,
                "model": _cpu_tree(module),
                "provenance": dict(selection.provenance),
                "value": float(selection.best),
            }
            target = directory / DIRECTORY
            target.mkdir(parents=True, exist_ok=True)
            _atomic_torch(target / WEIGHTS, payload)
            record = {key: value for key, value in payload.items() if key != "model"}
            record["weights"] = WEIGHTS
            _atomic_json(target / METADATA, record)
        except (OSError, RuntimeError, TypeError, ValueError, RuntimeFailure) as exc:
            error = f"{type(exc).__name__}: {exc}"
    if int(getattr(runtime, "world_size", 1)) > 1:
        gathered = runtime.gather_objects(error)
        error = next((item for item in gathered if item), None)
    _barrier(runtime)
    if error:
        raise RuntimeFailure(f"research selection export failed: {error}")


def _module(trainer: Any) -> torch.nn.Module:
    for name in ("model", "generator"):
        handle = getattr(trainer, name, None)
        if isinstance(handle, PreparedModel):
            return handle.state_module
        if isinstance(handle, torch.nn.Module):
            return handle
    raise RuntimeFailure("research export needs the trainer model or generator")


def _cpu_tree(module: torch.nn.Module) -> dict[str, Any]:
    method = getattr(module, "export_for_inference", None)
    exported = method() if callable(method) else module.state_dict()
    if not isinstance(exported, dict) or not exported:
        raise RuntimeFailure("inference export is empty")
    return {str(key): _cpu_value(value) for key, value in exported.items()}


def _cpu_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(key): _cpu_value(item) for key, item in value.items()}
    raise RuntimeFailure("inference export contains a value that is not a tensor")


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _barrier(runtime: Any) -> None:
    barrier = getattr(runtime, "barrier", None)
    if callable(barrier):
        barrier()
