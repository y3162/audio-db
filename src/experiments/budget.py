"""Planned and observed training volume for one experiment.

Planned figures come from the sampler and the step cap. Observed figures come
from a run directory. Missing observations stay null; they are not estimated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def fixed_pair_kept_samples(
    count: int,
    micro_batch: int,
    group_size: int,
    *,
    keep_partial: bool = True,
) -> int:
    """Utterances that enter training.

    The default keeps a short tail. ``keep_partial`` false drops an incomplete
    micro-batch and an incomplete trailing group.
    """

    if count < 1 or micro_batch < 1 or group_size < 1:
        raise ValueError("sample count and batch sizes must be positive")
    if not isinstance(keep_partial, bool):
        raise ValueError("keep_partial must be a boolean")
    if keep_partial:
        return count
    packed = count // micro_batch
    return (packed - packed % group_size) * micro_batch


def logical_steps(
    samples: int,
    micro_batch: int,
    group_size: int,
    *,
    keep_partial: bool = True,
) -> int:
    if keep_partial:
        if samples < 1 or micro_batch < 1 or group_size < 1:
            raise ValueError("sample count and batch sizes must be positive")
        packed = (samples + micro_batch - 1) // micro_batch
        return (packed + group_size - 1) // group_size
    kept = fixed_pair_kept_samples(samples, micro_batch, group_size, keep_partial=False)
    return kept // (micro_batch * group_size)


def observed_budget(run_directory: str | Path) -> dict[str, Any]:
    """Read counters a finished or partial run actually wrote."""

    directory = Path(run_directory)
    metrics = directory / "metrics.jsonl"
    status_path = directory / "status.json"
    gpu_path = directory / "gpu_memory.json"
    result: dict[str, Any] = {
        "run_directory": str(directory),
        "present": directory.is_dir(),
        "logical_steps": None,
        "optimizer_updates": None,
        "samples_seen": None,
        "validation_count": None,
        "training_status": None,
        "gpu_memory_mib": None,
    }
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        result["training_status"] = status.get("state") or status.get("status")
    if gpu_path.is_file():
        payload = json.loads(gpu_path.read_text(encoding="utf-8"))
        result["gpu_memory_mib"] = payload.get("max_memory_used_mib", payload.get("max"))
    if not metrics.is_file():
        return result
    last: dict[str, Any] | None = None
    samples = 0
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        last = row
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str) and sample_id:
            samples += len([part for part in sample_id.split(",") if part])
    if last is None:
        return result
    result["logical_steps"] = last.get("logical_step")
    updates = last.get("optimizer_updates")
    if isinstance(updates, dict):
        result["optimizer_updates"] = {str(key): int(value) for key, value in updates.items()}
    elif isinstance(updates, int):
        result["optimizer_updates"] = updates
    result["samples_seen"] = samples
    history = directory / "trainer_state.json"
    if history.is_file():
        payload = json.loads(history.read_text(encoding="utf-8"))
        completed = payload.get("completed_boundaries")
        if isinstance(completed, list):
            result["validation_count"] = sum(1 for row in completed if row.get("validation_done"))
    return result
