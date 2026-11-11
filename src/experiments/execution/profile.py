"""Optional step and checkpoint timing.

``AUDIO_DB_RESOURCE_PROFILE=1`` records CPU spans and this rank's peak
``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``. The default
path does not synchronize CUDA and does not write a file. Spans do not
change the loss, the optimizer, or the checkpoint payload.

Whole-device memory over time is the launcher's ``gpu_memory.jsonl``
(``nvidia-smi``), not this profile.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def enabled() -> bool:
    return os.environ.get("AUDIO_DB_RESOURCE_PROFILE") == "1"


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.started = time.perf_counter()

    def reset(self) -> None:
        self.events = []
        self.started = time.perf_counter()

    def add(self, name: str, seconds: float) -> None:
        if enabled():
            self.events.append({"name": name, "seconds": seconds})


_RECORDER = Recorder()


def recorder() -> Recorder:
    return _RECORDER


def reset() -> None:
    _RECORDER.reset()


class Span:
    """Time one block. CUDA synchronization runs only while profiling is on."""

    def __init__(self, name: str, device: torch.device | None = None) -> None:
        self.name = name
        self.device = device
        self._started = 0.0

    def __enter__(self) -> Span:
        if enabled() and self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if not enabled():
            return
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        _RECORDER.add(self.name, time.perf_counter() - self._started)


def dump(directory: Path, runtime: object) -> Path | None:
    """Write this rank's profile. Returns the path, or None when profiling is off."""

    if not enabled():
        return None
    rank = int(getattr(runtime, "rank", 0) or 0)
    world = int(getattr(runtime, "world_size", 1) or 1)
    device = getattr(runtime, "device", None)
    allocated = None
    reserved = None
    if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        index = int(device.index or 0)
        allocated = int(torch.cuda.max_memory_allocated(index))
        reserved = int(torch.cuda.max_memory_reserved(index))
    payload = {
        "elapsed_seconds": time.perf_counter() - _RECORDER.started,
        "events": list(_RECORDER.events),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "rank": rank,
        "world_size": world,
    }
    directory.mkdir(parents=True, exist_ok=True)
    name = "resource_profile.json" if world == 1 or rank == 0 else f"resource_profile.rank{rank}.json"
    if world > 1 and rank == 0:
        path = directory / "resource_profile.json"
    else:
        path = directory / name
    if world > 1 and rank != 0:
        path = directory / f"resource_profile.rank{rank}.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
