"""Whole-device GPU memory sampled while the launcher supervises a run.

Each sample is ``nvidia-smi`` ``memory.used`` and ``memory.total`` for a GPU.
That is the memory the driver reports for the device, including other
processes on it. It is not ``torch.cuda.memory_allocated`` or
``memory_reserved``; those peaks stay in the optional resource profile.

Samples are appended as complete JSON lines and the summary is replaced
after each sample, so a killed launcher keeps every finished line. A missing
``nvidia-smi``, a failed query, or an empty ``CUDA_VISIBLE_DEVICES`` does not
stop training. Nothing is written until this launch has reached ``RUNNING``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from src.experiments.execution.status import now, read_status, replace_text

SCHEMA = "audio-db-gpu-memory-v1"
SERIES_NAME = "gpu_memory.jsonl"
SUMMARY_NAME = "gpu_memory.json"
DEFAULT_INTERVAL_SECONDS = 10.0
_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
_QUERY_TIMEOUT_SECONDS = 5.0
_POLL_SECONDS = 0.2
METRICS = {
    "memory_used_mib": (
        "nvidia-smi memory.used for the whole GPU, MiB. "
        "Not torch.cuda.memory_allocated or memory_reserved."
    ),
    "memory_total_mib": "nvidia-smi memory.total for the whole GPU, MiB.",
}


def resolve_interval(cli: float | None, environ: Mapping[str, str]) -> float:
    """Seconds between samples. ``0`` disables. The environment is the fallback."""

    if cli is not None:
        value = float(cli)
    else:
        raw = environ.get("AUDIO_DB_GPU_MEMORY_INTERVAL", "").strip()
        if not raw:
            value = DEFAULT_INTERVAL_SECONDS
        else:
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError("AUDIO_DB_GPU_MEMORY_INTERVAL must be a number of seconds") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("gpu memory interval must be a finite number >= 0")
    return value


def query_nvidia_smi() -> str:
    """Return the ``nvidia-smi`` CSV. Raises when the command fails or times out."""

    try:
        completed = subprocess.run(
            _QUERY,
            check=False,
            capture_output=True,
            text=True,
            timeout=_QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(str(exc)) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "nvidia-smi failed").strip()
        raise RuntimeError(detail)
    return completed.stdout


def parse_gpu_query(text: str) -> list[dict[str, Any]]:
    """Parse ``index, uuid, memory.used, memory.total`` rows. MiB may be ``[N/A]``."""

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_used_mib": _mib(parts[2]),
                "memory_total_mib": _mib(parts[3]),
            }
        )
    return rows


def select_devices(rows: list[dict[str, Any]], visible: str | None) -> list[dict[str, Any]]:
    """Keep every row when ``visible`` is ``None``. Otherwise match indexes or UUIDs."""

    if visible is None:
        return list(rows)
    tokens = [part.strip() for part in visible.split(",") if part.strip()]
    indexes: set[int] = set()
    uuids: set[str] = set()
    for token in tokens:
        if token.isdigit():
            indexes.add(int(token))
        else:
            uuids.add(token)
    return [row for row in rows if row["index"] in indexes or row["uuid"] in uuids]


class GpuMemoryLog:
    """One experiment's series. ``start`` returns a log that writes nothing when disabled."""

    def __init__(
        self,
        directory: Path,
        interval: float,
        launch_id: str,
        *,
        visible: str | None,
        enabled: bool,
        query: Callable[[], str],
    ) -> None:
        self.directory = directory
        self.interval = interval
        self.launch_id = launch_id
        self.visible = visible
        self.enabled = enabled
        self._query = query
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._header = False
        self._samples = 0
        self._last_error: str | None = None
        self._peaks: dict[int, dict[str, Any]] = {}

    @classmethod
    def start(
        cls,
        directory: Path,
        interval: float,
        launch_id: str,
        *,
        environ: Mapping[str, str] | None = None,
        query: Callable[[], str] | None = None,
    ) -> GpuMemoryLog:
        env = os.environ if environ is None else environ
        visible = _visible(env)
        # An empty list is the CPU lane: the test hid every GPU. Do not query.
        enabled = interval > 0 and visible != ""
        log = cls(
            directory,
            interval,
            launch_id,
            visible=None if visible is None else visible,
            enabled=enabled,
            query=query or query_nvidia_smi,
        )
        if enabled:
            log._thread = threading.Thread(target=log._loop, name="audio-db-gpu-memory", daemon=True)
            log._thread.start()
        return log

    def close(self) -> None:
        """Stop the thread and, if this launch trained, record one last sample."""

        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_QUERY_TIMEOUT_SECONDS + 3.0)
        if self.enabled and self._belongs():
            self._capture()

    def _loop(self) -> None:
        next_at = 0.0
        while not self._stop.is_set():
            if self._running() and time.monotonic() >= next_at:
                self._capture()
                next_at = time.monotonic() + self.interval
            if self._stop.wait(_POLL_SECONDS):
                return

    def _running(self) -> bool:
        status = read_status(self.directory) or {}
        return status.get("launch_id") == self.launch_id and status.get("status") == "RUNNING"

    def _belongs(self) -> bool:
        status = read_status(self.directory) or {}
        return status.get("launch_id") == self.launch_id and status.get("status") in {
            "RUNNING",
            "COMPLETED",
            "INTERRUPTED",
            "FAILED",
        }

    def _capture(self) -> None:
        try:
            rows = select_devices(parse_gpu_query(self._query()), self.visible)
            error = None
            if self.visible not in (None, "") and not rows:
                error = "nvidia-smi returned no device in CUDA_VISIBLE_DEVICES"
        except Exception as exc:
            rows = []
            error = f"{type(exc).__name__}: {exc}"
        recorded_at = now()
        with self._lock:
            try:
                self._store(recorded_at, rows, error)
            except OSError:
                return

    def _store(self, recorded_at: str, rows: list[dict[str, Any]], error: str | None) -> None:
        if not self._header:
            self._append(
                {
                    "schema": SCHEMA,
                    "kind": "header",
                    "interval_seconds": self.interval,
                    "cuda_visible_devices": self.visible,
                    "scope": "host" if self.visible is None else "visible",
                    "metrics": METRICS,
                    "launch_id": self.launch_id,
                }
            )
            self._header = True
        self._append(
            {
                "schema": SCHEMA,
                "kind": "sample",
                "recorded_at": recorded_at,
                "devices": rows,
                "error": error,
            }
        )
        self._samples += 1
        self._last_error = error
        for row in rows:
            used = row["memory_used_mib"]
            if not isinstance(used, int):
                continue
            previous = self._peaks.get(row["index"])
            if previous is None or used >= int(previous["memory_used_mib"]):
                self._peaks[row["index"]] = dict(row)
        self._write_summary_locked()

    def _write_summary_locked(self) -> None:
        payload = {
            "schema": SCHEMA,
            "series": SERIES_NAME,
            "interval_seconds": self.interval,
            "cuda_visible_devices": self.visible,
            "scope": "host" if self.visible is None else "visible",
            "metrics": METRICS,
            "sample_count": self._samples,
            "peaks": [self._peaks[index] for index in sorted(self._peaks)],
            "last_error": self._last_error,
            "launch_id": self.launch_id,
            "updated_at": now(),
        }
        try:
            replace_text(self.directory / SUMMARY_NAME, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except OSError:
            return

    def _append(self, payload: dict[str, Any]) -> None:
        path = self.directory / SERIES_NAME
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _visible(environ: Mapping[str, str]) -> str | None:
    """``None`` means the variable is unset. ``""`` means every GPU is hidden."""

    if "CUDA_VISIBLE_DEVICES" not in environ:
        return None
    return environ["CUDA_VISIBLE_DEVICES"].strip()


def _mib(text: str) -> int | None:
    token = text.strip().strip("[]")
    if token.upper() == "N/A" or not token:
        return None
    return int(token)
