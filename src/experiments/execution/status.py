"""Run records of one experiment directory: status, completion, and per-rank failures.

``status.json`` is the verdict. A process writes ``RUNNING`` when training
starts and ``FAILED`` when it raises. Run directly, it also writes the final
``COMPLETED`` or ``INTERRUPTED``. Under the launcher it reports its outcome
and the launcher writes the verdict after every process has exited.

``failure.rank<N>.json`` is what rank ``N`` raised. Rank 0 alone owns
``status.json``, so the error of another rank would otherwise only be in the
job log.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FAILURE_GLOB = "failure.rank*.json"


def write_status(directory: Path, status: str, **details: Any) -> None:
    """Replace ``status.json`` in ``directory``.

    The writer's host, pid, and process start time are recorded, so a reader
    can tell a ``RUNNING`` left behind by a killed process from a live one.
    """

    record: dict[str, Any] = {"status": status, **process_identity(), "updated_at": now()}
    record.update(details)
    replace_text(directory / "status.json", json.dumps(record, sort_keys=True) + "\n")


def read_status(directory: Path) -> dict[str, Any] | None:
    """Return ``status.json`` as a mapping, or ``None`` when it is missing or unreadable."""

    return _read_mapping(directory / "status.json")


def write_complete(directory: Path, logical_step: int) -> None:
    """Publish ``complete.json``. Its presence refuses a later resume."""

    body = json.dumps({"logical_step": logical_step}, sort_keys=True) + "\n"
    replace_text(directory / "complete.json", body)


def record_failure(directory: Path, rank: int, error: BaseException) -> None:
    """Record what this rank raised. Writing it must not hide the error itself."""

    record = {
        "rank": int(rank),
        "kind": failure_kind(error),
        "error": f"{type(error).__name__}: {error}"[:2000],
        **process_identity(),
        "time": now(),
    }
    try:
        replace_text(directory / f"failure.rank{int(rank)}.json", json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def read_failures(directory: Path) -> list[dict[str, Any]]:
    """Recorded rank failures, earliest first. The first one is usually the cause."""

    found = [record for record in map(_read_mapping, sorted(directory.glob(_FAILURE_GLOB))) if record is not None]
    return sorted(found, key=lambda record: (str(record.get("time")), int(record.get("rank", 0))))


def clear_failures(directory: Path) -> None:
    """Drop the failure records of an earlier run."""

    for path in directory.glob(_FAILURE_GLOB):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def failure_kind(error: BaseException) -> str:
    """``cuda_oom``, ``nccl``, ``interrupt``, or ``exception``."""

    import torch

    text = f"{type(error).__name__}: {error}"
    if isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in text:
        return "cuda_oom"
    if "NCCL" in text:
        return "nccl"
    if isinstance(error, KeyboardInterrupt):
        return "interrupt"
    return "exception"


def process_identity() -> dict[str, Any]:
    """Host, pid, and start time of this process."""

    pid = os.getpid()
    return {"host": socket.gethostname(), "pid": pid, "process_start": process_start(pid)}


def process_start(pid: int) -> str | None:
    """Kernel start time of ``pid`` in clock ticks since boot, or ``None`` without ``/proc``.

    A pid is reused after its process exits. The pair of pid and start time
    is not.
    """

    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    # ``pid (comm) state ppid ...``; comm may itself contain spaces and parentheses.
    fields = stat.rpartition(")")[2].split()
    if len(fields) < 20 or fields[0] == "Z":
        return None
    return fields[19]


def process_alive(pid: object, started: object) -> bool:
    """True when ``pid`` exists on this host and is the process that was recorded."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    current = process_start(pid)
    if current is None:
        # No /proc entry: a zombie, or a system without /proc where the pid is all there is.
        return not Path("/proc/self/stat").is_file()
    return started is None or str(started) == current


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def replace_text(target: Path, body: str) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, target)


def _read_mapping(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None
