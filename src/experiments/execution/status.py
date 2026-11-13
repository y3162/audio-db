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


def write_complete(directory: Path, logical_step: int, **details: Any) -> None:
    """Publish ``complete.json``. Its presence refuses a later resume."""

    body = json.dumps({"logical_step": logical_step, **details}, sort_keys=True) + "\n"
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
    """Kernel start time of ``pid`` in clock ticks since boot, or ``None`` when it is not running.

    A pid is reused after its process exits. The pair of pid and start time
    is not. A zombie has exited, so it has no start time here.
    """

    kind, start = _proc_stat(pid)
    return start if kind == "running" else None


def process_alive(pid: object, started: object) -> bool:
    """True when ``pid`` is the recorded process and has not exited.

    A missing start time still matches. Callers that must not confuse a reused
    pid with the original process use :func:`process_match`.
    """

    return process_match(pid, started, missing_start="alive") == "alive"


def process_match(pid: object, started: object, *, missing_start: str = "unknown") -> str:
    """``alive``, ``dead``, or ``unknown`` for one recorded pid.

    ``unknown`` means the pid exists but its start time cannot be checked, or
    no start time was recorded on a host that has ``/proc``. A zombie is
    ``dead``: the process has exited and only the entry remains.
    """

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return "dead"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "unknown"
    kind, current = _proc_stat_retry(pid)
    if kind == "missing" and not Path("/proc/self/stat").is_file():
        # No procfs: a live pid cannot be told apart from any other occupant.
        return "alive"
    if kind in ("missing", "zombie"):
        return "dead"
    if kind != "running":
        return "unknown"
    if not Path("/proc/self/stat").is_file():
        return "alive"
    if started is None:
        return missing_start
    return "alive" if str(started) == current else "dead"


def _proc_stat(pid: int) -> tuple[str, str | None]:
    """``running``, ``zombie``, ``missing``, or ``unreadable``, and the start field when present."""

    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "missing", None
    except (OSError, ValueError):
        return "unreadable", None
    # ``pid (comm) state ppid ...``; comm may itself contain spaces and parentheses.
    fields = stat.rpartition(")")[2].split()
    if len(fields) < 20:
        return "unreadable", None
    if fields[0] == "Z":
        return "zombie", fields[19]
    return "running", fields[19]


def _proc_stat_retry(pid: int) -> tuple[str, str | None]:
    """Read ``/proc`` a few times. One torn read is not evidence that the process exited."""

    last = ("unreadable", None)
    for _ in range(3):
        kind, start = _proc_stat(pid)
        if kind != "unreadable":
            return kind, start
        last = (kind, start)
    return last


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
