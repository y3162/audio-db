"""Exclusive write lock for reproduction.duckdb and its snapshots.

The lock file is ``<database directory>/.reproduction.lock``. One process holds
it. The same process may enter the lock again.
"""

from __future__ import annotations

import fcntl
import threading
from contextlib import contextmanager
from pathlib import Path

from src.experiments.data.errors import ReproductionError
from src.experiments.data.paths import LOCK_NAME

_local = threading.local()


@contextmanager
def reproduction_lock(directory: Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    current = getattr(_local, "directory", None)
    depth = getattr(_local, "depth", 0)
    if depth > 0 and current == str(directory):
        _local.depth = depth + 1
        try:
            yield
        finally:
            _local.depth -= 1
        return
    handle = (directory / LOCK_NAME).open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ReproductionError(
            f"reproduction write lock is held: {directory / LOCK_NAME}"
        ) from exc
    _local.directory = str(directory)
    _local.depth = 1
    try:
        yield
    finally:
        _local.depth = 0
        _local.directory = None
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
