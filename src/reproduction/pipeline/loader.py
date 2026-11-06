"""Render planned micro-batches.

Workers do not draw random numbers. Closing the loader discards nothing that
has already been committed by the trainer; unconsumed futures are simply dropped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class GroupLoader:
    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("load workers must be positive")
        self.workers = workers
        self._pool: ThreadPoolExecutor | None = None

    def map(self, function: Callable[[T], R], items: Sequence[T]) -> list[R]:
        if self.workers > 1 and len(items) > 1:
            return list(self._executor().map(function, items))
        return [function(item) for item in items]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="audio-db-render",
            )
        return self._pool
