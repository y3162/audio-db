import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")

IO_WORKERS = min(os.cpu_count() or 1, 64)


def iter_in_threads(
    items: Iterable[T],
    worker: Callable[[T], list[U]],
    *,
    max_workers: int = IO_WORKERS,
) -> Iterator[U]:
    item_iter = iter(items)
    in_flight_limit = max_workers * 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: set = set()
        listing_done = False
        while pending or not listing_done:
            while not listing_done and len(pending) < in_flight_limit:
                try:
                    item = next(item_iter)
                except StopIteration:
                    listing_done = True
                    break
                pending.add(executor.submit(worker, item))
            if not pending:
                break
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield from future.result()


def iter_libri_chapter_dirs(root: Path) -> Iterator[Path]:
    subsets: list[os.DirEntry] = []
    with os.scandir(root) as entries:
        for subset in entries:
            if subset.is_dir() and subset.name.startswith(("train-", "dev-", "test-")):
                subsets.append(subset)
    if not subsets:
        return
    with ThreadPoolExecutor(max_workers=len(subsets)) as executor:
        futures = [executor.submit(_list_chapter_dirs, subset) for subset in subsets]
        for future in as_completed(futures):
            yield from future.result()


def _list_chapter_dirs(subset: os.DirEntry) -> list[Path]:
    chapter_dirs: list[Path] = []
    with os.scandir(subset.path) as speakers:
        for speaker in speakers:
            if not speaker.is_dir():
                continue
            with os.scandir(speaker.path) as chapters:
                for chapter in chapters:
                    if chapter.is_dir():
                        chapter_dirs.append(Path(chapter.path))
    return chapter_dirs
