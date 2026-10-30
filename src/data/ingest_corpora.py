import sys
import time
from collections.abc import Callable, Iterator

import duckdb

from src.data.constants import DUCKDB_MAIN_PATH
from src.data.corpora import AUDIO_FILE_ROW_ITERATORS
from src.data.db.bulk_insert import copy_insert_batched
from src.data.db.schema import create_corpora_schema
from src.data.models.audio_file import AudioFileRow


def _insert_progress_reporter() -> tuple[Callable[[int], None], Callable[[], None]]:
    use_tty = sys.stdout.isatty()

    def on_progress(inserted: int) -> None:
        if not use_tty:
            return
        print(f"\r  inserted {inserted:,} rows", end="", flush=True)

    def finish() -> None:
        if use_tty:
            print(flush=True)

    return on_progress, finish


def main() -> None:
    corpora = sorted(AUDIO_FILE_ROW_ITERATORS)
    con = duckdb.connect(DUCKDB_MAIN_PATH)
    create_corpora_schema(con, recreate=True)
    total = 0
    for corpus in corpora:
        started = time.perf_counter()
        print(f"Ingesting {corpus}...", flush=True)
        rows: Iterator[AudioFileRow] = AUDIO_FILE_ROW_ITERATORS[corpus]()
        on_progress, finish_progress = _insert_progress_reporter()
        inserted = copy_insert_batched(con, rows, on_progress=on_progress)
        finish_progress()
        elapsed = time.perf_counter() - started
        total += inserted
        print(f"  inserted {inserted:,} rows in {elapsed:.1f}s", flush=True)
    con.close()
    print(f"Done. {total:,} rows total.", flush=True)


if __name__ == "__main__":
    main()
