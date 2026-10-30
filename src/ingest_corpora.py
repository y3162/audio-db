import argparse
import time
from collections.abc import Iterator

import duckdb

from src.constants import DUCKDB_MAIN_PATH
from src.corpora import AUDIO_FILE_ROW_ITERATORS
from src.db.bulk_insert import copy_insert_batched
from src.db.schema import create_corpora_schema
from src.models.audio_file import AudioFileRow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest audio file metadata from data/corpora into DuckDB")
    parser.add_argument(
        "--corpus",
        action="append",
        choices=sorted(AUDIO_FILE_ROW_ITERATORS),
        help="limit to one or more corpora (default: all)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete existing rows for selected corpora before insert",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    corpora = args.corpus or list(AUDIO_FILE_ROW_ITERATORS)
    con = duckdb.connect(DUCKDB_MAIN_PATH)
    replacing_all = args.replace and args.corpus is None
    create_corpora_schema(con, recreate=replacing_all)
    if args.replace and not replacing_all:
        placeholders = ", ".join("?" for _ in corpora)
        con.execute(f"DELETE FROM corpora.audio_files WHERE corpus IN ({placeholders})", corpora)
    total = 0
    for corpus in corpora:
        started = time.perf_counter()
        print(f"Ingesting {corpus}...", flush=True)
        rows: Iterator[AudioFileRow] = AUDIO_FILE_ROW_ITERATORS[corpus]()
        inserted = copy_insert_batched(con, rows)
        elapsed = time.perf_counter() - started
        total += inserted
        print(f"  inserted {inserted:,} rows in {elapsed:.1f}s", flush=True)
    con.close()
    print(f"Done. {total:,} rows total.", flush=True)


if __name__ == "__main__":
    main()
