"""Build a recipe database from an empty file.

Order: corpus links, corpus index, eval sets from configs/db/init.yaml.
The production file data/db/resource.duckdb is not replaced by this command.
Pass --replace only after a backup, and only when the new file has been checked.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from src.data.constants import CORPORA_ROOT, DUCKDB_MAIN_PATH
from src.data.init_eval import initialize
from src.data.ingest_corpora import main as ingest_main
from src.data.link_corpora import DEFAULT_DATASET_ROOT, link_corpora


def bootstrap(database: Path, dataset_root: Path) -> None:
    started = time.perf_counter()
    if database.resolve() == DUCKDB_MAIN_PATH.resolve() and database.exists():
        raise SystemExit(f"refusing to rebuild existing {database}; choose a new path")
    print("== links", flush=True)
    for line in link_corpora(dataset_root, CORPORA_ROOT):
        print(line, flush=True)
    print("== ingest", database, flush=True)
    import sys

    previous = sys.argv
    sys.argv = [previous[0], str(database)]
    try:
        ingest_main()
    finally:
        sys.argv = previous
    print("== eval sets", flush=True)
    initialize(database=database)
    elapsed = time.perf_counter() - started
    print(f"bootstrap finished in {elapsed:.1f}s", flush=True)


def replace_production(new_database: Path, backup: Path) -> None:
    if not new_database.is_file():
        raise FileNotFoundError(new_database)
    if DUCKDB_MAIN_PATH.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(DUCKDB_MAIN_PATH, backup)
    shutil.copy2(new_database, DUCKDB_MAIN_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/db/resource-next.duckdb"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--backup", type=Path, default=Path("data/db/backups/resource-before-bootstrap.duckdb"))
    args = parser.parse_args()
    bootstrap(args.database, args.dataset_root)
    if args.replace:
        replace_production(args.database, args.backup)
        print(f"replaced {DUCKDB_MAIN_PATH} backup {args.backup}", flush=True)


if __name__ == "__main__":
    main()
