"""Build a recipe database from an empty file.

Order: corpus links, corpus index, eval sets from configs/db/init.yaml.
The production file data/db/resource.duckdb is not replaced by this command.
Pass --replace only after a backup, and only when the new file has been checked.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from src.data.constants import CORPORA_ROOT, DUCKDB_MAIN_PATH
from src.data.corpora_catalog import INDEXED
from src.data.init_eval import initialize
from src.data.ingest_corpora import main as ingest_main
from src.data.link_corpora import DEFAULT_DATASET_ROOT, link_corpora

MANAGED_CORPORA = frozenset(spec.link_name for spec in INDEXED)


def bootstrap(database: Path, dataset_root: Path, *, rebuild: bool = False) -> None:
    started = time.perf_counter()
    if database.exists() and not rebuild:
        raise SystemExit(f"{database} already exists; pass --rebuild to construct it again")
    if database.resolve() == DUCKDB_MAIN_PATH.resolve():
        raise SystemExit(f"refusing to build {database} in place; choose another path")
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


def validate_database(path: Path) -> None:
    import duckdb

    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"{path} is not a database file")
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = {
            (row[0], row[1])
            for row in connection.execute(
                """
                SELECT table_schema, table_name FROM information_schema.tables
                WHERE table_schema IN ('corpora', 'eval')
                """
            ).fetchall()
        }
        if ("corpora", "audio_files") not in tables or ("eval", "cases") not in tables:
            raise ValueError(f"{path} is missing corpora.audio_files or eval.cases")
        names = {row[0] for row in connection.execute("SELECT DISTINCT corpus FROM corpora.audio_files").fetchall()}
        extra = names - MANAGED_CORPORA
        if extra or not names:
            raise ValueError(f"{path} corpora are {sorted(names)}")
        count = connection.execute("SELECT count(*) FROM corpora.audio_files").fetchone()[0]
        if int(count) < 1:
            raise ValueError(f"{path} has no audio rows")
    finally:
        connection.close()


def holders(path: Path) -> list[int]:
    """Process ids, other than this one, that have the file open."""
    if not path.exists():
        return []
    target = path.resolve()
    found: list[int] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        fd_dir = entry / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                if descriptor.resolve() == target:
                    found.append(pid)
                    break
            except OSError:
                continue
    return found


def replace_production(new_database: Path, backup: Path, production: Path = DUCKDB_MAIN_PATH) -> None:
    """Validate, back up, then atomically replace the production file."""
    validate_database(new_database)
    if production.exists():
        busy = holders(production)
        if busy:
            raise RuntimeError(f"{production} is open by {busy}")
        if backup.exists():
            raise FileExistsError(backup)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(production, backup)
    incoming = production.with_name(production.name + ".incoming")
    shutil.copy2(new_database, incoming)
    os.replace(incoming, production)
    validate_database(production)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/db/resource-next.duckdb"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--backup", type=Path, default=Path("data/db/backups/resource-before-bootstrap.duckdb"))
    args = parser.parse_args()
    bootstrap(args.database, args.dataset_root, rebuild=args.rebuild)
    if args.replace:
        replace_production(args.database, args.backup)
        print(f"replaced {DUCKDB_MAIN_PATH} backup {args.backup}", flush=True)


if __name__ == "__main__":
    main()
