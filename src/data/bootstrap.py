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
import tempfile
from pathlib import Path

from src.data.constants import CORPORA_ROOT, DUCKDB_MAIN_PATH
from src.data.corpora_catalog import INDEXED
from src.data.init_eval import initialize
from src.data.ingest_corpora import main as ingest_main
from src.data.link_corpora import DEFAULT_DATASET_ROOT, link_corpora

MANAGED_CORPORA = frozenset(spec.link_name for spec in INDEXED)


def bootstrap(database: Path, dataset_root: Path, *, rebuild: bool = False, backup: Path | None = None) -> None:
    started = time.perf_counter()
    if database.exists() and not rebuild:
        raise SystemExit(f"{database} already exists; pass --rebuild to construct it again")
    if database.resolve() == DUCKDB_MAIN_PATH.resolve():
        raise SystemExit(f"refusing to build {database} in place; choose another path")
    if database.exists() and backup is None:
        raise ValueError("rebuilding an existing database requires an explicit backup path")
    database.parent.mkdir(parents=True, exist_ok=True)
    print("== links", flush=True)
    for line in link_corpora(dataset_root, CORPORA_ROOT):
        print(line, flush=True)
    # DuckDB creates its own file; keep it on the destination filesystem.
    with tempfile.TemporaryDirectory(prefix=f".{database.name}.build-", dir=database.parent) as tmp:
        candidate = Path(tmp) / "candidate.duckdb"
        print("== ingest", candidate, flush=True)
        import sys
        previous = sys.argv
        sys.argv = [previous[0], str(candidate)]
        try:
            ingest_main()
        finally:
            sys.argv = previous
        print("== eval sets", flush=True)
        initialize(database=candidate)
        validate_database(candidate)
        if database.exists():
            replace_production(candidate, backup, database)
        else:
            os.replace(candidate, database)
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
        required = {("corpora", "audio_files")} | {("eval", name) for name in ("cases", "sets", "set_items", "runs", "results")}
        if required - tables:
            raise ValueError(f"{path} is missing tables: {sorted(required - tables)}")
        names = {row[0] for row in connection.execute("SELECT DISTINCT corpus FROM corpora.audio_files").fetchall()}
        extra = names - MANAGED_CORPORA
        if names != MANAGED_CORPORA:
            raise ValueError(f"{path} corpora are {sorted(names)}")
        count = connection.execute("SELECT count(*) FROM corpora.audio_files").fetchone()[0]
        if int(count) < 1:
            raise ValueError(f"{path} has no audio rows")
        duplicates = connection.execute("SELECT count(*) - count(DISTINCT path) FROM corpora.audio_files").fetchone()[0]
        if duplicates:
            raise ValueError(f"{path} contains duplicate corpus paths")
        invalid = connection.execute("""
            SELECT count(*) FROM eval.set_items i
            LEFT JOIN eval.sets s ON s.name=i.set_name
            LEFT JOIN eval.cases c ON c.case_id=i.case_id
            WHERE s.name IS NULL OR c.case_id IS NULL
        """).fetchone()[0]
        if invalid:
            raise ValueError(f"{path} contains dangling set references")
        bad_order = connection.execute("""
            SELECT count(*) FROM (SELECT set_name FROM eval.set_items GROUP BY set_name
                HAVING min(position) <> 0 OR max(position) <> count(*) - 1
                    OR count(DISTINCT position) <> count(*))
        """).fetchone()[0]
        if bad_order:
            raise ValueError(f"{path} contains invalid set positions")
        from src.data.recipe import canonical_recipe, case_content_hash, case_id_for, json_hash
        import json
        rows = connection.execute("SELECT case_id, kind, dataset, split, sample_rate, transcript, speaker_id, recipe_json, content_hash, metadata FROM eval.cases").fetchall()
        for case_id, kind, dataset, split, rate, transcript, speaker, recipe_json, digest, metadata in rows:
            recipe = canonical_recipe(json.loads(recipe_json))
            body = dict(kind=kind, dataset=dataset, split=split, sample_rate=rate, transcript=transcript,
                        speaker_id=speaker, recipe=recipe, metadata=json.loads(metadata))
            if case_id_for(recipe) != case_id or case_content_hash(body) != digest or rate != recipe["output"]["sample_rate"]:
                raise ValueError(f"invalid case hash or sample rate: {case_id}")
        sets = connection.execute("SELECT name, description, content_hash FROM eval.sets").fetchall()
        if not sets:
            raise ValueError(f"{path} has no evaluation sets")
        for name, description, digest in sets:
            ids = [row[0] for row in connection.execute("SELECT case_id FROM eval.set_items WHERE set_name=? ORDER BY position", [name]).fetchall()]
            if digest != json_hash({"case_ids": ids, "description": description}):
                raise ValueError(f"invalid set hash: {name}")
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
    if new_database.resolve() == production.resolve() or backup.resolve() in {new_database.resolve(), production.resolve()}:
        raise ValueError("source, destination and backup must be distinct")
    validate_database(new_database)
    production.parent.mkdir(parents=True, exist_ok=True)
    if production.exists() and holders(production):
        raise RuntimeError(f"{production} is open by {holders(production)}")
    # Unique incoming names prevent concurrent attempts from clobbering each other.
    fd, incoming_name = tempfile.mkstemp(prefix=f".{production.name}.incoming-", dir=production.parent)
    os.close(fd)
    incoming = Path(incoming_name)
    try:
        shutil.copy2(new_database, incoming)
        with incoming.open("rb") as stream:
            os.fsync(stream.fileno())
        validate_database(incoming)
        if production.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation avoids overwriting an earlier backup.
            with production.open("rb") as source:
                target = backup.open("xb")
                try:
                    with target:
                        shutil.copyfileobj(source, target)
                        target.flush()
                        os.fsync(target.fileno())
                except BaseException:
                    backup.unlink(missing_ok=True)
                    raise
            if holders(production):
                raise RuntimeError(f"{production} became open during replacement")
        os.replace(incoming, production)
        directory_fd = os.open(production.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        incoming.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/db/resource-next.duckdb"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    if (args.replace or (args.rebuild and args.database.exists())) and args.backup is None:
        parser.error("--backup is required when replacing an existing database")
    if args.replace and args.database.exists():
        parser.error("--replace requires a new --database path; rebuild it separately first")
    bootstrap(args.database, args.dataset_root, rebuild=args.rebuild, backup=args.backup)
    if args.replace:
        replace_production(args.database, args.backup)
        print(f"replaced {DUCKDB_MAIN_PATH} backup {args.backup}", flush=True)


if __name__ == "__main__":
    main()
