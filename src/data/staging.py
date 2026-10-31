"""Persistent staging files and DuckDB registration.

Staging files are the source for rebuilding a database. Registering them is
idempotent and atomic: the same content is a no-op, different content for the
same id is an error, and a failed batch leaves the database unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.db.eval_schema import create_eval_schema
from src.data.db.schema import create_corpora_schema
from src.data.models.audio_file import AUDIO_FILE_COLUMNS
from src.data.recipe import RecipeError, canonical_recipe, case_content_hash, case_id_for, json_hash

STAGING_VERSION = 1
CASE_FIELDS = (
    "case_id",
    "kind",
    "dataset",
    "split",
    "sample_rate",
    "transcript",
    "speaker_id",
    "recipe_json",
    "content_hash",
    "metadata",
)
SET_FIELDS = ("name", "description", "content_hash")
ITEM_FIELDS = ("set_name", "position", "case_id")
RUN_FIELDS = ("run_id", "config_hash", "config_json", "content_hash")
RESULT_FIELDS = ("record_id", "run_id", "case_id", "payload_json", "content_hash")


class StagingError(ValueError):
    """A staging directory is incomplete or disagrees with an existing row."""


def export_database(database: str | Path, staging: str | Path) -> Path:
    """Copy corpora and evaluation tables out of a recipe-schema database."""
    directory = Path(staging)
    directory.mkdir(parents=True, exist_ok=True)
    connection = _connect(database, read_only=True)
    try:
        _reject_legacy(connection)
        corpora = directory / "corpora" / "audio_files.parquet"
        cases = directory / "evaluation" / "cases.parquet"
        sets = directory / "evaluation" / "sets.parquet"
        items = directory / "evaluation" / "set_items.parquet"
        runs = directory / "results" / "runs.parquet"
        results = directory / "results" / "results.parquet"
        for path in (cases, sets, items, runs, results):
            path.parent.mkdir(parents=True, exist_ok=True)
        files = {
            "cases": "evaluation/cases.parquet",
            "sets": "evaluation/sets.parquet",
            "set_items": "evaluation/set_items.parquet",
            "runs": "results/runs.parquet",
            "results": "results/results.parquet",
        }
        has_corpora = connection.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_schema = 'corpora' AND table_name = 'audio_files'
            """
        ).fetchone()[0]
        if has_corpora:
            corpora.parent.mkdir(parents=True, exist_ok=True)
            _copy(connection, "SELECT * FROM corpora.audio_files", corpora)
            files["corpora"] = "corpora/audio_files.parquet"
        _copy(connection, _case_select(), cases)
        _copy(connection, "SELECT name, description, content_hash FROM eval.sets", sets)
        _copy(connection, "SELECT set_name, position, case_id FROM eval.set_items", items)
        _copy(connection, "SELECT run_id, config_hash, config_json::VARCHAR AS config_json, content_hash FROM eval.runs", runs)
        _copy(
            connection,
            """
            SELECT record_id, run_id, case_id, payload_json::VARCHAR AS payload_json, content_hash
            FROM eval.results
            """,
            results,
        )
    finally:
        connection.close()
    metadata = {"staging_version": STAGING_VERSION, "files": files}
    target = directory / "metadata.json"
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def register_staging(staging: str | Path, database: str | Path) -> dict[str, int]:
    """Validate staging files and insert them. Existing matching rows stay."""
    directory = Path(staging)
    metadata = _metadata(directory)
    files = metadata["files"]
    cases = _read_cases(directory / files["cases"])
    sets = _read_sets(directory / files["sets"])
    items = _read_items(directory / files["set_items"], cases, sets)
    runs = _read_runs(directory / files["runs"]) if files.get("runs") else []
    results = _read_results(directory / files["results"], runs) if files.get("results") else []
    corpora = directory / files["corpora"] if files.get("corpora") else None
    if corpora is not None and not corpora.is_file():
        raise StagingError(f"missing staging file: {corpora}")
    destination = Path(database)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(destination, read_only=False)
    try:
        create_eval_schema(connection)
        if corpora is not None:
            create_corpora_schema(connection, recreate=False)
        connection.execute("BEGIN TRANSACTION")
        try:
            counts = {
                "corpora": _merge_corpora(connection, corpora) if corpora is not None else 0,
                "cases": _merge_cases(connection, cases),
                "sets": _merge_sets(connection, sets, items),
                "runs": _merge_runs(connection, runs),
                "results": _merge_results(connection, results),
            }
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    return counts


def rebuild_database(staging: str | Path, database: str | Path) -> dict[str, int]:
    """Create a new database from staging. An existing file is left untouched."""
    destination = Path(database)
    if destination.exists():
        raise StagingError(f"refusing to overwrite an existing database: {destination}")
    return register_staging(staging, destination)


def register_run(database: str | Path, run_dir: str | Path) -> dict[str, int]:
    """Register one finished run from its config.json and results.jsonl."""
    directory = Path(run_dir)
    config_path = directory / "config.json"
    results_path = directory / "results.jsonl"
    if not config_path.is_file() or not results_path.is_file():
        raise StagingError(f"run directory is missing config.json or results.jsonl: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise StagingError("config.json must be an object")
    config_hash = json_hash(config)
    run_id = directory.name
    run = {
        "run_id": run_id,
        "config_hash": config_hash,
        "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
        "content_hash": json_hash({"config": config, "run_id": run_id}),
    }
    records = []
    for lineno, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StagingError(f"invalid results.jsonl line {lineno}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("record_id"), str):
            raise StagingError(f"results.jsonl line {lineno} needs a record_id")
        if payload.get("run_id") not in (None, run_id):
            raise StagingError(f"results.jsonl line {lineno} run_id does not match {run_id}")
        payload["run_id"] = run_id
        case_id = payload.get("utterance_id")
        if not isinstance(case_id, str) or len(case_id) != 64:
            case_id = None
        records.append(
            {
                "record_id": payload["record_id"],
                "run_id": run_id,
                "case_id": case_id,
                "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                "content_hash": json_hash(payload),
            }
        )
    if not records:
        raise StagingError(f"results.jsonl is empty: {results_path}")
    destination = Path(database)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(destination, read_only=False)
    try:
        create_eval_schema(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            counts = {"runs": _merge_runs(connection, [run]), "results": _merge_results(connection, records)}
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    return counts


def _metadata(directory: Path) -> dict[str, Any]:
    path = directory / "metadata.json"
    if not path.is_file():
        raise StagingError(f"staging metadata is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StagingError(f"invalid staging metadata: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("staging_version") != STAGING_VERSION:
        raise StagingError("unsupported staging_version")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise StagingError("staging metadata needs a files object")
    for name in ("cases", "sets", "set_items"):
        if name not in files:
            raise StagingError(f"staging metadata is missing files.{name}")
        if not (directory / files[name]).is_file():
            raise StagingError(f"missing staging file: {files[name]}")
    return payload


def _read_cases(path: Path) -> list[dict[str, Any]]:
    rows = _rows(path, CASE_FIELDS)
    if not rows:
        return []
    cases = []
    seen: set[str] = set()
    for row in rows:
        recipe = _object(row["recipe_json"], f"case {row['case_id']} recipe_json")
        metadata = _object(row["metadata"], f"case {row['case_id']} metadata")
        try:
            recipe = canonical_recipe(recipe)
            case_id = case_id_for(recipe)
        except RecipeError as exc:
            raise StagingError(f"case {row['case_id']} recipe is invalid: {exc}") from exc
        if case_id != row["case_id"]:
            raise StagingError(f"case_id does not match the recipe: {row['case_id']}")
        if case_id in seen:
            raise StagingError(f"duplicate case_id in staging: {case_id}")
        seen.add(case_id)
        rate = row["sample_rate"]
        if rate is not None:
            rate = int(rate)
        if rate != recipe["output"]["sample_rate"]:
            raise StagingError(f"case {case_id} sample_rate does not match the recipe")
        body = {
            "kind": row["kind"],
            "dataset": row["dataset"],
            "split": row["split"],
            "sample_rate": rate,
            "transcript": row["transcript"],
            "speaker_id": None if row["speaker_id"] is None else str(row["speaker_id"]),
            "recipe": recipe,
            "metadata": metadata,
        }
        digest = case_content_hash(body)
        if digest != row["content_hash"]:
            raise StagingError(f"case {case_id} content_hash does not match its fields")
        cases.append({**body, "case_id": case_id, "content_hash": digest, "recipe_json": recipe})
    return cases


def _read_sets(path: Path) -> list[dict[str, Any]]:
    rows = _rows(path, SET_FIELDS)
    seen: set[str] = set()
    sets = []
    for row in rows:
        name = row["name"]
        if not isinstance(name, str) or name in seen:
            raise StagingError(f"duplicate or invalid set name: {name!r}")
        seen.add(name)
        sets.append({"name": name, "description": row["description"]})
    return sets


def _read_items(path: Path, cases: list[dict[str, Any]], sets: list[dict[str, Any]]) -> dict[str, list[str]]:
    known_cases = {item["case_id"] for item in cases}
    known_sets = {item["name"] for item in sets}
    grouped: dict[str, list[tuple[int, str]]] = {name: [] for name in known_sets}
    for row in _rows(path, ITEM_FIELDS):
        name = row["set_name"]
        if name not in known_sets:
            raise StagingError(f"set item refers to an unknown set: {name}")
        case_id = row["case_id"]
        if case_id not in known_cases:
            raise StagingError(f"set item refers to an unknown case: {case_id}")
        position = row["position"]
        if isinstance(position, bool) or not isinstance(position, int):
            raise StagingError(f"set {name} has a non-integer position")
        grouped[name].append((position, case_id))
    ordered: dict[str, list[str]] = {}
    for name, pairs in grouped.items():
        pairs.sort()
        positions = [item[0] for item in pairs]
        if positions != list(range(len(pairs))):
            raise StagingError(f"set {name} positions are not 0..n-1")
        case_ids = [item[1] for item in pairs]
        if len(case_ids) != len(set(case_ids)):
            raise StagingError(f"set {name} repeats a case")
        ordered[name] = case_ids
    return ordered


def _read_runs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise StagingError(f"missing staging file: {path}")
    runs = []
    seen: set[str] = set()
    for row in _rows(path, RUN_FIELDS):
        config = _object(row["config_json"], f"run {row['run_id']} config_json")
        if row["run_id"] in seen:
            raise StagingError(f"duplicate run_id: {row['run_id']}")
        seen.add(row["run_id"])
        digest = json_hash({"config": config, "run_id": row["run_id"]})
        if digest != row["content_hash"] or json_hash(config) != row["config_hash"]:
            raise StagingError(f"run {row['run_id']} content_hash does not match")
        runs.append(
            {
                "run_id": row["run_id"],
                "config_hash": row["config_hash"],
                "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
                "content_hash": digest,
            }
        )
    return runs


def _read_results(path: Path, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise StagingError(f"missing staging file: {path}")
    known = {item["run_id"] for item in runs}
    records = []
    seen: set[str] = set()
    for row in _rows(path, RESULT_FIELDS):
        payload = _object(row["payload_json"], f"result {row['record_id']} payload_json")
        if row["record_id"] in seen:
            raise StagingError(f"duplicate record_id: {row['record_id']}")
        seen.add(row["record_id"])
        if row["run_id"] not in known:
            raise StagingError(f"result {row['record_id']} refers to an unknown run")
        digest = json_hash(payload)
        if digest != row["content_hash"]:
            raise StagingError(f"result {row['record_id']} content_hash does not match")
        records.append(
            {
                "record_id": row["record_id"],
                "run_id": row["run_id"],
                "case_id": row["case_id"],
                "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                "content_hash": digest,
            }
        )
    return records


def _merge_corpora(connection, path: Path) -> int:
    connection.execute("CREATE TEMP TABLE staging_corpora AS SELECT * FROM read_parquet(?)", [str(path)])
    columns = list(AUDIO_FILE_COLUMNS)
    present = {row[0] for row in connection.execute("DESCRIBE staging_corpora").fetchall()}
    missing = [name for name in columns if name not in present]
    if missing:
        raise StagingError(f"corpora parquet is missing columns: {', '.join(missing)}")
    signature = _column_signature(columns)
    conflicts = connection.execute(
        f"""
        SELECT count(*) FROM staging_corpora AS incoming
        JOIN corpora.audio_files AS stored USING (path)
        WHERE {signature.replace('__T__', 'incoming')} <> {signature.replace('__T__', 'stored')}
        """
    ).fetchone()[0]
    if conflicts:
        raise StagingError(f"{int(conflicts)} corpora rows disagree with the database")
    before = connection.execute("SELECT count(*) FROM corpora.audio_files").fetchone()[0]
    connection.execute(
        f"""
        INSERT INTO corpora.audio_files ({", ".join(columns)})
        SELECT {", ".join(f"incoming.{name}" for name in columns)}
        FROM staging_corpora AS incoming
        WHERE incoming.path NOT IN (SELECT path FROM corpora.audio_files)
        """
    )
    after = connection.execute("SELECT count(*) FROM corpora.audio_files").fetchone()[0]
    connection.execute("DROP TABLE staging_corpora")
    return int(after) - int(before)


def _merge_cases(connection, cases: list[dict[str, Any]]) -> int:
    inserted = 0
    for case in cases:
        row = connection.execute(
            "SELECT content_hash FROM eval.cases WHERE case_id = ?",
            [case["case_id"]],
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO eval.cases (
                    case_id, kind, dataset, split, sample_rate, transcript, speaker_id,
                    recipe_json, content_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON, ?, ?::JSON)
                """,
                [
                    case["case_id"],
                    case["kind"],
                    case["dataset"],
                    case["split"],
                    case["sample_rate"],
                    case["transcript"],
                    case["speaker_id"],
                    json.dumps(case["recipe_json"], ensure_ascii=False, sort_keys=True),
                    case["content_hash"],
                    json.dumps(case["metadata"], ensure_ascii=False, sort_keys=True),
                ],
            )
            inserted += 1
        elif row[0] != case["content_hash"]:
            raise StagingError(f"case {case['case_id']} is already registered with different content")
    return inserted


def _merge_sets(connection, sets: list[dict[str, Any]], items: dict[str, list[str]]) -> int:
    inserted = 0
    for item in sets:
        case_ids = items.get(item["name"], [])
        digest = json_hash({"case_ids": case_ids, "description": item["description"]})
        row = connection.execute(
            "SELECT content_hash FROM eval.sets WHERE name = ?",
            [item["name"]],
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO eval.sets (name, description, content_hash) VALUES (?, ?, ?)",
                [item["name"], item["description"], digest],
            )
            for position, case_id in enumerate(case_ids):
                connection.execute(
                    "INSERT INTO eval.set_items (set_name, position, case_id) VALUES (?, ?, ?)",
                    [item["name"], position, case_id],
                )
            inserted += 1
        elif row[0] != digest:
            raise StagingError(f"evaluation set {item['name']} is already registered with different content")
    return inserted


def _merge_runs(connection, runs: list[dict[str, Any]]) -> int:
    inserted = 0
    for run in runs:
        row = connection.execute(
            "SELECT content_hash FROM eval.runs WHERE run_id = ?",
            [run["run_id"]],
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO eval.runs (run_id, config_hash, config_json, content_hash)
                VALUES (?, ?, ?::JSON, ?)
                """,
                [run["run_id"], run["config_hash"], run["config_json"], run["content_hash"]],
            )
            inserted += 1
        elif row[0] != run["content_hash"]:
            raise StagingError(f"run {run['run_id']} is already registered with different content")
    return inserted


def _merge_results(connection, records: list[dict[str, Any]]) -> int:
    inserted = 0
    for record in records:
        row = connection.execute(
            "SELECT content_hash FROM eval.results WHERE record_id = ?",
            [record["record_id"]],
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO eval.results (record_id, run_id, case_id, payload_json, content_hash)
                VALUES (?, ?, ?, ?::JSON, ?)
                """,
                [
                    record["record_id"],
                    record["run_id"],
                    record["case_id"],
                    record["payload_json"],
                    record["content_hash"],
                ],
            )
            inserted += 1
        elif row[0] != record["content_hash"]:
            raise StagingError(f"result {record['record_id']} is already registered with different content")
    return inserted


def _case_select() -> str:
    return """
        SELECT
            case_id, kind, dataset, split, sample_rate, transcript, speaker_id,
            recipe_json::VARCHAR AS recipe_json,
            content_hash,
            metadata::VARCHAR AS metadata
        FROM eval.cases
    """


def _copy(connection, query: str, path: Path) -> None:
    if path.exists():
        path.unlink()
    connection.execute(f"COPY ({query}) TO ? (FORMAT PARQUET)", [str(path)])


def _rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    connection = _connect(":memory:", read_only=False)
    try:
        try:
            cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
        except Exception as exc:
            raise StagingError(f"cannot read {path.name}: {exc}") from exc
        names = [item[0] for item in cursor.description]
        missing = [field for field in fields if field not in names]
        if missing:
            raise StagingError(f"{path.name} is missing columns: {', '.join(missing)}")
        rows = []
        for record in cursor.fetchall():
            item = dict(zip(names, record, strict=True))
            rows.append({field: item[field] for field in fields})
    finally:
        connection.close()
    return rows


def _object(value: Any, where: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise StagingError(f"{where} is not JSON") from exc
    if not isinstance(value, dict):
        raise StagingError(f"{where} must be an object")
    return value


def _column_signature(columns: list[str]) -> str:
    parts = ", ".join(f"COALESCE(CAST(__T__.{name} AS VARCHAR), '')" for name in columns)
    return f"sha256(concat_ws('|', {parts}))"


def _reject_legacy(connection) -> None:
    columns = connection.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'eval' AND table_name = 'cases'
        """
    ).fetchall()
    if columns and "recipe_json" not in {row[0] for row in columns}:
        raise StagingError("refusing to export the previous fixed-column eval.cases schema")


def _connect(path: str | Path, *, read_only: bool):
    import duckdb

    try:
        return duckdb.connect(str(path), read_only=read_only)
    except duckdb.Error as exc:
        raise StagingError(f"failed to open DuckDB: {exc}") from exc
