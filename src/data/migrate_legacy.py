"""One-time export of the previous eval schema into recipe staging.

This is not a compatibility layer. The runtime does not read the old columns.
It opens ``data/db/resource.duckdb`` read-only, writes staging files, and builds
separate databases. It does not replace the existing database.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.data.recipe import canonical_recipe, case_content_hash, case_id_for, json_hash, sha256_file
from src.data.staging import rebuild_database, register_staging

DEFAULT_SOURCE = Path("data/db/resource.duckdb")
DEFAULT_STAGING = Path("data/staging")
PEAK_LIMIT = 0.99


def migrate(source: Path, staging: Path, rebuilt: Path, reversed_db: Path) -> str:
    if not source.is_file():
        raise SystemExit(f"source database not found: {source}")
    for path in (staging, rebuilt, reversed_db):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    rows, sets, items = _read_legacy(source)
    cases = [_convert_case(source.parent, row) for row in rows]
    _write_staging(source, staging, cases, sets, items)
    rebuild_database(staging, rebuilt)
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        reversed_staging = Path(temporary) / "staging"
        _copy_reversed_cases(staging, reversed_staging)
        register_staging(reversed_staging, reversed_db)
    report = _compare(source, rebuilt, reversed_db, cases)
    again = register_staging(staging, rebuilt)
    if any(again.values()):
        raise SystemExit(f"second registration inserted rows: {again}")
    print(report)
    return report


def _read_legacy(source: Path):
    import duckdb

    connection = duckdb.connect(str(source), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT case_id, kind, dataset, split, sample_rate, transcript, speaker_id,
                   snr_db, audio_path, clean_path, noisy_path, noise_path, corpus_path,
                   mix_protocol, mix_seed, noise_start, max_seconds, prevent_clip, hard_clip, metadata
            FROM eval.cases
            ORDER BY case_id
            """
        ).fetchall()
        names = [item[0] for item in connection.description]
        cases = [dict(zip(names, row, strict=True)) for row in rows]
        set_rows = connection.execute("SELECT name, description FROM eval.sets ORDER BY name").fetchall()
        item_rows = connection.execute(
            "SELECT set_name, position, case_id FROM eval.set_items ORDER BY set_name, position"
        ).fetchall()
    finally:
        connection.close()
    return cases, set_rows, item_rows


def _convert_case(database_dir: Path, row: dict) -> dict:
    if row["hard_clip"]:
        raise SystemExit(f"{row['case_id']} uses hard clipping, which has no recipe operation")
    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    if not isinstance(metadata, dict):
        raise SystemExit(f"{row['case_id']} metadata is not an object")
    metadata = dict(metadata)
    metadata["source_id"] = row["case_id"]
    metadata["legacy_case_id"] = row["case_id"]
    if row["corpus_path"]:
        metadata["corpus_path"] = row["corpus_path"]
    kind = row["kind"]
    if kind == "synthetic":
        recipe = _synthetic_recipe(database_dir, row)
    elif kind == "paired":
        recipe = _recorded(database_dir, row, reference="dry_clean", clean=True, noisy=True)
    elif kind == "noisy":
        recipe = _recorded(database_dir, row, reference="none", clean=False, noisy=True)
    elif kind == "clean":
        recipe = _recorded(database_dir, row, reference="dry_clean", clean=True, noisy=False)
    else:
        raise SystemExit(f"unknown kind {kind}")
    rate = recipe["output"]["sample_rate"]
    body = {
        "kind": kind,
        "dataset": row["dataset"],
        "split": row["split"],
        "sample_rate": rate,
        "transcript": row["transcript"],
        "speaker_id": None if row["speaker_id"] is None else str(row["speaker_id"]),
        "recipe": recipe,
        "metadata": metadata,
    }
    return {
        **body,
        "case_id": case_id_for(recipe),
        "content_hash": case_content_hash(body),
        "legacy_case_id": row["case_id"],
    }


def _synthetic_recipe(database_dir: Path, row: dict) -> dict:
    if row["mix_protocol"] not in (None, "rms_v1"):
        raise SystemExit(f"{row['case_id']} has protocol {row['mix_protocol']!r}")
    if row["noise_start"] is None or row["mix_seed"] is None or row["snr_db"] is None:
        raise SystemExit(f"{row['case_id']} is missing a stored noise crop, seed, or SNR")
    clean = _source(database_dir, row["clean_path"])
    noise = _source(database_dir, row["noise_path"])
    rate = int(row["sample_rate"]) if row["sample_rate"] is not None else _rate(database_dir, row["clean_path"])
    operations = [
        {
            "type": "additive_noise",
            "version": 1,
            "source": "noise",
            "snr_db": float(row["snr_db"]),
            "noise_start": int(row["noise_start"]),
            "noise_start_unit": "sample",
            "seed": int(row["mix_seed"]),
        }
    ]
    if row["prevent_clip"]:
        operations.append({"type": "peak_normalize", "version": 1, "limit": PEAK_LIMIT})
    output = {"sample_rate": rate, "reference": "dry_clean"}
    if row["max_seconds"] is not None:
        output["max_seconds"] = float(row["max_seconds"])
    return canonical_recipe(
        {"recipe_version": 1, "sources": {"clean": clean, "noise": noise}, "operations": operations, "output": output}
    )


def _recorded(database_dir: Path, row: dict, *, reference: str, clean: bool, noisy: bool) -> dict:
    sources = {}
    primary = None
    if clean:
        primary = row["clean_path"] or row["audio_path"]
        sources["clean"] = _source(database_dir, primary)
    if noisy:
        primary = row["noisy_path"] or row["audio_path"]
        sources["noisy"] = _source(database_dir, primary)
    if not sources or primary is None:
        raise SystemExit(f"{row['case_id']} has no audio path")
    rate = int(row["sample_rate"]) if row["sample_rate"] is not None else _rate(database_dir, primary)
    return canonical_recipe(
        {
            "recipe_version": 1,
            "sources": sources,
            "operations": [],
            "output": {"sample_rate": rate, "reference": reference},
        }
    )


def _source(database_dir: Path, stored: str | None) -> dict[str, str]:
    if not stored:
        raise SystemExit("audio path is missing")
    path = Path(os_norm(database_dir / stored))
    if not path.is_file():
        raise SystemExit(f"audio file not found: {stored}")
    return {"path": Path(stored).as_posix(), "sha256": sha256_file(path)}


def _rate(database_dir: Path, stored: str) -> int:
    import torchaudio

    info = torchaudio.info(str(Path(os_norm(database_dir / stored))))
    return int(info.sample_rate)


def os_norm(path: Path) -> str:
    import os

    return os.path.normpath(path)


def _write_staging(source: Path, staging: Path, cases: list[dict], sets, items) -> None:
    import duckdb

    by_legacy = {case["legacy_case_id"]: case["case_id"] for case in cases}
    staging.mkdir(parents=True)
    (staging / "corpora").mkdir()
    (staging / "evaluation").mkdir()
    (staging / "results").mkdir()
    connection = duckdb.connect()
    try:
        connection.execute(f"ATTACH '{source.as_posix()}' AS src (READ_ONLY)")
        connection.execute(
            "COPY (SELECT * FROM src.corpora.audio_files) TO ? (FORMAT PARQUET)",
            [str(staging / "corpora" / "audio_files.parquet")],
        )
        connection.execute("DETACH src")
        connection.execute(
            """
            CREATE TABLE cases (
                case_id VARCHAR, kind VARCHAR, dataset VARCHAR, split VARCHAR,
                sample_rate INTEGER, transcript VARCHAR, speaker_id VARCHAR,
                recipe_json VARCHAR, content_hash VARCHAR, metadata VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                [
                    case["case_id"],
                    case["kind"],
                    case["dataset"],
                    case["split"],
                    case["sample_rate"],
                    case["transcript"],
                    case["speaker_id"],
                    json.dumps(case["recipe"], ensure_ascii=False, sort_keys=True),
                    case["content_hash"],
                    json.dumps(case["metadata"], ensure_ascii=False, sort_keys=True),
                ]
                for case in cases
            ],
        )
        connection.execute(
            "COPY cases TO ? (FORMAT PARQUET)",
            [str(staging / "evaluation" / "cases.parquet")],
        )
        set_rows = []
        item_rows = []
        grouped: dict[str, list[str]] = {}
        for set_name, position, legacy_id in items:
            grouped.setdefault(set_name, []).append(by_legacy[legacy_id])
            if position != len(grouped[set_name]) - 1:
                raise SystemExit(f"set {set_name} positions are not contiguous")
        for name, description in sets:
            case_ids = grouped.get(name, [])
            set_rows.append((name, description, json_hash({"case_ids": case_ids, "description": description})))
            item_rows.extend((name, index, case_id) for index, case_id in enumerate(case_ids))
        connection.execute("CREATE TABLE sets (name VARCHAR, description VARCHAR, content_hash VARCHAR)")
        connection.executemany("INSERT INTO sets VALUES (?, ?, ?)", set_rows)
        connection.execute("COPY sets TO ? (FORMAT PARQUET)", [str(staging / "evaluation" / "sets.parquet")])
        connection.execute("CREATE TABLE items (set_name VARCHAR, position INTEGER, case_id VARCHAR)")
        connection.executemany("INSERT INTO items VALUES (?, ?, ?)", item_rows)
        connection.execute("COPY items TO ? (FORMAT PARQUET)", [str(staging / "evaluation" / "set_items.parquet")])
        connection.execute(
            """
            CREATE TABLE runs (run_id VARCHAR, config_hash VARCHAR, config_json VARCHAR, content_hash VARCHAR)
            """
        )
        connection.execute("COPY runs TO ? (FORMAT PARQUET)", [str(staging / "results" / "runs.parquet")])
        connection.execute(
            """
            CREATE TABLE results (
                record_id VARCHAR, run_id VARCHAR, case_id VARCHAR,
                payload_json VARCHAR, content_hash VARCHAR
            )
            """
        )
        connection.execute("COPY results TO ? (FORMAT PARQUET)", [str(staging / "results" / "results.parquet")])
    finally:
        connection.close()
    metadata = {
        "staging_version": 1,
        "files": {
            "corpora": "corpora/audio_files.parquet",
            "cases": "evaluation/cases.parquet",
            "sets": "evaluation/sets.parquet",
            "set_items": "evaluation/set_items.parquet",
            "runs": "results/runs.parquet",
            "results": "results/results.parquet",
        },
        "note": "Exported once from the previous eval.cases columns. The runtime does not read those columns.",
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_reversed_cases(staging: Path, destination: Path) -> None:
    """Copy staging and reverse only the case-row order. Set positions stay."""
    import shutil

    import duckdb

    shutil.copytree(staging, destination)
    cases = destination / "evaluation" / "cases.parquet"
    temporary = destination / "evaluation" / "cases-reversed.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            "COPY (SELECT * FROM read_parquet(?) ORDER BY case_id DESC) TO ? (FORMAT PARQUET)",
            [str(cases), str(temporary)],
        )
    finally:
        connection.close()
    temporary.replace(cases)


def _compare(source: Path, rebuilt: Path, reversed_db: Path, cases: list[dict]) -> str:
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute(f"ATTACH '{source.as_posix()}' AS old (READ_ONLY)")
        connection.execute(f"ATTACH '{rebuilt.as_posix()}' AS rebuilt (READ_ONLY)")
        connection.execute(f"ATTACH '{reversed_db.as_posix()}' AS reversed (READ_ONLY)")
        corpus_gap = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM (SELECT * FROM old.corpora.audio_files EXCEPT SELECT * FROM rebuilt.corpora.audio_files)),
                (SELECT count(*) FROM (SELECT * FROM rebuilt.corpora.audio_files EXCEPT SELECT * FROM old.corpora.audio_files))
            """
        ).fetchone()
        case_gap = connection.execute(
            """
            SELECT count(*) FROM (
                SELECT case_id, content_hash FROM rebuilt.eval.cases
                EXCEPT
                SELECT case_id, content_hash FROM reversed.eval.cases
            )
            """
        ).fetchone()[0]
        item_gap = connection.execute(
            """
            SELECT count(*) FROM (
                SELECT * FROM rebuilt.eval.set_items
                EXCEPT
                SELECT * FROM reversed.eval.set_items
            )
            """
        ).fetchone()[0]
        old_items = connection.execute(
            """
            SELECT i.set_name, i.position, c.case_id, c.kind, c.dataset, c.split, c.transcript, c.speaker_id,
                   c.clean_path, c.noisy_path, c.audio_path, c.noise_path, c.snr_db, c.noise_start, c.mix_seed,
                   c.max_seconds, c.prevent_clip
            FROM old.eval.set_items AS i
            JOIN old.eval.cases AS c ON c.case_id = i.case_id
            ORDER BY i.set_name, i.position
            """
        ).fetchall()
        new_items = connection.execute(
            """
            SELECT i.set_name, i.position, c.case_id, c.kind, c.dataset, c.split, c.transcript, c.speaker_id,
                   c.recipe_json::VARCHAR
            FROM rebuilt.eval.set_items AS i
            JOIN rebuilt.eval.cases AS c ON c.case_id = i.case_id
            ORDER BY i.set_name, i.position
            """
        ).fetchall()
    finally:
        connection.close()
    if corpus_gap != (0, 0):
        raise SystemExit(f"corpora rows differ: {corpus_gap}")
    if case_gap or item_gap:
        raise SystemExit(f"rebuild order changed content: cases {case_gap}, items {item_gap}")
    if len(old_items) != len(new_items):
        raise SystemExit("set item counts differ")
    converted = {case["legacy_case_id"]: case for case in cases}
    for old, new in zip(old_items, new_items, strict=True):
        recipe = json.loads(new[8])
        expected = converted[old[2]]
        if new[0] != old[0] or new[1] != old[1]:
            raise SystemExit(f"set order changed at {old[0]}:{old[1]}")
        if (new[3], new[4], new[5], new[6], None if new[7] is None else str(new[7])) != (
            old[3],
            old[4],
            old[5],
            old[6],
            None if old[7] is None else str(old[7]),
        ):
            raise SystemExit(f"case fields changed for {old[2]}")
        if new[2] != expected["case_id"]:
            raise SystemExit(f"case id drifted for {old[2]}")
        if old[3] == "synthetic":
            operation = recipe["operations"][0]
            if operation["snr_db"] != float(f"{float(old[12]):.6f}"):
                raise SystemExit(f"SNR changed for {old[2]}")
            if operation["noise_start"] != int(old[13]) or operation["seed"] != int(old[14]):
                raise SystemExit(f"noise crop changed for {old[2]}")
            if recipe["sources"]["clean"]["path"] != old[8] or recipe["sources"]["noise"]["path"] != old[11]:
                raise SystemExit(f"synthetic paths changed for {old[2]}")
        else:
            source_name = "noisy" if old[3] == "noisy" else "clean"
            stored = old[9] or old[10] if source_name == "noisy" else old[8] or old[10]
            if recipe["sources"][source_name]["path"] != stored:
                raise SystemExit(f"path changed for {old[2]}")
            if recipe["operations"]:
                raise SystemExit(f"recorded case {old[2]} gained operations")
    lines = [
        f"legacy cases: {len(cases)}",
        f"corpora rows match: {corpus_gap == (0, 0)}",
        f"rebuilds match independent of case order: {case_gap == 0 and item_gap == 0}",
        f"set items match the legacy order and values: {len(new_items)}",
        "production database was not modified",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    source = DEFAULT_SOURCE if not args else Path(args[0])
    try:
        migrate(
            source,
            DEFAULT_STAGING,
            source.parent / "rebuild-a.duckdb",
            source.parent / "rebuild-b.duckdb",
        )
    except (OSError, ValueError, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return 0
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
