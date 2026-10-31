"""Read-only selection from corpora.audio_files.

This module never creates or recreates the database. Ingest stays a separate
manual step. Paths stored in the table are relative to the corpora root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.models.audio_file import AUDIO_FILE_COLUMNS
from src.evaluation.errors import DatasetError
from src.evaluation.sample import AudioSample


def load_duckdb(
    path: str | Path,
    *,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
    role: str = "clean",
    corpora_root: str | Path = "data/corpora",
) -> list[AudioSample]:
    database = Path(path)
    if not database.is_file():
        raise DatasetError(
            f"DuckDB not found: {database}. Create it with make db-init, or select an "
            "existing database. This pipeline does not run corpus ingest."
        )
    rows = _query(database, filters or {}, limit)
    if not rows:
        raise DatasetError(f"DuckDB query returned no rows from {database}")
    root = Path(corpora_root)
    return [_row_to_sample(row, root, role) for row in rows]


def fetch_audio_files(connection, filters: dict[str, Any] | None, limit: int | None) -> list[dict[str, Any]]:
    """Read corpora.audio_files through an open connection. Rows are ordered by path."""
    columns = ", ".join(AUDIO_FILE_COLUMNS)
    clauses: list[str] = []
    params: list[Any] = []
    for key in AUDIO_FILE_COLUMNS:
        if not filters or key not in filters or filters[key] is None:
            continue
        clauses.append(f"{key} = ?")
        params.append(filters[key])
    sql = f"SELECT {columns} FROM corpora.audio_files"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY path"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    try:
        cursor = connection.execute(sql, params)
    except Exception as exc:
        raise DatasetError(f"DuckDB query failed: {exc}") from exc
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _query(database: Path, filters: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:
        raise DatasetError("duckdb is not installed in this environment") from exc
    try:
        connection = duckdb.connect(str(database), read_only=True)
    except duckdb.Error as exc:
        raise DatasetError(f"failed to open DuckDB read-only: {exc}") from exc
    try:
        return fetch_audio_files(connection, filters, limit)
    finally:
        connection.close()


def _row_to_sample(row: dict[str, Any], corpora_root: Path, role: str) -> AudioSample:
    stored = Path(row["path"])
    resolved = stored if stored.is_absolute() else corpora_root / stored
    path = resolved.as_posix()
    utterance_id = row["path"]
    sample = AudioSample(
        utterance_id=utterance_id,
        dataset=row["corpus"],
        split=row.get("subset") or row.get("eval_split"),
        audio_path=path,
        sample_rate=int(row["sample_rate"]) if row.get("sample_rate") is not None else None,
        transcript=row.get("transcript"),
        speaker_id=row.get("speaker_id"),
        metadata={
            "relative_path": row["path"],
            "file_name": row.get("file_name"),
            "utterance_id": row.get("utterance_id"),
            "track_type": row.get("track_type"),
            "noise_type": row.get("noise_type"),
            "scene_code": row.get("scene_code"),
        },
    )
    if role == "clean":
        sample.clean_path = path
    elif role == "noisy":
        sample.noisy_path = path
    elif role == "noise":
        sample.noise_path = path
    else:
        raise DatasetError(f"unknown dataset role: {role}")
    return sample
