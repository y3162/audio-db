"""Parquet helpers and row normalization for the reproduction database."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.experiments.data.db.schema import TableSpec
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import canonical_text, digest

_SQL_TYPE = {
    "row_count": "BIGINT",
    "sample_rate": "INTEGER",
    "frames": "BIGINT",
    "channels": "INTEGER",
    "size_bytes": "BIGINT",
    "version": "INTEGER",
    "member_count": "BIGINT",
    "total_seconds": "DOUBLE",
    "position": "INTEGER",
    "algorithm_version": "INTEGER",
    "generator_version": "INTEGER",
    "seed": "BIGINT",
    "shard_size": "INTEGER",
    "gen_index": "BIGINT",
    "world_size": "INTEGER",
}
_BOOLEAN = {"selection_uses_test"}


def plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def normalize_row(row: dict[str, Any], spec: TableSpec) -> dict[str, Any]:
    normalized = {}
    for column in spec.columns:
        value = plain(row.get(column))
        if column in spec.json_columns and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ReproductionError(f"{spec.name}.{column} is not JSON") from exc
        normalized[column] = value
    return normalized


def row_digest(row: dict[str, Any], spec: TableSpec) -> str:
    return digest(normalize_row(row, spec))


def write_parquet(path: Path, spec: TableSpec, rows: list[dict[str, Any]]) -> None:
    import duckdb

    definitions = []
    for column in spec.columns:
        if column in spec.json_columns:
            kind = "VARCHAR"
        elif column in _BOOLEAN:
            kind = "BOOLEAN"
        else:
            kind = _SQL_TYPE.get(column, "VARCHAR")
        definitions.append(f"{column} {kind}")
    connection = duckdb.connect()
    try:
        connection.execute(f"CREATE TABLE staged ({', '.join(definitions)})")
        if rows:
            placeholders = ", ".join("?" for _ in spec.columns)
            payload = []
            for row in rows:
                item = []
                for column in spec.columns:
                    value = row.get(column)
                    if column in spec.json_columns and value is not None:
                        value = canonical_text(value)
                    item.append(value)
                payload.append(item)
            connection.executemany(
                f"INSERT INTO staged VALUES ({placeholders})",
                payload,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        if partial.exists():
            partial.unlink()
        connection.execute("COPY staged TO ? (FORMAT PARQUET)", [str(partial)])
    except Exception:
        partial = path.with_name(path.name + ".partial")
        if partial.exists():
            partial.unlink()
        raise
    finally:
        connection.close()
    os.replace(partial, path)


def read_parquet(path: Path, spec: TableSpec) -> list[dict[str, Any]]:
    import duckdb

    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            f"SELECT {', '.join(spec.columns)} FROM read_parquet(?)",
            [str(path)],
        )
        rows = []
        for record in cursor.fetchall():
            rows.append(normalize_row(dict(zip(spec.columns, record, strict=True)), spec))
    except ReproductionError:
        raise
    except Exception as exc:
        raise ReproductionError(f"cannot read {path}: {exc}") from exc
    finally:
        connection.close()
    return rows


def read_table(connection, spec: TableSpec) -> list[dict[str, Any]]:
    order = ", ".join(spec.pk)
    cursor = connection.execute(f"SELECT {', '.join(spec.columns)} FROM repro.{spec.name} ORDER BY {order}")
    return [normalize_row(dict(zip(spec.columns, record, strict=True)), spec) for record in cursor.fetchall()]
