"""Register staging into reproduction.duckdb and rebuild it.

Registration is one transaction. The same id with the same content is a no-op.
The same id with different content is an error and leaves the database unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.reproduction.db.audit import audits_for
from src.reproduction.db.graph import Graph, logical_bindings
from src.reproduction.db.io import read_table, row_digest, write_parquet
from src.reproduction.db.lock import reproduction_lock
from src.reproduction.db.schema import INSERT_ORDER, TABLE_ORDER, TABLES, TableSpec, assert_ready, create_schema, mark_ready
from src.reproduction.db.staging import load_graph, write_graph
from src.reproduction.db.validate import validate_graph
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest

_NAMED = ("material_sets", "protocols", "case_sets", "datasets")


def initialize_database(database: str | Path) -> None:
    """Create an empty ready database. An existing ready database is left as it is."""
    path = Path(database)
    with reproduction_lock(path.parent):
        if path.exists():
            connection = _connect(path, read_only=True)
            try:
                assert_ready(connection)
            finally:
                connection.close()
            return
        partial = path.with_name(path.name + ".partial")
        if partial.exists():
            partial.unlink()
        connection = _connect(partial, read_only=False)
        try:
            create_schema(connection)
            mark_ready(connection)
        except Exception:
            connection.close()
            if partial.exists():
                partial.unlink()
            raise
        connection.close()
        os.replace(partial, path)


def register_staging(
    staging: str | Path,
    database: str | Path,
    *,
    resource_db: str | Path | None = None,
) -> dict[str, int]:
    """Validate staging and insert missing rows. Failures roll back."""
    staging = Path(staging)
    path = Path(database)
    resource = None if resource_db is None else str(resource_db)
    with reproduction_lock(path.parent):
        if not path.is_file():
            raise ReproductionError(f"database does not exist: {path}. Run reproduction-init-db first.")
        incoming = load_graph(staging)
        _apply_updates_in_memory(incoming)
        connection = _connect(path, read_only=False)
        try:
            assert_ready(connection)
            existing = _read_graph(connection)
            merged = existing.extend(incoming)
            merged.assets = _collapse_assets(merged.assets)
            generated_ids = {
                asset["asset_id"] for asset in incoming.assets if asset["storage_root"] == "generated"
            }
            validate_graph(merged, resource_db=resource, require_verified=generated_ids)
            computed = audits_for(merged, [row["dataset_id"] for row in incoming.datasets], resource)
            _stage_audits(staging, computed)
            incoming.audits.extend(computed)
            existing_rows = _flat(existing)
            incoming_rows = _flat(incoming)
            _reject_conflicts(existing_rows, incoming_rows)
            _reject_name_collisions(existing_rows, incoming_rows)
            connection.execute("BEGIN TRANSACTION")
            try:
                counts = _insert_missing(connection, existing_rows, incoming_rows)
                _apply_hash_updates(connection, incoming.hash_updates)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()
    return counts


def rebuild_database(
    staging: str | Path,
    database: str | Path,
    *,
    reference: str | Path,
    resource_db: str | Path | None = None,
) -> dict[str, str]:
    """Build a new database from staging and require it to match ``reference``."""
    destination = Path(database)
    source = Path(reference)
    if destination.exists():
        raise ReproductionError(f"refusing to overwrite an existing database: {destination}")
    if not source.is_file():
        raise ReproductionError(f"reference database does not exist: {source}")
    partial = destination.with_name(destination.name + ".partial")
    with reproduction_lock(destination.parent):
        if partial.exists():
            partial.unlink()
        try:
            initialize_database(partial)
            register_staging(staging, partial, resource_db=resource_db)
            rebuilt = table_hashes(partial)
            original = table_hashes(source)
            if rebuilt != original:
                mismatched = sorted(name for name in rebuilt if rebuilt[name] != original.get(name))
                raise ReproductionError(
                    "rebuilt database does not match the reference: " + ", ".join(mismatched)
                )
            os.replace(partial, destination)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise
    return rebuilt


def read_graph(database: str | Path) -> Graph:
    """Read a ready database. The caller does not need the write lock."""
    path = Path(database)
    connection = _connect(path, read_only=True)
    try:
        assert_ready(connection)
        return _read_graph(connection)
    finally:
        connection.close()


def table_hashes(database: str | Path) -> dict[str, str]:
    """Content hash of every table, ordered by primary key."""
    path = Path(database)
    connection = _connect(path, read_only=True)
    try:
        assert_ready(connection)
        hashes = {}
        for name in TABLE_ORDER:
            spec = TABLES[name]
            hashes[name] = digest(read_table(connection, spec))
        return hashes
    finally:
        connection.close()


def _stage_audits(staging: Path, audits: list[dict[str, Any]]) -> None:
    if not audits:
        return
    write_graph(staging, Graph(audits=audits))


def _read_graph(connection) -> Graph:
    graph = Graph(
        catalogs=read_table(connection, TABLES["catalog_snapshots"]),
        assets=read_table(connection, TABLES["assets"]),
        derivations=read_table(connection, TABLES["derivations"]),
        protocols=read_table(connection, TABLES["protocols"]),
        cases=read_table(connection, TABLES["cases"]),
        builds=read_table(connection, TABLES["builds"]),
        audits=read_table(connection, TABLES["audits"]),
        runs=read_table(connection, TABLES["runs"]),
    )
    members: dict[str, list[dict[str, Any]]] = {}
    for member in read_table(connection, TABLES["material_set_members"]):
        members.setdefault(member["material_set_id"], []).append(member)
    for material in read_table(connection, TABLES["material_sets"]):
        material["members"] = sorted(members.get(material["material_set_id"], []), key=lambda item: item["position"])
        graph.materials.append(material)
    items: dict[str, list[dict[str, Any]]] = {}
    for item in read_table(connection, TABLES["case_set_items"]):
        items.setdefault(item["case_set_id"], []).append(item)
    for case_set in read_table(connection, TABLES["case_sets"]):
        case_set["items"] = sorted(items.get(case_set["case_set_id"], []), key=lambda item: item["position"])
        graph.case_sets.append(case_set)
    bindings: dict[str, list[dict[str, Any]]] = {}
    for binding in read_table(connection, TABLES["dataset_bindings"]):
        bindings.setdefault(binding["dataset_id"], []).append(binding)
    for dataset in read_table(connection, TABLES["datasets"]):
        dataset["bindings"] = logical_bindings(bindings.get(dataset["dataset_id"], []))
        graph.datasets.append(dataset)
    return graph


def _flat(graph: Graph) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in INSERT_ORDER}
    rows["catalog_snapshots"] = [_project(row, TABLES["catalog_snapshots"]) for row in graph.catalogs]
    rows["derivations"] = [_project(row, TABLES["derivations"]) for row in graph.derivations]
    rows["builds"] = [_project(row, TABLES["builds"]) for row in graph.builds]
    rows["assets"] = [_project(row, TABLES["assets"]) for row in graph.assets]
    for material in graph.materials:
        rows["material_sets"].append(_project(material, TABLES["material_sets"]))
        for member in material["members"]:
            rows["material_set_members"].append(
                {
                    "material_set_id": material["material_set_id"],
                    "position": member["position"],
                    "asset_id": member["asset_id"],
                    "group_key": member.get("group_key"),
                    "attrs_json": member.get("attrs_json"),
                }
            )
    rows["protocols"] = [_project(row, TABLES["protocols"]) for row in graph.protocols]
    rows["cases"] = [_project(row, TABLES["cases"]) for row in graph.cases]
    for case_set in graph.case_sets:
        rows["case_sets"].append(_project(case_set, TABLES["case_sets"]))
        for item in case_set["items"]:
            rows["case_set_items"].append(
                {
                    "case_set_id": case_set["case_set_id"],
                    "position": item["position"],
                    "case_id": item["case_id"],
                }
            )
    for dataset in graph.datasets:
        rows["datasets"].append(_project(dataset, TABLES["datasets"]))
        for binding in dataset["bindings"]:
            rows["dataset_bindings"].append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "slot": binding["slot"],
                    "case_set_id": binding.get("case_set_id"),
                    "material_set_id": binding.get("material_set_id"),
                    "params_json": binding.get("params_json"),
                }
            )
    rows["audits"] = [_project(row, TABLES["audits"]) for row in graph.audits]
    rows["runs"] = [_project(row, TABLES["runs"]) for row in graph.runs]
    return {name: _dedupe(table_rows, TABLES[name]) for name, table_rows in rows.items()}


def _project(row: dict[str, Any], spec: TableSpec) -> dict[str, Any]:
    return {column: row.get(column) for column in spec.columns}


def _primary_key(row: dict[str, Any], spec: TableSpec) -> tuple[Any, ...]:
    return tuple(row[column] for column in spec.pk)


def _dedupe(rows: list[dict[str, Any]], spec: TableSpec) -> list[dict[str, Any]]:
    found: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = _primary_key(row, spec)
        previous = found.get(key)
        if previous is None:
            found[key] = row
            continue
        if row_digest(previous, spec) != row_digest(row, spec):
            raise ReproductionError(f"{spec.name} {key} is staged with different content")
    return list(found.values())


def _reject_conflicts(
    existing: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
) -> None:
    for name in INSERT_ORDER:
        spec = TABLES[name]
        current = {_primary_key(row, spec): row for row in existing.get(name, [])}
        for row in incoming.get(name, []):
            key = _primary_key(row, spec)
            stored = current.get(key)
            if stored is None or _compatible(stored, row, spec):
                continue
            raise ReproductionError(f"{name} {key} is already registered with different content")


def _reject_name_collisions(
    existing: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
) -> None:
    for name in _NAMED:
        spec = TABLES[name]
        id_column = spec.pk[0]
        known = {(row["name"], row["version"]): row[id_column] for row in existing.get(name, [])}
        for row in incoming.get(name, []):
            key = (row["name"], row["version"])
            if key in known and known[key] != row[id_column]:
                raise ReproductionError(
                    f"{name} name {row['name']} version {row['version']} is already registered"
                )


def _insert_missing(connection, existing, incoming) -> dict[str, int]:
    counts = {}
    for name in INSERT_ORDER:
        spec = TABLES[name]
        present = {_primary_key(row, spec) for row in existing.get(name, [])}
        fresh = [row for row in incoming.get(name, []) if _primary_key(row, spec) not in present]
        counts[name] = len(fresh)
        if not fresh:
            continue
        _insert_rows(connection, spec, fresh)
    return counts


def _insert_rows(connection, spec: TableSpec, rows: list[dict[str, Any]]) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / f"{spec.name}.parquet"
        write_parquet(path, spec, rows)
        selected = []
        for column in spec.columns:
            if column in spec.json_columns:
                selected.append(f"src.{column}::JSON")
            else:
                selected.append(f"src.{column}")
        connection.execute(
            f"""
            INSERT INTO repro.{spec.name} ({", ".join(spec.columns)})
            SELECT {", ".join(selected)}
            FROM read_parquet(?) AS src
            """,
            [str(path)],
        )


def _apply_updates_in_memory(graph: Graph) -> None:
    """Fill null asset hashes from staged updates before validation."""
    updates = {item["asset_id"]: item for item in graph.hash_updates}
    for asset in graph.assets:
        update = updates.get(asset["asset_id"])
        if update is None:
            continue
        if asset.get("size_bytes") is not None and int(asset["size_bytes"]) != int(update["size_bytes"]):
            raise ReproductionError(f"hash update size does not match {asset['asset_id']}")
        if asset.get("sha256") is None:
            asset["sha256"] = update["sha256"]
            asset["integrity"] = "sha256"
            continue
        if asset["sha256"] != update["sha256"] or asset["integrity"] != "sha256":
            raise ReproductionError(f"asset {asset['asset_id']} sha256 does not match the staged hash")


def _collapse_assets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        previous = found.get(row["asset_id"])
        if previous is None or digest(previous) == digest(row):
            found[row["asset_id"]] = previous or row
            continue
        if _is_hash_backfill(previous, row):
            found[row["asset_id"]] = row
            continue
        if _is_hash_backfill(row, previous):
            continue
        raise ReproductionError(f"assets ({row['asset_id']},) is staged with different content")
    return list(found.values())


def _compatible(stored: dict[str, Any], row: dict[str, Any], spec: TableSpec) -> bool:
    if row_digest(stored, spec) == row_digest(row, spec):
        return True
    return spec.name == "assets" and _is_hash_backfill(stored, row)


def _is_hash_backfill(stored: dict[str, Any], filled: dict[str, Any]) -> bool:
    """True when ``filled`` is ``stored`` plus a null sha256 that is now known."""
    if stored.get("sha256") is not None or filled.get("sha256") is None:
        return False
    if stored.get("integrity") != "size" or filled.get("integrity") != "sha256":
        return False
    left = dict(stored)
    left["sha256"] = filled["sha256"]
    left["integrity"] = "sha256"
    return digest(left) == digest(filled)


def _apply_hash_updates(connection, updates: list[dict[str, Any]]) -> None:
    for update in sorted(updates, key=lambda item: item["asset_id"]):
        row = connection.execute(
            "SELECT sha256, integrity, size_bytes FROM repro.assets WHERE asset_id = ?",
            [update["asset_id"]],
        ).fetchone()
        if row is None:
            raise ReproductionError(f"hash update references a missing asset {update['asset_id']}")
        sha256, integrity, size_bytes = row
        if size_bytes is not None and int(size_bytes) != int(update["size_bytes"]):
            raise ReproductionError(f"hash update size does not match {update['asset_id']}")
        if sha256 is None:
            connection.execute(
                """
                UPDATE repro.assets
                SET sha256 = ?, integrity = 'sha256'
                WHERE asset_id = ? AND sha256 IS NULL
                """,
                [update["sha256"], update["asset_id"]],
            )
            continue
        if sha256 == update["sha256"] and integrity == "sha256":
            continue
        raise ReproductionError(f"asset {update['asset_id']} sha256 does not match the staged hash")


def _connect(path: Path, *, read_only: bool):
    import duckdb

    try:
        return duckdb.connect(str(path), read_only=read_only)
    except Exception as exc:
        raise ReproductionError(f"cannot open {path}: {exc}") from exc
