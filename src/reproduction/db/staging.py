"""Content-addressed staging for the reproduction database.

Staging is the rebuild source. Files are written once. A second write of the
same logical rows is a no-op. Different contents at the same id are an error.
``meta.json`` is provenance and is not part of any id.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.reproduction.db.graph import Graph, logical_bindings
from src.reproduction.db.io import read_parquet, write_parquet
from src.reproduction.db.schema import TABLES, TableSpec
from src.reproduction.db.validate import CATALOG_FIELDS
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import asset_batch_id, canonical_text, digest
from src.reproduction.paths import ROOT

CATALOG_SPEC = TableSpec("catalog_rows", ("path",), CATALOG_FIELDS)


def write_graph(staging: str | Path, graph: Graph) -> None:
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str]] = []
    for catalog in graph.catalogs:
        rows = graph.catalog_rows.get(catalog["catalog_id"])
        if rows is None:
            raise ReproductionError(f"catalog {catalog['catalog_id']} has no rows")
        directory = staging / "catalog" / catalog["catalog_id"]
        _write_json(directory / "catalog.json", _public_row(catalog, ("catalog_id", "source_kind", "corpora_json", "row_count", "source_db_sha256")))
        _write_rows(directory / "audio_files.parquet", CATALOG_SPEC, rows, "path")
        _write_meta(directory)
        entries.append(("catalog", catalog["catalog_id"]))
    if graph.assets:
        batch_id = asset_batch_id(graph.assets)
        directory = staging / "assets" / batch_id
        _write_rows(directory / "assets.parquet", TABLES["assets"], graph.assets, "asset_id")
        if asset_batch_id(_read_rows(directory / "assets.parquet", TABLES["assets"])) != batch_id:
            raise ReproductionError(f"asset batch {batch_id} does not match its rows")
        _write_meta(directory)
        entries.append(("asset_batch", batch_id))
    for derivation in graph.derivations:
        _write_json(staging / "derivations" / f"{derivation['derivation_id']}.json", derivation)
        entries.append(("derivation", derivation["derivation_id"]))
    for material in graph.materials:
        directory = staging / "materials" / material["material_set_id"]
        body = {key: material[key] for key in TABLES["material_sets"].columns}
        _write_json(directory / "set.json", body)
        members = [
            {
                "material_set_id": material["material_set_id"],
                "position": member["position"],
                "asset_id": member["asset_id"],
                "group_key": member.get("group_key"),
                "attrs_json": member.get("attrs_json"),
            }
            for member in material["members"]
        ]
        _write_rows(directory / "members.parquet", TABLES["material_set_members"], members, "position")
        entries.append(("material_set", material["material_set_id"]))
    for protocol in graph.protocols:
        _write_json(staging / "protocols" / f"{protocol['protocol_id']}.json", protocol)
        entries.append(("protocol", protocol["protocol_id"]))
    cases = {case["case_id"]: case for case in graph.cases}
    for case_set in graph.case_sets:
        directory = staging / "cases" / case_set["case_set_id"]
        body = {key: case_set[key] for key in TABLES["case_sets"].columns}
        _write_json(directory / "set.json", body)
        items = [
            {
                "case_set_id": case_set["case_set_id"],
                "position": item["position"],
                "case_id": item["case_id"],
            }
            for item in case_set["items"]
        ]
        owned = []
        for item in items:
            case = cases.get(item["case_id"])
            if case is None:
                raise ReproductionError(f"case set {case_set['name']} is missing case {item['case_id']}")
            owned.append(case)
        _write_rows(directory / "items.parquet", TABLES["case_set_items"], items, "position")
        _write_rows(directory / "cases.parquet", TABLES["cases"], owned, "case_id")
        entries.append(("case_set", case_set["case_set_id"]))
    for dataset in graph.datasets:
        directory = staging / "datasets" / dataset["dataset_id"]
        body = {key: dataset[key] for key in TABLES["datasets"].columns}
        _write_json(directory / "dataset.json", body)
        _write_json(directory / "bindings.json", list(dataset["bindings"]))
        entries.append(("dataset", dataset["dataset_id"]))
    for build in graph.builds:
        directory = staging / "builds" / build["build_id"]
        body = {key: build[key] for key in TABLES["builds"].columns}
        _write_json(directory / "build.json", body)
        marker = graph.verified.get(build["build_id"])
        if marker is not None:
            _write_json(directory / "verified" / "complete.json", marker)
        entries.append(("build", build["build_id"]))
    for audit in graph.audits:
        _write_json(staging / "audits" / f"{audit['audit_id']}.json", audit)
        entries.append(("audit", audit["audit_id"]))
    for run in graph.runs:
        _write_json(staging / "runs" / f"{run['run_id']}.json", run)
        entries.append(("run", run["run_id"]))
    if graph.hash_updates:
        ordered = sorted(graph.hash_updates, key=lambda item: item["asset_id"])
        update_id = digest(ordered)
        _write_json(staging / "asset_hashes" / f"{update_id}.json", {"updates": ordered})
        entries.append(("asset_hash", update_id))
    _update_index(staging, entries)


def load_graph(staging: str | Path) -> Graph:
    staging = Path(staging)
    index_path = staging / "INDEX.json"
    if not index_path.is_file():
        raise ReproductionError(f"staging INDEX.json is missing: {staging}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("entries"), list):
        raise ReproductionError("staging INDEX.json must contain entries")
    discovered: list[tuple[str, str]] = []
    graph = Graph()
    catalog_root = staging / "catalog"
    if catalog_root.is_dir():
        for directory in _dirs(catalog_root):
            catalog = _read_json(directory / "catalog.json")
            if catalog.get("catalog_id") != directory.name:
                raise ReproductionError(f"catalog directory does not match its id: {directory.name}")
            rows = read_parquet(directory / "audio_files.parquet", CATALOG_SPEC)
            graph.catalogs.append(catalog)
            graph.catalog_rows[catalog["catalog_id"]] = rows
            discovered.append(("catalog", directory.name))
    asset_root = staging / "assets"
    if asset_root.is_dir():
        for directory in _dirs(asset_root):
            rows = read_parquet(directory / "assets.parquet", TABLES["assets"])
            if asset_batch_id(rows) != directory.name:
                raise ReproductionError(f"asset batch {directory.name} does not match its contents")
            graph.assets.extend(rows)
            discovered.append(("asset_batch", directory.name))
    derivation_root = staging / "derivations"
    if derivation_root.is_dir():
        for path in sorted(derivation_root.glob("*.json")):
            if path.name.endswith(".partial"):
                continue
            row = _read_json(path)
            if row.get("derivation_id") != path.stem:
                raise ReproductionError(f"derivation file does not match its id: {path.name}")
            graph.derivations.append(row)
            discovered.append(("derivation", path.stem))
    material_root = staging / "materials"
    if material_root.is_dir():
        for directory in _dirs(material_root):
            body = _read_json(directory / "set.json")
            if body.get("material_set_id") != directory.name:
                raise ReproductionError(f"material set directory does not match its id: {directory.name}")
            members = read_parquet(directory / "members.parquet", TABLES["material_set_members"])
            body["members"] = sorted(members, key=lambda item: item["position"])
            graph.materials.append(body)
            discovered.append(("material_set", directory.name))
    protocol_root = staging / "protocols"
    if protocol_root.is_dir():
        for path in sorted(protocol_root.glob("*.json")):
            if path.name.endswith(".partial"):
                continue
            row = _read_json(path)
            if row.get("protocol_id") != path.stem:
                raise ReproductionError(f"protocol file does not match its id: {path.name}")
            graph.protocols.append(row)
            discovered.append(("protocol", path.stem))
    case_root = staging / "cases"
    if case_root.is_dir():
        for directory in _dirs(case_root):
            body = _read_json(directory / "set.json")
            if body.get("case_set_id") != directory.name:
                raise ReproductionError(f"case set directory does not match its id: {directory.name}")
            items = read_parquet(directory / "items.parquet", TABLES["case_set_items"])
            body["items"] = sorted(items, key=lambda item: item["position"])
            graph.case_sets.append(body)
            graph.cases.extend(read_parquet(directory / "cases.parquet", TABLES["cases"]))
            discovered.append(("case_set", directory.name))
    dataset_root = staging / "datasets"
    if dataset_root.is_dir():
        for directory in _dirs(dataset_root):
            body = _read_json(directory / "dataset.json")
            if body.get("dataset_id") != directory.name:
                raise ReproductionError(f"dataset directory does not match its id: {directory.name}")
            bindings = _read_json(directory / "bindings.json")
            if not isinstance(bindings, list):
                raise ReproductionError(f"dataset {directory.name} bindings must be a list")
            body["bindings"] = logical_bindings(bindings)
            graph.datasets.append(body)
            discovered.append(("dataset", directory.name))
    build_root = staging / "builds"
    if build_root.is_dir():
        for directory in _dirs(build_root):
            body = _read_json(directory / "build.json")
            if body.get("build_id") != directory.name:
                raise ReproductionError(f"build directory does not match its id: {directory.name}")
            graph.builds.append(body)
            marker = directory / "verified" / "complete.json"
            if marker.is_file():
                graph.verified[body["build_id"]] = _read_json(marker)
            discovered.append(("build", directory.name))
    for kind, folder, field in (
        ("audit", "audits", "audit_id"),
        ("run", "runs", "run_id"),
    ):
        root = staging / folder
        if not root.is_dir():
            continue
        target = graph.audits if kind == "audit" else graph.runs
        for path in sorted(root.glob("*.json")):
            if path.name.endswith(".partial"):
                continue
            row = _read_json(path)
            if row.get(field) != path.stem:
                raise ReproductionError(f"{kind} file does not match its id: {path.name}")
            target.append(row)
            discovered.append((kind, path.stem))
    hash_root = staging / "asset_hashes"
    if hash_root.is_dir():
        for path in sorted(hash_root.glob("*.json")):
            if path.name.endswith(".partial"):
                continue
            body = _read_json(path)
            updates = body.get("updates")
            if not isinstance(updates, list) or digest(updates) != path.stem:
                raise ReproductionError(f"asset hash file does not match its contents: {path.name}")
            graph.hash_updates.extend(updates)
            discovered.append(("asset_hash", path.stem))
    indexed = {(entry["kind"], entry["id"]) for entry in index["entries"]}
    found = set(discovered)
    if indexed != found:
        missing = sorted(indexed - found)
        extra = sorted(found - indexed)
        raise ReproductionError(f"staging INDEX does not match files; missing={missing}; extra={extra}")
    return graph


def _dirs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".") and not path.name.endswith(".partial")
    )


def _public_row(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    return {column: row.get(column) for column in columns}


def _write_rows(path: Path, spec: TableSpec, rows: list[dict[str, Any]], order_key: str) -> None:
    ordered = sorted(rows, key=lambda row: row[order_key])
    if path.is_file():
        existing = sorted(read_parquet(path, spec), key=lambda row: row[order_key])
        if digest(existing) != digest(ordered):
            raise ReproductionError(f"staging file already exists with different content: {path}")
        return
    write_parquet(path, spec, ordered)


def _read_rows(path: Path, spec: TableSpec) -> list[dict[str, Any]]:
    return read_parquet(path, spec)


def _write_json(path: Path, value: Any) -> None:
    text = canonical_text(value) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ReproductionError(f"staging file already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionError(f"cannot read {path}: {exc}") from exc


def _write_meta(directory: Path) -> None:
    path = directory / "meta.json"
    if path.exists():
        return
    _write_json(
        path,
        {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
        },
    )


def _update_index(staging: Path, entries: list[tuple[str, str]]) -> None:
    path = staging / "INDEX.json"
    current: list[dict[str, str]] = []
    if path.is_file():
        current = list(_read_json(path)["entries"])
    seen = {(entry["kind"], entry["id"]) for entry in current}
    for kind, ident in entries:
        if any(entry["id"] == ident and entry["kind"] != kind for entry in current):
            raise ReproductionError(f"staging id {ident} is already indexed as a different kind")
        if (kind, ident) not in seen:
            current.append({"kind": kind, "id": ident})
            seen.add((kind, ident))
    ordered = sorted(current, key=lambda entry: (entry["kind"], entry["id"]))
    text = canonical_text({"entries": ordered}) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("INDEX.json.partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
