"""Check asset files and record the result as an audit.

A hash mismatch is stored as FAIL and the asset row is left unchanged.
A null sha256 with a matching size is filled from a staged hash update.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.experiments.data.db.graph import Graph, index_rows
from src.experiments.data.db.lock import reproduction_lock
from src.experiments.data.db.registry import read_graph, register_staging
from src.experiments.data.db.snapshot import resolve_dataset
from src.experiments.data.db.staging import write_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import audit_id_for
from src.experiments.data.paths import DEFAULT_CORPORA, DEFAULT_DERIVED, DEFAULT_GENERATED, DEFAULT_STAGING


def verify_files(
    database: str | Path,
    *,
    staging: str | Path = DEFAULT_STAGING,
    dataset: str | None = None,
    build: str | None = None,
    check_hash: bool = False,
    corpora: str | Path = DEFAULT_CORPORA,
    generated: str | Path = DEFAULT_GENERATED,
    derived: str | Path = DEFAULT_DERIVED,
) -> bool:
    """Return True when every checked file matches. FAIL audits stay in the database."""
    dataset_name = (dataset or "").strip()
    build_name = (build or "").strip()
    if bool(dataset_name) == bool(build_name):
        raise ReproductionError("specify exactly one of DATASET or BUILD")
    database = Path(database)
    roots = {
        "corpora": Path(corpora),
        "generated": Path(generated),
        "derived": Path(derived),
    }
    with reproduction_lock(database.parent):
        graph = read_graph(database)
        subject_kind, subject_id, assets = _scope(graph, dataset_name, build_name)
        checks = []
        updates = []
        for asset in assets:
            check, update = _check_asset(asset, roots, check_hash=check_hash)
            checks.append(check)
            if update is not None:
                updates.append(update)
        failed = any(check["status"] == "FAIL" for check in checks)
        if failed:
            updates = []
        report = {"status": "FAIL" if failed else "PASS", "hash": bool(check_hash), "checks": checks}
        audit = {
            "audit_id": audit_id_for(
                subject_kind=subject_kind,
                subject_id=subject_id,
                audit_kind="hash_verify",
                report=report,
            ),
            "subject_kind": subject_kind,
            "subject_id": subject_id,
            "audit_kind": "hash_verify",
            "status": report["status"],
            "report_json": report,
        }
        write_graph(staging, Graph(audits=[audit], hash_updates=updates))
        register_staging(staging, database)
    return not failed


def _scope(graph: Graph, dataset_name: str, build_name: str) -> tuple[str, str, list[dict[str, Any]]]:
    assets = index_rows(graph.assets, "asset_id", "assets")
    if build_name:
        builds = index_rows(graph.builds, "build_id", "builds")
        matches = [row for row in graph.builds if row["build_id"] == build_name or row["generator"] == build_name]
        if build_name not in builds and len(matches) != 1:
            raise ReproductionError(f"build selector {build_name} matched {len(matches)} rows")
        build_id = build_name if build_name in builds else matches[0]["build_id"]
        chosen = [assets[key] for key in sorted(assets) if assets[key].get("build_id") == build_id]
        return "build", build_id, chosen
    dataset = resolve_dataset(graph.datasets, dataset_name)
    case_sets = index_rows(graph.case_sets, "case_set_id", "case_sets")
    materials = index_rows(graph.materials, "material_set_id", "materials")
    cases = index_rows(graph.cases, "case_id", "cases")
    found: set[str] = set()
    for binding in dataset["bindings"]:
        if binding.get("case_set_id"):
            for item in case_sets[binding["case_set_id"]]["items"]:
                case = cases[item["case_id"]]
                for key in ("clean_asset_id", "noisy_asset_id", "noise_asset_id"):
                    if case.get(key):
                        found.add(case[key])
        if binding.get("material_set_id"):
            for member in materials[binding["material_set_id"]]["members"]:
                found.add(member["asset_id"])
    return "dataset", dataset["dataset_id"], [assets[key] for key in sorted(found)]


def _check_asset(asset: dict[str, Any], roots: dict[str, Path], *, check_hash: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = roots[asset["storage_root"]] / asset["rel_path"]
    base = {"asset_id": asset["asset_id"], "size_bytes": asset.get("size_bytes"), "sha256": asset.get("sha256")}
    if not path.is_file():
        return {**base, "status": "FAIL", "reason": "missing"}, None
    actual_size = path.stat().st_size
    if asset.get("size_bytes") is None or int(asset["size_bytes"]) != actual_size:
        return {**base, "status": "FAIL", "reason": "size", "actual_size": actual_size}, None
    if not check_hash:
        return {**base, "status": "PASS", "reason": "size"}, None
    actual_hash = _sha256(path)
    if asset.get("sha256"):
        if asset["sha256"] != actual_hash:
            return {**base, "status": "FAIL", "reason": "sha256", "actual_sha256": actual_hash}, None
        return {**base, "status": "PASS", "reason": "match"}, None
    if asset["integrity"] != "size":
        return {**base, "status": "FAIL", "reason": "sha256_absent"}, None
    filled = {**base, "status": "PASS", "reason": "filled", "sha256": actual_hash}
    update = {"asset_id": asset["asset_id"], "sha256": actual_hash, "size_bytes": actual_size}
    return filled, update


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
