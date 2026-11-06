"""Export an immutable snapshot from reproduction.duckdb.

The writer may import DuckDB. The files it produces are read by
``src.reproduction.snapshot_reader``, which does not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.reproduction.db.graph import Graph, index_rows
from src.reproduction.db.lock import reproduction_lock
from src.reproduction.db.registry import read_graph
from src.reproduction.db.staging import git_commit
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import SCHEMA_VERSION, canonical_text, generated_rel_path
from src.reproduction.paths import DEFAULT_CORPORA, DEFAULT_DERIVED, DEFAULT_GENERATED, DEFAULT_SNAPSHOTS
from src.reproduction.snapshot_reader import (
    FORMAT,
    arrays_from_assets,
    arrays_from_cases,
    arrays_from_items,
    arrays_from_members,
    file_hashes,
    load_snapshot,
    snapshot_hash,
)

_CASE_COLUMNS = (
    "case_id",
    "doc_kind",
    "case_role",
    "protocol_id",
    "teacher_kind",
    "sample_rate",
    "frames",
    "speaker_id",
    "utterance_key",
    "clean_asset_id",
    "noisy_asset_id",
    "noise_asset_id",
    "build_id",
    "gen_index",
)


def export_snapshot(
    database: str | Path,
    dataset: str,
    snapshots: str | Path = DEFAULT_SNAPSHOTS,
    *,
    allow_unknown_leak_check: bool = False,
    corpora: str | Path = DEFAULT_CORPORA,
    generated: str | Path = DEFAULT_GENERATED,
    derived: str | Path = DEFAULT_DERIVED,
) -> str:
    """Write ``snapshots/<dataset_id>`` once. The same logical snapshot is a no-op."""
    database = Path(database)
    root = Path(snapshots)
    with reproduction_lock(database.parent):
        graph = read_graph(database)
        selected = resolve_dataset(graph.datasets, dataset)
        _require_ready(selected)
        _require_audits(graph, selected, allow_unknown=allow_unknown_leak_check)
        parts, checks = _parts(graph, selected)
        _check_files(checks, Path(corpora), Path(generated), Path(derived))
        digest_value = snapshot_hash(parts)
        destination = root / selected["dataset_id"]
        if destination.exists():
            _require_same_snapshot(destination, digest_value)
            return digest_value
        root.mkdir(parents=True, exist_ok=True)
        partial = root / f".{selected['dataset_id']}.partial"
        if partial.exists():
            _remove_tree(partial)
        _write_snapshot(
            partial,
            parts,
            digest_value,
            allow_unknown_leak_check=allow_unknown_leak_check,
        )
        loaded = load_snapshot(partial)
        if loaded.snapshot_hash != digest_value:
            _remove_tree(partial)
            raise ReproductionError("snapshot self-check did not match the logical hash")
        os.replace(partial, destination)
    return digest_value


def resolve_dataset(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    if not selector or not str(selector).strip():
        raise ReproductionError("DATASET is required")
    selector = str(selector).strip()
    if "@" in selector:
        name, raw_version = selector.rsplit("@", 1)
        if not raw_version.isdigit():
            raise ReproductionError(f"dataset version is not an integer: {selector}")
        version = int(raw_version)
        matches = [row for row in rows if row["name"] == name and int(row["version"]) == version]
    else:
        matches = [row for row in rows if row["dataset_id"] == selector or row["name"] == selector]
    if len(matches) != 1:
        raise ReproductionError(f"dataset selector {selector} matched {len(matches)} rows")
    return matches[0]


def _require_ready(dataset: dict[str, Any]) -> None:
    if dataset["declared_status"] != "ready":
        raise ReproductionError(
            f"dataset {dataset['name']} declared_status is {dataset['declared_status']}; snapshot requires ready"
        )


def _require_audits(graph: Graph, dataset: dict[str, Any], *, allow_unknown: bool) -> None:
    audits = [
        row
        for row in graph.audits
        if row["subject_kind"] == "dataset" and row["subject_id"] == dataset["dataset_id"]
    ]
    _require_pass(audits, "count_check", dataset["name"])
    _require_pass(audits, "list_hash_check", dataset["name"])
    if any(row["status"] == "FAIL" for row in audits):
        raise ReproductionError(f"dataset {dataset['name']} has a FAIL audit")
    if dataset["profile"] == "leakage_safe":
        leak = [row["status"] for row in audits if row["audit_kind"] == "leak_check"]
        if any(status in ("FAIL", "WARN") for status in leak):
            raise ReproductionError(f"dataset {dataset['name']} leak_check is not usable")
        if "PASS" not in leak:
            if not (allow_unknown and "UNKNOWN" in leak):
                raise ReproductionError(
                    f"dataset {dataset['name']} leak_check is UNKNOWN; pass --allow-unknown-leak-check to export"
                )
    builds = _bound_builds(graph, dataset)
    if builds:
        for build_id in sorted(builds):
            rows = [
                row
                for row in graph.audits
                if row["subject_kind"] == "build"
                and row["subject_id"] == build_id
                and row["audit_kind"] == "build_verification"
            ]
            if any(row["status"] == "FAIL" for row in rows) or not any(row["status"] == "PASS" for row in rows):
                raise ReproductionError(f"build {build_id} has no PASS build_verification")


def _require_pass(audits: list[dict[str, Any]], kind: str, name: str) -> None:
    rows = [row for row in audits if row["audit_kind"] == kind]
    if any(row["status"] != "PASS" for row in rows) or not rows:
        raise ReproductionError(f"dataset {name} {kind} is not PASS")


def _bound_builds(graph: Graph, dataset: dict[str, Any]) -> set[str]:
    case_sets = index_rows(graph.case_sets, "case_set_id", "case_sets")
    cases = index_rows(graph.cases, "case_id", "cases")
    found = set()
    for binding in dataset["bindings"]:
        if not binding.get("case_set_id"):
            continue
        for item in case_sets[binding["case_set_id"]]["items"]:
            case = cases[item["case_id"]]
            if case["case_role"] == "generated" and case.get("build_id"):
                found.add(case["build_id"])
    return found


def _parts(graph: Graph, dataset: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str, int, str]]]:
    protocols = index_rows(graph.protocols, "protocol_id", "protocols")
    case_sets = index_rows(graph.case_sets, "case_set_id", "case_sets")
    materials = index_rows(graph.materials, "material_set_id", "materials")
    cases = index_rows(graph.cases, "case_id", "cases")
    assets = index_rows(graph.assets, "asset_id", "assets")
    builds = index_rows(graph.builds, "build_id", "builds")
    protocol = protocols[dataset["protocol_id"]]
    bindings = []
    item_rows = []
    member_rows = []
    case_ids: list[str] = []
    asset_ids: set[str] = set()
    for binding in sorted(dataset["bindings"], key=lambda item: item["slot"]):
        bindings.append(
            {
                "slot": binding["slot"],
                "case_set_id": binding.get("case_set_id"),
                "material_set_id": binding.get("material_set_id"),
            }
        )
        if binding.get("case_set_id"):
            case_set = case_sets[binding["case_set_id"]]
            for item in sorted(case_set["items"], key=lambda row: row["position"]):
                item_rows.append(
                    {
                        "case_set_id": case_set["case_set_id"],
                        "position": int(item["position"]),
                        "case_id": item["case_id"],
                    }
                )
                case_ids.append(item["case_id"])
        if binding.get("material_set_id"):
            material = materials[binding["material_set_id"]]
            for member in sorted(material["members"], key=lambda row: row["position"]):
                member_rows.append(
                    {
                        "material_set_id": material["material_set_id"],
                        "position": int(member["position"]),
                        "asset_id": member["asset_id"],
                        "group_key": member.get("group_key"),
                    }
                )
                asset_ids.add(member["asset_id"])
    chosen_cases = []
    docs = []
    seen_cases: set[str] = set()
    for case_id in case_ids:
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        case = cases[case_id]
        chosen_cases.append(_case_snapshot(case))
        for key in ("clean_asset_id", "noisy_asset_id", "noise_asset_id"):
            if case.get(key):
                asset_ids.add(case[key])
        delivery = (protocol.get("rule_json") or {}).get("delivery")
        if case["case_role"] != "generated" or delivery == "on_demand":
            docs.append({"case_id": case["case_id"], "doc_json": case["doc_json"]})
    chosen_cases.sort(key=lambda row: row["case_id"])
    docs.sort(key=lambda row: row["case_id"])
    plain_assets = []
    generated_assets = []
    checks: list[tuple[str, str, int, str]] = []
    for asset_id in sorted(asset_ids):
        asset = assets[asset_id]
        if asset.get("size_bytes") is None:
            raise ReproductionError(f"snapshot asset {asset_id} has no size_bytes")
        size = int(asset["size_bytes"])
        checks.append((asset["storage_root"], asset["rel_path"], size, asset_id))
        if asset["storage_root"] == "generated":
            build = builds[asset["build_id"]]
            extra = asset["extra_json"]
            generated_assets.append(
                {
                    "asset_id": asset["asset_id"],
                    "rel_path": asset["rel_path"],
                    "build_id": asset["build_id"],
                    "generator": build["generator"],
                    "stream": extra["stream"],
                    "gen_index": int(extra["gen_index"]),
                    "shard_size": int(build["shard_size"]),
                    "sample_rate": int(asset["sample_rate"]),
                    "frames": int(asset["frames"]),
                    "channels": int(asset["channels"]),
                    "sha256": asset["sha256"],
                    "integrity": asset["integrity"],
                    "size_bytes": size,
                }
            )
            continue
        plain_assets.append(
            {
                "asset_id": asset["asset_id"],
                "rel_path": asset["rel_path"],
                "sample_rate": int(asset["sample_rate"]),
                "frames": int(asset["frames"]),
                "channels": int(asset["channels"]),
                "sha256": asset.get("sha256"),
                "integrity": asset["integrity"],
            }
        )
    for row in generated_assets:
        expected = generated_rel_path(
            generator=row["generator"],
            build_id=row["build_id"],
            stream=row["stream"],
            gen_index=row["gen_index"],
            shard_size=row["shard_size"],
        )
        if expected != row["rel_path"]:
            raise ReproductionError(f"generated asset path mismatch: {row['asset_id']}")
    document = {
        "dataset_id": dataset["dataset_id"],
        "name": dataset["name"],
        "version": int(dataset["version"]),
        "protocol_id": dataset["protocol_id"],
        "algorithm": protocol["algorithm"],
        "algorithm_version": int(protocol["algorithm_version"]),
        "kind": protocol["kind"],
        "rule_json": protocol["rule_json"],
        "profile": dataset["profile"],
        "material_origin": dataset["material_origin"],
        "claim_label": dataset["claim_label"],
        "declared_status": dataset["declared_status"],
        "selection_uses_test": bool(dataset["selection_uses_test"]),
        "selection_note": dataset.get("selection_note"),
        "deviations_json": dataset["deviations_json"],
        "bindings": bindings,
    }
    parts = {
        "dataset": document,
        "assets": plain_assets,
        "cases": chosen_cases,
        "case_set_items": item_rows,
        "material_set_members": member_rows,
        "docs": docs,
        "generated": generated_assets,
    }
    return parts, checks


def _check_files(
    checks: list[tuple[str, str, int, str]],
    corpora: Path,
    generated: Path,
    derived: Path,
) -> None:
    roots = {"corpora": corpora, "generated": generated, "derived": derived}
    for storage_root, rel_path, size, asset_id in checks:
        path = roots[storage_root] / rel_path
        if not path.is_file():
            raise ReproductionError(f"snapshot asset file is missing: {path}")
        actual = path.stat().st_size
        if actual != size:
            raise ReproductionError(f"snapshot asset {asset_id} size is {actual}, database has {size}")


def _write_snapshot(directory: Path, parts: dict[str, Any], digest_value: str, *, allow_unknown_leak_check: bool) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    logical = file_hashes(parts)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_hash": digest_value,
        "files": logical,
        "provenance": {
            "format": FORMAT,
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "allow_unknown_leak_check": allow_unknown_leak_check,
        },
    }
    _write_text(directory / "dataset.json", canonical_text(parts["dataset"]) + "\n")
    _write_text(directory / "generated.json", canonical_text(parts["generated"]) + "\n")
    lines = [canonical_text(row) for row in parts["docs"]]
    _write_text(directory / "docs.jsonl", ("\n".join(lines) + "\n") if lines else "")
    _save_npz(directory / "assets.npz", arrays_from_assets(_public_assets(parts["assets"])))
    _save_npz(directory / "cases.npz", arrays_from_cases(parts["cases"]))
    _save_npz(directory / "case_set_items.npz", arrays_from_items(parts["case_set_items"]))
    _save_npz(directory / "material_set_members.npz", arrays_from_members(parts["material_set_members"]))
    _write_text(directory / "manifest.json", canonical_text(manifest) + "\n")


def _case_snapshot(case: dict[str, Any]) -> dict[str, Any]:
    row = {column: case.get(column) for column in _CASE_COLUMNS}
    row["sample_rate"] = int(row["sample_rate"])
    row["frames"] = None if row["frames"] is None else int(row["frames"])
    row["gen_index"] = None if row["gen_index"] is None else int(row["gen_index"])
    return row


def _public_assets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": row["asset_id"],
            "rel_path": row["rel_path"],
            "sample_rate": row["sample_rate"],
            "frames": row["frames"],
            "channels": row["channels"],
            "sha256": row["sha256"],
            "integrity": row["integrity"],
        }
        for row in rows
    ]


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _require_same_snapshot(directory: Path, digest_value: str) -> None:
    try:
        loaded = load_snapshot(directory)
    except ReproductionError as exc:
        raise ReproductionError(f"existing snapshot cannot be read: {directory}") from exc
    if loaded.snapshot_hash != digest_value:
        raise ReproductionError(f"existing snapshot {directory} has a different snapshot_hash")


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()
