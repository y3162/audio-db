"""Audit rows derived from a validated graph.

Structural audits are PASS because validation has already rejected mismatches.
Leak status is computed again so the stored report matches the decision.
"""

from __future__ import annotations

from typing import Any

from src.reproduction.db.graph import Graph, index_rows
from src.reproduction.db.validate import leak_report
from src.reproduction.ids import audit_id_for


def audits_for(graph: Graph, dataset_ids: list[str], resource_db: str | None) -> list[dict[str, Any]]:
    datasets = index_rows(graph.datasets, "dataset_id", "datasets")
    case_sets = index_rows(graph.case_sets, "case_set_id", "case_sets")
    materials = index_rows(graph.materials, "material_set_id", "materials")
    cases = index_rows(graph.cases, "case_id", "cases")
    assets = index_rows(graph.assets, "asset_id", "assets")
    builds = index_rows(graph.builds, "build_id", "builds")
    rows: list[dict[str, Any]] = []
    for dataset_id in sorted(set(dataset_ids)):
        dataset = datasets[dataset_id]
        rows.append(_audit("dataset", dataset_id, "count_check", _count_report(dataset, case_sets, materials)))
        rows.append(_audit("dataset", dataset_id, "list_hash_check", _list_report(dataset, case_sets, materials)))
        report = leak_report(
            dataset,
            case_sets,
            materials,
            cases,
            assets,
            resource_db=resource_db,
        )
        rows.append(_audit("dataset", dataset_id, "leak_check", report))
    build_ids: set[str] = set()
    for dataset_id in sorted(set(dataset_ids)):
        for case_id in _dataset_case_ids(datasets[dataset_id], case_sets):
            case = cases[case_id]
            if case["case_role"] == "generated" and case.get("build_id"):
                build_ids.add(case["build_id"])
    for build_id in sorted(build_ids):
        marker = graph.verified[build_id]
        asset_count = sum(1 for asset in assets.values() if asset.get("build_id") == build_id)
        case_count = sum(1 for case in cases.values() if case.get("build_id") == build_id and case["case_role"] == "generated")
        rows.append(
            _audit(
                "build",
                build_id,
                "build_verification",
                {
                    "status": "PASS",
                    "generator": builds[build_id]["generator"],
                    "shard_ids": sorted(marker["shard_ids"]),
                    "asset_count": asset_count,
                    "case_count": case_count,
                },
            )
        )
    return rows


def _dataset_case_ids(dataset: dict[str, Any], case_sets: dict[str, dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for binding in dataset["bindings"]:
        if binding.get("case_set_id"):
            found.update(item["case_id"] for item in case_sets[binding["case_set_id"]]["items"])
    return found


def _count_report(dataset, case_sets, materials) -> dict[str, Any]:
    case_reports = []
    material_reports = []
    for binding in sorted(dataset["bindings"], key=lambda item: item["slot"]):
        if binding.get("case_set_id"):
            case_set = case_sets[binding["case_set_id"]]
            case_reports.append(
                {
                    "slot": binding["slot"],
                    "id": case_set["case_set_id"],
                    "member_count": case_set["member_count"],
                    "items": len(case_set["items"]),
                }
            )
        if binding.get("material_set_id"):
            material = materials[binding["material_set_id"]]
            material_reports.append(
                {
                    "slot": binding["slot"],
                    "id": material["material_set_id"],
                    "member_count": material["member_count"],
                    "items": len(material["members"]),
                }
            )
    return {"status": "PASS", "case_sets": case_reports, "material_sets": material_reports}


def _list_report(dataset, case_sets, materials) -> dict[str, Any]:
    lists = []
    for binding in sorted(dataset["bindings"], key=lambda item: item["slot"]):
        if binding.get("case_set_id"):
            case_set = case_sets[binding["case_set_id"]]
            lists.append(
                {
                    "slot": binding["slot"],
                    "id": case_set["case_set_id"],
                    "list_sha256": case_set["list_sha256"],
                }
            )
        if binding.get("material_set_id"):
            material = materials[binding["material_set_id"]]
            lists.append(
                {
                    "slot": binding["slot"],
                    "id": material["material_set_id"],
                    "list_sha256": material["list_sha256"],
                }
            )
    return {"status": "PASS", "lists": lists}


def _audit(subject_kind: str, subject_id: str, audit_kind: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": audit_id_for(
            subject_kind=subject_kind,
            subject_id=subject_id,
            audit_kind=audit_kind,
            report=report,
        ),
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "audit_kind": audit_kind,
        "status": report["status"],
        "report_json": report,
    }
