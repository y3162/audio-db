"""Material identities for a snapshot.

Training writes these into ``data_provenance.json``. This module does not open
DuckDB. Derived files contribute a source hash only when the sibling stamp
matches the file hash stored in the snapshot.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.experiments.data.ids import digest
from src.experiments.data.assets.store import AssetStore

_SLOTS = (
    ("train.cases", "train"),
    ("valid.cases", "selection"),
    ("test.cases", "test"),
)
_ASSET_KEYS = ("clean_asset_id", "noisy_asset_id", "noise_asset_id")
_HEX = frozenset("0123456789abcdef")


def snapshot_provenance_parts(
    store: AssetStore,
    *,
    lineage_cache: dict[str, dict[str, Any]] | None = None,
    workers: int = 16,
) -> dict[str, Any]:
    """Case-set ids and content hashes for train, selection, and test."""
    cache = {} if lineage_cache is None else lineage_cache
    dataset = store.dataset
    materials: list[dict[str, Any]] = []
    sets: list[dict[str, Any]] = []
    limitations: list[str] = []
    seen: set[tuple[str, str]] = set()
    pending: list[tuple[str, str, str, dict[str, Any]]] = []
    for slot, role in _SLOTS:
        case_set_id = store.bindings.get(slot)
        if not case_set_id:
            limitations.append(f"{slot} is not bound")
            continue
        case_ids = store.case_ids(slot)
        sets.append(
            {
                "slot": slot,
                "role": role,
                "case_set_id": case_set_id,
                "member_count": len(case_ids),
                "case_ids_sha256": digest(case_ids),
            }
        )
        for case_id in case_ids:
            case = store.case(case_id)
            ids = [case.get(key) for key in _ASSET_KEYS]
            for asset_id in ids:
                if not asset_id:
                    continue
                token = (role, asset_id)
                if token in seen:
                    continue
                seen.add(token)
                pending.append((role, slot, case_set_id, store.asset(asset_id)))
    members: dict[str, list[str]] = {}
    for row in getattr(store, "members", []):
        members.setdefault(str(row["material_set_id"]), []).append(str(row["asset_id"]))
    for binding in dataset.get("bindings") or []:
        material_set_id = binding.get("material_set_id")
        if not material_set_id:
            continue
        slot = str(binding["slot"])
        role = slot.split(".", 1)[0]
        if role == "valid":
            role = "selection"
        for asset_id in members.get(str(material_set_id), []):
            token = (role, asset_id)
            if token in seen:
                continue
            seen.add(token)
            pending.append((role, slot, str(material_set_id), store.asset(asset_id)))
    lineages = _lineages(store, [item[3] for item in pending], cache, workers)
    counts = {"source": 0, "derived": 0, "unknown": 0}
    for (role, slot, case_set_id, asset), lineage in zip(pending, lineages, strict=True):
        kind = lineage["lineage"]
        counts[kind] = counts.get(kind, 0) + 1
        materials.append(
            {
                "role": role,
                "slot": slot,
                "case_set_id": case_set_id,
                "asset_id": asset["asset_id"],
                "sha256": asset.get("sha256") if _is_sha(asset.get("sha256")) else None,
                "source_sha256": lineage.get("source_sha256"),
                "derivation_id": lineage.get("derivation_id"),
                "lineage": kind,
            }
        )
    materials.sort(key=lambda row: (row["role"], row["asset_id"]))
    if counts["unknown"]:
        limitations.append("some assets have no verified content or source hash")
    if dataset.get("selection_uses_test") is True:
        limitations.append("checkpoint selection uses the test materials")
    if dataset.get("profile") == "leakage_safe":
        limitations.append(
            "leakage_safe describes this dataset's own split; it does not prove independence from another evaluation set"
        )
    if not dataset.get("claim_label"):
        limitations.append("snapshot dataset has no claim_label")
    return {
        "dataset_id": dataset.get("dataset_id"),
        "dataset_name": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "snapshot_hash": store.snapshot_hash,
        "protocol_id": dataset.get("protocol_id"),
        "profile": dataset.get("profile"),
        "claim_label": dataset.get("claim_label"),
        "selection_uses_test": dataset.get("selection_uses_test"),
        "materials": materials,
        "sets": sets,
        "lineage_counts": counts,
        "limitations": limitations,
    }


def _lineages(
    store: AssetStore,
    assets: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    missing = []
    seen: set[str] = set()
    for asset in assets:
        asset_id = asset["asset_id"]
        if asset_id in cache or asset_id in seen:
            continue
        seen.add(asset_id)
        missing.append(asset)
    derived = [asset for asset in missing if str(asset["asset_id"]).startswith("derived/")]
    derived_ids = {asset["asset_id"] for asset in derived}
    for asset in missing:
        if asset["asset_id"] in derived_ids:
            continue
        cache[asset["asset_id"]] = _plain_lineage(asset)
    if derived:
        if len(derived) == 1 or workers <= 1:
            for asset in derived:
                cache[asset["asset_id"]] = _derived_lineage(store, asset)
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(derived), 16)) as pool:
                found = list(pool.map(lambda asset: _derived_lineage(store, asset), derived))
            for asset, lineage in zip(derived, found, strict=True):
                cache[asset["asset_id"]] = lineage
    return [cache[asset["asset_id"]] for asset in assets]


def _plain_lineage(asset: dict[str, Any]) -> dict[str, Any]:
    if _is_sha(asset.get("sha256")):
        return {"lineage": "source", "source_sha256": None, "derivation_id": None}
    return {"lineage": "unknown", "source_sha256": None, "derivation_id": None}


def _derived_lineage(store: AssetStore, asset: dict[str, Any]) -> dict[str, Any]:
    sha = asset.get("sha256")
    if not _is_sha(sha):
        return {"lineage": "unknown", "source_sha256": None, "derivation_id": None}
    try:
        path = store.path(asset["asset_id"])
    except Exception:
        return {"lineage": "unknown", "source_sha256": None, "derivation_id": None}
    stamp_path = path.with_name(path.name + ".stamp.json")
    if not stamp_path.is_file():
        return {"lineage": "unknown", "source_sha256": None, "derivation_id": None}
    try:
        body = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"lineage": "unknown", "source_sha256": None, "derivation_id": None}
    source = body.get("source_sha256") if isinstance(body, dict) else None
    if not isinstance(body, dict) or body.get("output_sha256") != sha or not _is_sha(source):
        return {"lineage": "unknown", "source_sha256": None, "derivation_id": None}
    derivation = body.get("derivation_id")
    return {
        "lineage": "derived",
        "source_sha256": source,
        "derivation_id": derivation if isinstance(derivation, str) and derivation else None,
    }


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX for char in value)
