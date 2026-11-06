"""In-memory index of one snapshot.

Asset ids are ``{storage_root}/{relative path}``. Learning resolves files from
that prefix and does not open DuckDB.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.reproduction.errors import ReproductionError
from src.reproduction.paths import DEFAULT_CORPORA, DEFAULT_DERIVED, DEFAULT_GENERATED
from src.reproduction.snapshot_reader import LoadedSnapshot

_ROOTS = {"corpora": "corpora", "derived": "derived", "generated": "generated"}


class AssetStore:
    """Cases, case-set order, and file paths for one loaded snapshot."""

    def __init__(
        self,
        dataset: dict[str, Any],
        assets: list[dict[str, Any]],
        cases: list[dict[str, Any]],
        items: list[dict[str, Any]],
        *,
        snapshot_hash: str,
        derived: Path,
        corpora: Path,
        generated: Path,
        docs: list[dict[str, Any]] | None = None,
        members: list[dict[str, Any]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.snapshot_hash = snapshot_hash
        self.roots = {"derived": Path(derived), "corpora": Path(corpora), "generated": Path(generated)}
        self.assets = {row["asset_id"]: row for row in assets}
        if len(self.assets) != len(assets):
            raise ReproductionError("snapshot asset ids are not unique")
        self.cases = {row["case_id"]: row for row in cases}
        if len(self.cases) != len(cases):
            raise ReproductionError("snapshot case ids are not unique")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["case_set_id"]].append(item)
        self.items: dict[str, list[dict[str, Any]]] = {}
        for case_set_id, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: int(row["position"]))
            positions = [int(row["position"]) for row in ordered]
            if positions != list(range(len(positions))):
                raise ReproductionError(f"case set {case_set_id} positions are not 0..n-1")
            self.items[case_set_id] = ordered
        self.bindings = {row["slot"]: row["case_set_id"] for row in dataset["bindings"]}
        self.docs = {}
        for row in docs or []:
            case_id = row["case_id"]
            if case_id in self.docs:
                raise ReproductionError(f"snapshot repeats a realization for {case_id}")
            self.docs[case_id] = row["doc_json"]
        self.members = list(members or [])

    @classmethod
    def from_snapshot(
        cls,
        loaded: LoadedSnapshot,
        *,
        derived: str | Path = DEFAULT_DERIVED,
        corpora: str | Path = DEFAULT_CORPORA,
        generated: str | Path = DEFAULT_GENERATED,
    ) -> AssetStore:
        assets = list(loaded.assets)
        for row in loaded.generated:
            assets.append(
                {
                    "asset_id": row["asset_id"],
                    "rel_path": row["rel_path"],
                    "sample_rate": int(row["sample_rate"]),
                    "frames": int(row["frames"]),
                    "channels": int(row["channels"]),
                    "sha256": row.get("sha256"),
                    "integrity": row["integrity"],
                }
            )
        return cls(
            loaded.dataset,
            assets,
            loaded.cases,
            loaded.case_set_items,
            snapshot_hash=loaded.snapshot_hash,
            derived=Path(derived),
            corpora=Path(corpora),
            generated=Path(generated),
            docs=loaded.docs,
            members=loaded.material_set_members,
        )

    def case_ids(self, slot: str) -> list[str]:
        case_set_id = self.bindings.get(slot)
        if not case_set_id:
            raise ReproductionError(f"snapshot has no {slot} binding")
        rows = self.items.get(case_set_id)
        if rows is None:
            raise ReproductionError(f"snapshot is missing case set {case_set_id}")
        return [row["case_id"] for row in rows]

    def case(self, case_id: str) -> dict[str, Any]:
        row = self.cases.get(case_id)
        if row is None:
            raise ReproductionError(f"snapshot is missing case {case_id}")
        return row

    def asset(self, asset_id: str) -> dict[str, Any]:
        row = self.assets.get(asset_id)
        if row is None:
            raise ReproductionError(f"snapshot is missing asset {asset_id}")
        return row

    def path(self, asset_id: str) -> Path:
        row = self.asset(asset_id)
        prefix, separator, rest = asset_id.partition("/")
        if separator != "/" or prefix not in _ROOTS:
            raise ReproductionError(f"asset id {asset_id} has no storage root")
        if rest != row["rel_path"]:
            raise ReproductionError(f"asset id {asset_id} does not match its relative path")
        return self.roots[prefix] / row["rel_path"]
