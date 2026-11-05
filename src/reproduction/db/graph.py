"""In-memory reproduction graph built from staging or from DuckDB."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest


@dataclass
class Graph:
    catalogs: list[dict[str, Any]] = field(default_factory=list)
    catalog_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    assets: list[dict[str, Any]] = field(default_factory=list)
    derivations: list[dict[str, Any]] = field(default_factory=list)
    materials: list[dict[str, Any]] = field(default_factory=list)
    protocols: list[dict[str, Any]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    case_sets: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    builds: list[dict[str, Any]] = field(default_factory=list)
    verified: dict[str, dict[str, Any]] = field(default_factory=dict)
    audits: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    hash_updates: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: "Graph") -> "Graph":
        return Graph(
            catalogs=self.catalogs + other.catalogs,
            catalog_rows={**self.catalog_rows, **other.catalog_rows},
            assets=self.assets + other.assets,
            derivations=self.derivations + other.derivations,
            materials=self.materials + other.materials,
            protocols=self.protocols + other.protocols,
            cases=self.cases + other.cases,
            case_sets=self.case_sets + other.case_sets,
            datasets=self.datasets + other.datasets,
            builds=self.builds + other.builds,
            verified={**self.verified, **other.verified},
            audits=self.audits + other.audits,
            runs=self.runs + other.runs,
            hash_updates=self.hash_updates + other.hash_updates,
        )


def logical_bindings(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Binding identity is the slot. Row ids and null params are not part of it."""
    rows = []
    for binding in bindings:
        row = {
            "slot": binding["slot"],
            "case_set_id": binding.get("case_set_id"),
            "material_set_id": binding.get("material_set_id"),
        }
        if binding.get("params_json") is not None:
            row["params_json"] = binding["params_json"]
        rows.append(row)
    return sorted(rows, key=lambda item: item["slot"])


def index_rows(rows: list[dict[str, Any]], key: str, table: str) -> dict[Any, dict[str, Any]]:
    found: dict[Any, dict[str, Any]] = {}
    for row in rows:
        ident = row[key]
        previous = found.get(ident)
        if previous is None:
            found[ident] = row
            continue
        if digest(previous) != digest(row):
            raise ReproductionError(f"{table} {ident} is staged with different content")
    return found
