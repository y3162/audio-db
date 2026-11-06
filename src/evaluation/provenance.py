"""Checkpoint data provenance and evaluation independence.

The JSON file is the record evaluation can read without DuckDB and without
``src.reproduction``. ``independent`` is true only when content hashes and
known source hashes show no overlap. ``UNKNOWN`` is not an independent result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FORMAT = "audio-db-data-provenance-v1"
SCHEMA_VERSION = 1
UNKNOWN = "UNKNOWN"
PROVENANCE_NAME = "data_provenance.json"
_HEX = frozenset("0123456789abcdef")


def canonical_bytes(value: Any) -> bytes:
    """Stable JSON bytes. The trailing newline is part of the hash."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return text.encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def parse_provenance_bytes(payload: bytes) -> dict[str, Any]:
    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("data provenance must be a JSON object")
    return document


def sidecar_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".data_provenance.json")


def locate_provenance_files(checkpoint: str | Path) -> list[Path]:
    path = Path(checkpoint)
    found = []
    for candidate in (sidecar_path(path), path.parent / PROVENANCE_NAME):
        if candidate.is_file() and candidate not in found:
            found.append(candidate)
    return found


def load_provenance(checkpoint: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(document, error)``. A missing file is an error string, not an exception."""
    files = locate_provenance_files(checkpoint)
    if not files:
        return None, "data_provenance.json is missing"
    parsed: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        try:
            document = parse_provenance_bytes(path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return None, f"data provenance is unreadable: {path.name}: {exc}"
        parsed.append((path, document))
    digests = {content_sha256(document) for _, document in parsed}
    if len(digests) > 1:
        return None, "checkpoint provenance files disagree"
    return parsed[0][1], None


def identities_from_samples(samples: list[Any]) -> list[dict[str, Any]]:
    """Content identities carried by evaluation samples.

    A recipe hash alone is ``unspecified`` lineage: it can prove overlap, not
    independence. ``metadata['material_identity']`` is an explicit attestation.
    """
    materials: list[dict[str, Any]] = []
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        set_name = getattr(sample, "dataset", None) or "unnamed"
        label = getattr(sample, "utterance_id", None)
        attested = metadata.get("material_identity")
        if attested is not None:
            rows = attested if isinstance(attested, list) else [attested]
            for row in rows:
                if not isinstance(row, dict):
                    materials.append(_eval_row(set_name, label, None, None, "unknown"))
                    continue
                materials.append(
                    _eval_row(
                        row.get("set_name") or set_name,
                        label,
                        row.get("sha256"),
                        row.get("source_sha256"),
                        row.get("lineage") or "unknown",
                    )
                )
            continue
        recipe = metadata.get("recipe")
        sources = recipe.get("sources") if isinstance(recipe, dict) else None
        if isinstance(sources, dict) and sources:
            for source in sources.values():
                sha = source.get("sha256") if isinstance(source, dict) else None
                materials.append(_eval_row(set_name, label, sha, None, "unspecified"))
            continue
        materials.append(_eval_row(set_name, label, None, None, "unspecified"))
    return materials


def judge_independence(
    provenance: dict[str, Any] | None,
    eval_materials: list[dict[str, Any]] | None,
    *,
    load_problem: str | None = None,
) -> dict[str, Any]:
    """Decide whether an evaluation set is independent of train and selection data."""
    if load_problem:
        return _unknown(load_problem, comparison="NOT_TESTED" if "missing" in load_problem else "INCONCLUSIVE")
    if provenance is None:
        return _unknown("data provenance was not provided", comparison="NOT_TESTED")
    if eval_materials is None:
        base = _known_fields(provenance)
        base.update(
            _unknown(
                "evaluation materials were not provided",
                comparison="NOT_TESTED",
            )
        )
        base["selection_uses_test"] = _optional_bool(provenance.get("selection_uses_test"))
        base["profile"] = _optional_str(provenance.get("profile"))
        base["claim_label"] = _optional_str(provenance.get("claim_label"))
        base["split_independent"] = _split_status(provenance)
        base["confirmed_independent"] = False
        return base
    problem = _document_problem(provenance)
    if problem:
        return _unknown(problem)
    if provenance.get("pipeline") != "reproduction" or not provenance.get("materials"):
        result = _unknown(
            "checkpoint has no content-addressed train or selection materials",
            comparison="INCONCLUSIVE",
        )
        result["selection_uses_test"] = _optional_bool(provenance.get("selection_uses_test"))
        result["profile"] = _optional_str(provenance.get("profile"))
        result["claim_label"] = _optional_str(provenance.get("claim_label"))
        result["split_independent"] = UNKNOWN
        return result

    selection_keys, selection_complete, selection_rows = _role_index(provenance, "selection")
    train_keys, train_complete, train_rows = _role_index(provenance, "train")
    if not selection_rows and not train_rows:
        result = _unknown("provenance lists neither train nor selection materials")
        result["selection_uses_test"] = _optional_bool(provenance.get("selection_uses_test"))
        return result

    overlap_sets: list[str] = []
    train_sets: list[str] = []
    overlap_sha: set[str] = set()
    overlap_cases: set[str] = set()
    train_sha: set[str] = set()
    eval_complete = bool(eval_materials)
    for item in eval_materials:
        known, complete = _eval_keys(item)
        if not complete:
            eval_complete = False
        set_name = str(item.get("set_name") or "unnamed")
        hit_selection = known & selection_keys
        hit_train = known & train_keys
        if hit_selection:
            if set_name not in overlap_sets:
                overlap_sets.append(set_name)
            overlap_sha.update(hit_selection)
            overlap_cases.update(_case_sets_for(selection_rows, hit_selection))
        if hit_train:
            if set_name not in train_sets:
                train_sets.append(set_name)
            train_sha.update(hit_train)
    definite = bool(overlap_sets or train_sets)
    fields = _known_fields(provenance)
    fields["split_independent"] = _split_status(provenance)
    if definite:
        fields.update(
            {
                "independent": False,
                "confirmed_independent": False,
                "checkpoint_selection_overlap": overlap_sets,
                "train_overlap": train_sets,
                "overlap_sha256": sorted(overlap_sha | train_sha),
                "overlap_case_sets": sorted(overlap_cases),
                "comparison": "TESTED",
                "reason": _overlap_reason(overlap_sets, train_sets),
                "missing": [],
            }
        )
        return fields
    missing = []
    if not selection_complete:
        missing.append("selection lineage is incomplete")
    if not train_complete:
        missing.append("train lineage is incomplete")
    if not eval_complete:
        missing.append("evaluation lineage is incomplete or unspecified")
    if missing:
        fields.update(
            _unknown(
                "independence cannot be confirmed: " + "; ".join(missing),
                comparison="INCONCLUSIVE",
            )
        )
        fields["selection_uses_test"] = _optional_bool(provenance.get("selection_uses_test"))
        fields["profile"] = _optional_str(provenance.get("profile"))
        fields["claim_label"] = _optional_str(provenance.get("claim_label"))
        fields["split_independent"] = _split_status(provenance)
        fields["missing"] = missing
        return fields
    fields.update(
        {
            "independent": True,
            "confirmed_independent": True,
            "checkpoint_selection_overlap": [],
            "train_overlap": [],
            "overlap_sha256": [],
            "overlap_case_sets": [],
            "comparison": "TESTED",
            "reason": "content and source hashes of the evaluation materials are disjoint from train and selection",
            "missing": [],
        }
    )
    return fields


def judge_checkpoint(checkpoint: str | Path, eval_materials: list[dict[str, Any]] | None) -> dict[str, Any]:
    document, problem = load_provenance(checkpoint)
    result = judge_independence(document, eval_materials, load_problem=problem)
    result["checkpoint"] = str(checkpoint)
    return result


def judge_model_checkpoints(
    checkpoints: dict[str, str],
    samples: list[Any],
) -> dict[str, Any]:
    if not checkpoints:
        result = _unknown("model has no checkpoint", comparison="NOT_TESTED")
        result["checkpoints"] = []
        return result
    materials = identities_from_samples(samples)
    parts = [judge_checkpoint(path, materials) for path in checkpoints.values()]
    combined = combine_judgments(parts)
    combined["checkpoints"] = parts
    return combined


def combine_judgments(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    if not judgments:
        return _unknown("no independence judgments", comparison="NOT_TESTED")
    if any(item.get("independent") is False for item in judgments):
        flag: bool | str = False
    elif all(item.get("independent") is True for item in judgments):
        flag = True
    else:
        flag = UNKNOWN
    if flag is True:
        overlap: list[str] | None = []
        train: list[str] | None = []
        hashes: list[str] | None = []
        cases: list[str] | None = []
    elif flag is False:
        overlap = _union(judgments, "checkpoint_selection_overlap")
        train = _union(judgments, "train_overlap")
        hashes = _union(judgments, "overlap_sha256")
        cases = _union(judgments, "overlap_case_sets")
    else:
        overlap = train = hashes = cases = None
    selection = _merge_bool(judgments, "selection_uses_test")
    return {
        "schema_version": SCHEMA_VERSION,
        "independent": flag,
        "confirmed_independent": flag is True,
        "checkpoint_selection_overlap": overlap,
        "train_overlap": train,
        "overlap_sha256": hashes,
        "overlap_case_sets": cases,
        "selection_uses_test": selection,
        "profile": _merge_text(judgments, "profile"),
        "claim_label": _merge_text(judgments, "claim_label"),
        "split_independent": _merge_status(judgments, "split_independent"),
        "comparison": _merge_comparison(judgments, flag),
        "reason": judgments[0].get("reason") if len(judgments) == 1 else _combine_reason(flag),
        "missing": _union(judgments, "missing") if flag == UNKNOWN else [],
    }


def summarize_record_independence(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    judgments = []
    for record in records:
        stored = record.get("data_independence")
        if isinstance(stored, dict):
            judgments.append(stored)
    if not judgments:
        return None
    summary = combine_judgments(judgments)
    summary["models"] = judgments
    summary["unknown_is_not_independent"] = summary["independent"] == UNKNOWN
    return summary


def apply_independence(record: dict[str, Any], judgment: dict[str, Any]) -> None:
    record["independent"] = judgment.get("independent")
    record["checkpoint_selection_overlap"] = judgment.get("checkpoint_selection_overlap")
    record["data_independence"] = judgment


def is_confirmed_independent(value: Any) -> bool:
    """True only for a confirmed independent evaluation. ``UNKNOWN`` is false."""
    return value is True


def _eval_row(set_name: str, label: Any, sha: Any, source: Any, lineage: Any) -> dict[str, Any]:
    return {
        "set_name": set_name,
        "label": label,
        "sha256": sha if _is_sha(sha) else None,
        "source_sha256": source if _is_sha(source) else None,
        "lineage": lineage if isinstance(lineage, str) else "unknown",
    }


def _document_problem(document: dict[str, Any]) -> str | None:
    if document.get("format") != FORMAT or document.get("schema_version") != SCHEMA_VERSION:
        return "data provenance schema is not supported"
    if not isinstance(document.get("materials"), list):
        return "data provenance materials are missing"
    return None


def _known_fields(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_uses_test": _optional_bool(provenance.get("selection_uses_test")),
        "profile": _optional_str(provenance.get("profile")),
        "claim_label": _optional_str(provenance.get("claim_label")),
        "run_id": _optional_str(provenance.get("run_id")),
        "dataset_id": _optional_str(provenance.get("dataset_id")),
        "snapshot_hash": _optional_str(provenance.get("snapshot_hash")),
    }


def _unknown(reason: str, *, comparison: str = "INCONCLUSIVE") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "independent": UNKNOWN,
        "confirmed_independent": False,
        "checkpoint_selection_overlap": None,
        "train_overlap": None,
        "overlap_sha256": None,
        "overlap_case_sets": None,
        "selection_uses_test": None,
        "profile": None,
        "claim_label": None,
        "split_independent": UNKNOWN,
        "comparison": comparison,
        "reason": reason,
        "missing": [reason],
    }


def _role_index(
    provenance: dict[str, Any], role: str
) -> tuple[set[str], bool, list[dict[str, Any]]]:
    rows = [row for row in provenance.get("materials") or [] if isinstance(row, dict) and row.get("role") == role]
    keys: set[str] = set()
    complete = bool(rows)
    for row in rows:
        known, row_complete = _material_keys(row)
        keys.update(known)
        complete = complete and row_complete
    return keys, complete, rows


def _material_keys(row: dict[str, Any]) -> tuple[set[str], bool]:
    sha = row.get("sha256") if _is_sha(row.get("sha256")) else None
    source = row.get("source_sha256") if _is_sha(row.get("source_sha256")) else None
    lineage = row.get("lineage")
    known = set()
    if sha:
        known.add(sha)
    if source:
        known.add(source)
    if lineage == "source" and sha:
        return known, True
    if lineage == "derived" and sha and source and source != sha:
        return known, True
    return known, False


def _eval_keys(row: dict[str, Any]) -> tuple[set[str], bool]:
    return _material_keys(row)


def _case_sets_for(rows: list[dict[str, Any]], hashes: set[str]) -> set[str]:
    found = set()
    for row in rows:
        known, _complete = _material_keys(row)
        if known & hashes:
            case_set = row.get("case_set_id")
            if isinstance(case_set, str) and case_set:
                found.add(case_set)
    return found


def _split_status(provenance: dict[str, Any]) -> bool | str:
    if not isinstance(provenance.get("materials"), list):
        return UNKNOWN
    groups = {}
    complete = True
    for role in ("train", "selection", "test"):
        keys, role_complete, rows = _role_index(provenance, role)
        if not rows:
            return UNKNOWN
        groups[role] = keys
        complete = complete and role_complete
    if not complete:
        return UNKNOWN
    train, selection, test = groups["train"], groups["selection"], groups["test"]
    if train & selection or train & test:
        return False
    if _optional_bool(provenance.get("selection_uses_test")) is True:
        return False
    if selection & test:
        return False
    return True


def _overlap_reason(selection_sets: list[str], train_sets: list[str]) -> str:
    parts = []
    if selection_sets:
        parts.append("evaluation materials overlap checkpoint selection: " + ", ".join(selection_sets))
    if train_sets:
        parts.append("evaluation materials overlap training materials: " + ", ".join(train_sets))
    return "; ".join(parts)


def _combine_reason(flag: bool | str) -> str:
    if flag is False:
        return "at least one checkpoint overlaps the evaluation materials"
    if flag is True:
        return "every checkpoint is independent of the evaluation materials"
    return "independence is unknown for at least one checkpoint"


def _union(judgments: list[dict[str, Any]], key: str) -> list[str]:
    found: list[str] = []
    for item in judgments:
        values = item.get(key) or []
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value not in found:
                found.append(value)
    return found


def _merge_bool(judgments: list[dict[str, Any]], key: str) -> bool | None:
    values = [item.get(key) for item in judgments]
    if any(value is True for value in values):
        return True
    if any(value is None for value in values):
        return None
    if values and all(value is False for value in values):
        return False
    return None


def _merge_text(judgments: list[dict[str, Any]], key: str) -> str | None:
    values = {item.get(key) for item in judgments if isinstance(item.get(key), str)}
    if len(values) == 1:
        return next(iter(values))
    return None


def _merge_status(judgments: list[dict[str, Any]], key: str) -> bool | str:
    values = [item.get(key) for item in judgments]
    if any(value is False for value in values):
        return False
    if any(value == UNKNOWN or value is None for value in values):
        return UNKNOWN
    if values and all(value is True for value in values):
        return True
    return UNKNOWN


def _merge_comparison(judgments: list[dict[str, Any]], flag: bool | str) -> str:
    if flag is True or flag is False:
        return "TESTED"
    values = {item.get("comparison") for item in judgments}
    if values == {"NOT_TESTED"}:
        return "NOT_TESTED"
    return "INCONCLUSIVE"


def _optional_bool(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    return None


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX for char in value)
