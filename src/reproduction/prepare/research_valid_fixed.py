"""Fixed research validation subset for TF-Locoformer DNS checkpoint selection.

``research_valid_fixed_v1`` is a pure function of the sorted parent validation
case ids. It does not use ``random`` or a NumPy Generator. The same population
yields the same 1024 ids in every process. Worker-local shards must not be
sampled separately and unioned.

The parent dataset name ``tf_locoformer.dns.accept_recipe`` is a constant.
This module does not rename it, does not write mixture wavs, and does not
write inside an existing snapshot directory.

Registration of ``tf_locoformer.dns.accept_recipe.research_valid_fixed@1`` is
performed only by ``materialize``'s caller. This module records BLOCKED when
``reproduction.duckdb`` is absent or the claim label is not in the schema.
It does not invent a snapshot hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.reproduction.errors import ReproductionError
from src.reproduction.ids import (
    CLAIM_LABELS,
    case_set_id_for,
    dataset_id_for,
    digest,
    protocol_id_for,
)
from src.reproduction.paths import ROOT

ALGORITHM_NAME = "research_valid_fixed_v1"
SELECTION_SEED = 20261107
SELECTION_COUNT = 1024
PARENT_DATASET_NAME = "tf_locoformer.dns.accept_recipe"
PARENT_DATASET_SELECTOR = "tf_locoformer.dns.accept_recipe@1"
NEW_DATASET_NAME = "tf_locoformer.dns.accept_recipe.research_valid_fixed"
NEW_DATASET_SELECTOR = "tf_locoformer.dns.accept_recipe.research_valid_fixed@1"
CLAIM_LABEL = "research_checkpoint_selection"
PARENT_SNAPSHOT_HASH = "3343a55efb4c57d9ffef9179fb2f39c3e0186d36f587a0f4ad9f0aa14e9926f5"
PARENT_DATASET_ID = "5469f0416329a8661df2cc91cd5bd9a149d6ea223c72e69ebdd5696bea6bdfbe"
PARENT_PROTOCOL_ID = "cadda632ebd08a9066e5b7a81851920762cc2a5fa5bddc21f8488bf1a1630116"

CHUNKS_PER_UTTERANCE = 14
VALIDATION_RANKS = 4
SECONDS_PER_CHUNK = Decimal("0.1602")
TRAIN_HOURS = Decimal("384.4")
CHECKPOINT_BYTES = 179910164
CHECKPOINT_WRITE_SECONDS = Decimal("2.29")
CHECKPOINT_EPOCHS_WRITTEN = 150
FULL_VALIDATION_UTTERANCES = 36000

SELECTION_RULE = """
research_valid_fixed_v1

Population: the validation case_id values of tf_locoformer.dns.accept_recipe@1.
Input order is ignored. A duplicate case_id is an error.
Case ids that also appear in the parent train or test split are removed before
ranking. If that removal leaves fewer than 1024 ids, the published set is the
entire remainder and the count is documented as smaller. Otherwise the count
stays 1024. When the validation split is disjoint from train and test, the
ranked population is exactly the sorted validation case ids.

Rank: SHA-256 digest of the UTF-8 bytes of "{seed}\\n{case_id}".
The seed is the non-negative decimal integer 20261107, with no sign and no
leading zeros. The digest is the raw 32 bytes, ordered lexicographically.
Ties break by case_id ascending. Take the first 1024 surviving case ids.
Publish those ids sorted lexicographically. Do not publish them in rank order.

case_id_list_sha256 is the hex SHA-256 of the UTF-8 bytes of
"\\n".join(published_case_ids) + "\\n".

Call the function on the full population. Hashing each worker's shard and
uniting the lowest ids is a different set. The random module and NumPy
Generator are not used.
""".strip()

PROTOCOL_VALIDATION_NOTE = (
    "research_valid_fixed_v1 selects 1024 case ids from the parent validation "
    "split with seed 20261107. Rank is SHA-256 of UTF-8 '{seed}\\n{case_id}', "
    "lowest 1024, then sorted lexicographically. Train and test case ids stay "
    "the parent sets. This is research checkpoint selection. It is not "
    "cv_synthetic_small and it is not the parent 36000-utterance validation."
)

RECORDING_KEYS = ("recording_id", "recording", "rec_id")
RIR_KEYS = ("rir", "rir_id", "reverb", "rt60")
LEAK_CHECKS = (
    "case_id_train",
    "case_id_test",
    "clean_sha256_in_case_document",
    "speaker",
    "recording",
    "noise_file_name",
)

LIST_JSON = ROOT / "configs" / "data" / "lists" / "tf_locoformer" / "research_valid_fixed_v1.json"
PROTOCOL_YAML = ROOT / "configs" / "data" / "protocols" / "dns.tflocoformer.research_valid_fixed.yaml"
STAGING_JSON = (
    ROOT / "data" / "staging" / "reproduction" / "research_valid_fixed_v1" / "dataset_definition.json"
)
REPORT_MD = ROOT / "docs" / "training" / "reproduction" / "preflight" / "AGENT4_RESEARCH_VALID.md"


def selection_rank(case_id: str, seed: int = SELECTION_SEED) -> bytes:
    """Raw SHA-256 digest used to order one case id. Lower sorts first."""
    if not isinstance(case_id, str) or not case_id:
        raise ReproductionError("case_id must be a non-empty string")
    payload = f"{_seed_text(seed)}\n{case_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def case_id_list_sha256(case_ids: Iterable[str]) -> str:
    """SHA-256 hex of the lexicographically sorted case ids, one per line."""
    items = _require_case_ids(list(case_ids))
    if len(set(items)) != len(items):
        raise ReproductionError("case_id list contains duplicates")
    payload = ("\n".join(sorted(items)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_case_ids(
    case_ids: Iterable[str],
    *,
    seed: int = SELECTION_SEED,
    count: int = SELECTION_COUNT,
) -> list[str]:
    """Return ``count`` case ids from ``case_ids``.

    The result is sorted lexicographically. It does not depend on the input
    order. The caller's list is not modified.
    """
    ordered = _sorted_unique(case_ids)
    take = _positive_int(count, "selection count")
    _seed_text(seed)
    if take > len(ordered):
        raise ReproductionError(
            f"selection count {take} is larger than the population {len(ordered)}"
        )
    ranked = sorted(ordered, key=lambda case_id: (selection_rank(case_id, seed), case_id))
    return sorted(ranked[:take])


def select_research_validation(
    valid_ids: Iterable[str],
    train_ids: Iterable[str],
    test_ids: Iterable[str],
    *,
    seed: int = SELECTION_SEED,
    count: int = SELECTION_COUNT,
) -> dict[str, Any]:
    """Select the research validation set from the parent validation split.

    Train and test ids are removed first. The count stays ``count`` unless
    fewer than ``count`` validation ids remain after that removal.
    """
    population = _sorted_unique(valid_ids)
    train_list = _require_case_ids(list(train_ids))
    test_list = _require_case_ids(list(test_ids))
    train = set(train_list)
    test = set(test_list)
    if len(train) != len(train_list) or len(test) != len(test_list):
        raise ReproductionError("train or test case ids contain duplicates")
    excluded = [case_id for case_id in population if case_id in train or case_id in test]
    clean = [case_id for case_id in population if case_id not in train and case_id not in test]
    take = _positive_int(count, "selection count")
    reduced = len(clean) < take
    chosen_count = len(clean) if reduced else take
    selected = select_case_ids(clean, seed=seed, count=chosen_count) if chosen_count else []
    overlap = case_id_overlap(selected, train, test)
    return {
        "algorithm": ALGORITHM_NAME,
        "seed": seed,
        "requested_count": take,
        "count": len(selected),
        "population_count": len(population),
        "excluded_case_ids": excluded,
        "reduced_because_leak": reduced,
        "case_ids": selected,
        "case_id_list_sha256": case_id_list_sha256(selected) if selected else None,
        "case_id_overlap": overlap,
    }


def case_id_overlap(
    selected: Iterable[str],
    train_ids: Iterable[str],
    test_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    selected_ids = set(selected)
    train_hit = sorted(selected_ids & set(train_ids))
    test_hit = sorted(selected_ids & set(test_ids))
    return {
        "train": _overlap_status(train_hit),
        "test": _overlap_status(test_hit),
    }


def speaker_independence(
    left_ids: Sequence[str | None],
    right_ids: Sequence[str | None],
) -> dict[str, Any]:
    """PASS only when every id is present and the two sets are disjoint.

    A missing speaker id is UNKNOWN. UNKNOWN is never PASS.
    """
    left, left_missing = _present_ids(left_ids)
    right, right_missing = _present_ids(right_ids)
    if left_missing or right_missing or not left_ids or not right_ids:
        return {"status": "UNKNOWN", "overlap": [], "missing": True}
    overlap = sorted(set(left) & set(right))
    if overlap:
        return {"status": "FAIL", "overlap": overlap, "missing": False}
    return {"status": "PASS", "overlap": [], "missing": False}


def clean_sha_independence(
    left_hashes: Sequence[str | None],
    right_hashes: Sequence[str | None],
) -> dict[str, Any]:
    """PASS only when every clean SHA-256 is present and the sets are disjoint.

    An absent hash is UNKNOWN. UNKNOWN is never PASS.
    """
    if (
        not left_hashes
        or not right_hashes
        or any(not isinstance(item, str) or not item for item in list(left_hashes) + list(right_hashes))
    ):
        return {"status": "UNKNOWN", "missing": True, "overlap_count": None}
    overlap = sorted(set(left_hashes) & set(right_hashes))
    if overlap:
        return {"status": "OVERLAP", "missing": False, "overlap_count": len(overlap)}
    return {"status": "PASS", "missing": False, "overlap_count": 0}


def recording_status(document_keys: Iterable[str]) -> dict[str, Any]:
    """UNKNOWN when no recording id field exists. Never PASS from absence."""
    keys = set(document_keys)
    found = sorted(key for key in RECORDING_KEYS if key in keys)
    if not found:
        return {
            "status": "UNKNOWN",
            "missing": True,
            "reason": "案件文書に recording_id が無い",
        }
    return {"status": "PRESENT", "missing": False, "fields": found}


def noise_file_overlap(*, shared_names: int, cases_sharing: int) -> dict[str, Any]:
    """Record shared noise filenames. Zero is PASS. A positive count is OVERLAP."""
    if shared_names < 0 or cases_sharing < 0:
        raise ReproductionError("noise overlap counts must be non-negative")
    status = "OVERLAP" if shared_names or cases_sharing else "PASS"
    return {
        "status": status,
        "shared_noise_names": shared_names,
        "selected_cases_sharing_train_noise": cases_sharing,
    }


def leakage_safe_allowed(checks: Mapping[str, Mapping[str, Any]]) -> bool:
    """True only when every required independence check is PASS."""
    for name in LEAK_CHECKS:
        status = checks.get(name, {}).get("status")
        if status != "PASS":
            return False
    return True


def validation_budget(utterances: int, *, epochs: int = 1) -> dict[str, Any]:
    """Extrapolate validation time from 0.1602 s/chunk and 14 chunks/utterance.

    Wall-clock is one rank's chunks times 0.1602 s. Four ranks are assumed to
    run those chunks in parallel. The utterance count must divide evenly.
    """
    count = _non_negative_int(utterances, "utterances")
    epoch_count = _positive_int(epochs, "epochs")
    chunks = count * CHUNKS_PER_UTTERANCE
    if chunks % VALIDATION_RANKS != 0:
        raise ReproductionError("validation chunks are not divisible by 4 ranks")
    per_rank = chunks // VALIDATION_RANKS
    one_pass_seconds = Decimal(per_rank) * SECONDS_PER_CHUNK
    one_pass_hours = one_pass_seconds / Decimal(3600)
    all_epochs_hours = one_pass_hours * Decimal(epoch_count)
    return {
        "utterances": count,
        "chunks_per_utterance": CHUNKS_PER_UTTERANCE,
        "chunks": chunks,
        "ranks": VALIDATION_RANKS,
        "chunks_per_rank": per_rank,
        "epochs": epoch_count,
        "seconds_per_chunk": _decimal_text(SECONDS_PER_CHUNK),
        "one_pass_seconds": _decimal_text(one_pass_seconds),
        "one_pass_hours": _decimal_text(one_pass_hours),
        "all_epochs_hours": _decimal_text(all_epochs_hours),
        "label": "extrapolated",
    }


def time_estimate() -> dict[str, Any]:
    """150-epoch estimate for the 1024-utterance subset. Not a new measurement."""
    subset = validation_budget(SELECTION_COUNT, epochs=CHECKPOINT_EPOCHS_WRITTEN)
    full = validation_budget(FULL_VALIDATION_UTTERANCES, epochs=CHECKPOINT_EPOCHS_WRITTEN)
    io_seconds = CHECKPOINT_WRITE_SECONDS * Decimal(CHECKPOINT_EPOCHS_WRITTEN)
    io_hours = io_seconds / Decimal(3600)
    validation_hours = Decimal(subset["all_epochs_hours"])
    total = TRAIN_HOURS + validation_hours + io_hours
    margin_train = TRAIN_HOURS * Decimal("1.3")
    margin_valid = validation_hours * Decimal("1.5")
    margin_total = margin_train + margin_valid + io_hours
    kept_upper = 3 + 1 + 5 + 1
    return {
        "label": "extrapolated",
        "not_remeasured": True,
        "source": "P10-C TF-Locoformer section; 0.1602 s/chunk and 2.29 s/write are the published timings",
        "subset": subset,
        "full_validation": full,
        "published_full_validation_hours": "841",
        "formula_full_validation_hours": full["all_epochs_hours"],
        "training_hours": _decimal_text(TRAIN_HOURS),
        "training_note": "384.4 h is the published 2,100,000-update extrapolation and is left unchanged",
        "checkpoint": {
            "bytes_per_write": CHECKPOINT_BYTES,
            "seconds_per_write": _decimal_text(CHECKPOINT_WRITE_SECONDS),
            "epochs_written": CHECKPOINT_EPOCHS_WRITTEN,
            "write_seconds": _decimal_text(io_seconds),
            "write_hours_rounded_7dp": _round(io_hours, 7),
            "kept_roles": "latest 3 + best + 5 nbest + average",
            "kept_files_upper_bound_if_roles_do_not_overlap": kept_upper,
            "kept_bytes_upper_bound": kept_upper * CHECKPOINT_BYTES,
            "files_not_retained": 150,
            "average_file_extra_write": "not added; the published 2.29 s figure is applied once per epoch",
        },
        "total_hours_rounded_4dp": _round(total, 4),
        "total_hours_components": {
            "training_hours": _decimal_text(TRAIN_HOURS),
            "validation_150_hours": _decimal_text(validation_hours),
            "checkpoint_io_seconds": _decimal_text(io_seconds),
        },
        "margin_1_3_train_1_5_validation": {
            "training_hours": _decimal_text(margin_train),
            "validation_hours": _decimal_text(margin_valid),
            "checkpoint_io_hours_rounded_7dp": _round(io_hours, 7),
            "total_hours_rounded_4dp": _round(margin_total, 4),
            "note": "Checkpoint I/O has no published margin factor and is not scaled",
        },
    }


def compare_histograms(parent: Mapping[Any, int], selected: Mapping[Any, int]) -> dict[str, Any]:
    """Counts for the selected subset against the parent validation split."""
    parent_counter = Counter({key: int(value) for key, value in parent.items()})
    selected_counter = Counter({key: int(value) for key, value in selected.items()})
    values = sorted(set(parent_counter) | set(selected_counter), key=_histogram_key)
    parent_total = sum(parent_counter.values())
    selected_total = sum(selected_counter.values())
    rows = []
    max_abs = Decimal(0)
    for value in values:
        parent_count = int(parent_counter[value])
        selected_count = int(selected_counter[value])
        parent_fraction = _fraction(parent_count, parent_total)
        selected_fraction = _fraction(selected_count, selected_total)
        delta = None
        if parent_fraction is not None and selected_fraction is not None:
            delta = selected_fraction - parent_fraction
            max_abs = max(max_abs, abs(delta))
        rows.append(
            {
                "value": value,
                "parent_valid": parent_count,
                "selected": selected_count,
                "parent_fraction": _fraction_text(parent_fraction),
                "selected_fraction": _fraction_text(selected_fraction),
                "fraction_delta": _fraction_text(delta),
            }
        )
    return {
        "parent_valid_total": parent_total,
        "selected_total": selected_total,
        "max_abs_fraction_delta": _fraction_text(max_abs) if parent_total and selected_total else None,
        "rows": rows,
    }


def find_parent_snapshot(snapshots_root: str | Path) -> Path:
    """Locate the accept_recipe snapshot by manifest. Do not write there."""
    root = Path(snapshots_root)
    if not root.is_dir():
        raise ReproductionError(f"snapshot root is missing: {root}")
    matches = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = child / "manifest.json"
        dataset_path = child / "dataset.json"
        if not manifest_path.is_file() or not dataset_path.is_file():
            continue
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        if dataset.get("name") != PARENT_DATASET_NAME or int(dataset.get("version", -1)) != 1:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("snapshot_hash") != PARENT_SNAPSHOT_HASH:
            raise ReproductionError(
                f"{child.name} is {PARENT_DATASET_NAME}@1 but its manifest hash is not {PARENT_SNAPSHOT_HASH}"
            )
        matches.append(child)
    if len(matches) != 1:
        raise ReproductionError(f"expected one accept_recipe snapshot, found {len(matches)}")
    return matches[0]


def parent_split_case_ids(snapshot_dir: str | Path) -> dict[str, Any]:
    """Read parent split case ids from the snapshot. The directory is not modified."""
    directory = Path(snapshot_dir)
    dataset = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if dataset.get("name") != PARENT_DATASET_NAME or int(dataset.get("version", -1)) != 1:
        raise ReproductionError("snapshot dataset is not tf_locoformer.dns.accept_recipe@1")
    if manifest.get("snapshot_hash") != PARENT_SNAPSHOT_HASH:
        raise ReproductionError("manifest snapshot_hash does not match the recorded parent hash")
    if dataset.get("dataset_id") != PARENT_DATASET_ID:
        raise ReproductionError("parent dataset_id does not match the recorded id")
    if dataset.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise ReproductionError("parent protocol_id does not match the recorded id")
    case_set_ids: dict[str, str] = {}
    materials: dict[str, str] = {}
    for binding in dataset["bindings"]:
        slot = str(binding["slot"])
        if binding.get("case_set_id"):
            case_set_ids[slot.split(".", 1)[0]] = str(binding["case_set_id"])
        if binding.get("material_set_id"):
            materials[slot] = str(binding["material_set_id"])
    from src.reproduction.snapshot_reader import items_from_arrays

    items = items_from_arrays(_load_npz(directory / "case_set_items.npz"))
    groups: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    split_of_set = {case_set_ids[split]: split for split in case_set_ids}
    for item in items:
        split = split_of_set.get(item["case_set_id"])
        if split is None:
            raise ReproductionError("snapshot item is not bound to train, valid, or test")
        groups[split].append(str(item["case_id"]))
    splits = {}
    for split, ids in groups.items():
        if len(ids) != len(set(ids)):
            raise ReproductionError(f"parent {split} case ids are not unique")
        splits[split] = sorted(ids)
    return {
        "train": splits["train"],
        "valid": splits["valid"],
        "test": splits["test"],
        "case_set_ids": case_set_ids,
        "materials": materials,
        "protocol_id": str(dataset["protocol_id"]),
        "dataset_id": str(dataset["dataset_id"]),
        "rule_json": dataset["rule_json"],
        "snapshot_hash": str(manifest["snapshot_hash"]),
        "selection_uses_test": bool(dataset["selection_uses_test"]),
    }


def materialize(root: str | Path | None = None) -> dict[str, Any]:
    """Audit the parent snapshot and write the research-validation definition.

    Writes the git-tracked list, a protocol yaml, a staging definition, and
    the Japanese report. Does not open ``reproduction.duckdb`` for write and
    does not create a snapshot.
    """
    base = ROOT if root is None else Path(root)
    snapshots = base / "data" / "db" / "snapshots"
    snapshot = find_parent_snapshot(snapshots)
    _assert_outside_snapshot(LIST_JSON, snapshot)
    _assert_outside_snapshot(PROTOCOL_YAML, snapshot)
    _assert_outside_snapshot(STAGING_JSON, snapshot)
    _assert_outside_snapshot(REPORT_MD, snapshot)
    fingerprint = _directory_fingerprint(snapshot)
    database = base / "data" / "db" / "reproduction.duckdb"
    resource = base / "data" / "db" / "resource.duckdb"
    parent = parent_split_case_ids(snapshot)
    if [len(parent["train"]), len(parent["valid"]), len(parent["test"])] != [324000, 36000, 150]:
        raise ReproductionError(
            "parent split counts are not train 324000 / valid 36000 / test 150: "
            f"{len(parent['train'])} / {len(parent['valid'])} / {len(parent['test'])}"
        )
    selection = select_research_validation(parent["valid"], parent["train"], parent["test"])
    if selection["reduced_because_leak"] or selection["excluded_case_ids"] or selection["count"] != SELECTION_COUNT:
        raise ReproductionError("parent validation leaked into train or test; refusing to publish 1024")
    attributes = _audit_documents(
        snapshot / "docs.jsonl",
        parent,
        set(selection["case_ids"]),
        _index_asset_sha256(snapshot),
    )
    speakers = _speaker_counts(snapshot, parent, set(selection["case_ids"]))
    if speakers["valid"]["present"] + speakers["valid"]["absent"] != len(parent["valid"]):
        raise ReproductionError("parent validation speaker counts do not cover every validation case")
    if speakers["selected"]["present"] + speakers["selected"]["absent"] != selection["count"]:
        raise ReproductionError("selected speaker counts do not cover every selected case")
    if attributes["speech_sha256_documents"]["parent_valid"] or attributes["speech_sha256_documents"]["selected"]:
        raise ReproductionError("validation case documents contain clean sha256; refusing a blanket UNKNOWN label")
    payload = _definition(parent, selection, attributes, speakers, fingerprint, database, resource)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_owned(base / "configs" / "data" / "lists" / "tf_locoformer" / "research_valid_fixed_v1.json", text)
    _write_owned(
        base / "data" / "staging" / "reproduction" / "research_valid_fixed_v1" / "dataset_definition.json",
        text,
    )
    _write_owned(
        base / "configs" / "data" / "protocols" / "dns.tflocoformer.research_valid_fixed.yaml",
        _protocol_yaml(payload),
    )
    _write_owned(
        base / "docs" / "training" / "reproduction" / "preflight" / "AGENT4_RESEARCH_VALID.md",
        render_report(payload),
    )
    return payload


def render_report(payload: Mapping[str, Any]) -> str:
    """Japanese record of the protocol, the leak labels, and the time estimate."""
    leak = payload["leak"]
    time = payload["time_estimate"]
    subset = time["subset"]
    lines = [
        "# TF-Locoformer 研究用固定検証（Agent 4）",
        "",
        "日付: 2026-11-07。学習は開始していない。GPU は使っていない。混合 WAV と波形キャッシュは作っていない。",
        "",
        "## 結論",
        "",
        f"- Dataset id は `{payload['dataset_id']}`。登録状態は **BLOCKED**。snapshot hash は発行していない。",
        f"- 親は `{payload['parent']['dataset_selector']}`。manifest の snapshot hash は `{payload['parent']['snapshot_hash']}`。検証 36,000 件から {payload['selection']['count']} 件を選んだ。",
        f"- リーク判定は case id の train / test が PASS、案件文書の clean SHA-256 が UNKNOWN、話者が UNKNOWN、録音 ID が UNKNOWN、雑音ファイル名が {leak['checks']['noise_file_name']['status']}。`leakage_safe` ではない。",
            f"- 150 epoch の検証時間は式で外挿すると {subset['all_epochs_hours']} 時間。公開されている全件検証 841 時間に対する外挿であり、再計測ではない。",
        "",
        "## Dataset",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        f"| dataset id | `{payload['dataset_id']}` |",
        f"| claim | `{payload['claim_label']}` |",
        f"| profile | `{payload['profile']}`。スキーマ上の非 leakage 値であり、公式再現という意味ではない |",
        f"| selection_uses_test | {str(payload['selection_uses_test']).lower()} |",
        f"| 登録 | {payload['registration_status']} |",
        f"| snapshot hash | BLOCKED。null のまま。作っていない |",
        f"| 親 dataset | `{payload['parent']['dataset_selector']}` |",
        f"| 親 dataset_id | `{payload['parent']['dataset_id']}` |",
        f"| 親 protocol_id | `{payload['parent']['protocol_id']}` |",
        f"| 親 snapshot hash | `{payload['parent']['snapshot_hash']}` |",
        f"| 親の件数 | train {payload['parent']['counts']['train']} / valid {payload['parent']['counts']['valid']} / test {payload['parent']['counts']['test']} |",
        f"| この定義の valid | {payload['selection']['count']} |",
        f"| train / test の case set | 親と同じ id を指す想定。新しいスナップショットは書いていない |",
        f"| case_id_list_sha256 | `{payload['case_id_list_sha256']}` |",
        "",
        "未登録の内容ハッシュは、登録できた場合に `dataset_id_for` が返す値である。スナップショットハッシュではない。",
        "",
        f"- protocol 内容 id: `{payload['unregistered_content_ids']['protocol_id']}`",
        f"- valid case set 内容 id: `{payload['unregistered_content_ids']['valid_case_set_id']}`",
        f"- dataset 内容 id: `{payload['unregistered_content_ids']['dataset_id']}`",
        "",
        "## 選択規則",
        "",
        "アルゴリズム名は `research_valid_fixed_v1`。seed は整数 `20261107`。候補数は 1024。母集団は親の validation 36,000 件だけである。先頭 1024 件は取らない。",
        "",
        "```text",
        SELECTION_RULE,
        "```",
        "",
        f"公開リストは辞書順である。SHA-256 は `{payload['case_id_list_sha256']}`。",
        "同じ整列済み case id を渡す限り、プロセス数には依存しない。worker ごとの部分母集団から下位を集める操作は、この規則と違う集合になる。",
        "",
        f"train / test と重なった validation case id は {len(payload['selection']['excluded_case_ids'])} 件だった。件数は 1024 のままである。",
        "",
        "## 登録が BLOCKED である理由",
        "",
    ]
    for reason in payload["blockers"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "`register_staging` は呼んでいない。`reproduction.duckdb` は作っていない。`INDEX.json` も作っていない。staging に書いた JSON は定義の写しであり、登録グラフではない。",
            "",
            "## リーク",
            "",
            "PASS は、その検査の入力が揃っていて重なりが 0 のときだけ使う。欠測を PASS にはしない。",
            "",
            "| 検査 | 状態 | 内容 |",
            "|---|---|---|",
        ]
    )
    descriptions = {
        "case_id_train": "選択 case id と親 train の交差",
        "case_id_test": "選択 case id と親 test の交差",
        "clean_sha256_in_case_document": "案件文書の clean SHA-256",
        "speaker": "speaker_id",
        "recording": "recording id",
        "noise_file_name": "雑音ファイル名の train との共有",
    }
    for name in LEAK_CHECKS:
        check = leak["checks"][name]
        lines.append(f"| {descriptions[name]} | {check['status']} | {_check_detail(name, check)} |")
    lines.extend(
        [
            "",
            f"leakage_safe: {str(payload['leakage_safe']).lower()}。上の検査がすべて PASS ではないため、この名前では呼ばない。",
            "",
            "### ファイル名と、案件文書に無い SHA-256",
            "",
            leak["research_impact"]["noise"],
            "",
            leak["research_impact"]["clean"],
            "",
            (
                f"親 validation 36,000 件と train の clean ファイル名交差は {leak['train_valid_file_overlap']['clean_names']} 件、"
                f"雑音ファイル名交差は {leak['train_valid_file_overlap']['noise_names']} 件である。"
                "選択 1024 件の clean 交差が 0 なのは、その交差を除く規則ではなく、ハッシュ抽出の結果である。"
            ),
            "",
            leak["research_impact"]["speaker_recording"],
            "",
            "| 追加の計数 | 値 |",
            "|---|---|",
            f"| train と共有する clean ファイル名 | {leak['clean_file_name']['shared_names']} |",
            f"| clean ファイル名を train と共有する選択案件 | {leak['clean_file_name']['selected_cases_sharing']} / {payload['selection']['count']} |",
            f"| 共有 clean ファイル名の sha256（名前リスト） | {_optional_hash(leak['clean_file_name']['shared_name_sha256'])} |",
            f"| スナップショット asset へファイル名で結んだ clean SHA の共有 | {leak['clean_sha256_via_asset_join']['status']} / 共有 {leak['clean_sha256_via_asset_join']['shared_hashes']} |",
            f"| train と共有する雑音ファイル名 | {leak['checks']['noise_file_name']['shared_noise_names']} |",
            f"| 雑音ファイル名を train と共有する選択案件 | {leak['checks']['noise_file_name']['selected_cases_sharing_train_noise']} / {payload['selection']['count']} |",
            f"| 共有雑音ファイル名の sha256（名前リスト） | `{leak['noise_file_name_sha256']}` |",
            f"| スナップショット asset へ結んだ雑音 SHA の共有 | {leak['noise_sha256_via_asset_join']['status']} / 共有 {leak['noise_sha256_via_asset_join']['shared_hashes']} |",
            f"| 未解決のファイル名 | clean {leak['unresolved_names']['clean']} / noise {leak['unresolved_names']['noise']} |",
            f"| 親 validation 全体と train の clean ファイル名共有 | {leak['train_valid_file_overlap']['clean_names']} |",
            f"| 親 validation 全体と train の雑音ファイル名共有 | {leak['train_valid_file_overlap']['noise_names']} |",
            "",
            "案件文書の clean SHA-256 が UNKNOWN であることは、asset 表への結合結果では覆らない。結合はファイル名が指す元素材の計数であり、文書内ハッシュの PASS ではない。",
            "",
            "親 Dataset の DB `leak_check` は、このホストに DB が無いので再実行していない。受理レシピの検証記録では asset 交差が 0 で話者は WARN だった。生成案件は asset 列を持たず、素材集合は train スロットにだけ付く。そのため DB の asset 交差 0 は、案件文書の中の clean / noise ファイル再利用が 0 であることを意味しない。ここでの計数は案件文書のファイル名である。",
            "",
            "## 分布",
            "",
            "比較相手は親の validation 36,000 件である。train だけとの比較ではない。割合は件数からの表示用丸め（小数 6 桁）である。",
            "",
            _count_summary(payload),
            "",
            "### SNR（dB）",
            "",
            _markdown_histogram(payload["distribution"]["snr_db"]),
            "",
            "### 目標レベル（dBFS）",
            "",
            _markdown_histogram(payload["distribution"]["target_level_dbfs"]),
            "",
            "### clean の source_rate",
            "",
            _markdown_histogram(payload["distribution"]["clean_source_rate"]),
            "",
            "### noise の source_rate",
            "",
            _markdown_histogram(payload["distribution"]["noise_source_rate"]),
            "",
            "### clean 断片数",
            "",
            _markdown_histogram(payload["distribution"]["clean_piece_count"]),
            "",
            "### noise 断片数",
            "",
            _markdown_histogram(payload["distribution"]["noise_piece_count"]),
            "",
            "### RIR",
            "",
            f"案件文書に RIR フィールド（{', '.join(RIR_KEYS)}）は無かった。parent valid の該当文書は {payload['distribution']['rir_documents']['parent_valid']}、選択集合は {payload['distribution']['rir_documents']['selected']}。状態は UNKNOWN であり、PASS ではない。DNS のこの受理計画は RIR を混合しない、という既存の説明を、この欠測の PASS には使わない。",
            "",
            "## 時間見積",
            "",
            "再計測ではない。P10-C の公開値を、発話数に比例させて外挿した。1 chunk 0.1602 秒、1 発話 14 chunk、4 rank が並列で、1 回の壁時計は rank あたり chunk 数 × 0.1602 秒である。36,000 発話では rank あたり 126,000 chunk になり、公開の 5.61 時間 / 150 epoch 841 時間と同じ式である。式どおりに計算すると 1 回 5.607 時間、150 epoch で 841.05 時間になる。841 時間は公開文書の丸めである。",
            "",
            "| 項目 | 1024 発話 | 親の 36000 発話 |",
            "|---|---:|---:|",
            f"| 発話数 | {subset['utterances']} | {time['full_validation']['utterances']} |",
            f"| chunk 数 | {subset['chunks']} | {time['full_validation']['chunks']} |",
            f"| rank あたり chunk | {subset['chunks_per_rank']} | {time['full_validation']['chunks_per_rank']} |",
            f"| 検証 1 回（時間） | {subset['one_pass_hours']} | {time['full_validation']['one_pass_hours']} |",
            f"| 検証 150 epoch（時間） | {subset['all_epochs_hours']} | {time['full_validation']['all_epochs_hours']} |",
            "",
            f"学習時間は {time['training_hours']} 時間のままである。",
            f"Checkpoint は epoch ごとに 1 回書き、その後に剪定する想定である。書き込み時間は 2.29 秒 × 150 = {time['checkpoint']['write_seconds']} 秒（{_round(Decimal(time['checkpoint']['write_seconds']) / Decimal(3600), 7)} 時間）。150 ファイルは残さない。保持は最新 3、best、nbest 5、平均の役割で、役割が重ならないときの上限は {time['checkpoint']['kept_files_upper_bound_if_roles_do_not_overlap']} ファイル、{time['checkpoint']['kept_bytes_upper_bound']} バイトである。平均ファイルを epoch 書き込みとは別に足す計測は無いので、足していない。",
            "",
            f"合計は学習 {time['total_hours_components']['training_hours']} + 検証 {time['total_hours_components']['validation_150_hours']} + checkpoint 書き込み {time['total_hours_components']['checkpoint_io_seconds']} 秒で、{_round_note(time['total_hours_rounded_4dp'])} 時間。",
            f"公開の全件検証 841 時間に対し、この検証 150 epoch は {subset['all_epochs_hours']} 時間である。同じ式の全件側は {time['formula_full_validation_hours']} 時間で、差は検証時間の短縮分である。",
            f"P10-C の余裕（学習 1.3、検証 1.5）を同じ外挿に掛けると、合計は約 {time['margin_1_3_train_1_5_validation']['total_hours_rounded_4dp']} 時間になる。checkpoint 書き込みには係数を掛けていない。これも外挿である。",
            "",
            "## 主張していないこと",
            "",
            "- 公式再現であるとは主張しない。`cv_synthetic_small` でもない。公式の共有カウンタとのビット一致も主張しない。",
            (
                "- `leakage_safe` とは主張しない。話者、録音 ID、案件文書の clean SHA-256 は UNKNOWN である。"
                "雑音ファイル名は train と共有する。"
                f"選択 1024 件の clean ファイル名と train の交差は {leak['clean_file_name']['shared_names']}、"
                f"親 validation 全体では {leak['train_valid_file_overlap']['clean_names']} である。"
                "文書内 SHA-256 が無いので、clean の独立は PASS にしない。"
            ),
            "- 親の 36,000 件 validation を置き換えたとは主張しない。親の protocol id と snapshot hash は変えていない。",
            "- 新しい snapshot hash は無い。未登録の内容 id をスナップショットの所在として扱わない。",
            "- 150 epoch は走らせていない。時間は公開計測からの外挿である。",
            "- この集合での WER、PESQ、STOI、early stopping の実際の epoch 数は測っていない。",
            "",
            "## 変更していないもの",
            "",
            "- `data/db/resource.duckdb`",
            "- `data/db/reproduction.duckdb`（この作業木には無かった。作っていない）",
            "- 親スナップショットディレクトリのファイル",
            "- `data/corpora`、`data/derived`、`data/checkpoints`、`data/results`",
            "- 受理レシピの build yaml",
            "",
        ]
    )
    return "\n".join(lines)


def _definition(
    parent: Mapping[str, Any],
    selection: Mapping[str, Any],
    attributes: Mapping[str, Any],
    speakers: Mapping[str, Any],
    fingerprint: list[dict[str, Any]],
    database: Path,
    resource: Path,
) -> dict[str, Any]:
    case_ids = list(selection["case_ids"])
    valid_set = _valid_case_set(case_ids, parent["case_set_ids"]["valid"])
    protocol = _protocol(parent["rule_json"], parent["protocol_id"])
    content_dataset_id = _content_dataset_id(protocol["protocol_id"], valid_set["case_set_id"], parent)
    checks = _leak_checks(selection, attributes, speakers)
    safe = leakage_safe_allowed(checks)
    blockers = _blockers(database, ROOT / "data" / "staging" / "reproduction" / "INDEX.json", safe)
    return {
        "generator": ALGORITHM_NAME,
        "dataset_id": NEW_DATASET_SELECTOR,
        "dataset_name": NEW_DATASET_NAME,
        "version": 1,
        "registered": False,
        "registration_status": "BLOCKED",
        "snapshot_hash": None,
        "snapshot_status": "BLOCKED",
        "claim_label": CLAIM_LABEL,
        "profile": "official_faithful",
        "profile_note": "official_faithful is the schema profile that is not leakage_safe. It does not mean official reproduction.",
        "material_origin": "derived_from_official",
        "selection_uses_test": False,
        "declared_status_if_registered": "ready",
        "blockers": blockers,
        "leakage_safe": safe,
        "parent": {
            "dataset_name": PARENT_DATASET_NAME,
            "dataset_selector": PARENT_DATASET_SELECTOR,
            "dataset_id": parent["dataset_id"],
            "protocol_id": parent["protocol_id"],
            "snapshot_hash": parent["snapshot_hash"],
            "snapshot_directory": f"data/db/snapshots/{parent['dataset_id']}",
            "counts": {
                "train": len(parent["train"]),
                "valid": len(parent["valid"]),
                "test": len(parent["test"]),
            },
            "case_set_ids": parent["case_set_ids"],
            "materials": parent["materials"],
            "selection_uses_test": parent["selection_uses_test"],
            "manifest_fingerprint": fingerprint,
        },
        "selection": {
            "algorithm": ALGORITHM_NAME,
            "seed": SELECTION_SEED,
            "count": selection["count"],
            "requested_count": selection["requested_count"],
            "population": "validation split of tf_locoformer.dns.accept_recipe@1",
            "population_count": selection["population_count"],
            "excluded_case_ids": selection["excluded_case_ids"],
            "reduced_because_leak": selection["reduced_because_leak"],
            "rule": SELECTION_RULE,
        },
        "case_ids": case_ids,
        "case_id_list_sha256": selection["case_id_list_sha256"],
        "case_id_list_sha256_rule": "sha256 hex of UTF-8 bytes of newline-joined lexicographic case ids plus a trailing newline",
        "intended_case_sets": {
            "train": parent["case_set_ids"]["train"],
            "valid": valid_set["case_set_id"],
            "test": parent["case_set_ids"]["test"],
            "note": "Train and test ids are the parent case sets. Only validation changes. No snapshot was exported.",
        },
        "unregistered_content_ids": {
            "protocol_id": protocol["protocol_id"],
            "valid_case_set_id": valid_set["case_set_id"],
            "dataset_id": content_dataset_id,
            "note": "Registrar hashes of this definition. Not inserted. Not a snapshot hash.",
        },
        "protocol": protocol,
        "valid_case_set": {
            "name": valid_set["name"],
            "case_set_id": valid_set["case_set_id"],
            "list_sha256_digest": valid_set["list_sha256"],
            "selection_rule_json": valid_set["selection_rule_json"],
        },
        "leak": {
            "checks": checks,
            "clean_file_name": attributes["clean_file_name"],
            "clean_sha256_via_asset_join": attributes["clean_sha_join"],
            "noise_sha256_via_asset_join": attributes["noise_sha_join"],
            "noise_file_name_sha256": attributes["noise_shared_name_sha256"],
            "unresolved_names": attributes["unresolved_names"],
            "train_valid_file_overlap": attributes["train_valid_file_overlap"],
            "speaker_counts": speakers,
            "research_impact": attributes["research_impact"],
        },
        "distribution": attributes["distribution"],
        "time_estimate": time_estimate(),
        "database": {
            "reproduction_duckdb_exists": database.is_file(),
            "resource_duckdb_exists": resource.is_file(),
            "resource_duckdb_mtime_ns": resource.stat().st_mtime_ns if resource.is_file() else None,
            "resource_duckdb_size": resource.stat().st_size if resource.is_file() else None,
        },
    }


def _leak_checks(
    selection: Mapping[str, Any],
    attributes: Mapping[str, Any],
    speakers: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    overlap = selection["case_id_overlap"]
    valid_speakers = speakers["valid"]
    selected_speakers = speakers["selected"]
    if valid_speakers["absent"] or selected_speakers["absent"]:
        speaker = {
            "status": "UNKNOWN",
            "missing": True,
            "present_valid": valid_speakers["present"],
            "absent_valid": valid_speakers["absent"],
            "present_selected": selected_speakers["present"],
            "absent_selected": selected_speakers["absent"],
        }
    else:
        speaker = speaker_independence(speakers["ids"]["valid"], speakers["ids"]["selected"])
    clean_document = clean_sha_independence([None], [None])
    clean_document["documents_with_speech_sha256"] = {
        "parent_valid": attributes["speech_sha256_documents"]["parent_valid"],
        "selected": attributes["speech_sha256_documents"]["selected"],
    }
    recording = recording_status(attributes["document_keys"]["valid"])
    recording["selected_keys_have_recording"] = bool(
        set(RECORDING_KEYS) & set(attributes["document_keys"]["selected"])
    )
    noise = noise_file_overlap(
        shared_names=attributes["noise_shared_names"],
        cases_sharing=attributes["noise_cases_sharing"],
    )
    return {
        "case_id_train": overlap["train"],
        "case_id_test": overlap["test"],
        "clean_sha256_in_case_document": clean_document,
        "speaker": speaker,
        "recording": recording,
        "noise_file_name": noise,
    }


def _audit_documents(
    docs_path: Path,
    parent: Mapping[str, Any],
    selected: set[str],
    assets: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    split_of = {}
    for split in ("train", "valid", "test"):
        for case_id in parent[split]:
            split_of[case_id] = split
    train_clean: set[str] = set()
    train_noise: set[str] = set()
    valid_clean: set[str] = set()
    valid_noise: set[str] = set()
    selected_clean: set[str] = set()
    selected_noise: set[str] = set()
    selected_cases: list[dict[str, set[str]]] = []
    histograms = {
        split: {
            "snr_db": Counter(),
            "target_level_dbfs": Counter(),
            "clean_source_rate": Counter(),
            "noise_source_rate": Counter(),
            "clean_piece_count": Counter(),
            "noise_piece_count": Counter(),
        }
        for split in ("valid", "selected")
    }
    keys = {split: set() for split in ("train", "valid", "test", "selected")}
    speech_sha_docs = Counter()
    rir_docs = Counter()
    unresolved = Counter()
    seen: set[str] = set()
    with docs_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row["case_id"])
            if case_id in seen:
                raise ReproductionError(f"duplicate case document at line {line_number}")
            seen.add(case_id)
            split = split_of.get(case_id)
            if split is None:
                raise ReproductionError(f"document {case_id} is outside the parent splits")
            parsed = _parse_document(row.get("doc_json") or {})
            keys[split].update(parsed["keys"])
            if split == "train":
                train_clean.update(parsed["clean_names"])
                train_noise.update(parsed["noise_names"])
            elif split == "valid":
                valid_clean.update(parsed["clean_names"])
                valid_noise.update(parsed["noise_names"])
                _add_histogram(histograms["valid"], parsed)
                if parsed["speech_sha256"]:
                    speech_sha_docs["parent_valid"] += 1
                if parsed["rir"]:
                    rir_docs["parent_valid"] += 1
            if case_id in selected:
                if split != "valid":
                    raise ReproductionError(f"selected case {case_id} is not in parent validation")
                keys["selected"].update(parsed["keys"])
                selected_clean.update(parsed["clean_names"])
                selected_noise.update(parsed["noise_names"])
                selected_cases.append(
                    {"clean": set(parsed["clean_names"]), "noise": set(parsed["noise_names"])}
                )
                _add_histogram(histograms["selected"], parsed)
                if parsed["speech_sha256"]:
                    speech_sha_docs["selected"] += 1
                if parsed["rir"]:
                    rir_docs["selected"] += 1
            if line_number % 50000 == 0:
                print(f"audited_docs\t{line_number}", flush=True)
    expected = set(split_of)
    if seen != expected:
        raise ReproductionError(
            f"case documents do not match the parent splits: missing {len(expected - seen)} extra {len(seen - expected)}"
        )
    if len(selected_cases) != len(selected):
        raise ReproductionError("selected cases were not all present in docs.jsonl")
    for role, names in (("clean", train_clean | valid_clean | selected_clean), ("noise", train_noise | valid_noise | selected_noise)):
        for name in names:
            if name not in assets[role]:
                unresolved[role] += 1
    shared_clean = train_clean & selected_clean
    shared_noise = train_noise & selected_noise
    train_valid_clean = train_clean & valid_clean
    train_valid_noise = train_noise & valid_noise
    clean_cases = sum(1 for item in selected_cases if item["clean"] & train_clean)
    noise_cases = sum(1 for item in selected_cases if item["noise"] & train_noise)
    clean_join = _hash_join(shared_clean, assets["clean"], unresolved["clean"])
    noise_join = _hash_join(shared_noise, assets["noise"], unresolved["noise"])
    impact = _impact(
        shared_clean=len(shared_clean),
        clean_cases=clean_cases,
        shared_noise=len(shared_noise),
        noise_cases=noise_cases,
        selected_count=len(selected),
    )
    return {
        "clean_file_name": {
            "status": "OVERLAP" if shared_clean else "NONE",
            "shared_names": len(shared_clean),
            "selected_cases_sharing": clean_cases,
            "shared_name_sha256": case_id_list_sha256(sorted(shared_clean)) if shared_clean else None,
            "note": "Filename overlap is not a case-document SHA-256 result.",
        },
        "clean_sha_join": clean_join,
        "noise_sha_join": noise_join,
        "noise_shared_names": len(shared_noise),
        "noise_cases_sharing": noise_cases,
        "noise_shared_name_sha256": case_id_list_sha256(sorted(shared_noise)) if shared_noise else None,
        "unresolved_names": {"clean": int(unresolved["clean"]), "noise": int(unresolved["noise"])},
        "train_valid_file_overlap": {
            "clean_names": len(train_valid_clean),
            "noise_names": len(train_valid_noise),
        },
        "speech_sha256_documents": {
            "parent_valid": int(speech_sha_docs["parent_valid"]),
            "selected": int(speech_sha_docs["selected"]),
        },
        "document_keys": {split: sorted(values) for split, values in keys.items()},
        "research_impact": impact,
        "distribution": {
            "name_counts": {
                "train_unique_clean_names": len(train_clean),
                "train_unique_noise_names": len(train_noise),
                "parent_valid_unique_clean_names": len(valid_clean),
                "parent_valid_unique_noise_names": len(valid_noise),
                "selected_unique_clean_names": len(selected_clean),
                "selected_unique_noise_names": len(selected_noise),
            },
            "snr_db": compare_histograms(histograms["valid"]["snr_db"], histograms["selected"]["snr_db"]),
            "target_level_dbfs": compare_histograms(
                histograms["valid"]["target_level_dbfs"], histograms["selected"]["target_level_dbfs"]
            ),
            "clean_source_rate": compare_histograms(
                histograms["valid"]["clean_source_rate"], histograms["selected"]["clean_source_rate"]
            ),
            "noise_source_rate": compare_histograms(
                histograms["valid"]["noise_source_rate"], histograms["selected"]["noise_source_rate"]
            ),
            "clean_piece_count": compare_histograms(
                histograms["valid"]["clean_piece_count"], histograms["selected"]["clean_piece_count"]
            ),
            "noise_piece_count": compare_histograms(
                histograms["valid"]["noise_piece_count"], histograms["selected"]["noise_piece_count"]
            ),
            "rir_documents": {
                "parent_valid": int(rir_docs["parent_valid"]),
                "selected": int(rir_docs["selected"]),
                "status": "UNKNOWN",
            },
            "document_keys": {split: sorted(values) for split, values in keys.items()},
        },
    }


def _parse_document(document: Mapping[str, Any]) -> dict[str, Any]:
    sources = document.get("sources") if isinstance(document.get("sources"), dict) else {}
    clean = _pieces(sources.get("speech"))
    if not clean:
        clean = _pieces(sources.get("clean"))
    noise = _pieces(sources.get("noise"))
    draws = document.get("draws") if isinstance(document.get("draws"), dict) else {}
    keys = set()
    _collect_keys(document, keys)
    return {
        "keys": keys,
        "clean_names": [str(piece["name"]) for piece in clean if piece.get("name")],
        "noise_names": [str(piece["name"]) for piece in noise if piece.get("name")],
        "clean_rates": [int(piece["source_rate"]) for piece in clean if piece.get("source_rate") is not None],
        "noise_rates": [int(piece["source_rate"]) for piece in noise if piece.get("source_rate") is not None],
        "snr_db": draws.get("snr_db"),
        "target_level_dbfs": draws.get("target_level_dbfs"),
        "speech_sha256": [str(piece["sha256"]) for piece in clean if isinstance(piece.get("sha256"), str)],
        "rir": bool(keys & set(RIR_KEYS)),
        "clean_piece_count": len(clean),
        "noise_piece_count": len(noise),
    }


def _pieces(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _collect_keys(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            _collect_keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_keys(child, found)


def _add_histogram(target: dict[str, Counter], parsed: Mapping[str, Any]) -> None:
    if isinstance(parsed["snr_db"], int) and not isinstance(parsed["snr_db"], bool):
        target["snr_db"][int(parsed["snr_db"])] += 1
    if isinstance(parsed["target_level_dbfs"], int) and not isinstance(parsed["target_level_dbfs"], bool):
        target["target_level_dbfs"][int(parsed["target_level_dbfs"])] += 1
    target["clean_piece_count"][int(parsed["clean_piece_count"])] += 1
    target["noise_piece_count"][int(parsed["noise_piece_count"])] += 1
    for rate in parsed["clean_rates"]:
        target["clean_source_rate"][int(rate)] += 1
    for rate in parsed["noise_rates"]:
        target["noise_source_rate"][int(rate)] += 1


def _index_asset_sha256(snapshot_dir: Path) -> dict[str, dict[str, str]]:
    from src.reproduction.snapshot_reader import assets_from_arrays

    rows = assets_from_arrays(_load_npz(snapshot_dir / "assets.npz"))
    clean: dict[str, str] = {}
    noise: dict[str, str] = {}
    for row in rows:
        rel = str(row["rel_path"])
        name = rel.rsplit("/", 1)[-1]
        sha = row.get("sha256")
        if not isinstance(sha, str):
            continue
        if rel.startswith("DNS2020/datasets/clean/"):
            _put_sha(clean, name, sha)
        elif rel.startswith("DNS2020/datasets/noise/"):
            _put_sha(noise, name, sha)
    return {"clean": clean, "noise": noise}


def _put_sha(bucket: dict[str, str], name: str, sha: str) -> None:
    previous = bucket.get(name)
    if previous is not None and previous != sha:
        raise ReproductionError(f"asset filename {name} has two sha256 values")
    bucket[name] = sha


def _hash_join(names: set[str], index: Mapping[str, str], unresolved: int) -> dict[str, Any]:
    if unresolved:
        return {
            "status": "UNKNOWN",
            "shared_hashes": None,
            "reason": "at least one case-document filename did not resolve to a snapshot asset",
        }
    hashes = {index[name] for name in names}
    return {
        "status": "OVERLAP" if hashes else "NONE",
        "shared_hashes": len(hashes),
        "note": "Resolved from snapshot asset sha256 by filename. Not stored in the case document. NONE is not PASS.",
    }


def _impact(
    *,
    shared_clean: int,
    clean_cases: int,
    shared_noise: int,
    noise_cases: int,
    selected_count: int,
) -> dict[str, str]:
    return {
        "noise": (
            f"選択 {selected_count} 件のうち {noise_cases} 件が、train と少なくとも 1 つの雑音ファイル名を共有する。"
            f"共有する雑音ファイル名は {shared_noise} 件である。"
            "DNS の受理計画は train と validation が同じ雑音プールから切り出されるので、この共有は想定内である。"
            "共有があるため、この検証集合は学習データと音源が独立ではない。"
            "checkpoint の選択にだけ使い、独立な汎化評価とは呼ばない。"
        ),
        "clean": (
            f"選択 {selected_count} 件のうち {clean_cases} 件が、train と少なくとも 1 つの clean ファイル名を共有する。"
            f"共有する clean ファイル名は {shared_clean} 件である。"
            "案件文書に clean の SHA-256 は無いので、その検査は UNKNOWN のままである。"
            "ファイル名の一致を SHA-256 の PASS にはしない。"
        ),
        "speaker_recording": (
            "案件の speaker_id は無く、文書に recording_id も無い。"
            "ファイル名に reader という綴りがあっても、話者 ID としては読まない。"
            "話者と録音 ID は UNKNOWN であり、PASS ではない。"
        ),
    }


def _speaker_counts(
    snapshot_dir: Path,
    parent: Mapping[str, Any],
    selected: set[str],
) -> dict[str, Any]:
    from src.reproduction.snapshot_reader import unpack_strings

    arrays = _load_npz(snapshot_dir / "cases.npz")
    case_ids = unpack_strings(arrays["case_id_data"], arrays["case_id_start"], arrays["case_id_end"])
    speakers = unpack_strings(arrays["speaker_id_data"], arrays["speaker_id_start"], arrays["speaker_id_end"])
    split_of = {}
    for split in ("train", "valid", "test"):
        for case_id in parent[split]:
            split_of[case_id] = split
    counts = {
        split: {"present": 0, "absent": 0}
        for split in ("train", "valid", "test", "selected")
    }
    ids = {split: set() for split in ("train", "valid", "test", "selected")}
    for case_id, speaker in zip(case_ids, speakers, strict=True):
        split = split_of.get(case_id)
        if split is None:
            continue
        present = isinstance(speaker, str) and bool(speaker)
        counts[split]["present" if present else "absent"] += 1
        if present:
            ids[split].add(speaker)
        if case_id in selected:
            counts["selected"]["present" if present else "absent"] += 1
            if present:
                ids["selected"].add(speaker)
    return {
        "train": counts["train"],
        "valid": counts["valid"],
        "test": counts["test"],
        "selected": counts["selected"],
        "ids": {split: sorted(values) for split, values in ids.items()},
    }


def _valid_case_set(case_ids: Sequence[str], parent_valid_case_set_id: str) -> dict[str, Any]:
    ordered = list(case_ids)
    if ordered != sorted(ordered):
        raise ReproductionError("published case ids must already be sorted")
    rule = {"rule": "fixed_subset", "of": parent_valid_case_set_id, "k": len(ordered)}
    body = {
        "name": "tf_locoformer.dns.accept_recipe.research_valid_fixed.valid",
        "version": 1,
        "split_role": "valid",
        "selection_rule_json": rule,
        "list_sha256": digest(ordered),
    }
    body["case_set_id"] = case_set_id_for(
        name=body["name"],
        version=1,
        split_role="valid",
        rule=rule,
        case_ids=ordered,
    )
    return body


def _protocol(parent_rule: Mapping[str, Any], parent_protocol_id: str) -> dict[str, Any]:
    from src.reproduction.ids import validate_rule
    from src.reproduction.prepare.dns2020 import OFFICIAL_REF

    rule = json.loads(json.dumps(parent_rule))
    validation = dict(rule["validation"])
    validation["note"] = PROTOCOL_VALIDATION_NOTE
    rule["validation"] = validation
    validate_rule("pregenerated.dns_tflocoformer", rule)
    protocol_id = protocol_id_for(
        algorithm="pregenerated.dns_tflocoformer",
        algorithm_version=1,
        kind="pregenerated",
        rule=rule,
        deviation_class="research_improvement",
        parent=parent_protocol_id,
    )
    return {
        "protocol_id": protocol_id,
        "name": "dns.tflocoformer.ondemand.accept.research_valid_fixed",
        "version": 1,
        "kind": "pregenerated",
        "algorithm": "pregenerated.dns_tflocoformer",
        "algorithm_version": 1,
        "deviation_class": "research_improvement",
        "parent_protocol_id": parent_protocol_id,
        "rule_json": rule,
        "official_ref_json": OFFICIAL_REF,
        "registered": False,
    }


def _content_dataset_id(protocol_id: str, valid_case_set_id: str, parent: Mapping[str, Any]) -> str:
    bindings = [
        {"slot": "train.cases", "case_set_id": parent["case_set_ids"]["train"], "material_set_id": None},
        {"slot": "valid.cases", "case_set_id": valid_case_set_id, "material_set_id": None},
        {"slot": "test.cases", "case_set_id": parent["case_set_ids"]["test"], "material_set_id": None},
        {"slot": "train.speech", "case_set_id": None, "material_set_id": parent["materials"]["train.speech"]},
        {"slot": "train.noise", "case_set_id": None, "material_set_id": parent["materials"]["train.noise"]},
    ]
    return dataset_id_for(
        protocol_id=protocol_id,
        bindings=bindings,
        profile="official_faithful",
        material_origin="derived_from_official",
        claim_label=CLAIM_LABEL,
        selection_uses_test=False,
        declared_status="ready",
    )


def _blockers(database: Path, index_path: Path, leakage_safe: bool) -> list[str]:
    reasons = []
    if not database.is_file():
        reasons.append(
            "この作業木に data/db/reproduction.duckdb が無い。空の DB は作っていない。親の行が無いファイルへは追記できない。"
        )
    if not index_path.is_file():
        reasons.append(
            "data/staging/reproduction/INDEX.json が無い。register_staging は INDEX の無いディレクトリを読めない。書いた JSON は登録グラフではない。"
        )
    if CLAIM_LABEL not in CLAIM_LABELS:
        reasons.append(
            "claim_label research_checkpoint_selection は CLAIM_LABELS と repro.datasets の CHECK に無い。スキーマは変えていない。既存の公式再現ラベルへは付け替えていない。"
        )
    if leakage_safe:
        reasons.append("leakage_safe の検査がすべて PASS だったが、claim のスキーマ制約で登録していない。")
    else:
        reasons.append(
            "話者、案件文書の clean SHA-256、録音 ID のいずれかが UNKNOWN であるか、雑音ファイルが train と共有する。leakage_safe としては登録しない。"
        )
    reasons.append("スナップショットは export していない。snapshot hash は作っていない。")
    return reasons


def _protocol_yaml(payload: Mapping[str, Any]) -> str:
    protocol = payload["protocol"]
    return "\n".join(
        [
            "# Research checkpoint-selection protocol for TF-Locoformer DNS.",
            "# Not inserted into reproduction.duckdb. snapshot_hash is BLOCKED.",
            "# Not cv_synthetic_small. Not an official reproduction. Not leakage_safe.",
            "# official_faithful below is only the schema profile that is not leakage_safe.",
            f"name: {protocol['name']}",
            "version: 1",
            "kind: pregenerated",
            "algorithm: pregenerated.dns_tflocoformer",
            "algorithm_version: 1",
            "deviation_class: research_improvement",
            f"parent_protocol_id: {protocol['parent_protocol_id']}",
            f"parent_dataset: {PARENT_DATASET_SELECTOR}",
            f"parent_snapshot_hash: {payload['parent']['snapshot_hash']}",
            f"dataset: {NEW_DATASET_SELECTOR}",
            f"claim_label: {CLAIM_LABEL}",
            "profile: official_faithful",
            "selection_uses_test: false",
            "registration: BLOCKED",
            "selection:",
            f"  algorithm: {ALGORITHM_NAME}",
            f"  seed: {SELECTION_SEED}",
            f"  count: {payload['selection']['count']}",
            "  population: parent validation case ids",
            f"  case_id_list_sha256: {payload['case_id_list_sha256']}",
            "list: configs/data/lists/tf_locoformer/research_valid_fixed_v1.json",
            "",
        ]
    )


def _check_detail(name: str, check: Mapping[str, Any]) -> str:
    if name in ("case_id_train", "case_id_test"):
        return f"交差 {check['count']}"
    if name == "noise_file_name":
        return (
            f"共有ファイル名 {check['shared_noise_names']}、"
            f"共有する選択案件 {check['selected_cases_sharing_train_noise']}"
        )
    if name == "clean_sha256_in_case_document":
        docs = check.get("documents_with_speech_sha256", {})
        return f"文書にある件数 parent valid {docs.get('parent_valid', 0)} / selected {docs.get('selected', 0)}"
    if name == "speaker":
        if check["status"] == "UNKNOWN":
            return (
                f"valid present {check.get('present_valid', 0)} / absent {check.get('absent_valid', 0)}、"
                f"selected present {check.get('present_selected', 0)} / absent {check.get('absent_selected', 0)}"
            )
        return "重なり 0"
    if name == "recording":
        return str(check.get("reason") or check["status"])
    return check["status"]


def _count_summary(payload: Mapping[str, Any]) -> str:
    names = payload["distribution"]["name_counts"]
    rir = payload["distribution"]["rir_documents"]
    sha = payload["leak"]["checks"]["clean_sha256_in_case_document"]["documents_with_speech_sha256"]
    speakers = payload["leak"]["speaker_counts"]
    lines = [
        "| 項目 | parent valid | selected |",
        "|---|---:|---:|",
        f"| 案件数 | {payload['parent']['counts']['valid']} | {payload['selection']['count']} |",
        f"| ユニーク clean ファイル名 | {names['parent_valid_unique_clean_names']} | {names['selected_unique_clean_names']} |",
        f"| ユニーク noise ファイル名 | {names['parent_valid_unique_noise_names']} | {names['selected_unique_noise_names']} |",
        f"| RIR フィールドがある文書 | {rir['parent_valid']} | {rir['selected']} |",
        f"| clean SHA-256 が文書にある案件 | {sha['parent_valid']} | {sha['selected']} |",
        f"| speaker_id がある案件 | {speakers['valid']['present']} | {speakers['selected']['present']} |",
    ]
    return "\n".join(lines)


def _markdown_histogram(table: Mapping[str, Any]) -> str:
    lines = [
        f"最大絶対割合差 {table['max_abs_fraction_delta']}。parent 合計 {table['parent_valid_total']}、選択合計 {table['selected_total']}。",
        "",
        "| 値 | parent valid | selected | parent 割合 | selected 割合 | 差 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table["rows"]:
        lines.append(
            f"| {row['value']} | {row['parent_valid']} | {row['selected']} | {row['parent_fraction']} | {row['selected_fraction']} | {row['fraction_delta']} |"
        )
    return "\n".join(lines)


def _directory_fingerprint(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return rows


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _write_owned(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == text:
            return
        if ALGORITHM_NAME not in current:
            raise ReproductionError(f"refusing to rewrite an unrelated file: {path}")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _assert_outside_snapshot(path: Path, snapshot: Path) -> None:
    resolved = path.resolve()
    root = snapshot.resolve()
    if resolved == root or root in resolved.parents:
        raise ReproductionError(f"refusing to write inside the parent snapshot: {path}")


def _sorted_unique(case_ids: Iterable[str]) -> list[str]:
    items = _require_case_ids(list(case_ids))
    ordered = sorted(items)
    if len(set(ordered)) != len(ordered):
        raise ReproductionError("case_id population contains duplicates")
    return ordered


def _require_case_ids(case_ids: list[Any]) -> list[str]:
    for case_id in case_ids:
        if not isinstance(case_id, str) or not case_id:
            raise ReproductionError("case_id must be a non-empty string")
    return case_ids


def _seed_text(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ReproductionError("selection seed must be a non-negative integer")
    return str(seed)


def _positive_int(value: int, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReproductionError(f"{where} must be a positive integer")
    return value


def _non_negative_int(value: int, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReproductionError(f"{where} must be a non-negative integer")
    return value


def _overlap_status(hits: list[str]) -> dict[str, Any]:
    return {"status": "PASS" if not hits else "FAIL", "count": len(hits), "ids": hits}


def _present_ids(values: Sequence[str | None]) -> tuple[list[str], bool]:
    present = []
    missing = False
    for value in values:
        if isinstance(value, str) and value:
            present.append(value)
        else:
            missing = True
    return present, missing


def _histogram_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))


def _fraction(count: int, total: int) -> Decimal | None:
    if total <= 0:
        return None
    return Decimal(count) / Decimal(total)


def _fraction_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return _round(value, 6)


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _round(value: Decimal, places: int) -> str:
    quant = Decimal(1).scaleb(-places)
    return format(value.quantize(quant, rounding=ROUND_HALF_UP), "f")


def _round_note(value: str) -> str:
    return value


def _optional_hash(value: str | None) -> str:
    if value is None:
        return "なし（共有 0）"
    return f"`{value}`"


def locate_reproduction_database(root: str | Path | None = None) -> dict[str, Any]:
    """Look only at the documented database locations.

    This does not create a file and does not search the wider filesystem.
    """
    base = ROOT if root is None else Path(root)
    candidates = (
        base / "data" / "db" / "reproduction.duckdb",
        base / "data" / "db" / "backups" / "reproduction.duckdb",
        base / "data" / "staging" / "reproduction" / "reproduction.duckdb",
    )
    found = [str(path) for path in candidates if path.is_file()]
    return {
        "candidates": [str(path) for path in candidates],
        "found": found,
        "registration": "AVAILABLE" if found else "BLOCKED",
    }


def research_child_graph(parent: Any, *, parent_name: str, child_name: str, selected_case_ids: list[str]) -> Any:
    """New protocol, validation case set, and dataset. Parent rows are not copied as datasets.

    ``selected_case_ids`` must already be the published order. They have to be
    members of the parent's validation case set. Train and test bindings stay
    on the parent's ids.
    """
    from src.reproduction.db.graph import Graph
    from src.reproduction.ids import protocol_id_for

    if not isinstance(parent_name, str) or not parent_name:
        raise ReproductionError("parent dataset name is required")
    if not isinstance(child_name, str) or not child_name or child_name == parent_name:
        raise ReproductionError("child dataset name must be a new name")
    selected = _require_case_ids(list(selected_case_ids))
    if selected != sorted(selected):
        raise ReproductionError("published case ids must already be sorted")
    if len(set(selected)) != len(selected):
        raise ReproductionError("selected case ids contain duplicates")
    datasets = [
        row for row in parent.datasets if row["name"] == parent_name and int(row["version"]) == 1
    ]
    if len(datasets) != 1:
        raise ReproductionError(f"parent dataset {parent_name}@1 was not found once")
    dataset = datasets[0]
    valid_binding = next((item for item in dataset["bindings"] if item["slot"] == "valid.cases"), None)
    if valid_binding is None or not valid_binding.get("case_set_id"):
        raise ReproductionError("parent dataset has no validation case set")
    valid_id = valid_binding["case_set_id"]
    case_sets = {row["case_set_id"]: row for row in parent.case_sets}
    valid_set = case_sets.get(valid_id)
    if valid_set is None:
        raise ReproductionError("parent validation case set is missing")
    population = [item["case_id"] for item in valid_set["items"]]
    unknown = [case_id for case_id in selected if case_id not in set(population)]
    if unknown:
        raise ReproductionError("selected case ids are not in the parent validation split")
    test_binding = next((item for item in dataset["bindings"] if item["slot"] == "test.cases"), None)
    if test_binding is not None and test_binding.get("case_set_id"):
        test_set = case_sets.get(test_binding["case_set_id"])
        test_ids = set() if test_set is None else {item["case_id"] for item in test_set["items"]}
        if set(selected) & test_ids:
            raise ReproductionError("research validation overlaps the parent test split")
    protocols = {row["protocol_id"]: row for row in parent.protocols}
    parent_protocol = protocols.get(dataset["protocol_id"])
    if parent_protocol is None:
        raise ReproductionError("parent protocol is missing")
    protocol = {
        "name": f"{child_name}.protocol",
        "version": 1,
        "kind": parent_protocol["kind"],
        "algorithm": parent_protocol["algorithm"],
        "algorithm_version": parent_protocol["algorithm_version"],
        "deviation_class": "research_improvement",
        "parent_protocol_id": parent_protocol["protocol_id"],
        "rule_json": json.loads(json.dumps(parent_protocol["rule_json"])),
        "official_ref_json": json.loads(json.dumps(parent_protocol["official_ref_json"])),
        "deviations_json": [
            {
                "id": ALGORITHM_NAME,
                "class": "research_improvement",
                "text": (
                    "validation case ids are a fixed research subset of the parent "
                    "validation split; not leakage_safe and not cv_synthetic_small"
                ),
            }
        ],
    }
    protocol["protocol_id"] = protocol_id_for(
        algorithm=protocol["algorithm"],
        algorithm_version=int(protocol["algorithm_version"]),
        kind=protocol["kind"],
        rule=protocol["rule_json"],
        deviation_class="research_improvement",
        parent=parent_protocol["protocol_id"],
    )
    rule = {"rule": "fixed_subset", "of": valid_id, "k": len(selected)}
    case_set_name = f"{child_name}.valid"
    case_set = {
        "case_set_id": case_set_id_for(
            name=case_set_name,
            version=1,
            split_role="valid",
            rule=rule,
            case_ids=selected,
        ),
        "name": case_set_name,
        "version": 1,
        "split_role": "valid",
        "origin": "derived_from_official",
        "selection_rule_json": rule,
        "provenance_json": {"source": ALGORITHM_NAME, "note": "subset of the parent validation case set"},
        "member_count": len(selected),
        "list_sha256": digest(selected),
        "items": [{"position": index, "case_id": case_id} for index, case_id in enumerate(selected)],
    }
    bindings = []
    for binding in dataset["bindings"]:
        if binding["slot"] == "valid.cases":
            bindings.append(
                {"slot": "valid.cases", "case_set_id": case_set["case_set_id"], "material_set_id": None}
            )
        else:
            bindings.append(
                {
                    "slot": binding["slot"],
                    "case_set_id": binding.get("case_set_id"),
                    "material_set_id": binding.get("material_set_id"),
                }
            )
    child = {
        "dataset_id": dataset_id_for(
            protocol_id=protocol["protocol_id"],
            bindings=bindings,
            profile="official_faithful",
            material_origin="derived_from_official",
            claim_label=CLAIM_LABEL,
            selection_uses_test=False,
            declared_status="ready",
        ),
        "name": child_name,
        "version": 1,
        "protocol_id": protocol["protocol_id"],
        "profile": "official_faithful",
        "material_origin": "derived_from_official",
        "claim_label": CLAIM_LABEL,
        "declared_status": "ready",
        "blocking_json": [],
        "selection_uses_test": False,
        "selection_note": (
            f"{ALGORITHM_NAME}; not leakage_safe; not cv_synthetic_small; "
            f"case_id_list_sha256={case_id_list_sha256(selected)}"
        ),
        "deviations_json": list(protocol["deviations_json"]),
        "bindings": bindings,
    }
    by_id = {row["case_id"]: row for row in parent.cases}
    cases = []
    for case_id in selected:
        case = by_id.get(case_id)
        if case is None:
            raise ReproductionError(f"selected case {case_id} is missing from the parent graph")
        cases.append(json.loads(json.dumps(case)))
    return Graph(protocols=[protocol], cases=cases, case_sets=[case_set], datasets=[child])


def register_research_validation_dataset(
    database: str | Path,
    *,
    parent_name: str,
    child_name: str,
    selected_case_ids: list[str],
    staging: str | Path,
) -> dict[str, Any]:
    """Append the research dataset. Refuse to create a missing database.

    The claim check is extended in place when an older database does not
    list it yet. Existing dataset rows are not updated. A second call with
    the same child is a no-op.
    """
    path = Path(database)
    if not path.is_file():
        raise ReproductionError(f"database does not exist: {path}. Refusing to create it.")
    import duckdb

    from src.reproduction.db.lock import reproduction_lock
    from src.reproduction.db.registry import read_graph, register_staging
    from src.reproduction.db.schema import ensure_dataset_claim_labels
    from src.reproduction.db.staging import write_graph

    with reproduction_lock(path.parent):
        connection = duckdb.connect(str(path))
        try:
            ensure_dataset_claim_labels(connection)
        finally:
            connection.close()
    parent = read_graph(path)
    before = {
        row["dataset_id"]: digest(row)
        for row in parent.datasets
    }
    child = research_child_graph(
        parent,
        parent_name=parent_name,
        child_name=child_name,
        selected_case_ids=selected_case_ids,
    )
    write_graph(staging, child)
    counts = register_staging(staging, path)
    after = read_graph(path)
    after_ids = {row["dataset_id"]: digest(row) for row in after.datasets}
    for dataset_id, fingerprint in before.items():
        if after_ids.get(dataset_id) != fingerprint:
            raise ReproductionError(f"parent dataset {dataset_id} changed during research registration")
    stored = next(row for row in after.datasets if row["name"] == child_name and int(row["version"]) == 1)
    if stored["claim_label"] != CLAIM_LABEL or stored["dataset_id"] != child.datasets[0]["dataset_id"]:
        raise ReproductionError("registered research dataset does not match the staged id")
    return {
        "dataset_id": stored["dataset_id"],
        "dataset_name": child_name,
        "protocol_id": stored["protocol_id"],
        "valid_case_set_id": next(
            item["case_set_id"] for item in stored["bindings"] if item["slot"] == "valid.cases"
        ),
        "claim_label": stored["claim_label"],
        "inserted": counts,
        "parent_dataset_ids": sorted(before),
    }


def main() -> None:
    payload = materialize()
    leak = payload["leak"]["checks"]
    print(f"dataset\t{payload['dataset_id']}", flush=True)
    print(f"registration\t{payload['registration_status']}", flush=True)
    print(f"snapshot\tBLOCKED", flush=True)
    print(f"count\t{payload['selection']['count']}", flush=True)
    print(f"list_sha256\t{payload['case_id_list_sha256']}", flush=True)
    for name in LEAK_CHECKS:
        print(f"leak\t{name}\t{leak[name]['status']}", flush=True)
    print(f"leakage_safe\t{payload['leakage_safe']}", flush=True)


if __name__ == "__main__":
    main()
