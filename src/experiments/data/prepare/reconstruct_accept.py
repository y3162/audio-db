"""Rebuild the DNS accept-recipe graph from its verified snapshot.

The original ``reproduction.duckdb`` is not in this worktree. This module does
not invent a replacement and then call it the original file. It rebuilds only
the rows whose content-addressed ids are fixed by the snapshot plus the
registrar that produced those ids. Fields the snapshot does not store
(catalog id, asset ``sample_format``, build-audit sample counts) stay marked
unverified. An id that does not match the snapshot aborts before any database
file is created.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.data.recipe import sha256_file
from src.experiments.data.build.dns2020 import (
    AUDIO_SECONDS,
    SAMPLE_RATE,
    build_definition,
    order_sha256,
    reader_id,
    split_train_valid,
)
from src.experiments.data.db.graph import Graph
from src.experiments.data.db.snapshot import _parts
from src.experiments.data.db.validate import validate_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest
from src.experiments.data.prepare.dns2020 import (
    ACCEPT_NOTE,
    accept_protocol_pair,
    asset_row,
    build_row,
    case_set_for,
    condition_case,
    material_set,
    protocol_pair,
    recorded_pair,
)
from src.experiments.data.prepare.dns_run import (
    ACCEPT_NOISE_NAME,
    ACCEPT_RECIPE_NAME,
    ACCEPT_SPEECH_NAME,
    P7_TEST_SET_NAME,
    RECIPE_FILES,
    _catalog_rows,
)
from src.experiments.data.prepare.research_valid_fixed import (
    PARENT_DATASET_ID,
    PARENT_SNAPSHOT_HASH,
)
from src.experiments.data.snapshot import PART_NAMES, file_hashes, load_snapshot, snapshot_hash

ORIGIN = "reconstructed_from_verified_snapshot"
PUBLISHED_LIST_SHA256 = "73e8d44a3c2196af874f8cddb7a1cc9a6b1745a220d6f1b88fda7a9a502f7ad4"
_ACCEPT_SEED = 17
_SHARD_SIZE = 1000
_SHA_SAMPLE = 32


def snapshot_fingerprint(directory: str | Path) -> list[tuple[str, int, int]]:
    """Name, size, and mtime of each file directly inside a snapshot directory."""
    root = Path(directory)
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append((path.name, stat.st_size, stat.st_mtime_ns))
    return rows


def reconstruct_parent_graph(
    snapshot_dir: str | Path,
    *,
    corpora: str | Path,
    resource_db: str | Path,
) -> tuple[Graph, dict[str, Any]]:
    """Return the parent graph and a report. Does not write a database."""
    directory = Path(snapshot_dir)
    print(f"load snapshot {directory}", flush=True)
    loaded = load_snapshot(directory)
    if loaded.snapshot_hash != PARENT_SNAPSHOT_HASH:
        raise ReproductionError(
            f"snapshot hash {loaded.snapshot_hash} is not the accept-recipe parent {PARENT_SNAPSHOT_HASH}"
        )
    if loaded.dataset["dataset_id"] != PARENT_DATASET_ID:
        raise ReproductionError("snapshot dataset_id is not the accept-recipe parent")
    if loaded.dataset["name"] != ACCEPT_RECIPE_NAME or int(loaded.dataset["version"]) != 1:
        raise ReproductionError("snapshot dataset name is not tf_locoformer.dns.accept_recipe@1")
    if loaded.generated:
        raise ReproductionError("accept-recipe snapshot contains generated assets; refusing to invent them")
    parent_protocol, accept = accept_protocol_pair()
    _ondemand_parent, ondemand = protocol_pair()
    if digest(parent_protocol) != digest(_ondemand_parent):
        raise ReproductionError("accept and P7 formula protocols diverged")
    if accept["protocol_id"] != loaded.dataset["protocol_id"]:
        raise ReproductionError("accept protocol id does not match the snapshot")
    if accept["rule_json"] != loaded.dataset["rule_json"]:
        raise ReproductionError("accept protocol rule does not match the snapshot")
    bindings = {row["slot"]: row for row in loaded.dataset["bindings"]}
    speech_id = bindings["train.speech"]["material_set_id"]
    noise_id = bindings["train.noise"]["material_set_id"]
    train_set_id = bindings["train.cases"]["case_set_id"]
    valid_set_id = bindings["valid.cases"]["case_set_id"]
    test_set_id = bindings["test.cases"]["case_set_id"]
    print("read resource catalog", flush=True)
    pools = _accept_pools(Path(resource_db), Path(corpora))
    assets, asset_report = _assets_from_snapshot(loaded, pools, Path(corpora))
    by_id = {row["asset_id"]: row for row in assets}
    speech = _material_from_snapshot(
        loaded,
        speech_id,
        name=ACCEPT_SPEECH_NAME,
        role="speech",
        note=(
            "All 16 kHz clean files for the 3000 h acceptance recipe, "
            "sorted then shuffled with random.Random(seed)."
        ),
        assets_by_id=by_id,
        expected_rels=set(pools["clean"]),
    )
    noise = _material_from_snapshot(
        loaded,
        noise_id,
        name=ACCEPT_NOISE_NAME,
        role="noise",
        note=(
            "All mono noise files, including non-16 kHz files. Those are resampled at plan and "
            "render time with librosa 1.0.0 soxr_hq. Source files are not rewritten."
        ),
        assets_by_id=by_id,
        expected_rels=set(pools["noise"]),
    )
    definition = build_definition(
        seed=_ACCEPT_SEED,
        num_files=RECIPE_FILES,
        clean_count=len(speech["members"]),
        noise_count=len(noise["members"]),
        order_sha256=order_sha256(
            [Path(by_id[row["asset_id"]]["rel_path"]).name for row in speech["members"]],
            [Path(by_id[row["asset_id"]]["rel_path"]).name for row in noise["members"]],
        ),
        inputs={"speech": speech["material_set_id"], "noise": noise["material_set_id"]},
        shard_size=_SHARD_SIZE,
        audio_seconds=AUDIO_SECONDS,
        total_hours=3000.0,
        excluded_non_16k_noise=int(pools["excluded"]),
        acceptance="deterministic_case_stream_v1",
    )
    print("rebuild cases", flush=True)
    cases, case_sets, case_report = _cases_from_snapshot(
        loaded,
        accept_protocol_id=accept["protocol_id"],
        ondemand_protocol_id=ondemand["protocol_id"],
        train_set_id=train_set_id,
        valid_set_id=valid_set_id,
        test_set_id=test_set_id,
        assets_by_id=by_id,
        build_id=definition["build_id"],
    )
    deviations_match = accept["deviations_json"] + [
        {
            "id": "full_accept_recipe",
            "class": "algorithm_change",
            "text": (
                "This is the 3000 h plan under the P7.1 acceptance rules. "
                "It is not tf_locoformer.dns.recipe, which keeps length_plan_no_sample_retry, "
                "and not the 256-mixture accept_smoke sample. "
                "Validation is the sorted-uttid tail and is not cv_synthetic_small. "
                "The 150 no_reverb test pairs are the existing recorded set and are not used for selection."
            ),
        }
    ]
    note = ACCEPT_NOTE + " This is the 3000 h accepted DNS plan. It is not the P7 length plan and not accept_smoke."
    dataset = {
        "name": ACCEPT_RECIPE_NAME,
        "version": 1,
        "protocol_id": accept["protocol_id"],
        "profile": loaded.dataset["profile"],
        "material_origin": loaded.dataset["material_origin"],
        "claim_label": loaded.dataset["claim_label"],
        "declared_status": "ready",
        "blocking_json": [],
        "selection_uses_test": bool(loaded.dataset["selection_uses_test"]),
        "selection_note": loaded.dataset.get("selection_note"),
        "deviations_json": loaded.dataset["deviations_json"],
        "bindings": [
            {"slot": slot, "case_set_id": bindings[slot].get("case_set_id"), "material_set_id": bindings[slot].get("material_set_id")}
            for slot in (
                "train.cases",
                "valid.cases",
                "test.cases",
                "train.speech",
                "train.noise",
            )
        ],
    }
    from src.experiments.data.ids import dataset_id_for

    dataset["dataset_id"] = dataset_id_for(
        protocol_id=dataset["protocol_id"],
        bindings=dataset["bindings"],
        profile=dataset["profile"],
        material_origin=dataset["material_origin"],
        claim_label=dataset["claim_label"],
        selection_uses_test=bool(dataset["selection_uses_test"]),
        declared_status=dataset["declared_status"],
    )
    if dataset["dataset_id"] != PARENT_DATASET_ID:
        raise ReproductionError("recomputed parent dataset_id does not match the snapshot")
    catalog, catalog_rows = _catalog(assets)
    for asset in assets:
        asset["catalog_id"] = catalog["catalog_id"]
    protocols = [parent_protocol, accept]
    if case_report["test_protocol_id"] == ondemand["protocol_id"]:
        protocols.append(ondemand)
    elif case_report["test_protocol_id"] != accept["protocol_id"]:
        raise ReproductionError("test cases use a protocol that is not the P7 on-demand protocol or the accept protocol")
    graph = Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: catalog_rows},
        assets=assets,
        materials=[speech, noise],
        protocols=protocols,
        cases=cases,
        case_sets=case_sets,
        datasets=[dataset],
        builds=[build_row(definition)],
        verified={
            definition["build_id"]: {
                "status": "PASS",
                "delivery": "on_demand",
                "wav_files": 0,
                "shard_ids": ["conditions"],
                "source": ORIGIN,
                "render_statistics": "absent_from_snapshot",
            }
        },
    )
    print("validate graph", flush=True)
    validate_graph(graph, resource_db=str(resource_db))
    print("compare snapshot hash", flush=True)
    parts, _checks = _parts(graph, dataset)
    logical = file_hashes(parts)
    digest_value = snapshot_hash(parts)
    mismatched = [name for name in PART_NAMES if logical[name] != loaded.manifest["files"][name]]
    if digest_value != loaded.snapshot_hash or mismatched:
        raise ReproductionError(
            "reconstructed snapshot hash does not match the parent: " + ", ".join(mismatched or ["snapshot_hash"])
        )
    report = {
        "origin": ORIGIN,
        "original_database_identity": "unverified",
        "snapshot_hash": digest_value,
        "dataset_id": dataset["dataset_id"],
        "protocol_id": accept["protocol_id"],
        "build_id": definition["build_id"],
        "catalog_id": catalog["catalog_id"],
        "catalog_identity_with_original": "unverified",
        "excluded_non_16k_noise": pools["excluded"],
        "counts": case_report["counts"],
        "sha256_sample": asset_report,
        "selection_note_matches_registrar": loaded.dataset.get("selection_note") == note,
        "deviations_match_registrar": loaded.dataset["deviations_json"] == deviations_match,
        "test_protocol_id": case_report["test_protocol_id"],
        "fields_not_in_snapshot": [
            "catalog_id",
            "asset.sample_format",
            "asset.extra_json",
            "asset.speaker_id",
            "asset.subset",
            "case_set.provenance_json",
            "build.audit render sample counts",
        ],
    }
    print(f"parent snapshot round-trip {digest_value}", flush=True)
    return graph, report


def _accept_pools(resource_db: Path, corpus: Path) -> dict[str, Any]:
    rows, _other = _catalog_rows(resource_db, corpus)
    clean = [row for row in rows if row["subset"] == "clean"]
    noise16 = [row for row in rows if row["subset"] == "noise"]
    non16 = []
    excluded = 0
    for row in rows:
        if row["subset"] != "noise_excluded":
            continue
        if int(row["channels"]) != 1 or int(row["sample_rate"]) == SAMPLE_RATE:
            excluded += 1
            continue
        kept = dict(row)
        kept["subset"] = "noise"
        non16.append(kept)
    test_clean = [row for row in rows if row["subset"] == "test_no_reverb_clean"]
    test_noisy = [row for row in rows if row["subset"] == "test_no_reverb_noisy"]
    return {
        "clean": {row["rel"]: row for row in clean},
        "noise": {row["rel"]: row for row in noise16 + non16},
        "test_clean": {row["rel"]: row for row in test_clean},
        "test_noisy": {row["rel"]: row for row in test_noisy},
        "excluded": excluded,
    }


def _assets_from_snapshot(loaded, pools: dict[str, Any], corpora: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {}
    for name in ("clean", "noise", "test_clean", "test_noisy"):
        expected.update(pools[name])
    snapshot_rels = {}
    for row in loaded.assets:
        if row["integrity"] != "sha256" or not row["sha256"]:
            raise ReproductionError(f"snapshot asset {row['asset_id']} is not sha256-addressed")
        if not str(row["asset_id"]).startswith("corpora/"):
            raise ReproductionError(f"snapshot asset {row['asset_id']} is not a corpora path")
        rel = row["asset_id"][len("corpora/") :]
        if rel != row["rel_path"]:
            raise ReproductionError(f"snapshot asset_id does not match rel_path: {row['asset_id']}")
        snapshot_rels[rel] = row
    if set(snapshot_rels) != set(expected):
        missing = len(set(expected) - set(snapshot_rels))
        extra = len(set(snapshot_rels) - set(expected))
        raise ReproductionError(
            f"snapshot assets do not match the current DNS catalog (missing {missing}, extra {extra})"
        )
    speech_ids = {
        row["asset_id"]
        for row in loaded.material_set_members
        if row["material_set_id"] == _binding_material(loaded, "train.speech")
    }
    built = []
    for rel, snap in snapshot_rels.items():
        catalog = expected[rel]
        if int(snap["sample_rate"]) != int(catalog["sample_rate"]) or int(snap["frames"]) != int(catalog["frames"]):
            raise ReproductionError(f"snapshot asset {rel} does not match resource.duckdb frames or rate")
        if int(snap["channels"]) != int(catalog["channels"]):
            raise ReproductionError(f"snapshot asset {rel} does not match resource.duckdb channels")
        path = corpora / rel
        stat = path.stat()
        speaker = reader_id(Path(rel).name) if f"corpora/{rel}" in speech_ids else None
        row = asset_row(
            path,
            corpus_root=corpora,
            subset=str(catalog["subset"]),
            rate=int(snap["sample_rate"]),
            frames=int(snap["frames"]),
            channels=int(snap["channels"]),
            sha256=str(snap["sha256"]),
            speaker_id=speaker,
        )
        if row["asset_id"] != snap["asset_id"] or int(row["size_bytes"]) != stat.st_size:
            raise ReproductionError(f"asset row for {rel} does not match the file")
        built.append(row)
    sample = _sample_hashes(built, corpora)
    return built, sample


def _binding_material(loaded, slot: str) -> str:
    for row in loaded.dataset["bindings"]:
        if row["slot"] == slot:
            return row["material_set_id"]
    raise ReproductionError(f"snapshot is missing slot {slot}")


def _sample_hashes(assets: list[dict[str, Any]], corpora: Path) -> dict[str, Any]:
    ordered = sorted(assets, key=lambda row: row["asset_id"])
    if len(ordered) < _SHA_SAMPLE:
        chosen = ordered
    else:
        step = max(1, len(ordered) // _SHA_SAMPLE)
        chosen = [ordered[index] for index in range(0, len(ordered), step)][:_SHA_SAMPLE]
    mismatches = []
    for asset in chosen:
        actual = sha256_file(corpora / asset["rel_path"])
        if actual != asset["sha256"]:
            mismatches.append(asset["rel_path"])
    if mismatches:
        raise ReproductionError("snapshot sha256 does not match the file: " + ", ".join(mismatches[:5]))
    return {
        "checked": len(chosen),
        "population": len(ordered),
        "mismatches": 0,
        "scope": "deterministic sample of snapshot sha256 values, not every file",
    }


def _material_from_snapshot(
    loaded,
    material_set_id: str,
    *,
    name: str,
    role: str,
    note: str,
    assets_by_id: dict[str, dict[str, Any]],
    expected_rels: set[str],
) -> dict[str, Any]:
    members = [row for row in loaded.material_set_members if row["material_set_id"] == material_set_id]
    members.sort(key=lambda row: int(row["position"]))
    if [int(row["position"]) for row in members] != list(range(len(members))):
        raise ReproductionError(f"material set {material_set_id} positions are not 0..n-1")
    if any(row.get("group_key") for row in members):
        raise ReproductionError(f"material set {material_set_id} has group keys the registrar does not store")
    ordered = [assets_by_id[row["asset_id"]] for row in members]
    rels = {row["rel_path"] for row in ordered}
    if rels != expected_rels:
        raise ReproductionError(f"material set {name} members do not match the catalog pool")
    built = material_set(ordered, name=name, role=role, note=note)
    if built["material_set_id"] != material_set_id:
        raise ReproductionError(f"material set {name} id does not match the snapshot")
    return built


def _cases_from_snapshot(
    loaded,
    *,
    accept_protocol_id: str,
    ondemand_protocol_id: str,
    train_set_id: str,
    valid_set_id: str,
    test_set_id: str,
    assets_by_id: dict[str, dict[str, Any]],
    build_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    docs = {}
    for row in loaded.docs:
        case_id = row["case_id"]
        if case_id in docs:
            raise ReproductionError(f"snapshot repeats case document {case_id}")
        docs[case_id] = row["doc_json"]
    snap_cases = {row["case_id"]: row for row in loaded.cases}
    if len(snap_cases) != len(loaded.cases):
        raise ReproductionError("snapshot cases repeat a case_id")
    items: dict[str, list[str]] = {train_set_id: [], valid_set_id: [], test_set_id: []}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in loaded.case_set_items:
        grouped.setdefault(row["case_set_id"], []).append(row)
    for case_set_id, rows in grouped.items():
        rows.sort(key=lambda item: int(item["position"]))
        if [int(item["position"]) for item in rows] != list(range(len(rows))):
            raise ReproductionError(f"case set {case_set_id} positions are not 0..n-1")
        if case_set_id not in items:
            raise ReproductionError(f"snapshot case set {case_set_id} is not a parent binding")
        items[case_set_id] = [row["case_id"] for row in rows]
    generated_docs = []
    for case_id in items[train_set_id] + items[valid_set_id]:
        document = docs.get(case_id)
        if document is None:
            raise ReproductionError(f"generated case {case_id} has no document")
        generated_docs.append(document)
    train_docs, valid_docs, stats = split_train_valid(generated_docs)
    train_cases = [condition_case(item, protocol_id=accept_protocol_id) for item in train_docs]
    valid_cases = [condition_case(item, protocol_id=accept_protocol_id) for item in valid_docs]
    if [row["case_id"] for row in train_cases] != items[train_set_id]:
        raise ReproductionError("sorted-uttid train order does not match the snapshot")
    if [row["case_id"] for row in valid_cases] != items[valid_set_id]:
        raise ReproductionError("sorted-uttid validation order does not match the snapshot")
    if len(train_cases) != 324000 or len(valid_cases) != 36000:
        raise ReproductionError(f"split counts are {len(train_cases)}/{len(valid_cases)}, expected 324000/36000")
    for case in train_cases + valid_cases:
        if case["build_id"] != build_id:
            raise ReproductionError("realization build_id does not match the registrar definition")
        _require_same_case(case, snap_cases[case["case_id"]])
        if case["doc_json"] != docs[case["case_id"]]:
            raise ReproductionError(f"case {case['case_id']} document does not match the snapshot")
    test_cases = []
    test_protocol_ids = set()
    for case_id in items[test_set_id]:
        snap = snap_cases[case_id]
        test_protocol_ids.add(snap["protocol_id"])
        if snap["case_role"] != "recorded_pair" or snap["doc_kind"] != "recipe_v1":
            raise ReproductionError(f"test case {case_id} is not a recorded recipe")
        clean = assets_by_id[snap["clean_asset_id"]]
        noisy = assets_by_id[snap["noisy_asset_id"]]
        built = recorded_pair(clean, noisy, snap["protocol_id"])
        if built["case_id"] != case_id:
            raise ReproductionError(f"test case {case_id} recipe does not match the snapshot assets")
        _require_same_case(built, snap)
        if built["doc_json"] != docs[case_id]:
            raise ReproductionError(f"test case {case_id} document does not match the snapshot")
        test_cases.append(built)
    if len(test_protocol_ids) != 1:
        raise ReproductionError("test cases do not share one protocol")
    test_protocol_id = next(iter(test_protocol_ids))
    if test_protocol_id not in (ondemand_protocol_id, accept_protocol_id):
        raise ReproductionError("test protocol id is not the P7 on-demand protocol or the accept protocol")
    if len(test_cases) != 150:
        raise ReproductionError(f"test split has {len(test_cases)} cases, expected 150")
    train_set = case_set_for(
        f"{ACCEPT_RECIPE_NAME}.train",
        "train",
        train_cases,
        {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.9, "part": "head"},
        f"dns_data_prep count*9/10 head of sorted uttids; {stats}",
    )
    valid_set = case_set_for(
        f"{ACCEPT_RECIPE_NAME}.valid",
        "valid",
        valid_cases,
        {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.9, "part": "tail"},
        f"{ACCEPT_NOTE} {stats}",
    )
    test_set = case_set_for(
        P7_TEST_SET_NAME,
        "test",
        test_cases,
        {"rule": "explicit"},
        "official synthetic no_reverb pairs by fileid. with_reverb and real recordings are not included. Not used for selection.",
    )
    if train_set["case_set_id"] != train_set_id or valid_set["case_set_id"] != valid_set_id:
        raise ReproductionError("train or validation case set id does not match the snapshot")
    if test_set["case_set_id"] != test_set_id:
        raise ReproductionError("test case set id does not match the snapshot")
    return train_cases + valid_cases + test_cases, [train_set, valid_set, test_set], {
        "counts": {"train": len(train_cases), "valid": len(valid_cases), "test": len(test_cases)},
        "test_protocol_id": test_protocol_id,
        "split_stats": stats,
    }


def _require_same_case(built: dict[str, Any], snap: dict[str, Any]) -> None:
    from src.experiments.data.db.snapshot import _CASE_COLUMNS

    for column in _CASE_COLUMNS:
        if built.get(column) != snap.get(column):
            raise ReproductionError(
                f"case {snap.get('case_id')} {column} is {built.get(column)!r}, snapshot has {snap.get(column)!r}"
            )


def _catalog(assets: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.experiments.data.prepare.dns2020 import _catalog as build_catalog

    return build_catalog(list(assets))


def published_selection_ids(list_path: str | Path) -> list[str]:
    """Case ids from the published list. The file's registration status is not trusted."""
    import json

    payload = json.loads(Path(list_path).read_text(encoding="utf-8"))
    case_ids = payload.get("case_ids")
    if not isinstance(case_ids, list) or not all(isinstance(item, str) for item in case_ids):
        raise ReproductionError("published research list has no case_id array")
    from src.experiments.data.prepare.research_valid_fixed import case_id_list_sha256

    if case_id_list_sha256(case_ids) != PUBLISHED_LIST_SHA256:
        raise ReproductionError("published research case-id list hash does not match the fixed digest")
    if payload.get("case_id_list_sha256") != PUBLISHED_LIST_SHA256:
        raise ReproductionError("published list records a different case_id_list_sha256")
    return list(case_ids)


def directory_sha256(directory: str | Path) -> str:
    """Hash of the fingerprint. Used only to detect writes inside an existing snapshot."""
    return hashlib.sha256(repr(snapshot_fingerprint(directory)).encode("utf-8")).hexdigest()


def commit_reconstructed_database(
    *,
    snapshot_dir: str | Path,
    database: str | Path,
    staging: str | Path,
    snapshots: str | Path,
    corpora: str | Path,
    resource_db: str | Path,
    generated: str | Path,
    derived: str | Path,
    list_path: str | Path,
) -> dict[str, Any]:
    """Write staging, the reconstructed database, the research dataset, and its snapshot.

    Refuses to replace an existing database. The parent snapshot directory is
    compared before and after and is not a write target.
    """
    from src.experiments.data.db.registry import initialize_database, register_staging
    from src.experiments.data.db.snapshot import export_snapshot
    from src.experiments.data.db.staging import write_graph
    from src.experiments.data.prepare.research_valid_fixed import (
        NEW_DATASET_NAME,
        NEW_DATASET_SELECTOR,
        PARENT_DATASET_NAME,
        select_research_validation,
    )
    from src.experiments.data.prepare.research_valid_fixed import register_research_validation_dataset

    parent_dir = Path(snapshot_dir)
    database = Path(database)
    if database.exists():
        raise ReproductionError(f"refusing to replace an existing database: {database}")
    if not parent_dir.is_dir():
        raise ReproductionError(f"parent snapshot directory is missing: {parent_dir}")
    before = snapshot_fingerprint(parent_dir)
    graph, report = reconstruct_parent_graph(parent_dir, corpora=corpora, resource_db=resource_db)
    published = published_selection_ids(list_path)
    splits = _split_ids(graph)
    selection = select_research_validation(splits["valid"], splits["train"], splits["test"])
    if selection["case_ids"] != published or selection["case_id_list_sha256"] != PUBLISHED_LIST_SHA256:
        raise ReproductionError("recomputed research validation does not match the published case-id list")
    if selection["count"] != 1024 or selection["excluded_case_ids"] or selection["reduced_because_leak"]:
        raise ReproductionError("research validation is not the fixed 1024-id subset")
    print("write parent staging", flush=True)
    write_graph(staging, graph)
    created = False
    try:
        initialize_database(database)
        created = True
        print("register parent graph", flush=True)
        parent_counts = register_staging(staging, database, resource_db=str(resource_db))
        _require_database_snapshot(database)
        _record_origin(
            database,
            {
                "database_origin": ORIGIN,
                "original_database_identity": "unverified",
                "reconstructed_parent_snapshot_hash": PARENT_SNAPSHOT_HASH,
                "reconstructed_parent_dataset_id": PARENT_DATASET_ID,
            },
        )
    except Exception:
        if created and database.exists():
            database.unlink()
        raise
    print("register research dataset", flush=True)
    research = register_research_validation_dataset(
        database,
        parent_name=PARENT_DATASET_NAME,
        child_name=NEW_DATASET_NAME,
        selected_case_ids=published,
        staging=staging,
    )
    if research["claim_label"] != "research_checkpoint_selection":
        raise ReproductionError("research dataset claim was not stored")
    print("export research snapshot", flush=True)
    provenance = {
        "database_origin": ORIGIN,
        "original_database_identity": "unverified",
        "parent_snapshot_hash": PARENT_SNAPSHOT_HASH,
        "parent_dataset_id": PARENT_DATASET_ID,
    }
    first = export_snapshot(
        database,
        NEW_DATASET_SELECTOR,
        snapshots,
        corpora=corpora,
        generated=generated,
        derived=derived,
        provenance_extra=provenance,
    )
    second = export_snapshot(
        database,
        NEW_DATASET_SELECTOR,
        snapshots,
        corpora=corpora,
        generated=generated,
        derived=derived,
        provenance_extra=provenance,
    )
    if first != second:
        raise ReproductionError("research snapshot hash changed on the second export")
    after = snapshot_fingerprint(parent_dir)
    if after != before:
        raise ReproductionError("parent snapshot files changed while the research snapshot was exported")
    child_dir = Path(snapshots) / research["dataset_id"]
    if child_dir.resolve() == parent_dir.resolve():
        raise ReproductionError("research snapshot was written onto the parent directory")
    report = {
        **report,
        "parent_register_counts": parent_counts,
        "research": research,
        "research_snapshot_hash": first,
        "research_snapshot_directory": str(child_dir),
        "published_list_sha256": PUBLISHED_LIST_SHA256,
        "parent_snapshot_unchanged": True,
    }
    _write_result(Path(staging) / "research_valid_fixed_v1" / "registration_result.json", report)
    print(f"research dataset {research['dataset_id']} snapshot {first}", flush=True)
    return report


def _require_database_snapshot(database: Path) -> None:
    """The stored parent must export to the verified snapshot hash. Do not write that directory."""
    from src.experiments.data.db.registry import read_graph
    from src.experiments.data.db.snapshot import resolve_dataset

    stored = read_graph(database)
    dataset = resolve_dataset(stored.datasets, "tf_locoformer.dns.accept_recipe@1")
    parts, _checks = _parts(stored, dataset)
    digest_value = snapshot_hash(parts)
    if digest_value != PARENT_SNAPSHOT_HASH:
        logical = file_hashes(parts)
        raise ReproductionError(
            "database round-trip does not match the parent snapshot hash "
            f"({digest_value}); parts={logical}"
        )


def _split_ids(graph: Graph) -> dict[str, list[str]]:
    dataset = next(row for row in graph.datasets if row["name"] == "tf_locoformer.dns.accept_recipe")
    sets = {row["case_set_id"]: row for row in graph.case_sets}
    found = {}
    for binding in dataset["bindings"]:
        if not binding.get("case_set_id"):
            continue
        split = binding["slot"].split(".", 1)[0]
        found[split] = [item["case_id"] for item in sets[binding["case_set_id"]]["items"]]
    return found


def _record_origin(database: Path, fields: dict[str, str]) -> None:
    import duckdb

    connection = duckdb.connect(str(database))
    try:
        connection.execute("BEGIN TRANSACTION")
        for key, value in fields.items():
            connection.execute("DELETE FROM repro.schema_info WHERE key = ?", [key])
            connection.execute("INSERT INTO repro.schema_info (key, value) VALUES (?, ?)", [key, value])
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _write_result(path: Path, report: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    import sys

    from src.experiments.data.paths import (
        DEFAULT_CORPORA,
        DEFAULT_DB,
        DEFAULT_DERIVED,
        DEFAULT_GENERATED,
        DEFAULT_RESOURCE_DB,
        DEFAULT_SNAPSHOTS,
        DEFAULT_STAGING,
        ROOT,
    )

    parser = argparse.ArgumentParser(description="Reconstruct the DNS accept-recipe parent from its snapshot")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOTS / PARENT_DATASET_ID))
    parser.add_argument("--database", default=str(DEFAULT_DB))
    parser.add_argument("--staging", default=str(DEFAULT_STAGING))
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--corpora", default=str(DEFAULT_CORPORA))
    parser.add_argument("--resource-db", default=str(DEFAULT_RESOURCE_DB))
    parser.add_argument("--generated", default=str(DEFAULT_GENERATED))
    parser.add_argument("--derived", default=str(DEFAULT_DERIVED))
    parser.add_argument(
        "--list",
        default=str(ROOT / "configs" / "data" / "lists" / "tf_locoformer" / "research_valid_fixed_v1.json"),
    )
    args = parser.parse_args()
    try:
        if not args.commit:
            graph, report = reconstruct_parent_graph(
                args.snapshot, corpora=args.corpora, resource_db=args.resource_db
            )
            published = published_selection_ids(args.list)
            selection = select_preview(graph, published)
            report["selection"] = selection
            print(report["snapshot_hash"], flush=True)
            print(report["counts"], flush=True)
            return
        commit_reconstructed_database(
            snapshot_dir=args.snapshot,
            database=args.database,
            staging=args.staging,
            snapshots=args.snapshots,
            corpora=args.corpora,
            resource_db=args.resource_db,
            generated=args.generated,
            derived=args.derived,
            list_path=args.list,
        )
    except ReproductionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def select_preview(graph: Graph, published: list[str]) -> dict[str, Any]:
    from src.experiments.data.prepare.research_valid_fixed import select_research_validation

    splits = _split_ids(graph)
    selection = select_research_validation(splits["valid"], splits["train"], splits["test"])
    if selection["case_ids"] != published or selection["case_id_list_sha256"] != PUBLISHED_LIST_SHA256:
        raise ReproductionError("recomputed research validation does not match the published case-id list")
    return {
        "count": selection["count"],
        "list_sha256": selection["case_id_list_sha256"],
        "excluded": len(selection["excluded_case_ids"]),
    }


if __name__ == "__main__":
    main()
