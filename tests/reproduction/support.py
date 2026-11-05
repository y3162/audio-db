"""Synthetic reproduction graphs for P1 tests.

These rows are not VoiceBank, MS-SNSD, or any other real corpus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.data.recipe import canonical_recipe
from src.reproduction.db.graph import Graph
from src.reproduction.ids import (
    build_id_for,
    case_content_hash,
    case_set_id_for,
    catalog_id_for,
    dataset_id_for,
    digest,
    generated_rel_path,
    material_members_key,
    material_set_id_for,
    protocol_id_for,
    recipe_case_id,
    weakest_integrity,
)

RULE = {"sample_rate": 16000, "teacher": "dry_clean", "normalize": "noisy_rms_unit_v1"}
PRE_RULE = {"sample_rate": 16000, "teacher": "dry_clean"}
OFFICIAL_REF = {"repo": "synthetic", "path": "train.py", "commit": "synthetic"}
PROVENANCE = {"source": "synthetic", "path": "list.txt", "sha256": "ab" * 32}


def protocol_row(*, algorithm: str = "fixed_pair.mp_senet", rule: dict[str, Any] | None = None, name: str = "mp.synthetic") -> dict[str, Any]:
    spec_rule = RULE if rule is None else rule
    kind = "fixed_pair" if algorithm.startswith("fixed_pair.") else "pregenerated"
    protocol_id = protocol_id_for(
        algorithm=algorithm,
        algorithm_version=1,
        kind=kind,
        rule=spec_rule,
        deviation_class="official_effective",
        parent=None,
    )
    return {
        "protocol_id": protocol_id,
        "name": name,
        "version": 1,
        "kind": kind,
        "algorithm": algorithm,
        "algorithm_version": 1,
        "rule_json": spec_rule,
        "official_ref_json": OFFICIAL_REF,
        "deviation_class": "official_effective",
        "parent_protocol_id": None,
        "deviations_json": [],
    }


def write_asset(
    corpora: Path,
    rel: str,
    payload: bytes,
    *,
    speaker: str,
    integrity: str = "sha256",
) -> dict[str, Any]:
    path = corpora / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    return {
        "asset_id": f"corpora/{rel}",
        "storage_root": "corpora",
        "rel_path": rel,
        "corpus": "synthetic",
        "subset": "unit",
        "speaker_id": speaker,
        "utterance_key": Path(rel).stem,
        "sample_rate": 16000,
        "frames": 1600,
        "channels": 1,
        "sample_format": "PCM_16",
        "size_bytes": len(payload),
        "sha256": None if integrity == "size" else sha,
        "integrity": integrity,
        "catalog_id": None,
        "derivation_id": None,
        "derived_from": None,
        "build_id": None,
        "extra_json": {},
    }


def attach_catalog(assets: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        {
            "path": asset["rel_path"],
            "corpus": asset["corpus"],
            "subset": asset["subset"],
            "sample_rate": asset["sample_rate"],
            "frames": asset["frames"],
            "channels": asset["channels"],
            "speaker_id": asset["speaker_id"],
            "utterance_id": asset["utterance_key"],
        }
        for asset in sorted(assets, key=lambda item: item["rel_path"])
    ]
    catalog_id = catalog_id_for(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["corpus"]] = counts.get(row["corpus"], 0) + 1
    for asset in assets:
        asset["catalog_id"] = catalog_id
    catalog = {
        "catalog_id": catalog_id,
        "source_kind": "staging_parquet",
        "corpora_json": counts,
        "row_count": len(rows),
        "source_db_sha256": None,
    }
    return catalog, rows


def recorded_case(clean: dict[str, Any], noisy: dict[str, Any], protocol_id: str) -> dict[str, Any]:
    document = canonical_recipe(
        {
            "recipe_version": 1,
            "sources": {
                "clean": {"path": "../" + clean["asset_id"], "sha256": clean["sha256"]},
                "noisy": {"path": "../" + noisy["asset_id"], "sha256": noisy["sha256"]},
            },
            "operations": [],
            "output": {"sample_rate": 16000, "reference": "dry_clean"},
        }
    )
    row = {
        "case_id": recipe_case_id(document),
        "doc_kind": "recipe_v1",
        "doc_json": document,
        "case_role": "recorded_pair",
        "protocol_id": protocol_id,
        "clean_asset_id": clean["asset_id"],
        "noisy_asset_id": noisy["asset_id"],
        "noise_asset_id": None,
        "teacher_kind": "dry_clean",
        "sample_rate": 16000,
        "frames": 1600,
        "speaker_id": clean["speaker_id"],
        "utterance_key": clean["utterance_key"],
        "transcript": None,
        "build_id": None,
        "gen_index": None,
        "content_hash": "",
    }
    row["content_hash"] = case_content_hash(row)
    return row


def case_set(name: str, split: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = [case["case_id"] for case in cases]
    rule = {"rule": "explicit"}
    return {
        "case_set_id": case_set_id_for(name=name, version=1, split_role=split, rule=rule, case_ids=case_ids),
        "name": name,
        "version": 1,
        "split_role": split,
        "origin": "official",
        "selection_rule_json": rule,
        "provenance_json": PROVENANCE,
        "member_count": len(case_ids),
        "list_sha256": digest(case_ids),
        "items": [{"position": index, "case_id": case_id} for index, case_id in enumerate(case_ids)],
    }


def dataset_row(
    name: str,
    protocol_id: str,
    bindings: list[dict[str, Any]],
    *,
    profile: str,
    claim: str,
    origin: str,
    selection: bool,
    status: str = "ready",
    blocking: list[Any] | None = None,
) -> dict[str, Any]:
    blocking_json = [] if blocking is None else blocking
    return {
        "dataset_id": dataset_id_for(
            protocol_id=protocol_id,
            bindings=bindings,
            profile=profile,
            material_origin=origin,
            claim_label=claim,
            selection_uses_test=selection,
            declared_status=status,
        ),
        "name": name,
        "version": 1,
        "protocol_id": protocol_id,
        "profile": profile,
        "material_origin": origin,
        "claim_label": claim,
        "declared_status": status,
        "blocking_json": blocking_json,
        "selection_uses_test": selection,
        "selection_note": None,
        "deviations_json": [],
        "bindings": bindings,
    }


def fixed_pair_graph(corpora: Path, *, safe: bool) -> Graph:
    """Official-faithful shares valid and test. Leakage-safe keeps three speakers apart."""
    protocol = protocol_row()
    prefix = "safe" if safe else "official"
    if safe:
        pairs = [("train", "spk_train"), ("valid", "spk_valid"), ("test", "spk_test")]
    else:
        pairs = [("train", "spk_train"), ("held", "spk_held")]
    assets = []
    by_role: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for role, speaker in pairs:
        clean = write_asset(corpora, f"{prefix}/{role}-clean.wav", f"{prefix}-{role}-clean".encode(), speaker=speaker)
        noisy = write_asset(corpora, f"{prefix}/{role}-noisy.wav", f"{prefix}-{role}-noisy".encode(), speaker=speaker)
        assets.extend((clean, noisy))
        by_role[role] = (clean, noisy)
    orphan = None
    if not safe:
        orphan = write_asset(corpora, f"{prefix}/orphan.wav", b"orphan-size-only", speaker="spk_orphan", integrity="size")
        assets.append(orphan)
    catalog, rows = attach_catalog(assets)
    cases = {
        role: recorded_case(clean, noisy, protocol["protocol_id"])
        for role, (clean, noisy) in by_role.items()
    }
    if safe:
        sets = [
            case_set("safe.train", "train", [cases["train"]]),
            case_set("safe.valid", "valid", [cases["valid"]]),
            case_set("safe.test", "test", [cases["test"]]),
        ]
        bindings = [_binding(item) for item in sets]
        dataset = dataset_row(
            "synthetic.safe",
            protocol["protocol_id"],
            bindings,
            profile="leakage_safe",
            claim="leakage_safe_comparison",
            origin="official",
            selection=False,
        )
        extra_cases = [cases["train"], cases["valid"], cases["test"]]
    else:
        sets = [
            case_set("official.train", "train", [cases["train"]]),
            case_set("official.valid", "valid", [cases["held"]]),
            case_set("official.test", "test", [cases["held"]]),
        ]
        dataset = dataset_row(
            "synthetic.official",
            protocol["protocol_id"],
            [_binding(item) for item in sets],
            profile="official_faithful",
            claim="official_reproduction",
            origin="official",
            selection=True,
        )
        blocked = dataset_row(
            "synthetic.blocked",
            protocol["protocol_id"],
            [],
            profile="leakage_safe",
            claim="leakage_safe_comparison",
            origin="unknown",
            selection=False,
            status="blocked",
            blocking=[{"id": "synthetic", "text": "not used by P1"}],
        )
        extra_cases = [cases["train"], cases["held"]]
        sets = sets
        dataset = dataset
    graph = Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: rows},
        assets=assets,
        protocols=[protocol],
        cases=extra_cases,
        case_sets=sets,
        datasets=[dataset] if safe else [dataset, blocked],
    )
    graph.orphan_id = None if orphan is None else orphan["asset_id"]  # type: ignore[attr-defined]
    return graph


def _binding(case_set_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot": f"{case_set_row['split_role']}.cases",
        "case_set_id": case_set_row["case_set_id"],
        "material_set_id": None,
    }


def generated_graph(root: Path) -> Graph:
    """One verified synthetic shard. This is not an ESPnet or MS-SNSD build."""
    corpora = root / "corpora"
    generated = root / "generated"
    protocol = protocol_row(algorithm="pregenerated.ms_snsd_tfgridnet", rule=PRE_RULE, name="tfgridnet.synthetic")
    speech = write_asset(corpora, "pool/speech.wav", b"pool-speech", speaker="spk_pool")
    catalog, rows = attach_catalog([speech])
    members = [material_members_key(speech["asset_id"], speech["sha256"], speech["size_bytes"])]
    rule = {"rule": "explicit"}
    material = {
        "material_set_id": material_set_id_for(
            name="pool.speech",
            version=1,
            role="speech",
            split_role="train",
            rule=rule,
            members=members,
        ),
        "name": "pool.speech",
        "version": 1,
        "role": "speech",
        "split_role": "train",
        "origin": "official",
        "selection_rule_json": rule,
        "provenance_json": PROVENANCE,
        "member_count": 1,
        "total_seconds": 1600 / 16000,
        "list_sha256": digest(members),
        "integrity_min": weakest_integrity(["sha256"]),
        "members": [{"position": 0, "asset_id": speech["asset_id"], "group_key": None, "attrs_json": None}],
    }
    build_id = build_id_for(
        generator="ms_snsd",
        generator_version=1,
        params={"note": "synthetic"},
        seed=0,
        inputs={"train.speech": material["material_set_id"]},
    )
    build = {
        "build_id": build_id,
        "generator": "ms_snsd",
        "generator_version": 1,
        "params_json": {"note": "synthetic"},
        "seed": 0,
        "inputs_json": {"train.speech": material["material_set_id"]},
        "shard_size": 10,
    }
    assets = [speech]
    cases = []
    for index, split in enumerate(("train", "valid", "test")):
        pair = []
        for stream in ("clean", "noisy"):
            asset = _generated_asset(generated, build, stream=stream, gen_index=index, payload=f"{split}-{stream}".encode())
            assets.append(asset)
            pair.append(asset)
        cases.append(_generated_case(pair[0], pair[1], protocol["protocol_id"], build_id, index))
    sets = [case_set(f"generated.{split}", split, [case]) for split, case in zip(("train", "valid", "test"), cases)]
    dataset = dataset_row(
        "synthetic.generated",
        protocol["protocol_id"],
        [_binding(item) for item in sets],
        profile="official_faithful",
        claim="espnet_public_se_recipe_reproduction",
        origin="unknown",
        selection=False,
    )
    return Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: rows},
        assets=assets,
        materials=[material],
        protocols=[protocol],
        cases=cases,
        case_sets=sets,
        datasets=[dataset],
        builds=[build],
        verified={build_id: {"shard_ids": ["s0000"]}},
    )


def _generated_asset(root: Path, build: dict[str, Any], *, stream: str, gen_index: int, payload: bytes) -> dict[str, Any]:
    rel = generated_rel_path(
        generator=build["generator"],
        build_id=build["build_id"],
        stream=stream,
        gen_index=gen_index,
        shard_size=build["shard_size"],
    )
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "asset_id": f"generated/{rel}",
        "storage_root": "generated",
        "rel_path": rel,
        "corpus": None,
        "subset": None,
        "speaker_id": f"spk_{gen_index}",
        "utterance_key": f"{stream}-{gen_index}",
        "sample_rate": 16000,
        "frames": 1600,
        "channels": 1,
        "sample_format": "PCM_16",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "integrity": "sha256",
        "catalog_id": None,
        "derivation_id": None,
        "derived_from": None,
        "build_id": build["build_id"],
        "extra_json": {"stream": stream, "gen_index": gen_index, "shard_id": "s0000"},
    }


def _generated_case(clean: dict[str, Any], noisy: dict[str, Any], protocol_id: str, build_id: str, gen_index: int) -> dict[str, Any]:
    document = {
        "realization_version": 1,
        "algorithm": "pregenerated.ms_snsd_tfgridnet",
        "algorithm_version": 1,
        "sample_rate": 16000,
        "sources": {"clean": {"asset": clean["asset_id"]}, "noisy": {"asset": noisy["asset_id"]}},
        "draws": {"index": gen_index},
        "output": {"teacher": "dry_clean", "length": 1600},
    }
    row = {
        "case_id": digest(document),
        "doc_kind": "realization_v1",
        "doc_json": document,
        "case_role": "generated",
        "protocol_id": protocol_id,
        "clean_asset_id": clean["asset_id"],
        "noisy_asset_id": noisy["asset_id"],
        "noise_asset_id": None,
        "teacher_kind": "dry_clean",
        "sample_rate": 16000,
        "frames": 1600,
        "speaker_id": clean["speaker_id"],
        "utterance_key": clean["utterance_key"],
        "transcript": None,
        "build_id": build_id,
        "gen_index": gen_index,
        "content_hash": "",
    }
    row["content_hash"] = case_content_hash(row)
    return row
