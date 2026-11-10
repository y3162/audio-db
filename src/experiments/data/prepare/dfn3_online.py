"""Register a small DeepFilterNet3 online-mixing substitute dataset.

Official DNS4, PTDB, the shipped ``dataset.cfg``, and the training RIR list are
not in this tree. The ready datasets are
``protocol_conformant_substitute_materials``. The official-material dataset
stays ``blocked``. This module does not write mixture wavs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import soundfile as sf

from src.experiments.data.db.graph import Graph, logical_bindings
from src.experiments.data.db.registry import initialize_database, register_staging
from src.experiments.data.db.snapshot import export_snapshot
from src.experiments.data.db.staging import write_graph
from src.experiments.data.db.validate import compute_selection_uses_test, validate_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import (
    DFN3_ONLINE_V1,
    case_content_hash,
    case_set_id_for,
    catalog_id_for,
    dataset_id_for,
    digest,
    material_members_key,
    material_set_id_for,
    protocol_id_for,
    weakest_integrity,
)
from src.experiments.data.paths import DEFAULT_CORPORA, ROOT

OFFICIAL_COMMIT = DFN3_ONLINE_V1["official_commit"]
STAGING = ROOT / "data/staging/reproduction-p8"
DATABASE = ROOT / "data/db/reproduction-p8.duckdb"
SUBSTITUTE_NAME = "dfn3.online.substitute"
SMOKE_NAME = "dfn3.online.smoke"
BLOCKED_NAME = "dfn3.online.official"

_LIMITS = {
    "substitute": {
        "speech": {"train": 200, "valid": 40, "test": 40},
        "noise": {"train": 40, "valid": 16, "test": 16},
        "rir": {"train": 40, "valid": 10, "test": 10},
    },
    "smoke": {
        "speech": {"train": 8, "valid": 4, "test": 4},
        "noise": {"train": 8, "valid": 4, "test": 4},
        "rir": {"train": 4, "valid": 4, "test": 4},
    },
}


def shipped_rule() -> dict[str, Any]:
    return {
        "sample_rate": 48000,
        "teacher": "official_mixed_target",
        "window": 144000,
        "order": "dfn3_online_ceil_v1",
        "epoch_unit": "speech_keys_times_sampling_factor",
        "global_batch": 64,
        "delivery": "on_demand",
        "validation": {
            "crop": "online_fixed_seed",
            "batch": 64,
            "interval": "epoch",
            "note": "valid and test ignore epoch in the sample seed",
        },
        "online": dict(DFN3_ONLINE_V1),
    }


def register_p8(
    *,
    corpora: Path = DEFAULT_CORPORA,
    staging: Path = STAGING,
    database: Path = DATABASE,
) -> dict[str, Any]:
    """Stage, register, and export the substitute datasets. Existing rows stay."""
    graph = build_graph(corpora)
    validate_graph(graph, resource_db=None, require_verified=set())
    write_graph(staging, graph)
    initialize_database(database)
    counts = register_staging(staging, database, resource_db=None)
    snapshots = {}
    for dataset in graph.datasets:
        if dataset["declared_status"] != "ready":
            continue
        snapshots[dataset["name"]] = export_snapshot(database, f"{dataset['name']}@1")
    return {"counts": counts, "snapshots": snapshots, "database": str(database)}


def build_graph(corpora: Path = DEFAULT_CORPORA) -> Graph:
    pools = _collect(Path(corpora))
    assets = [_asset(item) for item in _unique(pools["substitute"])]
    catalog, catalog_rows = _catalog(assets)
    for asset in assets:
        asset["catalog_id"] = catalog["catalog_id"]
    by_path = {asset["rel_path"]: asset for asset in assets}
    parent, child = _protocols()
    materials = []
    cases = []
    case_sets = []
    datasets = []
    for scale in ("substitute", "smoke"):
        scale_materials = []
        scale_cases = []
        scale_sets = []
        for split in ("train", "valid", "test"):
            for role in ("speech", "noise", "rir"):
                chosen = pools[scale][split][role]
                rows = [by_path[_rel(item)] for item in chosen]
                scale_materials.append(_material(rows, scale=scale, split=split, role=role))
            if split != "train":
                built_cases, case_set = _online_cases(
                    parent_id=child["protocol_id"],
                    scale=scale,
                    split=split,
                    count=len(pools[scale][split]["speech"]),
                )
                scale_cases.extend(built_cases)
                scale_sets.append(case_set)
                cases.extend(built_cases)
                case_sets.append(case_set)
        materials.extend(scale_materials)
        datasets.append(
            _dataset(
                name=SUBSTITUTE_NAME if scale == "substitute" else SMOKE_NAME,
                protocol=child,
                materials=scale_materials,
                case_sets=scale_sets,
                cases=scale_cases,
                claim="protocol_conformant_substitute_materials",
                origin="substitute",
                status="ready",
                blocking=[],
                note=(
                    "LibriSpeech speech, DEMAND noise, and OpenSLR28 simulated RIRs. "
                    "Not the DNS4 / PTDB / shipped dataset.cfg collection."
                ),
            )
        )
    datasets.append(_blocked(parent))
    graph = Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: catalog_rows},
        assets=assets,
        materials=materials,
        protocols=[parent, child],
        cases=cases,
        case_sets=case_sets,
        datasets=datasets,
    )
    return graph


def _protocols() -> tuple[dict[str, Any], dict[str, Any]]:
    rule = shipped_rule()
    parent_id = protocol_id_for(
        algorithm="online.dfn3",
        algorithm_version=1,
        kind="online_mixing",
        rule=rule,
        deviation_class="official_effective",
        parent=None,
    )
    parent = {
        "protocol_id": parent_id,
        "name": "dfn3.online",
        "version": 1,
        "kind": "online_mixing",
        "algorithm": "online.dfn3",
        "algorithm_version": 1,
        "rule_json": rule,
        "official_ref_json": {
            "repo": "Rikorose/DeepFilterNet",
            "commit": OFFICIAL_COMMIT,
            "path": "libDF/src/dataset.rs",
            "section": "TdDataset::get_sample",
            "document": "models/DeepFilterNet3 config.ini",
        },
        "deviation_class": "official_effective",
        "parent_protocol_id": None,
        "deviations_json": [],
    }
    deviations = [
        {
            "id": "numpy_pcg64",
            "class": "algorithm_change",
            "text": (
                "Sample draws use NumPy PCG64 seeded by SHA-256 of the protocol, "
                "snapshot, training seed, split, epoch, and sample index. This is "
                "not Rust StdRng seeded by seed_from_u64(config_seed + epoch_seed + index)."
            ),
            "ref": "libDF/src/dataloader.rs",
        },
        {
            "id": "scipy_resample",
            "class": "algorithm_change",
            "text": (
                "Speed and bandwidth resampling use scipy.signal.resample. "
                "The official resampler is rubato FftFixedInOut. The gate and the "
                "500 Hz rate grid match; the filter does not."
            ),
            "ref": "libDF/src/transforms.rs",
        },
        {
            "id": "epoch_order",
            "class": "algorithm_change",
            "text": (
                "Epoch order is mix_seed('dfn3-order', ...). The official main RNG "
                "is seeded once with 42 and is not restarted each epoch, so resume "
                "changes the official order. audio-db order does not depend on that history."
            ),
            "ref": "libDF/src/dataloader.rs",
        },
        {
            "id": "rir_cap_and_channel0",
            "class": "algorithm_change",
            "text": (
                "RIRs longer than 1,440,000 samples are cropped. Every file contributes "
                "channel 0. Official convolution keeps the HDF5 channel layout and can "
                "fail when it does not match the mono speech buffer."
            ),
            "ref": "libDF/src/augmentations.rs",
        },
        {
            "id": "replacement_cap",
            "class": "algorithm_change",
            "text": "Silent or unreadable speech and noise are replaced at most 64 times, then the sample errors.",
            "ref": "libDF/src/dataset.rs",
        },
    ]
    child_id = protocol_id_for(
        algorithm="online.dfn3",
        algorithm_version=1,
        kind="online_mixing",
        rule=rule,
        deviation_class="algorithm_change",
        parent=parent_id,
    )
    child = {
        "protocol_id": child_id,
        "name": "dfn3.online.audio-db",
        "version": 1,
        "kind": "online_mixing",
        "algorithm": "online.dfn3",
        "algorithm_version": 1,
        "rule_json": rule,
        "official_ref_json": parent["official_ref_json"],
        "deviation_class": "algorithm_change",
        "parent_protocol_id": parent_id,
        "deviations_json": deviations,
    }
    return parent, child


def _blocked(protocol: dict[str, Any]) -> dict[str, Any]:
    blocking = [
        {
            "item": "DNS Challenge 4 speech",
            "reason": "DNS4 is not linked under data/corpora. DNS2020 and DNS5 are different releases.",
        },
        {
            "item": "PTDB",
            "reason": "PTDB is not in the corpus catalog.",
        },
        {
            "item": "official dataset.cfg",
            "reason": "The shipped checkpoint does not include the training file list or sampling factors.",
        },
        {
            "item": "official RIR set",
            "reason": "The RIR collection used to train the shipped model is not identified.",
        },
        {
            "item": "VCTK factor",
            "reason": "VCTK is present, but the paper factor of 10 and the file list are not a dataset.cfg.",
        },
    ]
    return _dataset(
        name=BLOCKED_NAME,
        protocol=protocol,
        materials=[],
        case_sets=[],
        cases=[],
        claim="official_reproduction",
        origin="official",
        status="blocked",
        blocking=blocking,
        note="Official DeepFilterNet3 training materials are not available. No snapshot is exported.",
    )


def _dataset(
    *,
    name: str,
    protocol: dict[str, Any],
    materials: list[dict[str, Any]],
    case_sets: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    claim: str,
    origin: str,
    status: str,
    blocking: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    bindings = []
    for material in materials:
        bindings.append(
            {
                "slot": f"{material['split_role']}.{material['role']}",
                "case_set_id": None,
                "material_set_id": material["material_set_id"],
            }
        )
    for case_set in case_sets:
        bindings.append(
            {
                "slot": f"{case_set['split_role']}.cases",
                "case_set_id": case_set["case_set_id"],
                "material_set_id": None,
            }
        )
    encoded = logical_bindings(bindings)
    material_index = {row["material_set_id"]: row for row in materials}
    case_index = {row["case_set_id"]: row for row in case_sets}
    case_rows = {row["case_id"]: row for row in cases}
    provisional = {"name": name, "declared_status": status, "bindings": encoded}
    selection = compute_selection_uses_test(provisional, case_index, material_index, case_rows)
    dataset_id = dataset_id_for(
        protocol_id=protocol["protocol_id"],
        bindings=encoded,
        profile="official_faithful",
        material_origin=origin,
        claim_label=claim,
        selection_uses_test=selection,
        declared_status=status,
    )
    return {
        "dataset_id": dataset_id,
        "name": name,
        "version": 1,
        "protocol_id": protocol["protocol_id"],
        "profile": "official_faithful",
        "material_origin": origin,
        "claim_label": claim,
        "declared_status": status,
        "blocking_json": blocking,
        "selection_uses_test": selection,
        "selection_note": note,
        "deviations_json": [],
        "bindings": encoded,
    }


def _seconds(assets: list[dict[str, Any]]) -> float:
    total = 0.0
    for asset in assets:
        total += asset["frames"] / asset["sample_rate"]
    return total


def _material(assets: list[dict[str, Any]], *, scale: str, split: str, role: str) -> dict[str, Any]:
    keys = [material_members_key(asset["asset_id"], asset["sha256"], asset["size_bytes"]) for asset in assets]
    rule = {"rule": "explicit"}
    name = f"dfn3.{scale}.{split}.{role}"
    body = {
        "name": name,
        "version": 1,
        "role": role,
        "split_role": split,
        "origin": "substitute",
        "selection_rule_json": rule,
        "provenance_json": {
            "source": "local corpora substitute for DeepFilterNet3",
            "note": f"{scale} {split} {role}",
            "sha256": digest([asset["sha256"] for asset in assets]),
        },
        "member_count": len(assets),
        "total_seconds": _seconds(assets),
        "list_sha256": digest(keys),
        "integrity_min": weakest_integrity([asset["integrity"] for asset in assets]),
        "members": [
            {
                "position": index,
                "asset_id": asset["asset_id"],
                "group_key": asset.get("speaker_id"),
                "attrs_json": None,
            }
            for index, asset in enumerate(assets)
        ],
    }
    body["material_set_id"] = material_set_id_for(
        name=name,
        version=1,
        role=role,
        split_role=split,
        rule=rule,
        members=keys,
    )
    return body


def _online_cases(*, parent_id: str, scale: str, split: str, count: int):
    rows = []
    for index in range(count):
        document = {
            "realization_version": 1,
            "algorithm": "online.dfn3",
            "algorithm_version": 1,
            "sample_rate": 48000,
            "sources": {"split": split, "scale": scale},
            "draws": {"sample_index": index, "seed_rule": "split_sample_index"},
            "steps": ["dfn3_online_v1"],
            "output": {"length": 144000, "teacher": "official_mixed_target"},
        }
        case_id = digest(document)
        row = {
            "case_id": case_id,
            "doc_kind": "realization_v1",
            "doc_json": document,
            "case_role": "online_snapshot",
            "protocol_id": parent_id,
            "clean_asset_id": None,
            "noisy_asset_id": None,
            "noise_asset_id": None,
            "teacher_kind": "official_mixed_target",
            "sample_rate": 48000,
            "frames": 144000,
            "speaker_id": None,
            "utterance_key": f"{scale}-{split}-{index:05d}",
            "transcript": None,
            "build_id": None,
            "gen_index": None,
            "content_hash": "",
        }
        row["content_hash"] = case_content_hash(row)
        rows.append(row)
    rule = {"rule": "explicit"}
    name = f"dfn3.{scale}.{split}.online"
    case_ids = [row["case_id"] for row in rows]
    case_set = {
        "case_set_id": case_set_id_for(
            name=name, version=1, split_role=split, rule=rule, case_ids=case_ids
        ),
        "name": name,
        "version": 1,
        "split_role": split,
        "origin": "substitute",
        "selection_rule_json": rule,
        "provenance_json": {
            "source": "logical online sample index",
            "note": "The waveform is recomputed from the sample index. It is not stored.",
        },
        "member_count": len(rows),
        "list_sha256": digest(case_ids),
        "items": [
            {"position": index, "case_id": row["case_id"]}
            for index, row in enumerate(rows)
        ],
    }
    return rows, case_set


def _asset(item: dict[str, Any]) -> dict[str, Any]:
    path = item["path"]
    info = sf.info(path)
    payload = path.read_bytes()
    rel = path.relative_to(item["corpora"]).as_posix()
    return {
        "asset_id": f"corpora/{rel}",
        "storage_root": "corpora",
        "rel_path": rel,
        "corpus": item["corpus"],
        "subset": item["subset"],
        "speaker_id": item["speaker_id"],
        "utterance_key": path.stem,
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "sample_format": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "integrity": "sha256",
        "catalog_id": None,
        "derivation_id": None,
        "derived_from": None,
        "build_id": None,
        "extra_json": {"role": item["role"], "substitute_for": "deepfilternet3"},
    }


def _catalog(assets: list[dict[str, Any]]):
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
        for asset in assets
    ]
    ordered = sorted(rows, key=lambda row: row["path"])
    counts: dict[str, int] = {}
    for row in ordered:
        counts[row["corpus"]] = counts.get(row["corpus"], 0) + 1
    catalog = {
        "catalog_id": catalog_id_for(ordered),
        "source_kind": "staging_parquet",
        "corpora_json": counts,
        "row_count": len(ordered),
        "source_db_sha256": None,
    }
    return catalog, ordered


def _collect(corpora: Path) -> dict[str, Any]:
    speech = {
        "train": _take_speech(corpora / "LibriSpeech/train-clean-100", _LIMITS["substitute"]["speech"]["train"]),
        "valid": _take_speech(corpora / "LibriSpeech/dev-clean", _LIMITS["substitute"]["speech"]["valid"]),
        "test": _take_speech(corpora / "LibriSpeech/test-clean", _LIMITS["substitute"]["speech"]["test"]),
    }
    noise = _demand(corpora / "DEMAND")
    rir = {
        "train": _take_wavs(corpora / "OpenSLR28/RIRS_NOISES/simulated_rirs/smallroom", _LIMITS["substitute"]["rir"]["train"]),
        "valid": _take_wavs(corpora / "OpenSLR28/RIRS_NOISES/simulated_rirs/mediumroom", _LIMITS["substitute"]["rir"]["valid"]),
        "test": _take_wavs(corpora / "OpenSLR28/RIRS_NOISES/simulated_rirs/largeroom", _LIMITS["substitute"]["rir"]["test"]),
    }
    full = {}
    for split in ("train", "valid", "test"):
        full[split] = {
            "speech": [_described(corpora, path, "LibriSpeech", split, speaker, "speech") for speaker, path in speech[split]],
            "noise": [_described(corpora, path, "DEMAND", split, scene, "noise") for scene, path in noise[split]],
            "rir": [
                _described(
                    corpora,
                    path,
                    "OpenSLR28",
                    split,
                    f"{path.parent.parent.name}/{path.parent.name}",
                    "rir",
                )
                for path in rir[split]
            ],
        }
        for role, need in _LIMITS["substitute"].items():
            if len(full[split][role]) < need[split]:
                raise ReproductionError(
                    f"found {len(full[split][role])} {split} {role} files, need {need[split]}"
                )
    smoke = {}
    for split in ("train", "valid", "test"):
        smoke[split] = {
            role: full[split][role][: _LIMITS["smoke"][role][split]]
            for role in ("speech", "noise", "rir")
        }
    return {"substitute": full, "smoke": smoke}


def _described(corpora, path, corpus, subset, speaker, role) -> dict[str, Any]:
    return {
        "path": path,
        "corpora": corpora,
        "corpus": corpus,
        "subset": subset,
        "speaker_id": speaker,
        "role": role,
    }


def _take_speech(root: Path, limit: int) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise ReproductionError(f"missing speech directory {root}")
    found = []
    speakers = sorted(path for path in root.iterdir() if path.is_dir())
    for speaker in speakers:
        files = []
        for dirpath, dirnames, filenames in os.walk(speaker):
            dirnames.sort()
            for name in sorted(filenames):
                if name.lower().endswith(".flac"):
                    files.append(Path(dirpath) / name)
        for path in files:
            found.append((speaker.name, path))
            if len(found) >= limit:
                return found
    return found


def _demand(root: Path) -> dict[str, list[tuple[str, Path]]]:
    scenes = sorted(path for path in root.iterdir() if path.is_dir())
    if len(scenes) < 18:
        raise ReproductionError(f"DEMAND has {len(scenes)} scenes; expected at least 18")
    groups = {"train": scenes[:10], "valid": scenes[10:14], "test": scenes[14:18]}
    per_scene = {"train": 4, "valid": 4, "test": 4}
    out = {}
    for split, chosen in groups.items():
        rows = []
        for scene in chosen:
            wavs = sorted(scene.glob("*.wav"))[: per_scene[split]]
            rows.extend((scene.name, path) for path in wavs)
        out[split] = rows
    return out


def _take_wavs(root: Path, limit: int) -> list[Path]:
    found = []
    if not root.is_dir():
        raise ReproductionError(f"missing RIR directory {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(".wav"):
                found.append(Path(dirpath) / name)
                if len(found) >= limit:
                    return found
    return found


def _rel(item: dict[str, Any]) -> str:
    return item["path"].relative_to(item["corpora"]).as_posix()


def _unique(pools: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    found = {}
    for split in pools.values():
        for rows in split.values():
            for item in rows:
                found[_rel(item)] = item
    return [found[key] for key in sorted(found)]
