"""Register SEMamba++ online-degradation datasets.

The official VCTK 0.92 manifest, Arni RIRs, and fixed validation pairs are
not available. Ready datasets use substitute materials and are not an official
reproduction. This module does not write degraded wavs.
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
    SEMAMBA_GSR_V1,
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

OFFICIAL_COMMIT = SEMAMBA_GSR_V1["official_commit"]
STAGING = ROOT / "data/staging/reproduction-p9"
DATABASE = ROOT / "data/db/reproduction-p9.duckdb"
SUBSTITUTE_NAME = "semambapp.gsr.substitute"
SMOKE_NAME = "semambapp.gsr.smoke"
BLOCKED_NAME = "semambapp.gsr.official"

_LIMITS = {
    "substitute": {
        "speech": {"train": 16, "valid": 4, "test": 4},
        "noise": {"train": 12, "valid": 4, "test": 4},
        "rir": {"train": 8, "valid": 4, "test": 4},
    },
    "smoke": {
        "speech": {"train": 8, "valid": 2, "test": 2},
        "noise": {"train": 4, "valid": 2, "test": 2},
        "rir": {"train": 2, "valid": 2, "test": 2},
    },
}


def shipped_rule() -> dict[str, Any]:
    return {
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "normalize": "independent_peak_v1",
        "window": 48000,
        "order": "per_rank_independent_v1",
        "epoch_unit": "one_pass_over_speech",
        "global_batch": 8,
        "delivery": "on_demand",
        "validation": {
            "crop": "independent_peak_then_segment",
            "batch": 1,
            "interval": 5000,
            "note": "valid and test ignore epoch; official fixed pairs are absent",
        },
        "online": dict(SEMAMBA_GSR_V1),
    }


def register_p9(
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
    return {"counts": counts, "snapshots": snapshots, "database": str(database), "datasets": _ids(graph)}


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
                    protocol_id=child["protocol_id"],
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
                    "VCTK 0.80 speech, DNS2020 noise, and OpenSLR28 simulated RIRs. "
                    "Not VCTK 0.92, not the official WHAM! list, and not Arni RIRs."
                ),
            )
        )
    datasets.append(_blocked(parent))
    return Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: catalog_rows},
        assets=assets,
        materials=materials,
        protocols=[parent, child],
        cases=cases,
        case_sets=case_sets,
        datasets=datasets,
    )


def _ids(graph: Graph) -> list[dict[str, str]]:
    return [
        {
            "name": dataset["name"],
            "dataset_id": dataset["dataset_id"],
            "protocol_id": dataset["protocol_id"],
            "declared_status": dataset["declared_status"],
            "claim_label": dataset["claim_label"],
        }
        for dataset in graph.datasets
    ]


def _protocols() -> tuple[dict[str, Any], dict[str, Any]]:
    rule = shipped_rule()
    parent_id = protocol_id_for(
        algorithm="online.semambapp_gsr",
        algorithm_version=1,
        kind="online_mixing",
        rule=rule,
        deviation_class="official_effective",
        parent=None,
    )
    parent = {
        "protocol_id": parent_id,
        "name": "semambapp.gsr",
        "version": 1,
        "kind": "online_mixing",
        "algorithm": "online.semambapp_gsr",
        "algorithm_version": 1,
        "rule_json": rule,
        "official_ref_json": {
            "repo": "Eddieyjlee/SEMambapp",
            "commit": OFFICIAL_COMMIT,
            "path": "simulate_degradation.py",
            "section": "random_select_and_order",
            "document": "config.yaml",
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
                "Draws use one NumPy PCG64 per sample, seeded by SHA-256 of the "
                "protocol, snapshot, training seed, split, epoch, and speech index. "
                "Official code mixes unseeded CPython random, global np.random, and "
                "default_rng(None) for the noise offset. Bit sequences are not claimed."
            ),
            "ref": "dataset.py",
        },
        {
            "id": "discarded_draw_per_sample",
            "class": "algorithm_change",
            "text": (
                "The first random_select_and_order result is discarded inside the "
                "sample stream. Officially that call advances the shared process RNG, "
                "so it also changes later samples. Marginal probabilities match; the "
                "cross-sample coupling does not."
            ),
            "ref": "dataset.py",
        },
        {
            "id": "seeded_rank_shuffle",
            "class": "algorithm_change",
            "text": (
                "per_rank_independent_v1 builds a seeded permutation per rank and then "
                "applies a drop_last DistributedSampler. Official shuffles are unseeded. "
                "world_size 1 still reshuffles each epoch; the official 1-GPU loader does not. "
                "The official sampler pads (drop_last=False) before the loader drops a short batch."
            ),
            "ref": "train.py",
        },
        {
            "id": "rank_independent_waveform",
            "class": "algorithm_change",
            "text": (
                "A speech index renders the same mixture on every rank. Official ranks "
                "do not share a random stream, so an overlapped speech file still gets "
                "a different noise draw."
            ),
            "ref": "dataset.py",
        },
        {
            "id": "bandwidth_short_pad",
            "class": "algorithm_change",
            "text": (
                "If resampling returns fewer samples than the input, the tail is zero-padded. "
                "The official slice keeps the short array and the later length assert fails."
            ),
            "ref": "simulate_degradation_utils.py",
        },
        {
            "id": "substitute_validation",
            "class": "algorithm_change",
            "text": (
                "Valid and test are epoch-invariant online renders of held-out substitute "
                "files, peak-normalized and cropped to 48000. Official validation reads "
                "the first 100 sorted pre-degraded pairs at full length. Those files are absent."
            ),
            "ref": "dataset.py",
        },
        {
            "id": "waveform_not_stft",
            "class": "algorithm_change",
            "text": (
                "The dataset returns waveforms. SEMambaTask computes mag-phase STFT once. "
                "The official dataset also returns STFT arrays from the same window."
            ),
            "ref": "dataset.py",
        },
    ]
    child_id = protocol_id_for(
        algorithm="online.semambapp_gsr",
        algorithm_version=1,
        kind="online_mixing",
        rule=rule,
        deviation_class="algorithm_change",
        parent=parent_id,
    )
    child = {
        "protocol_id": child_id,
        "name": "semambapp.gsr.audio-db",
        "version": 1,
        "kind": "online_mixing",
        "algorithm": "online.semambapp_gsr",
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
            "item": "VCTK 0.92",
            "reason": "The linked VCTK README says version 0.80. p315 is absent and p280 is present. It is not registered as 0.92.",
        },
        {
            "item": "Arni RIR",
            "reason": "No Arni recording is linked. OpenSLR28 and DNS5 SLR26/SLR28 are different collections.",
        },
        {
            "item": "official train manifests",
            "reason": "train_speech.json, train_noise.json, and train_rir.json are a single newline. File lists and counts are unknown.",
        },
        {
            "item": "official validation pairs",
            "reason": "val_clean.json and val_degraded.json are a single newline. The fixed 100-pair validation set is absent.",
        },
        {
            "item": "WHAM! noise list",
            "reason": "WHAM48 high_res_wham is present, but the paper's WHAM! subset and split are not identified. It is not bound.",
        },
        {
            "item": "DNS2020 noise subset",
            "reason": "DNS2020 noise files exist, but the official training subset and split are unknown. DNS5 is a different release.",
        },
        {
            "item": "DNS5 RIR subset",
            "reason": "DNS5 impulse_responses contains SLR26 and SLR28. That is not proof of the training RIR list, and it is not Arni.",
        },
        {
            "item": "VCTK-GSR split",
            "reason": "The paper cites a 99/8 speaker split without the ordered speaker list used for VCTK-GSR.",
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
        note="Official SEMamba++ training and validation materials are not available. No snapshot is exported.",
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
    name = f"semambapp.{scale}.{split}.{role}"
    body = {
        "name": name,
        "version": 1,
        "role": role,
        "split_role": split,
        "origin": "substitute",
        "selection_rule_json": rule,
        "provenance_json": {
            "source": "local corpora substitute for SEMamba++",
            "note": f"{scale} {split} {role}; not an official manifest",
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


def _online_cases(*, protocol_id: str, scale: str, split: str, count: int):
    rows = []
    for index in range(count):
        document = {
            "realization_version": 1,
            "algorithm": "online.semambapp_gsr",
            "algorithm_version": 1,
            "sample_rate": 16000,
            "sources": {"split": split, "scale": scale},
            "draws": {"sample_index": index, "seed_rule": "split_sample_index_epoch0"},
            "steps": ["semamba_gsr_v1"],
            "output": {"length": 48000, "teacher": "dry_clean"},
        }
        case_id = digest(document)
        row = {
            "case_id": case_id,
            "doc_kind": "realization_v1",
            "doc_json": document,
            "case_role": "online_snapshot",
            "protocol_id": protocol_id,
            "clean_asset_id": None,
            "noisy_asset_id": None,
            "noise_asset_id": None,
            "teacher_kind": "dry_clean",
            "sample_rate": 16000,
            "frames": 48000,
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
    name = f"semambapp.{scale}.{split}.online"
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
        "items": [{"position": index, "case_id": row["case_id"]} for index, row in enumerate(rows)],
    }
    return rows, case_set


def _asset(item: dict[str, Any]) -> dict[str, Any]:
    path = item["path"]
    info = sf.info(path)
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_value.update(chunk)
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
        "sha256": digest_value.hexdigest(),
        "integrity": "sha256",
        "catalog_id": None,
        "derivation_id": None,
        "derived_from": None,
        "build_id": None,
        "extra_json": {
            "role": item["role"],
            "substitute_for": "semambapp",
            "corpus_note": item["note"],
        },
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
    speech = _vctk(corpora / "VCTK/wav48")
    noise = _noise(corpora / "DNS2020/datasets/noise")
    rir = {
        "train": _take_wavs(corpora / "OpenSLR28/RIRS_NOISES/simulated_rirs/smallroom", _LIMITS["substitute"]["rir"]["train"]),
        "valid": _take_wavs(corpora / "OpenSLR28/RIRS_NOISES/simulated_rirs/mediumroom", _LIMITS["substitute"]["rir"]["valid"]),
        "test": _take_wavs(corpora / "OpenSLR28/RIRS_NOISES/simulated_rirs/largeroom", _LIMITS["substitute"]["rir"]["test"]),
    }
    full = {}
    for split in ("train", "valid", "test"):
        full[split] = {
            "speech": [
                _described(corpora, path, "VCTK", split, speaker, "speech", "VCTK 0.80, not 0.92")
                for speaker, path in speech[split]
            ],
            "noise": [
                _described(corpora, path, "DNS2020", split, path.stem, "noise", "DNS2020 noise file, subset unknown")
                for path in noise[split]
            ],
            "rir": [
                _described(
                    corpora,
                    path,
                    "OpenSLR28",
                    split,
                    path.relative_to(corpora / "OpenSLR28/RIRS_NOISES/simulated_rirs").parent.as_posix(),
                    "rir",
                    "OpenSLR28 simulated RIR, not Arni",
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
            role: full[split][role][: _LIMITS["smoke"][role][split]] for role in ("speech", "noise", "rir")
        }
    return {"substitute": full, "smoke": smoke}


def _described(corpora, path, corpus, subset, speaker, role, note) -> dict[str, Any]:
    return {
        "path": path,
        "corpora": corpora,
        "corpus": corpus,
        "subset": subset,
        "speaker_id": speaker,
        "role": role,
        "note": note,
    }


def _vctk(root: Path) -> dict[str, list[tuple[str, Path]]]:
    if not root.is_dir():
        raise ReproductionError(f"missing VCTK directory {root}")
    speakers = sorted(path for path in root.iterdir() if path.is_dir())
    plan = [("train", 8, 2), ("valid", 4, 1), ("test", 4, 1)]
    cursor = 0
    out = {}
    for split, speaker_count, per_speaker in plan:
        chosen = speakers[cursor : cursor + speaker_count]
        cursor += speaker_count
        if len(chosen) < speaker_count:
            raise ReproductionError(f"VCTK does not have enough speakers for {split}")
        rows = []
        for speaker in chosen:
            files = sorted(speaker.glob("*.wav"))[:per_speaker]
            if len(files) < per_speaker:
                raise ReproductionError(f"{speaker} does not have {per_speaker} wav files")
            rows.extend((speaker.name, path) for path in files)
        out[split] = rows
    return out


def _noise(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise ReproductionError(f"missing DNS2020 noise directory {root}")
    files = sorted(path for path in root.glob("*.wav") if path.is_file())
    cuts = {
        "train": (0, _LIMITS["substitute"]["noise"]["train"]),
        "valid": (
            _LIMITS["substitute"]["noise"]["train"],
            _LIMITS["substitute"]["noise"]["train"] + _LIMITS["substitute"]["noise"]["valid"],
        ),
        "test": (
            _LIMITS["substitute"]["noise"]["train"] + _LIMITS["substitute"]["noise"]["valid"],
            _LIMITS["substitute"]["noise"]["train"]
            + _LIMITS["substitute"]["noise"]["valid"]
            + _LIMITS["substitute"]["noise"]["test"],
        ),
    }
    out = {}
    for split, (start, stop) in cuts.items():
        chosen = files[start:stop]
        if len(chosen) < stop - start:
            raise ReproductionError(f"DNS2020 noise does not have enough files for {split}")
        out[split] = chosen
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
