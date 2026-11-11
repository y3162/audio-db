"""Register a verified MS-SNSD build in the reproduction database.

The builder does not open DuckDB. This module turns a finished plan into
staging rows and leaves insertion to ``register_staging``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from src.data.recipe import sha256_file
from src.experiments.data.build.ms_snsd import (
    ALGORITHM,
    GENERATOR,
    GENERATOR_VERSION,
    OFFICIAL_COMMIT,
    SAMPLE_RATE,
    filter_test,
    render_realization,
    shape_lines,
    split_train_valid,
)
from src.experiments.data.db.graph import Graph
from src.experiments.data.db.validate import compute_selection_uses_test, validate_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import (
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

OFFICIAL_REF = {
    "repo": "espnet/espnet",
    "commit": OFFICIAL_COMMIT,
    "path": "egs2/ms_snsd/enh1",
    "section": "local/ms_snsd_noisyspeech_synthesizer.py",
}
RULE = {
    "sample_rate": SAMPLE_RATE,
    "teacher": "dry_clean",
    "window": 48000,
    "order": "espnet_chunk_v1",
    "epoch_unit": "source_utterance",
    "global_batch": 4,
    "validation": {
        "crop": "chunk",
        "batch": 2,
        "interval": "epoch",
        "note": "cv_ms_snsd is the tail of the sorted train noisy list",
    },
}
SEED_DEVIATIONS = [
    {
        "id": "seeded_draws",
        "class": "algorithm_change",
        "text": (
            "Official MS-SNSD generation uses unseeded np.random and an unstable glob order. "
            "audio-db sorts names, shuffles once with mix_seed('list', ...), and draws each clip "
            "from mix_seed('gen', seed, split, clip_index). An official unseeded corpus cannot "
            "be reproduced file-for-file."
        ),
    },
    {
        "id": "fid_paths",
        "class": "algorithm_change",
        "text": (
            "Generated files use fid{gen_index:07d}.wav under the build directory because "
            "generated_rel_path requires that layout. Utterance ids and the 80/20 sort use the "
            "official noisy{n}_SNRdb_{snr}_clnsp{n} names, with n starting at 1. Lexical sort of "
            "those names is not numeric, matching find | sort and sort -k1,1."
        ),
    },
]


def collect_sources(
    directory: Path,
    names: Sequence[str],
    *,
    rel_prefix: str,
    subset: str,
    known: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Hash and describe the wavs a build will read. The directory is not modified.

    ``known`` supplies header fields already checked against the file size.
    The bytes are still hashed.
    """
    assets = []
    frames: dict[str, int] = {}
    for index, name in enumerate(names, 1):
        path = directory / name
        if known is not None and name in known:
            row = known[name]
            rate = int(row["sample_rate"])
            file_frames = int(row["frames"])
            channels = int(row["channels"])
            size = int(row["size_bytes"])
            subtype = "PCM_16"
            if path.stat().st_size != size:
                raise ReproductionError(f"catalog size does not match {path}")
        else:
            info = sf.info(path)
            rate = int(info.samplerate)
            file_frames = int(info.frames)
            channels = int(info.channels)
            size = path.stat().st_size
            subtype = str(info.subtype or "")
        if rate != SAMPLE_RATE or channels < 1 or file_frames < 1:
            raise ReproductionError(f"{path} is not {SAMPLE_RATE} Hz audio with frames")
        rel = f"{rel_prefix}/{name}"
        asset = {
            "asset_id": f"corpora/{rel}",
            "storage_root": "corpora",
            "rel_path": rel,
            "corpus": "MS-SNSD",
            "subset": subset,
            "speaker_id": None,
            "utterance_key": Path(name).stem,
            "sample_rate": SAMPLE_RATE,
            "frames": file_frames,
            "channels": channels,
            "sample_format": subtype,
            "size_bytes": size,
            "sha256": sha256_file(path),
            "integrity": "sha256",
            "catalog_id": None,
            "derivation_id": None,
            "derived_from": None,
            "build_id": None,
            "extra_json": {},
        }
        assets.append(asset)
        frames[name] = file_frames
        if index % 2000 == 0 or index == len(names):
            print(f"hashed\t{subset}\t{index}\t{len(names)}", flush=True)
    return assets, frames


def material_set(
    assets: Sequence[dict[str, Any]],
    *,
    name: str,
    role: str,
    split_role: str,
    note: str,
) -> dict[str, Any]:
    if not assets:
        raise ReproductionError(f"material set {name} is empty")
    keys = [
        material_members_key(asset["asset_id"], asset.get("sha256"), asset.get("size_bytes"))
        for asset in assets
    ]
    seconds = float(f"{sum(asset['frames'] / asset['sample_rate'] for asset in assets):.6f}")
    rule = {"rule": "explicit"}
    provenance = {
        "source": "MS-SNSD",
        "repo": "microsoft/MS-SNSD",
        "path": assets[0]["rel_path"].rsplit("/", 1)[0],
        "note": note,
        "sha256": digest([asset["sha256"] for asset in assets]),
    }
    body = {
        "name": name,
        "version": 1,
        "role": role,
        "split_role": split_role,
        "origin": "official",
        "selection_rule_json": rule,
        "provenance_json": provenance,
        "member_count": len(assets),
        "total_seconds": seconds,
        "list_sha256": digest(keys),
        "integrity_min": weakest_integrity([asset["integrity"] for asset in assets]),
        "members": [
            {
                "position": index,
                "asset_id": asset["asset_id"],
                "group_key": None,
                "attrs_json": None,
            }
            for index, asset in enumerate(assets)
        ],
    }
    body["material_set_id"] = material_set_id_for(
        name=name,
        version=1,
        role=role,
        split_role=split_role,
        rule=rule,
        members=keys,
    )
    return body


def protocol_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """Official formula, then the seeded audio-db draw as its child."""
    parent_id = protocol_id_for(
        algorithm=ALGORITHM,
        algorithm_version=1,
        kind="pregenerated",
        rule=RULE,
        deviation_class="official_effective",
        parent=None,
    )
    parent = {
        "protocol_id": parent_id,
        "name": "ms_snsd.tfgridnet.formula",
        "version": 1,
        "kind": "pregenerated",
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "rule_json": RULE,
        "official_ref_json": OFFICIAL_REF,
        "deviation_class": "official_effective",
        "parent_protocol_id": None,
        "deviations_json": [],
    }
    child_id = protocol_id_for(
        algorithm=ALGORITHM,
        algorithm_version=1,
        kind="pregenerated",
        rule=RULE,
        deviation_class="algorithm_change",
        parent=parent_id,
    )
    child = {
        "protocol_id": child_id,
        "name": "ms_snsd.tfgridnet.seeded",
        "version": 1,
        "kind": "pregenerated",
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "rule_json": RULE,
        "official_ref_json": OFFICIAL_REF,
        "deviation_class": "algorithm_change",
        "parent_protocol_id": parent_id,
        "deviations_json": SEED_DEVIATIONS,
    }
    return parent, child


ONDEMAND_DEVIATION = {
    "id": "on_demand_render",
    "class": "algorithm_change",
    "text": (
        "Training reconstructs PCM_16 from the realization document and the source files. "
        "Generated wavs are not members of the dataset. Official generation is unseeded, so "
        "this path does not claim bit-exact files against an official dump. The mix formula "
        "itself is the seeded audio-db draw of snr_mixer."
    ),
}


def ondemand_protocol_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """Same official parent. The child adds on-demand delivery to the seeded rule."""
    parent, _stored = protocol_pair()
    rule = dict(RULE)
    rule["delivery"] = "on_demand"
    child_id = protocol_id_for(
        algorithm=ALGORITHM,
        algorithm_version=1,
        kind="pregenerated",
        rule=rule,
        deviation_class="algorithm_change",
        parent=parent["protocol_id"],
    )
    child = {
        "protocol_id": child_id,
        "name": "ms_snsd.tfgridnet.ondemand",
        "version": 1,
        "kind": "pregenerated",
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "rule_json": rule,
        "official_ref_json": OFFICIAL_REF,
        "deviation_class": "algorithm_change",
        "parent_protocol_id": parent["protocol_id"],
        "deviations_json": [*SEED_DEVIATIONS, ONDEMAND_DEVIATION],
    }
    return parent, child


def build_row(definition: dict[str, Any]) -> dict[str, Any]:
    inputs = definition.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ReproductionError("a registered MS-SNSD build needs material-set inputs")
    if definition["generator"] != GENERATOR or int(definition["generator_version"]) != GENERATOR_VERSION:
        raise ReproductionError("MS-SNSD build generator does not match the registrar")
    return {
        "build_id": definition["build_id"],
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "params_json": definition["params"],
        "seed": int(definition["seed"]),
        "inputs_json": dict(inputs),
        "shard_size": int(definition["shard_size"]),
    }


def condition_case(document: dict[str, Any], *, protocol_id: str) -> dict[str, Any]:
    """One planned mixture. The waveform is not an asset."""
    stored = {
        key: document[key]
        for key in (
            "realization_version",
            "algorithm",
            "algorithm_version",
            "sample_rate",
            "sources",
            "draws",
            "steps",
            "output",
        )
    }
    gen_index = int(stored["draws"]["gen_index"])
    row = {
        "case_id": digest(stored),
        "doc_kind": "realization_v1",
        "doc_json": stored,
        "case_role": "generated",
        "protocol_id": protocol_id,
        "clean_asset_id": None,
        "noisy_asset_id": None,
        "noise_asset_id": None,
        "teacher_kind": "dry_clean",
        "sample_rate": SAMPLE_RATE,
        "frames": int(stored["output"]["length"]),
        "speaker_id": None,
        "utterance_key": str(stored["draws"]["utt_id"]),
        "transcript": None,
        "build_id": str(stored["draws"]["build_id"]),
        "gen_index": gen_index,
    }
    row["content_hash"] = case_content_hash(row)
    return row


def audit_conditions(
    definition: dict[str, Any],
    documents: Sequence[dict[str, Any]],
    *,
    clean_dir: Path,
    noise_dir: Path,
    render_limit: int | None = None,
) -> dict[str, Any]:
    """Check every plan and render a sample in memory. Nothing is written.

    ``render_limit`` caps the mixtures that are actually mixed. File existence
    is still checked for every piece. A limit of ``None`` renders every plan.
    """
    seen: set[int] = set()
    clipped = 0
    peak = 0.0
    speech: set[str] = set()
    noise: set[str] = set()
    rendered = 0
    for document in documents:
        gen_index = int(document["draws"]["gen_index"])
        if gen_index in seen:
            raise ReproductionError(f"duplicate gen_index {gen_index}")
        seen.add(gen_index)
        length = int(document["output"]["length"])
        if length < 1:
            raise ReproductionError(f"gen_index {gen_index} has no samples")
        for piece in document["sources"]["speech"]:
            if "name" not in piece:
                continue
            name = str(piece["name"])
            speech.add(name)
            if not (clean_dir / name).is_file():
                raise ReproductionError(f"planned speech {name} is not in {clean_dir}")
        for piece in document["sources"]["noise"]:
            if "name" not in piece:
                continue
            name = str(piece["name"])
            noise.add(name)
            if not (noise_dir / name).is_file():
                raise ReproductionError(f"planned noise {name} is not in {noise_dir}")
        if render_limit is not None and rendered >= int(render_limit):
            continue
        mixed = render_realization(document, {"speech": clean_dir, "noise": noise_dir})
        rendered += 1
        for wave in mixed.values():
            if int(wave.shape[0]) != length:
                raise ReproductionError(f"gen_index {gen_index} rendered length does not match the plan")
            clipped += int(np.count_nonzero(np.abs(wave) >= 1.0))
            peak = max(peak, float(np.max(np.abs(wave))))
    scope = "all" if render_limit is None or rendered == len(documents) else "sample"
    return {
        "build_id": str(definition["build_id"]),
        "status": "PASS",
        "delivery": "on_demand",
        "wav_files": 0,
        "shard_ids": ["conditions"],
        "files": len(documents) * 3,
        "realizations": len(documents),
        "rendered": rendered,
        "render_scope": scope,
        "clipped_samples": clipped,
        "peak_abs": peak,
        "speech_files": len(speech),
        "noise_files": len(noise),
    }


def case_set_for(
    name: str,
    split_role: str,
    cases: Sequence[dict[str, Any]],
    rule: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    case_ids = [row["case_id"] for row in cases]
    body = {
        "name": name,
        "version": 1,
        "split_role": split_role,
        "origin": "derived_from_official",
        "selection_rule_json": rule,
        "provenance_json": {
            "source": "espnet",
            "repo": "espnet/espnet",
            "commit": OFFICIAL_COMMIT,
            "path": "egs2/ms_snsd/enh1/local/ms_snsd_data_prep.sh",
            "note": note,
        },
        "member_count": len(case_ids),
        "list_sha256": digest(case_ids),
        "items": [
            {"position": index, "case_id": case_id}
            for index, case_id in enumerate(case_ids)
        ],
    }
    body["case_set_id"] = case_set_id_for(
        name=name,
        version=1,
        split_role=split_role,
        rule=rule,
        case_ids=case_ids,
    )
    return body


def assemble_ondemand_graph(
    *,
    train_definition: dict[str, Any],
    train_documents: Sequence[dict[str, Any]],
    train_report: dict[str, Any],
    test_definition: dict[str, Any],
    test_documents: Sequence[dict[str, Any]],
    test_report: dict[str, Any],
    source_assets: Sequence[dict[str, Any]],
    train_clean: dict[str, Any],
    train_noise: dict[str, Any],
    test_clean: dict[str, Any],
    test_noise: dict[str, Any],
    dataset_name: str,
    resource_db: str | None,
    claim_label: str = "official_recipe_scaled_down",
) -> Graph:
    """Register plans and source files. Generated wavs are not assets."""
    for report in (train_report, test_report):
        if report.get("status") != "PASS" or report.get("delivery") != "on_demand":
            raise ReproductionError("refusing to register a build that was not checked on demand")
        if int(report.get("wav_files", 0)) != 0:
            raise ReproductionError("on-demand registration cannot include generated wavs")
    parent, child = ondemand_protocol_pair()
    catalog, catalog_rows = _catalog(list(source_assets))
    for asset in source_assets:
        asset["catalog_id"] = catalog["catalog_id"]
    train_kept, valid_kept, train_stats = split_train_valid(train_documents)
    test_kept, test_stats = filter_test(test_documents)
    train_cases = [condition_case(item, protocol_id=child["protocol_id"]) for item in train_kept]
    valid_cases = [condition_case(item, protocol_id=child["protocol_id"]) for item in valid_kept]
    test_cases = [condition_case(item, protocol_id=child["protocol_id"]) for item in test_kept]
    train_set = case_set_for(
        f"{dataset_name}.train",
        "train",
        train_cases,
        {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.8, "part": "head"},
        f"bash count*8/10 of sorted official utt ids, then duration filter; {json.dumps(train_stats, sort_keys=True)}",
    )
    valid_set = case_set_for(
        f"{dataset_name}.valid",
        "valid",
        valid_cases,
        {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.8, "part": "tail"},
        f"complement of the train head after the duration filter; {json.dumps(train_stats, sort_keys=True)}",
    )
    test_set = case_set_for(
        f"{dataset_name}.test",
        "test",
        test_cases,
        {"rule": "explicit"},
        f"all duration-kept test noisy files in utt-id order; {json.dumps(test_stats, sort_keys=True)}",
    )
    bindings = [
        {"slot": "train.cases", "case_set_id": train_set["case_set_id"], "material_set_id": None},
        {"slot": "valid.cases", "case_set_id": valid_set["case_set_id"], "material_set_id": None},
        {"slot": "test.cases", "case_set_id": test_set["case_set_id"], "material_set_id": None},
        {"slot": "train.speech", "case_set_id": None, "material_set_id": train_clean["material_set_id"]},
        {"slot": "train.noise", "case_set_id": None, "material_set_id": train_noise["material_set_id"]},
        {"slot": "test.speech", "case_set_id": None, "material_set_id": test_clean["material_set_id"]},
        {"slot": "test.noise", "case_set_id": None, "material_set_id": test_noise["material_set_id"]},
    ]
    deviations = _dataset_deviations(train_stats, claim_label=claim_label)
    deviations.append(dict(ONDEMAND_DEVIATION))
    dataset = {
        "name": dataset_name,
        "version": 1,
        "protocol_id": child["protocol_id"],
        "profile": "official_faithful",
        "material_origin": "derived_from_official",
        "claim_label": claim_label,
        "declared_status": "ready",
        "blocking_json": [],
        "selection_uses_test": False,
        "selection_note": _selection_note(claim_label) + " Waveforms are reconstructed on demand.",
        "deviations_json": deviations,
        "bindings": bindings,
    }
    materials = {
        row["material_set_id"]: row
        for row in (train_clean, train_noise, test_clean, test_noise)
    }
    case_sets = {row["case_set_id"]: row for row in (train_set, valid_set, test_set)}
    cases = {row["case_id"]: row for row in train_cases + valid_cases + test_cases}
    uses_test = compute_selection_uses_test(dataset, case_sets, materials, cases)
    dataset["selection_uses_test"] = uses_test
    dataset["dataset_id"] = dataset_id_for(
        protocol_id=child["protocol_id"],
        bindings=bindings,
        profile=dataset["profile"],
        material_origin=dataset["material_origin"],
        claim_label=dataset["claim_label"],
        selection_uses_test=uses_test,
        declared_status="ready",
    )
    graph = Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: catalog_rows},
        assets=list(source_assets),
        materials=[train_clean, train_noise, test_clean, test_noise],
        protocols=[parent, child],
        cases=train_cases + valid_cases + test_cases,
        case_sets=[train_set, valid_set, test_set],
        datasets=[dataset],
        builds=[build_row(train_definition), build_row(test_definition)],
        verified={
            train_definition["build_id"]: dict(train_report),
            test_definition["build_id"]: dict(test_report),
        },
    )
    validate_graph(graph, resource_db=resource_db)
    return graph


def write_shapes(directory: Path, cases: Sequence[dict[str, Any]], noisy_paths: dict[str, str]) -> None:
    """Write utt2num_samples and the three speech shape files in case-set order."""
    rows = []
    seen = set()
    for case in cases:
        utt_id = str(case["utterance_key"])
        if utt_id in seen:
            raise ReproductionError(f"duplicate utt id in a shape list: {utt_id}")
        seen.add(utt_id)
        rows.append((utt_id, int(case["frames"])))
    text = shape_lines(rows)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("utt2num_samples", "speech_mix_shape", "speech_ref1_shape", "noise_ref1_shape"):
        (directory / name).write_text(text, encoding="utf-8")
    scp = []
    for utt_id, _frames in rows:
        scp.append(f"{utt_id} {noisy_paths[utt_id]}")
    (directory / "wav.scp").write_text("\n".join(scp) + "\n", encoding="utf-8")


def _selection_note(claim_label: str) -> str:
    if claim_label == "official_recipe_scaled_down":
        return (
            "Scaled-down MS-SNSD build for the ESPnet public SE recipe. "
            "Validation is the 20% tail of generated train, not the test set. "
            "This is not the 30 h corpus."
        )
    return (
        "ESPnet public SE recipe on MS-SNSD. Validation is the tail of generated train, "
        "not the test set. This is not a TF-GridNet paper experiment."
    )


def _dataset_deviations(train_stats: dict[str, int], *, claim_label: str) -> list[dict[str, str]]:
    rows = []
    if claim_label == "official_recipe_scaled_down":
        rows.append(
            {
                "id": "scaled_down",
                "class": "research_improvement",
                "text": (
                    "Input lists are a short-file subset and max_clips stops early. "
                    "The official recipe generates 30 h of train and 1 h of test from the full lists."
                ),
            }
        )
    if train_stats.get("files_moved_to_keep_clip"):
        rows.append(
            {
                "id": "clip_atomic_split",
                "class": "algorithm_change",
                "text": (
                    "The ESPnet shell cuts the sorted noisy list at count*8/10. "
                    "That index can fall inside one clip's five SNR files, which share a clean waveform. "
                    "audio-db moves the cut to the nearer edge of that clip so train and valid do not share it."
                ),
            }
        )
    return rows


def _catalog(assets: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    return (
        {
            "catalog_id": catalog_id,
            "source_kind": "staging_parquet",
            "corpora_json": counts,
            "row_count": len(rows),
            "source_db_sha256": None,
        },
        rows,
    )
