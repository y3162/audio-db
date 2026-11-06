"""Turn a DNS-Challenge plan into reproduction-database rows.

Generated waveforms are not assets. Train and valid cases store the realization
document. The official no-reverb synthetic test is a recorded pair and is not
used for model selection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from src.data.recipe import canonical_recipe
from src.reproduction.build.dns2020 import (
    ALGORITHM,
    DNS_COMMIT,
    ESPNET_COMMIT,
    GENERATOR,
    GENERATOR_VERSION,
    SAMPLE_RATE,
    measure_realization,
    split_train_valid,
)
from src.reproduction.db.graph import Graph
from src.reproduction.db.validate import compute_selection_uses_test, validate_graph
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import (
    case_content_hash,
    case_set_id_for,
    catalog_id_for,
    dataset_id_for,
    digest,
    material_members_key,
    material_set_id_for,
    protocol_id_for,
    recipe_case_id,
    weakest_integrity,
)

OFFICIAL_REF = {
    "repo": "microsoft/DNS-Challenge",
    "commit": DNS_COMMIT,
    "path": "noisyspeech_synthesizer_multiprocessing.py",
    "section": "egs2/dns_ins20/enh1",
}
RULE = {
    "sample_rate": SAMPLE_RATE,
    "teacher": "dry_clean",
    "window": 64000,
    "order": "espnet_chunk_v1",
    "epoch_unit": "source_utterance",
    "global_batch": 4,
    "validation": {
        "crop": "chunk",
        "batch": 1,
        "interval": "epoch",
        "note": "cv_synthetic is the tail after count*9/10 of sorted uttids. This is not cv_synthetic_small.",
    },
}
SEED_DEVIATION = {
    "id": "seeded_per_case",
    "class": "algorithm_change",
    "text": (
        "Official DNS generation uses one shared multiprocessing counter and unseeded "
        "retries. audio-db stores a per-case plan from a sorted shuffle and "
        "mix_seed('dns', seed, gen_index). The stored spans rebuild the same PCM_16. "
        "They are not a bit match to an official dump."
    ),
}
ONDEMAND_DEVIATION = {
    "id": "on_demand_render",
    "class": "algorithm_change",
    "text": (
        "Training reconstructs PCM_16 from the realization and the source files. "
        "Mixture wavs are not dataset members and are not cached."
    ),
}
RETRY_DEVIATION = {
    "id": "no_sample_retry",
    "class": "algorithm_change",
    "text": (
        "Official build_audio skips a piece whose peak exceeds 0.99, and gen_audio "
        "rebuilds clean speech until activity is above 0.6. Noise activity is 0, so "
        "noise is not checked. A mix is rebuilt when clean, noise, or noisy is still "
        "clipped. Those retries depend on the samples and the shared counter. The "
        "stored plan does not retry."
    ),
}
EXCLUDE_DEVIATION = {
    "id": "exclude_non_16k_noise",
    "class": "algorithm_change",
    "text": (
        "Official audiolib resamples noise with librosa.core.resample(audio, fs, target). "
        "The librosa version used for the published mixtures is unknown. Noise that is "
        "not 16 kHz mono is excluded instead of resampled. Clean speech in this catalog "
        "is already 16 kHz."
    ),
}
FILEID_RE = re.compile(r"fileid_(\d+)\.wav$", re.IGNORECASE)


def protocol_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """Official formula, then the on-demand seeded child."""
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
        "name": "dns.tflocoformer.formula",
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
    rule = dict(RULE)
    rule["delivery"] = "on_demand"
    child_id = protocol_id_for(
        algorithm=ALGORITHM,
        algorithm_version=1,
        kind="pregenerated",
        rule=rule,
        deviation_class="algorithm_change",
        parent=parent_id,
    )
    child = {
        "protocol_id": child_id,
        "name": "dns.tflocoformer.ondemand",
        "version": 1,
        "kind": "pregenerated",
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "rule_json": rule,
        "official_ref_json": OFFICIAL_REF,
        "deviation_class": "algorithm_change",
        "parent_protocol_id": parent_id,
        "deviations_json": [SEED_DEVIATION, ONDEMAND_DEVIATION, RETRY_DEVIATION, EXCLUDE_DEVIATION],
    }
    return parent, child


def asset_row(
    path: Path,
    *,
    corpus_root: Path,
    subset: str,
    rate: int,
    frames: int,
    channels: int,
    sha256: str,
    speaker_id: str | None,
) -> dict[str, Any]:
    rel = path.relative_to(corpus_root).as_posix()
    if not rel.startswith("DNS2020/"):
        raise ReproductionError(f"{path} is not under DNS2020")
    return {
        "asset_id": f"corpora/{rel}",
        "storage_root": "corpora",
        "rel_path": rel,
        "corpus": "DNS2020",
        "subset": subset,
        "speaker_id": speaker_id,
        "utterance_key": path.stem,
        "sample_rate": int(rate),
        "frames": int(frames),
        "channels": int(channels),
        "sample_format": "PCM_16",
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
        "integrity": "sha256",
        "catalog_id": None,
        "derivation_id": None,
        "derived_from": None,
        "build_id": None,
        "extra_json": {},
    }


def material_set(
    assets: Sequence[dict[str, Any]],
    *,
    name: str,
    role: str,
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
    body = {
        "name": name,
        "version": 1,
        "role": role,
        "split_role": "train",
        "origin": "official",
        "selection_rule_json": rule,
        "provenance_json": {
            "source": "DNS-Challenge",
            "repo": "microsoft/DNS-Challenge",
            "commit": DNS_COMMIT,
            "path": assets[0]["rel_path"].rsplit("/", 1)[0],
            "note": note,
            "sha256": digest([asset["sha256"] for asset in assets]),
        },
        "member_count": len(assets),
        "total_seconds": seconds,
        "list_sha256": digest(keys),
        "integrity_min": weakest_integrity([asset["integrity"] for asset in assets]),
        "members": [
            {"position": index, "asset_id": asset["asset_id"], "group_key": None, "attrs_json": None}
            for index, asset in enumerate(assets)
        ],
    }
    body["material_set_id"] = material_set_id_for(
        name=name,
        version=1,
        role=role,
        split_role="train",
        rule=rule,
        members=keys,
    )
    return body


def condition_case(document: dict[str, Any], *, protocol_id: str) -> dict[str, Any]:
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
        "gen_index": int(stored["draws"]["gen_index"]),
    }
    row["content_hash"] = case_content_hash(row)
    return row


def recorded_pair(clean: dict[str, Any], noisy: dict[str, Any], protocol_id: str) -> dict[str, Any]:
    if int(clean["frames"]) != int(noisy["frames"]) or int(clean["sample_rate"]) != int(noisy["sample_rate"]):
        raise ReproductionError(f"{clean['rel_path']} and {noisy['rel_path']} are not an equal-length pair")
    document = canonical_recipe(
        {
            "recipe_version": 1,
            "sources": {
                "clean": {"path": "../" + clean["asset_id"], "sha256": clean["sha256"]},
                "noisy": {"path": "../" + noisy["asset_id"], "sha256": noisy["sha256"]},
            },
            "operations": [],
            "output": {"sample_rate": SAMPLE_RATE, "reference": "dry_clean"},
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
        "sample_rate": SAMPLE_RATE,
        "frames": int(clean["frames"]),
        "speaker_id": clean.get("speaker_id"),
        "utterance_key": clean["utterance_key"],
        "transcript": None,
        "build_id": None,
        "gen_index": None,
        "content_hash": "",
    }
    row["content_hash"] = case_content_hash(row)
    return row


def pair_test_files(clean_assets: Sequence[dict[str, Any]], noisy_assets: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair official no-reverb files by ``fileid_N``. Order is the integer id."""
    def index(asset: dict[str, Any]) -> str:
        match = FILEID_RE.search(Path(asset["rel_path"]).name)
        if match is None:
            raise ReproductionError(f"{asset['rel_path']} has no fileid_")
        return match.group(1)

    clean = {index(asset): asset for asset in clean_assets}
    noisy = {index(asset): asset for asset in noisy_assets}
    if set(clean) != set(noisy):
        raise ReproductionError("no_reverb clean and noisy file ids do not match")
    if len(clean) != len(clean_assets) or len(noisy) != len(noisy_assets):
        raise ReproductionError("duplicate no_reverb file id")
    return [(clean[key], noisy[key]) for key in sorted(clean, key=lambda item: int(item))]


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
            "commit": ESPNET_COMMIT,
            "path": "egs2/dns_ins20/enh1/local/dns_data_prep.sh",
            "note": note,
        },
        "member_count": len(case_ids),
        "list_sha256": digest(case_ids),
        "items": [{"position": index, "case_id": case_id} for index, case_id in enumerate(case_ids)],
    }
    body["case_set_id"] = case_set_id_for(
        name=name,
        version=1,
        split_role=split_role,
        rule=rule,
        case_ids=case_ids,
    )
    return body


def build_row(definition: dict[str, Any]) -> dict[str, Any]:
    inputs = definition.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ReproductionError("a registered DNS build needs material-set inputs")
    if definition["generator"] != GENERATOR or int(definition["generator_version"]) != GENERATOR_VERSION:
        raise ReproductionError("DNS build generator does not match the registrar")
    return {
        "build_id": definition["build_id"],
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "params_json": definition["params"],
        "seed": int(definition["seed"]),
        "inputs_json": dict(inputs),
        "shard_size": int(definition["shard_size"]),
    }


def audit_sample(
    definition: dict[str, Any],
    documents: Sequence[dict[str, Any]],
    *,
    speech: dict[str, Path],
    noise: dict[str, Path],
    render_limit: int,
) -> dict[str, Any]:
    """Check every plan's files exist. Render an evenly spaced sample in memory."""
    if render_limit < 1:
        raise ReproductionError("audit_realizations must be positive")
    seen: set[int] = set()
    for document in documents:
        gen_index = int(document["draws"]["gen_index"])
        if gen_index in seen:
            raise ReproductionError(f"duplicate gen_index {gen_index}")
        seen.add(gen_index)
        for kind, mapping in (("speech", speech), ("noise", noise)):
            for piece in document["sources"][kind]:
                if "name" not in piece:
                    continue
                if str(piece["name"]) not in mapping:
                    raise ReproductionError(f"planned {kind} {piece['name']} is not in the source map")
    chosen = _spread(len(documents), render_limit)
    rows = [measure_realization(documents[index], {"speech": speech, "noise": noise}) for index in chosen]
    rejected = sum(1 for row in rows if row["clean_activity_rejected"])
    source_clipped = sum(int(row["source_clipped_pieces"]) for row in rows)
    post_clipped = sum(int(row["post_mix_clipped"]) for row in rows)
    errors = [abs(float(row["snr_error_db"])) for row in rows]
    return {
        "build_id": str(definition["build_id"]),
        "status": "PASS",
        "delivery": "on_demand",
        "wav_files": 0,
        "shard_ids": ["conditions"],
        "realizations": len(documents),
        "rendered": len(rows),
        "render_scope": "all" if len(rows) == len(documents) else "sample",
        "clean_activity_rejected": rejected,
        "source_clipped_pieces": source_clipped,
        "post_mix_clipped": post_clipped,
        "max_snr_error_db": None if not errors else max(errors),
        "files": 0,
        "clipped_samples": post_clipped,
        "peak_abs": 0.0,
    }


def assemble_graph(
    *,
    recipe_name: str,
    recipe_claim: str,
    recipe_definition: dict[str, Any],
    recipe_documents: Sequence[dict[str, Any]],
    recipe_report: dict[str, Any],
    smoke_name: str,
    smoke_claim: str,
    smoke_definition: dict[str, Any],
    smoke_documents: Sequence[dict[str, Any]],
    smoke_report: dict[str, Any],
    speech: dict[str, Any],
    noise: dict[str, Any],
    source_assets: Sequence[dict[str, Any]],
    test_cases: Sequence[dict[str, Any]],
    resource_db: str | None,
    excluded_noise: int,
) -> Graph:
    for report in (recipe_report, smoke_report):
        if report.get("status") != "PASS" or report.get("delivery") != "on_demand" or int(report.get("wav_files", 0)) != 0:
            raise ReproductionError("refusing to register a DNS build that was not checked on demand")
    parent, child = protocol_pair()
    catalog, catalog_rows = _catalog(list(source_assets))
    for asset in source_assets:
        asset["catalog_id"] = catalog["catalog_id"]
    recipe_train, recipe_valid, recipe_stats = split_train_valid(recipe_documents)
    smoke_train, smoke_valid, smoke_stats = split_train_valid(smoke_documents)
    protocol_id = child["protocol_id"]
    recipe_train_cases = [condition_case(item, protocol_id=protocol_id) for item in recipe_train]
    recipe_valid_cases = [condition_case(item, protocol_id=protocol_id) for item in recipe_valid]
    smoke_train_cases = [condition_case(item, protocol_id=protocol_id) for item in smoke_train]
    smoke_valid_cases = [condition_case(item, protocol_id=protocol_id) for item in smoke_valid]
    test_set = case_set_for(
        "dns.tflocoformer.no_reverb",
        "test",
        test_cases,
        {"rule": "explicit"},
        "official synthetic no_reverb pairs by fileid. with_reverb and real recordings are not included. Not used for selection.",
    )
    sets = [
        case_set_for(
            f"{recipe_name}.train",
            "train",
            recipe_train_cases,
            {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.9, "part": "head"},
            f"dns_data_prep count*9/10 head of sorted uttids; {recipe_stats}",
        ),
        case_set_for(
            f"{recipe_name}.valid",
            "valid",
            recipe_valid_cases,
            {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.9, "part": "tail"},
            f"complement of the 9/10 head. This is cv_synthetic, not cv_synthetic_small; {recipe_stats}",
        ),
        case_set_for(
            f"{smoke_name}.train",
            "train",
            smoke_train_cases,
            {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.9, "part": "head"},
            f"same 9/10 rule on the smoke count; {smoke_stats}",
        ),
        case_set_for(
            f"{smoke_name}.valid",
            "valid",
            smoke_valid_cases,
            {"rule": "sorted_ratio", "key": "utt_id", "ratio": 0.9, "part": "tail"},
            f"smoke validation tail; {smoke_stats}",
        ),
        test_set,
    ]
    recipe = _dataset(
        recipe_name,
        protocol_id,
        recipe_claim,
        sets[0],
        sets[1],
        test_set,
        speech,
        noise,
        _deviations(recipe_claim, excluded_noise),
        _note(recipe_claim),
    )
    smoke = _dataset(
        smoke_name,
        protocol_id,
        smoke_claim,
        sets[2],
        sets[3],
        test_set,
        speech,
        noise,
        _deviations(smoke_claim, excluded_noise),
        _note(smoke_claim),
    )
    case_sets = {row["case_set_id"]: row for row in sets}
    cases = {
        row["case_id"]: row
        for row in recipe_train_cases + recipe_valid_cases + smoke_train_cases + smoke_valid_cases + list(test_cases)
    }
    for dataset in (recipe, smoke):
        uses_test = compute_selection_uses_test(dataset, case_sets, {
            speech["material_set_id"]: speech,
            noise["material_set_id"]: noise,
        }, cases)
        dataset["selection_uses_test"] = uses_test
        dataset["dataset_id"] = dataset_id_for(
            protocol_id=protocol_id,
            bindings=dataset["bindings"],
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
        materials=[speech, noise],
        protocols=[parent, child],
        cases=list(cases.values()),
        case_sets=sets,
        datasets=[recipe, smoke],
        builds=[build_row(recipe_definition), build_row(smoke_definition)],
        verified={
            recipe_definition["build_id"]: dict(recipe_report),
            smoke_definition["build_id"]: dict(smoke_report),
        },
    )
    validate_graph(graph, resource_db=resource_db)
    return graph


def _dataset(
    name: str,
    protocol_id: str,
    claim: str,
    train: dict[str, Any],
    valid: dict[str, Any],
    test: dict[str, Any],
    speech: dict[str, Any],
    noise: dict[str, Any],
    deviations: list[dict[str, str]],
    note: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "version": 1,
        "protocol_id": protocol_id,
        "profile": "official_faithful",
        "material_origin": "derived_from_official",
        "claim_label": claim,
        "declared_status": "ready",
        "blocking_json": [],
        "selection_uses_test": False,
        "selection_note": note,
        "deviations_json": deviations,
        "bindings": [
            {"slot": "train.cases", "case_set_id": train["case_set_id"], "material_set_id": None},
            {"slot": "valid.cases", "case_set_id": valid["case_set_id"], "material_set_id": None},
            {"slot": "test.cases", "case_set_id": test["case_set_id"], "material_set_id": None},
            {"slot": "train.speech", "case_set_id": None, "material_set_id": speech["material_set_id"]},
            {"slot": "train.noise", "case_set_id": None, "material_set_id": noise["material_set_id"]},
        ],
    }


def _deviations(claim: str, excluded_noise: int) -> list[dict[str, str]]:
    rows = [dict(SEED_DEVIATION), dict(ONDEMAND_DEVIATION), dict(RETRY_DEVIATION)]
    if excluded_noise:
        text = dict(EXCLUDE_DEVIATION)
        text["text"] = text["text"] + f" This catalog excludes {excluded_noise} files."
        rows.append(text)
    if claim == "official_recipe_scaled_down":
        rows.append(
            {
                "id": "scaled_down",
                "class": "research_improvement",
                "text": (
                    "The mixture count is smaller than the 3000 h recipe. The 9/10 split is "
                    "recomputed on this count. A gen_index uses the same spans and draws as "
                    "the full recipe when the pool and seed match, but the case_id differs "
                    "because the build id is inside the realization. This split is not a prefix "
                    "of the 3000 h train set."
                ),
            }
        )
    return rows


def _note(claim: str) -> str:
    common = (
        "Validation is the tail of the sorted generated uttids (cv_synthetic), not the test set "
        "and not the unpublished cv_synthetic_small split. Noise files are shared by train and "
        "valid through the realization names; valid does not bind a material set, so that share "
        "is not an asset-id overlap. "
    )
    if claim == "official_recipe_scaled_down":
        return common + "This is a short DNS plan for the trainer, not the 3000 h recipe."
    return (
        common
        + "claim_label espnet_public_se_recipe_reproduction is the existing ESPnet-recipe label. "
        "This dataset is DNS-Challenge, not MS-SNSD."
    )


def _spread(count: int, limit: int) -> list[int]:
    if count < 1:
        raise ReproductionError("there is nothing to audit")
    if limit >= count:
        return list(range(count))
    if limit == 1:
        return [0]
    chosen = sorted({int(round(index * (count - 1) / (limit - 1))) for index in range(limit)})
    return chosen


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


def paths_by_name(assets: Sequence[dict[str, Any]], corpus_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for asset in assets:
        name = Path(asset["rel_path"]).name
        if name in found:
            raise ReproductionError(f"duplicate source basename {name}")
        found[name] = corpus_root / asset["rel_path"]
    return found
