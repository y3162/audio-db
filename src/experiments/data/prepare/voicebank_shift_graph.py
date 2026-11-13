"""Build the reproduction graph for shared VoiceBank SNR-shift cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from src.data.recipe import sha256_file
from src.experiments.data.db.graph import Graph
from src.experiments.data.db.validate import validate_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import case_content_hash, case_set_id_for, digest, protocol_id_for, validate_rule
from src.experiments.data.prepare.assemble import _catalog, _dataset
from src.experiments.data.prepare.voicebank_shift_recipe import snapshot_realization

ALGORITHM = "fixed_pair.voicebank_snr_shift"
MODELS = ("mp_senet", "semambapp", "tf_locoformer", "tf_gridnet")


def build_shift_graph(metadata: dict[str, Any], staged: list[dict[str, Any]],
                      source_graph: Graph, corpora: Path, *, resource_db: str | None) -> Graph:
    """Build one graph from verified staging; source files are SHA-256 checked."""
    parent_by_name = {p["name"]: p for p in source_graph.protocols}
    parent = parent_by_name.get("mp_senet.fixed_pair.ondemand")
    if parent is None:
        raise ReproductionError("VoiceBank reference protocol is missing")
    parent_parent = next((p for p in source_graph.protocols if p["protocol_id"] == parent.get("parent_protocol_id")), None)
    if parent_parent is None:
        raise ReproductionError("VoiceBank reference protocol parent is missing")
    existing = {asset["asset_id"]: asset for asset in source_graph.assets}
    sources: dict[str, dict[str, Any]] = {}
    cases = []
    by_split: dict[str, list[str]] = {split: [] for split in ("train", "valid", "test")}
    protocols = [_protocol(model, parent) for model in MODELS]
    for item in staged:
        document = snapshot_realization(item)
        for name, reference in document["sources"].items():
            asset_id = reference["asset"]
            if asset_id in sources:
                if sources[asset_id]["sha256"] != reference["sha256"]:
                    raise ReproductionError(f"source {asset_id} has conflicting hashes")
                continue
            path = corpora / asset_id.removeprefix("corpora/")
            if not path.is_file() or sha256_file(path) != reference["sha256"]:
                raise ReproductionError(f"source file SHA-256 does not match: {asset_id}")
            if name == "clean":
                asset = dict(existing.get(asset_id) or {})
                if not asset or asset.get("sha256") != reference["sha256"]:
                    raise ReproductionError(f"clean asset is absent or inconsistent: {asset_id}")
            else:
                info = sf.info(path)
                asset = {"asset_id": asset_id, "storage_root": "corpora",
                         "rel_path": asset_id.removeprefix("corpora/"),
                         "corpus": asset_id.split("/")[1], "subset": None,
                         "speaker_id": None, "utterance_key": None,
                         "sample_rate": int(info.samplerate), "frames": int(info.frames),
                         "channels": int(info.channels), "sample_format": info.subtype,
                         "size_bytes": path.stat().st_size, "sha256": reference["sha256"],
                         "integrity": "sha256", "catalog_id": None,
                         "derivation_id": None, "derived_from": None, "build_id": None,
                         "extra_json": {"role": "noise"}}
            sources[asset_id] = asset
        case_id = digest(document)
        case = {"case_id": case_id, "doc_kind": "realization_v1", "doc_json": document,
                "case_role": "online_snapshot", "protocol_id": protocols[0]["protocol_id"],
                "clean_asset_id": None, "noisy_asset_id": None, "noise_asset_id": None,
                "teacher_kind": "dry_clean", "sample_rate": 16000,
                "frames": document["output"]["length"], "speaker_id": item["speaker_id"],
                "utterance_key": item["utterance_id"], "transcript": None,
                "build_id": None, "gen_index": None, "content_hash": ""}
        case["content_hash"] = case_content_hash(case)
        cases.append(case)
        by_split[item["split"]].append(case_id)
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ReproductionError("duplicate SNR-shift realization")
    sets = [_case_set(split, by_split[split], metadata["source_sha256"]) for split in by_split]
    bindings = [{"slot": f"{split}.cases", "case_set_id": case_set["case_set_id"], "material_set_id": None}
                for split, case_set in zip(by_split, sets, strict=True)]
    datasets = []
    overlap = metadata["noise_overlap_audit"]["comparisons"]["train_valid"]
    note = ("Validation speakers and utterances are disjoint; train and validation noise overlaps: "
            f"{overlap['target_utterances_with_noise_overlap']} utterances, "
            f"{overlap['overlap_samples']} samples at 48000 Hz ({overlap['overlap_seconds']} seconds, "
            f"fraction {overlap['target_noise_overlap_fraction']}); validation noise is not independent; "
            "test is not used for selection.")
    case_index = {case["case_id"]: case for case in cases}
    for model, protocol in zip(MODELS, protocols, strict=True):
        spec = {"name": f"voicebank.snr_shift.{model}", "version": 1, "profile": "leakage_safe",
                "material_origin": "derived_from_official", "claim_label": "leakage_safe_comparison",
                "selection_note": note,
                "deviations": [{"id": "shared-validation-noise", "class": "algorithm_change",
                                "text": note, "ref": "configs/data/voicebank_snr_shift.yaml"}]}
        datasets.append(_dataset(spec, protocol["protocol_id"], bindings, sets, case_index))
    assets = list(sources.values())
    catalog, catalog_rows = _catalog(assets, source_kind="staging_parquet", source_db_sha256=None)
    for asset in assets:
        asset["catalog_id"] = catalog["catalog_id"]
    graph = Graph(catalogs=[catalog], catalog_rows={catalog["catalog_id"]: catalog_rows},
                  assets=assets, protocols=[parent_parent, parent, *protocols], cases=cases,
                  case_sets=sets, datasets=datasets)
    validate_graph(graph, resource_db=resource_db)
    return graph


def _protocol(model: str, parent: dict[str, Any]) -> dict[str, Any]:
    rule = {"sample_rate": 16000, "teacher": "dry_clean", "delivery": "on_demand",
            "resample": parent["rule_json"]["resample"], "epoch_unit": "utterance"}
    if model == "mp_senet":
        rule.update(normalize="noisy_rms_unit_v1", window=32000,
                    order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
                    global_batch=4, validation={"crop": "full_length", "batch": 1, "interval": 5000})
    elif model == "semambapp":
        rule.update(normalize="independent_peak_v1", window=48000,
                    order="per_rank_independent_v1", global_batch=8,
                    validation={"crop": "full_length", "batch": 1})
    else:
        rule.update(normalize=None, window=64000 if model == "tf_locoformer" else 48000,
                    order="espnet_chunk_v1", global_batch=4,
                    validation={"crop": "full_length", "batch": 1})
    validate_rule(ALGORITHM, rule)
    deviation = "algorithm_change"
    protocol_id = protocol_id_for(algorithm=ALGORITHM, algorithm_version=1, kind="fixed_pair",
                                  rule=rule, deviation_class=deviation, parent=parent["protocol_id"])
    return {"protocol_id": protocol_id, "name": f"voicebank.snr_shift.{model}.protocol", "version": 1,
            "kind": "fixed_pair", "algorithm": ALGORITHM, "algorithm_version": 1,
            "rule_json": rule, "official_ref_json": {"document": "configs/data/voicebank_snr_shift.yaml"},
            "deviation_class": deviation, "parent_protocol_id": parent["protocol_id"],
            "deviations_json": [{"id": "snr-shift", "class": deviation,
                                 "text": "Canonical VoiceBank SNR-shift waveform; not the official mixture."}]}


def _case_set(split: str, ids: list[str], source_sha: str) -> dict[str, Any]:
    name = f"voicebank.snr_shift.{split}"
    rule = {"rule": "explicit"}
    return {"case_set_id": case_set_id_for(name=name, version=1, split_role=split, rule=rule, case_ids=ids),
            "name": name, "version": 1, "split_role": split, "origin": "derived_from_official",
            "selection_rule_json": rule,
            "provenance_json": {"source": "canonical VoiceBank SNR-shift recipe", "sha256": source_sha},
            "member_count": len(ids), "list_sha256": digest(ids),
            "items": [{"position": n, "case_id": case_id} for n, case_id in enumerate(ids)]}


def main() -> int:
    import argparse

    import yaml

    from src.experiments.data.db.registry import initialize_database, read_graph, register_staging
    from src.experiments.data.db.snapshot import export_snapshot
    from src.experiments.data.db.staging import write_graph
    from src.experiments.data.prepare.voicebank_shift_recipe import read_staged_index
    from src.experiments.data.snapshot import load_snapshot

    parser = argparse.ArgumentParser(description="Register and export VoiceBank SNR-shift snapshots")
    parser.add_argument("--recipe-staging", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-db", type=Path, required=True)
    parser.add_argument("--resource-db", type=Path, required=True)
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--graph-staging", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/voicebank_snr_shift.yaml"))
    args = parser.parse_args()
    for path in (args.graph_staging, args.database, args.database.with_name(args.database.name + ".partial"), args.snapshots):
        if path.exists():
            raise ReproductionError(f"refusing existing output: {path}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.dataset_root / config["recipes_rel"]
    metadata, cases = read_staged_index(args.recipe_staging, source, config["sha256"]["recipes.jsonl"])
    graph = build_shift_graph(metadata, cases, read_graph(args.reference_db), args.corpora,
                              resource_db=str(args.resource_db))
    # Graph staging is durable and written before a new registration database.
    write_graph(args.graph_staging, graph)
    initialize_database(args.database)
    register_staging(args.graph_staging, args.database, resource_db=str(args.resource_db))
    for dataset in graph.datasets:
        value = export_snapshot(args.database, dataset["dataset_id"], args.snapshots,
                                corpora=args.corpora, derived=args.derived)
        loaded = load_snapshot(args.snapshots / dataset["dataset_id"])
        print(f"snapshot {dataset['name']} dataset_id={dataset['dataset_id']} hash={value} "
              f"cases={len(loaded.cases)} selection_uses_test={loaded.dataset['selection_uses_test']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
