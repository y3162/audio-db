"""Plan and register DNS-Challenge conditions for TF-Locoformer.

The 3000 h recipe and the short trainer plan share one source catalog and one
material pool. Mixture wavs are not written. ``plan`` only builds the
conditions. ``register`` audits a sample in memory, then appends the graph.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

from src.data.recipe import sha256_file
from src.reproduction.build.dns2020 import (
    AUDIO_SECONDS,
    LEVEL_LOWER,
    LEVEL_UPPER,
    SAMPLE_RATE,
    SNR_LOWER,
    SNR_UPPER,
    build_definition,
    num_files_for_hours,
    order_sha256,
    plan_accepted_realizations,
    plan_realizations,
    reader_id,
    shuffle_pools,
)
from src.reproduction.db.graph import index_rows
from src.reproduction.db.registry import register_staging
from src.reproduction.db.snapshot import export_snapshot
from src.reproduction.db.staging import write_graph
from src.reproduction.db.validate import leak_report
from src.reproduction.errors import ReproductionError
from src.reproduction.paths import (
    DEFAULT_CORPORA,
    DEFAULT_DB,
    DEFAULT_GENERATED,
    DEFAULT_RESOURCE_DB,
    DEFAULT_SNAPSHOTS,
    DEFAULT_STAGING,
    ROOT,
)
from src.reproduction.prepare.dns2020 import (
    accept_protocol_pair,
    assemble_accept_graph,
    assemble_graph,
    asset_row,
    audit_sample,
    material_set,
    pair_test_files,
    paths_by_name,
    recorded_pair,
    protocol_pair,
)

RECIPE_FILES = num_files_for_hours(3000.0, AUDIO_SECONDS)


def load_build_config(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ReproductionError(f"{path} must be a mapping")
    return document


def run_config(
    config_path: Path,
    step: str,
    *,
    corpus: Path = DEFAULT_CORPORA,
    staging: Path = DEFAULT_STAGING,
    database: Path = DEFAULT_DB,
    snapshots: Path = DEFAULT_SNAPSHOTS,
    resource_db: Path = DEFAULT_RESOURCE_DB,
    generated: Path = DEFAULT_GENERATED,
) -> dict[str, Any]:
    if step not in ("plan", "register"):
        raise ReproductionError("step must be plan or register; stored wav generation is disabled")
    config = load_build_config(config_path)
    if config.get("acceptance"):
        return run_accept_config(
            config,
            step,
            corpus=corpus,
            staging=staging,
            database=database,
            snapshots=snapshots,
            resource_db=resource_db,
            generated=generated,
        )
    _check_config(config)
    if config.get("delivery") != "on_demand":
        raise ReproductionError("stored wav generation is disabled; set delivery: on_demand")
    started = time.perf_counter()
    listed = _list_sources(config, corpus, resource_db)
    print(
        f"listed clean={len(listed['clean'])} noise={len(listed['noise'])} "
        f"excluded_noise={listed['excluded_noise']} in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    hashed = time.perf_counter()
    cache = staging / "workspaces" / "dns2020" / "hashes.jsonl"
    digest_of = _hashes(
        [row["path"] for row in listed["clean"] + listed["noise"] + listed["test_clean"] + listed["test_noisy"]],
        cache=cache,
        workers=int(config.get("workers") or 8),
    )
    hash_seconds = time.perf_counter() - hashed
    print(f"hashed {len(digest_of)} files in {hash_seconds:.1f}s", flush=True)
    _spot_check(listed["clean"] + listed["noise"] + listed["test_clean"] + listed["test_noisy"])
    clean_names, noise_names = shuffle_pools(
        [row["name"] for row in listed["clean"]],
        [row["name"] for row in listed["noise"]],
        int(config["seed"]),
    )
    clean_by_name = {row["name"]: row for row in listed["clean"]}
    noise_by_name = {row["name"]: row for row in listed["noise"]}
    clean_assets = [
        _asset(clean_by_name[name], corpus, digest_of, speaker=reader_id(name)) for name in clean_names
    ]
    noise_assets = [_asset(noise_by_name[name], corpus, digest_of, speaker=None) for name in noise_names]
    test_clean = [_asset(row, corpus, digest_of, speaker=None) for row in listed["test_clean"]]
    test_noisy = [_asset(row, corpus, digest_of, speaker=None) for row in listed["test_noisy"]]
    speech = material_set(
        clean_assets,
        name="dns.tflocoformer.speech",
        role="speech",
        note="16 kHz clean files, sorted then shuffled with random.Random(seed)",
    )
    noise = material_set(
        noise_assets,
        name="dns.tflocoformer.noise",
        role="noise",
        note="16 kHz mono noise only. Other rates are excluded, not resampled.",
    )
    inputs = {"speech": speech["material_set_id"], "noise": noise["material_set_id"]}
    order = order_sha256(clean_names, noise_names)
    common = {
        "seed": int(config["seed"]),
        "clean_count": len(clean_names),
        "noise_count": len(noise_names),
        "order_sha256": order,
        "inputs": inputs,
        "shard_size": int(config["shard_size"]),
        "audio_seconds": float(config.get("audio_seconds") or AUDIO_SECONDS),
        "excluded_non_16k_noise": int(listed["excluded_noise"]),
    }
    recipe = build_definition(num_files=int(listed["num_files"]), total_hours=listed["total_hours"], **common)
    smoke = build_definition(num_files=int(config["smoke_num_files"]), total_hours=None, **common)
    clean_frames = {row["name"]: int(row["frames"]) for row in listed["clean"]}
    noise_frames = {row["name"]: int(row["frames"]) for row in listed["noise"]}
    planned = time.perf_counter()
    recipe_docs = plan_realizations(recipe, clean_names, clean_frames, noise_names, noise_frames)
    smoke_docs = plan_realizations(smoke, clean_names, clean_frames, noise_names, noise_frames)
    _same_draws(smoke_docs, recipe_docs)
    plan_seconds = time.perf_counter() - planned
    histogram = _histogram(recipe_docs)
    parent, child = protocol_pair()
    test_cases = [
        recorded_pair(clean, noisy, child["protocol_id"])
        for clean, noisy in pair_test_files(test_clean, test_noisy)
    ]
    summary = {
        "recipe_name": config["name"],
        "smoke_name": config["smoke_name"],
        "recipe_files": len(recipe_docs),
        "smoke_files": len(smoke_docs),
        "recipe_build_id": recipe["build_id"],
        "smoke_build_id": smoke["build_id"],
        "clean_files": len(clean_names),
        "noise_files": len(noise_names),
        "excluded_noise": listed["excluded_noise"],
        "test_pairs": len(test_cases),
        "other_counts": listed["other_counts"],
        "hash_seconds": hash_seconds,
        "plan_seconds": plan_seconds,
        "snr_counts": histogram["snr"],
        "level_counts": histogram["level"],
        "protocol_id": child["protocol_id"],
        "parent_protocol_id": parent["protocol_id"],
        "wav_files": 0,
    }
    print(
        f"planned recipe={summary['recipe_files']} smoke={summary['smoke_files']} in {plan_seconds:.1f}s",
        flush=True,
    )
    workspace = staging / "workspaces" / "dns2020"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "plan-stats.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_plan(workspace / "plans" / recipe["build_id"], recipe, recipe_docs if len(recipe_docs) <= 10000 else None)
    _write_plan(workspace / "plans" / smoke["build_id"], smoke, smoke_docs)
    if step == "plan":
        return summary
    speech_paths = paths_by_name(clean_assets, corpus)
    noise_paths = paths_by_name(noise_assets, corpus)
    limit = int(config["audit_realizations"])
    checked = time.perf_counter()
    recipe_report = audit_sample(
        recipe, recipe_docs, speech=speech_paths, noise=noise_paths, render_limit=limit,
    )
    smoke_report = audit_sample(
        smoke, smoke_docs, speech=speech_paths, noise=noise_paths, render_limit=min(limit, len(smoke_docs)),
    )
    summary["audit_seconds"] = time.perf_counter() - checked
    summary["recipe_audit"] = _audit_public(recipe_report)
    summary["smoke_audit"] = _audit_public(smoke_report)
    print(f"audited sample in {summary['audit_seconds']:.1f}s", flush=True)
    resource = str(resource_db) if resource_db.is_file() else None
    graph = _reuse_registered_assets(
        assemble_graph(
            recipe_name=str(config["name"]),
            recipe_claim=str(config["claim_label"]),
            recipe_definition=recipe,
            recipe_documents=recipe_docs,
            recipe_report=recipe_report,
            smoke_name=str(config["smoke_name"]),
            smoke_claim=str(config.get("smoke_claim_label") or "official_recipe_scaled_down"),
            smoke_definition=smoke,
            smoke_documents=smoke_docs,
            smoke_report=smoke_report,
            speech=speech,
            noise=noise,
            source_assets=clean_assets + noise_assets + test_clean + test_noisy,
            test_cases=test_cases,
            resource_db=resource,
            excluded_noise=int(listed["excluded_noise"]),
        ),
        database,
    )
    summary["leak"] = {
        dataset["name"]: leak_report(
            dataset,
            index_rows(graph.case_sets, "case_set_id", "case_sets"),
            index_rows(graph.materials, "material_set_id", "materials"),
            index_rows(graph.cases, "case_id", "cases"),
            index_rows(graph.assets, "asset_id", "assets"),
            resource_db=resource,
        )["status"]
        for dataset in graph.datasets
    }
    registered = time.perf_counter()
    write_graph(staging, graph)
    register_staging(staging, database, resource_db=resource)
    summary["register_seconds"] = time.perf_counter() - registered
    snapped = time.perf_counter()
    hashes = {}
    for dataset in graph.datasets:
        label = f"{dataset['name']}@{dataset['version']}"
        hashes[label] = export_snapshot(
            database,
            label,
            snapshots,
            corpora=corpus,
            generated=generated,
            derived=ROOT / "data" / "derived",
        )
    summary["snapshot_seconds"] = time.perf_counter() - snapped
    summary["snapshot_hash"] = hashes
    summary["dataset_id"] = {dataset["name"]: dataset["dataset_id"] for dataset in graph.datasets}
    summary["selection_uses_test"] = {dataset["name"]: dataset["selection_uses_test"] for dataset in graph.datasets}
    (workspace / "register-stats.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def run_accept_config(
    config: dict[str, Any],
    step: str,
    *,
    corpus: Path,
    staging: Path,
    database: Path,
    snapshots: Path,
    resource_db: Path,
    generated: Path,
) -> dict[str, Any]:
    """Register one small accepted dataset. The P7 datasets are not rebuilt."""
    if step not in ("plan", "register"):
        raise ReproductionError("step must be plan or register; stored wav generation is disabled")
    _check_accept_config(config)
    if config.get("delivery") != "on_demand":
        raise ReproductionError("stored wav generation is disabled; set delivery: on_demand")
    listed = _list_accept_sources(config, corpus, resource_db)
    cache_path = staging / "workspaces" / "dns2020-accept" / "hashes.jsonl"
    digest_of = _hashes(
        [row["path"] for row in listed["clean"] + listed["noise"] + listed["test_clean"] + listed["test_noisy"]],
        cache=cache_path,
        workers=int(config.get("workers") or 4),
    )
    _spot_check(listed["clean"] + listed["noise"] + listed["test_clean"] + listed["test_noisy"])
    clean_names, noise_names = shuffle_pools(
        [row["name"] for row in listed["clean"]],
        [row["name"] for row in listed["noise"]],
        int(config["seed"]),
    )
    clean_by_name = {row["name"]: row for row in listed["clean"]}
    noise_by_name = {row["name"]: row for row in listed["noise"]}
    clean_assets = [_asset(clean_by_name[name], corpus, digest_of, speaker=reader_id(name)) for name in clean_names]
    noise_assets = [_asset(noise_by_name[name], corpus, digest_of, speaker=None) for name in noise_names]
    test_clean = [_asset(row, corpus, digest_of, speaker=None) for row in listed["test_clean"]]
    test_noisy = [_asset(row, corpus, digest_of, speaker=None) for row in listed["test_noisy"]]
    speech = material_set(
        clean_assets,
        name="dns.tflocoformer.accept.speech",
        role="speech",
        note="16 kHz clean files used by the P7.1 acceptance sample",
    )
    noise = material_set(
        noise_assets,
        name="dns.tflocoformer.accept.noise",
        role="noise",
        note="16 kHz noise plus non-16 kHz files that are resampled while rendering. Source files are not rewritten.",
    )
    definition = build_definition(
        seed=int(config["seed"]),
        num_files=int(listed["num_files"]),
        clean_count=len(clean_names),
        noise_count=len(noise_names),
        order_sha256=order_sha256(clean_names, noise_names),
        inputs={"speech": speech["material_set_id"], "noise": noise["material_set_id"]},
        shard_size=int(config["shard_size"]),
        audio_seconds=float(config["audio_seconds"]),
        total_hours=None,
        excluded_non_16k_noise=int(listed["excluded_noise"]),
        acceptance=str(config["acceptance"]),
    )
    clean_paths = {row["name"]: row["path"] for row in listed["clean"]}
    noise_paths = {row["name"]: row["path"] for row in listed["noise"]}
    started = time.perf_counter()
    documents, totals = plan_accepted_realizations(
        definition,
        clean_names,
        clean_paths,
        noise_names,
        noise_paths,
        workers=int(config.get("workers") or 1),
    )
    plan_seconds = time.perf_counter() - started
    parent, child = accept_protocol_pair()
    summary = {
        "recipe_name": config["name"],
        "recipe_files": len(documents),
        "recipe_build_id": definition["build_id"],
        "clean_files": len(clean_names),
        "noise_files": len(noise_names),
        "non_16k_noise": int(listed["non_16k_noise"]),
        "excluded_noise": int(listed["excluded_noise"]),
        "protocol_id": child["protocol_id"],
        "parent_protocol_id": parent["protocol_id"],
        "renderer": child["rule_json"]["renderer"],
        "retry_policy": child["rule_json"]["retry_policy"],
        "plan_seconds": plan_seconds,
        "acceptance": _attempt_summary(totals),
        "wav_files": 0,
        "validation": "sorted_uttid_tail_tenth_not_cv_synthetic_small",
    }
    workspace = staging / "workspaces" / "dns2020-accept"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "plan-stats.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if step == "plan":
        return summary
    speech_paths = paths_by_name(clean_assets, corpus)
    noise_path_map = paths_by_name(noise_assets, corpus)
    report = audit_sample(
        definition,
        documents,
        speech=speech_paths,
        noise=noise_path_map,
        render_limit=min(int(config.get("audit_realizations") or len(documents)), len(documents)),
    )
    if report["clean_activity_rejected"] or report["post_mix_clipped"] or report["source_clipped_pieces"]:
        raise ReproductionError(f"accepted DNS sample still fails an official check: {report}")
    test_cases = [
        recorded_pair(clean, noisy, child["protocol_id"])
        for clean, noisy in pair_test_files(test_clean, test_noisy)
    ]
    resource = str(resource_db) if resource_db.is_file() else None
    graph = assemble_accept_graph(
        name=str(config["name"]),
        claim=str(config["claim_label"]),
        definition=definition,
        documents=documents,
        report=report,
        speech=speech,
        noise=noise,
        source_assets=clean_assets + noise_assets + test_clean + test_noisy,
        test_cases=test_cases,
        resource_db=resource,
    )
    summary["dataset_id"] = graph.datasets[0]["dataset_id"]
    summary["selection_uses_test"] = graph.datasets[0]["selection_uses_test"]
    summary["audit"] = _audit_public(report)
    write_graph(staging, graph)
    register_staging(staging, database, resource_db=resource)
    from src.reproduction.db.snapshot import export_snapshot

    summary["snapshot_hash"] = export_snapshot(
        database,
        f"{config['name']}@1",
        snapshots,
        allow_unknown_leak_check=True,
        corpora=corpus,
        generated=generated,
        derived=ROOT / "data" / "derived",
    )
    return summary


def _check_accept_config(config: dict[str, Any]) -> None:
    for key in ("name", "seed", "shard_size", "num_files", "clean_limit", "noise_limit", "audio_seconds"):
        if key not in config:
            raise ReproductionError(f"DNS acceptance config needs {key}")
    if str(config.get("acceptance")) != "deterministic_case_stream_v1":
        raise ReproductionError("acceptance must be deterministic_case_stream_v1")
    if config.get("source_scan") not in ("limited", "catalog"):
        raise ReproductionError("acceptance source_scan must be limited or catalog")
    if int(config["num_files"]) < 2 or int(config["clean_limit"]) < 1 or int(config["noise_limit"]) < 1:
        raise ReproductionError("acceptance needs at least two mixtures and one clean and noise file")
    if float(config["audio_seconds"]) <= 0.1 or float(config["audio_seconds"]) >= 31:
        raise ReproductionError("acceptance audio_seconds must survive the 0.1 s to 31 s duration filter")
    if str(config.get("claim_label") or "") != "official_recipe_scaled_down":
        raise ReproductionError("the acceptance sample uses official_recipe_scaled_down")
    if not config.get("include_test", True):
        raise ReproductionError("DNS datasets require the official no_reverb test slot")
    if int(config.get("include_non_16k_noise") or 0) < 0:
        raise ReproductionError("include_non_16k_noise must be non-negative")


def _list_accept_sources(config: dict[str, Any], corpus: Path, resource_db: Path) -> dict[str, Any]:
    clean_limit = int(config["clean_limit"])
    noise_limit = int(config["noise_limit"])
    extra = int(config.get("include_non_16k_noise") or 0)
    if config.get("source_scan") == "catalog":
        rows, _other = _catalog_rows(resource_db, corpus)
        clean = sorted((row for row in rows if row["subset"] == "clean"), key=lambda row: row["name"])[:clean_limit]
        noise = sorted((row for row in rows if row["subset"] == "noise"), key=lambda row: row["name"])[:noise_limit]
        non16 = []
        excluded = 0
        excluded_rows = sorted((row for row in rows if row["subset"] == "noise_excluded"), key=lambda row: row["name"])
        for row in excluded_rows:
            if int(row["channels"]) != 1:
                excluded += 1
                continue
            if len(non16) < extra:
                kept = dict(row)
                kept["subset"] = "noise"
                non16.append(kept)
            else:
                excluded += 1
        test_clean = [row for row in rows if row["subset"] == "test_no_reverb_clean"]
        test_noisy = [row for row in rows if row["subset"] == "test_no_reverb_noisy"]
    else:
        clean, noise, non16, excluded, test_clean, test_noisy = _accept_directory(
            corpus, clean_limit, noise_limit, extra
        )
    if len(clean) < clean_limit or len(noise) < noise_limit:
        raise ReproductionError("acceptance scan did not find the requested clean and 16 kHz noise files")
    if extra and len(non16) < extra:
        raise ReproductionError(f"acceptance scan found {len(non16)} non-16 kHz noise files, needed {extra}")
    return {
        "clean": clean,
        "noise": noise + non16,
        "excluded_noise": excluded,
        "non_16k_noise": len(non16),
        "test_clean": test_clean,
        "test_noisy": test_noisy,
        "num_files": int(config["num_files"]),
    }


def _accept_directory(
    corpus: Path, clean_limit: int, noise_limit: int, extra: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]], list[dict[str, Any]]]:
    root = corpus / "DNS2020" / "datasets"
    clean = _take_rate(root / "clean", "clean", clean_limit, 16000)
    noise = _take_rate(root / "noise", "noise", noise_limit, 16000)
    non16: list[dict[str, Any]] = []
    excluded = 0
    for path in sorted(item for item in (root / "noise").glob("*.wav") if item.is_file()):
        info = sf.info(path)
        if int(info.channels) != 1:
            excluded += 1
            continue
        if int(info.samplerate) == SAMPLE_RATE:
            continue
        if len(non16) >= extra:
            excluded += 1
            continue
        non16.append(_record(path, "noise", info))
    test_clean = _take(root / "test_set" / "synthetic" / "no_reverb" / "clean", "test_no_reverb_clean", None)
    test_noisy = _take(root / "test_set" / "synthetic" / "no_reverb" / "noisy", "test_no_reverb_noisy", None)
    return clean, noise, non16, excluded, test_clean, test_noisy


def _take_rate(directory: Path, subset: str, limit: int, rate: int) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise ReproductionError(f"DNS directory is missing: {directory}")
    rows = []
    for path in sorted(item for item in directory.glob("*.wav") if item.is_file()):
        info = sf.info(path)
        if int(info.samplerate) != rate or int(info.channels) != 1:
            continue
        rows.append(_record(path, subset, info))
        if len(rows) >= limit:
            break
    return rows


def _attempt_summary(totals: dict[str, Any]) -> dict[str, Any]:
    attempts = [int(value) for value in totals["outer_attempts"]]
    histogram: dict[str, int] = {}
    for value in attempts:
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    return {
        "cases": int(totals["cases"]),
        "failed_cases": int(totals["failed_cases"]),
        "activity_rejects": int(totals["activity_rejects"]),
        "post_clip_rejects": int(totals["post_clip_rejects"]),
        "short_rejects": int(totals["short_rejects"]),
        "clean_clip_skips": int(totals["clean_clip_skips"]),
        "noise_clip_skips": int(totals["noise_clip_skips"]),
        "resampled_noise_pieces": int(totals["resampled_noise_pieces"]),
        "attempt_histogram": histogram,
        "max_attempts": max(attempts) if attempts else 0,
    }


def _check_config(config: dict[str, Any]) -> None:
    for key in ("name", "smoke_name", "seed", "shard_size", "smoke_num_files", "audit_realizations"):
        if key not in config:
            raise ReproductionError(f"DNS config needs {key}")
    if int(config["smoke_num_files"]) < 2:
        raise ReproductionError("smoke_num_files must leave both a train and a valid mixture")
    if int(config["shard_size"]) < 1 or int(config["audit_realizations"]) < 1:
        raise ReproductionError("shard_size and audit_realizations must be positive")
    scan = config.get("source_scan")
    claim = str(config.get("claim_label") or "")
    if scan == "full":
        if config.get("clean_limit") is not None or config.get("noise_limit") is not None:
            raise ReproductionError("source_scan full does not take clean_limit or noise_limit")
        if claim != "espnet_public_se_recipe_reproduction":
            raise ReproductionError("the full DNS scan uses claim_label espnet_public_se_recipe_reproduction")
        if float(config.get("total_hours")) != 3000 or float(config.get("audio_seconds") or AUDIO_SECONDS) != AUDIO_SECONDS:
            raise ReproductionError("the full DNS recipe is 3000 hours of 30-second mixtures")
        for key, expected in (
            ("snr_lower", SNR_LOWER),
            ("snr_upper", SNR_UPPER),
            ("target_level_lower", LEVEL_LOWER),
            ("target_level_upper", LEVEL_UPPER),
        ):
            if key in config and int(config[key]) != expected:
                raise ReproductionError(f"{key} must be the TF-Locoformer DNS bound {expected}")
    elif scan == "limited":
        if claim == "espnet_public_se_recipe_reproduction":
            raise ReproductionError("a limited DNS scan cannot use the full ESPnet recipe claim")
        if config.get("clean_limit") is None or config.get("noise_limit") is None or config.get("num_files") is None:
            raise ReproductionError("a limited scan needs clean_limit, noise_limit, and num_files")
        if str(config.get("claim_label") or "official_recipe_scaled_down") != "official_recipe_scaled_down":
            raise ReproductionError("a limited DNS scan uses official_recipe_scaled_down")
        config["claim_label"] = "official_recipe_scaled_down"
    else:
        raise ReproductionError("source_scan must be full or limited")
    if not config.get("include_test", True):
        raise ReproductionError("DNS datasets require the official no_reverb test slot")
    if "silence_seconds" in config and float(config["silence_seconds"]) != 0.2:
        raise ReproductionError("DNS silence is 0.2 seconds")


def _list_sources(config: dict[str, Any], corpus: Path, resource_db: Path) -> dict[str, Any]:
    if config.get("source_scan") == "full":
        rows, other = _catalog_rows(resource_db, corpus)
        num_files = RECIPE_FILES
        total_hours = 3000.0
    else:
        rows = _scan_rows(corpus, clean_limit=int(config["clean_limit"]), noise_limit=int(config["noise_limit"]))
        other = {}
        num_files = int(config["num_files"])
        total_hours = None
    clean = [row for row in rows if row["subset"] == "clean"]
    noise = [row for row in rows if row["subset"] == "noise"]
    excluded = [row for row in rows if row["subset"] == "noise_excluded"]
    test_clean = [row for row in rows if row["subset"] == "test_no_reverb_clean"]
    test_noisy = [row for row in rows if row["subset"] == "test_no_reverb_noisy"]
    if len(clean) < 1 or len(noise) < 1 or len(test_clean) < 1:
        raise ReproductionError("DNS plan needs clean, 16 kHz noise, and no_reverb test files")
    if num_files < 2:
        raise ReproductionError("num_files must be at least 2")
    return {
        "clean": clean,
        "noise": noise,
        "excluded_noise": len(excluded),
        "test_clean": test_clean,
        "test_noisy": test_noisy,
        "num_files": num_files,
        "total_hours": total_hours,
        "other_counts": other,
    }


def _catalog_rows(resource_db: Path, corpus: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not resource_db.is_file():
        raise ReproductionError(f"resource catalog is missing: {resource_db}")
    import duckdb

    connection = duckdb.connect(str(resource_db), read_only=True)
    try:
        fetched = connection.execute(
            """
            SELECT path, sample_rate, frames, channels
            FROM corpora.audio_files
            WHERE path LIKE 'DNS2020/datasets/%'
            """
        ).fetchall()
    finally:
        connection.close()
    rows: list[dict[str, Any]] = []
    other: Counter[str] = Counter()
    seen: dict[str, set[str]] = {"clean": set(), "noise": set()}
    for path, rate, frames, channels in fetched:
        text = str(path)
        parts = text.split("/")
        name = parts[-1]
        if not name.endswith(".wav"):
            continue
        kind = _kind(parts)
        other[kind or "other"] += 1
        if kind not in ("clean", "noise", "test_no_reverb_clean", "test_no_reverb_noisy"):
            continue
        full = corpus / text
        if not full.is_file():
            raise ReproductionError(f"catalog file is missing: {full}")
        record = {
            "name": name,
            "path": full,
            "rel": text,
            "subset": kind,
            "sample_rate": int(rate),
            "frames": int(frames),
            "channels": int(channels),
        }
        if kind == "noise" and (int(rate) != SAMPLE_RATE or int(channels) != 1):
            record["subset"] = "noise_excluded"
        elif kind == "clean" and (int(rate) != SAMPLE_RATE or int(channels) != 1 or int(frames) < 1):
            raise ReproductionError(f"clean file {text} is not 16 kHz mono")
        elif kind.startswith("test_") and (int(rate) != SAMPLE_RATE or int(channels) != 1):
            raise ReproductionError(f"test file {text} is not 16 kHz mono")
        if kind in seen:
            if name in seen[kind]:
                raise ReproductionError(f"duplicate {kind} basename {name}")
            seen[kind].add(name)
        rows.append(record)
    if not rows:
        raise ReproductionError("resource.duckdb has no DNS2020 dataset rows")
    return rows, dict(other)


def _kind(parts: list[str]) -> str | None:
    if parts[:3] == ["DNS2020", "datasets", "clean"] and len(parts) == 4:
        return "clean"
    if parts[:3] == ["DNS2020", "datasets", "noise"] and len(parts) == 4:
        return "noise"
    if parts[:6] == ["DNS2020", "datasets", "test_set", "synthetic", "no_reverb", "clean"] and len(parts) == 7:
        return "test_no_reverb_clean"
    if parts[:6] == ["DNS2020", "datasets", "test_set", "synthetic", "no_reverb", "noisy"] and len(parts) == 7:
        return "test_no_reverb_noisy"
    if len(parts) >= 4 and parts[:3] == ["DNS2020", "datasets", "test_set"]:
        return "test_other"
    if len(parts) >= 3 and parts[2] == "blind_test_set":
        return "blind_test"
    return "other"


def _scan_rows(corpus: Path, *, clean_limit: int, noise_limit: int) -> list[dict[str, Any]]:
    root = corpus / "DNS2020" / "datasets"
    rows = []
    rows.extend(_take(root / "clean", "clean", clean_limit))
    rows.extend(_take(root / "noise", "noise", noise_limit))
    rows.extend(_take(root / "test_set" / "synthetic" / "no_reverb" / "clean", "test_no_reverb_clean", None))
    rows.extend(_take(root / "test_set" / "synthetic" / "no_reverb" / "noisy", "test_no_reverb_noisy", None))
    return rows


def _take(directory: Path, subset: str, limit: int | None) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise ReproductionError(f"DNS directory is missing: {directory}")
    rows = []
    for path in sorted(item for item in directory.glob("*.wav") if item.is_file()):
        info = sf.info(path)
        if int(info.samplerate) != SAMPLE_RATE or int(info.channels) != 1:
            if subset == "noise":
                rows.append(_record(path, "noise_excluded", info))
                continue
            raise ReproductionError(f"{path} is not 16 kHz mono")
        rows.append(_record(path, subset, info))
        kept = [row for row in rows if row["subset"] == subset]
        if limit is not None and len(kept) >= limit:
            break
    return [row for row in rows if row["subset"] == subset or row["subset"] == "noise_excluded"]


def _record(path: Path, subset: str, info: Any) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": path,
        "rel": None,
        "subset": subset,
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
    }


def _spot_check(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    picks = [rows[0], rows[len(rows) // 2], rows[-1]]
    for row in picks:
        info = sf.info(row["path"])
        if int(info.samplerate) != int(row["sample_rate"]) or int(info.frames) != int(row["frames"]):
            raise ReproductionError(f"catalog header does not match {row['path']}")
        if int(info.channels) != int(row["channels"]):
            raise ReproductionError(f"catalog channels do not match {row['path']}")
        if info.subtype not in (None, "", "PCM_16"):
            raise ReproductionError(f"{row['path']} subtype is {info.subtype}, expected PCM_16")


def _hashes(paths: list[Path], *, cache: Path, workers: int) -> dict[Path, str]:
    known = _read_cache(cache)
    missing = [path for path in paths if (str(path), path.stat().st_size) not in known]
    found: dict[tuple[str, int], str] = dict(known)
    if missing:
        workers = max(1, workers)

        def one(path: Path) -> tuple[Path, int, str]:
            return path, path.stat().st_size, sha256_file(path)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, size, digest in pool.map(one, missing):
                found[(str(path), size)] = digest
                done += 1
                if done % 5000 == 0 or done == len(missing):
                    print(f"hashed\t{done}\t{len(missing)}", flush=True)
        _write_cache(cache, found)
    digest_of = {}
    for path in paths:
        digest_of[path] = found[(str(path), path.stat().st_size)]
    return digest_of


def _read_cache(path: Path) -> dict[tuple[str, int], str]:
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        found[(str(row["path"]), int(row["size"]))] = str(row["sha256"])
    return found


def _write_cache(path: Path, rows: dict[tuple[str, int], str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"path": key[0], "size": key[1], "sha256": digest}, sort_keys=True)
        for key, digest in sorted(rows.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _asset(row: dict[str, Any], corpus: Path, digest_of: dict[Path, str], *, speaker: str | None) -> dict[str, Any]:
    return asset_row(
        row["path"],
        corpus_root=corpus,
        subset=row["subset"],
        rate=int(row["sample_rate"]),
        frames=int(row["frames"]),
        channels=int(row["channels"]),
        sha256=digest_of[row["path"]],
        speaker_id=speaker,
    )


def _same_draws(smoke: list[dict[str, Any]], recipe: list[dict[str, Any]]) -> None:
    for left, right in zip(smoke, recipe):
        if left["sources"] != right["sources"]:
            raise ReproductionError("smoke and recipe spans diverged for the same gen_index")
        for key in ("gen_index", "snr_db", "target_level_dbfs", "utt_id"):
            if left["draws"][key] != right["draws"][key]:
                raise ReproductionError(f"smoke and recipe {key} diverged")
        if left["draws"]["build_id"] == right["draws"]["build_id"]:
            raise ReproductionError("smoke and recipe unexpectedly share a build id")


def _histogram(documents: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    snr: Counter[str] = Counter()
    level: Counter[str] = Counter()
    for document in documents:
        snr[str(int(document["draws"]["snr_db"]))] += 1
        level[str(int(document["draws"]["target_level_dbfs"]))] += 1
        value = int(document["draws"]["snr_db"])
        if value < SNR_LOWER or value >= SNR_UPPER:
            raise ReproductionError(f"SNR {value} is outside the exclusive DNS upper bound")
        level_value = int(document["draws"]["target_level_dbfs"])
        if level_value < LEVEL_LOWER or level_value >= LEVEL_UPPER:
            raise ReproductionError(f"target level {level_value} is outside the exclusive DNS upper bound")
    return {"snr": dict(sorted(snr.items(), key=lambda item: int(item[0]))), "level": dict(sorted(level.items(), key=lambda item: int(item[0])))}


def _write_plan(directory: Path, definition: dict[str, Any], documents: list[dict[str, Any]] | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "build.json").write_text(json.dumps(definition, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if documents is None:
        (directory / "plan.jsonl").write_text(
            json.dumps({"omitted": True, "reason": "full conditions are the registered realization rows"}) + "\n",
            encoding="utf-8",
        )
        return
    lines = "".join(json.dumps(item, sort_keys=True) + "\n" for item in documents)
    (directory / "plan.jsonl").write_text(lines, encoding="utf-8")


def _audit_public(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "rendered",
        "realizations",
        "render_scope",
        "clean_activity_rejected",
        "source_clipped_pieces",
        "post_mix_clipped",
        "max_snr_error_db",
        "wav_files",
    )
    return {key: report[key] for key in keys}


def _reuse_registered_assets(graph, database: Path):
    """Keep corpus rows already stored when the file bytes still match."""
    if not database.is_file():
        return graph
    from src.reproduction.db.registry import read_graph

    existing = {row["asset_id"]: row for row in read_graph(database).assets}
    if not existing:
        return graph
    aligned = []
    for asset in graph.assets:
        previous = existing.get(asset["asset_id"])
        if previous is None:
            aligned.append(asset)
            continue
        for key in ("rel_path", "sha256", "frames", "size_bytes", "sample_rate"):
            if asset.get(key) != previous.get(key):
                raise ReproductionError(f"corpus asset {asset['asset_id']} does not match the registered {key}")
        aligned.append(previous)
    graph.assets = aligned
    return graph
