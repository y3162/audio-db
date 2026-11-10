"""Plan and register DNS-Challenge conditions for TF-Locoformer.

The 3000 h recipe and the short trainer plan share one source catalog and one
material pool. Mixture wavs are not written. ``plan`` only builds the
conditions. ``register`` audits a sample in memory, then appends the graph.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import resource
import socket
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from src.data.recipe import sha256_file
from src.experiments.data.build.dns2020 import (
    AUDIO_SECONDS,
    LEVEL_LOWER,
    LEVEL_UPPER,
    SAMPLE_RATE,
    SNR_LOWER,
    SNR_UPPER,
    _stored,
    activitydetector,
    build_definition,
    is_clipped,
    num_files_for_hours,
    order_sha256,
    plan_accepted_realizations,
    plan_accepted_span,
    plan_realizations,
    profile_acceptance,
    split_train_valid,
    reader_id,
    render_realization,
    resample_to_16k,
    shuffle_pools,
    snr_mixer,
)
from src.experiments.data.ids import digest
from src.experiments.data.db.graph import index_rows
from src.experiments.data.db.registry import register_staging
from src.experiments.data.db.snapshot import export_snapshot
from src.experiments.data.db.staging import write_graph
from src.experiments.data.db.validate import leak_report
from src.experiments.data.errors import ReproductionError
from src.experiments.data.paths import (
    DEFAULT_CORPORA,
    DEFAULT_DB,
    DEFAULT_GENERATED,
    DEFAULT_RESOURCE_DB,
    DEFAULT_SNAPSHOTS,
    DEFAULT_STAGING,
    ROOT,
)
from src.experiments.data.prepare.dns2020 import (
    accept_protocol_pair,
    assemble_accept_graph,
    assemble_graph,
    asset_row,
    audit_sample,
    material_set,
    measure_realization,
    pair_test_files,
    paths_by_name,
    recorded_pair,
    protocol_pair,
)
from src.experiments.data.prepare.dns_accept_store import (
    generator_code_identity,
    load_documents,
    missing_ranges,
    plan_identity,
    summarize_cases,
    write_shard,
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
    if step not in ("plan", "register", "benchmark"):
        raise ReproductionError("step must be plan, register, or benchmark; stored wav generation is disabled")
    config = load_build_config(config_path)
    if config.get("acceptance") and config.get("source_scan") == "full":
        return run_accept_recipe_config(
            config,
            step,
            corpus=corpus,
            staging=staging,
            database=database,
            snapshots=snapshots,
            resource_db=resource_db,
            generated=generated,
        )
    if step == "benchmark":
        raise ReproductionError("benchmark is only implemented for the full DNS acceptance recipe")
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
    from src.experiments.data.db.snapshot import export_snapshot

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


ACCEPT_RECIPE_NAME = "tf_locoformer.dns.accept_recipe"
ACCEPT_SPEECH_NAME = "dns.tflocoformer.accept_recipe.speech"
ACCEPT_NOISE_NAME = "dns.tflocoformer.accept_recipe.noise"
P7_TEST_SET_NAME = "dns.tflocoformer.no_reverb"
_SHARD_STATE: dict[str, Any] = {}


def run_accept_recipe_config(
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
    """Plan or register the 3000 h acceptance dataset. Existing datasets are not replaced."""
    if step not in ("plan", "register", "benchmark"):
        raise ReproductionError("step must be plan, register, or benchmark; stored wav generation is disabled")
    _check_accept_recipe_config(config)
    if config.get("delivery") != "on_demand":
        raise ReproductionError("stored wav generation is disabled; set delivery: on_demand")
    prepared = _prepare_accept_recipe(
        config,
        corpus=corpus,
        staging=staging,
        resource_db=resource_db,
    )
    code = generator_code_identity()
    parent, child = accept_protocol_pair()
    identity = plan_identity(
        prepared["definition"],
        protocol_id=child["protocol_id"],
        plan_shard=int(config["plan_shard"]),
        code_sha256=str(code["generator_code_sha256"]),
    )
    workspace = staging / "workspaces" / "dns2020-accept-recipe" / identity["build_id"] / identity["generator_code_sha256"][:12]
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "code-identity.json").write_text(
        json.dumps({**code, "identity": identity, "packages": _package_versions()}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if step == "benchmark":
        report = _benchmark_accept_recipe(config, prepared, workspace, corpus)
        (workspace / "benchmark.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        report["workspace"] = str(workspace)
        return report
    _assert_recipe_protocols(database, parent, child)
    shard_dir = workspace / "shards"
    started = time.perf_counter()
    cpu_before = _cpu_seconds()
    read_before = _proc_read_bytes()
    missing = missing_ranges(shard_dir, identity, int(identity["num_files"]))
    print(f"acceptance shards missing={len(missing)} total_grid={int(identity['num_files'])}", flush=True)
    if missing:
        _plan_accept_shards(config, prepared, identity, shard_dir, missing)
    plan_seconds = time.perf_counter() - started
    documents, stats = load_documents(shard_dir, identity, int(identity["num_files"]))
    metadata = summarize_cases(
        documents,
        stats,
        clean_names=prepared["clean_names"],
        noise_names=prepared["noise_names"],
    )
    if int(metadata["cases"]) != RECIPE_FILES or int(metadata["failed_cases"]) != 0:
        raise ReproductionError(f"acceptance metadata audit did not accept every case: {metadata}")
    summary = {
        **prepared["summary"],
        "protocol_id": child["protocol_id"],
        "parent_protocol_id": parent["protocol_id"],
        "renderer": child["rule_json"]["renderer"],
        "retry_policy": child["rule_json"]["retry_policy"],
        "plan_seconds": plan_seconds,
        "plan_cpu_seconds": _cpu_seconds() - cpu_before,
        "plan_read_bytes": _proc_read_bytes() - read_before,
        "acceptance": metadata,
        "wav_files": 0,
        "validation": "sorted_uttid_tail_tenth_not_cv_synthetic_small",
        "generator_code_sha256": code["generator_code_sha256"],
        "git_commit": code["git_commit"],
        "worktree_dirty": code["worktree_dirty"],
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "workers": int(config["workers"]),
        "packages": _package_versions(),
    }
    (workspace / "plan-stats.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"planned {metadata['cases']} max_retries={metadata['max_retries']} non16_cases={metadata['non16_cases']} in {plan_seconds:.1f}s",
        flush=True,
    )
    if step == "plan":
        return summary
    return _register_accept_recipe(
        config,
        prepared,
        documents,
        stats,
        summary,
        workspace=workspace,
        corpus=corpus,
        staging=staging,
        database=database,
        snapshots=snapshots,
        resource_db=resource_db,
        generated=generated,
        code=code,
    )


def _check_accept_recipe_config(config: dict[str, Any]) -> None:
    for key in ("name", "seed", "shard_size", "plan_shard", "audio_seconds", "total_hours", "workers", "audit_realizations"):
        if key not in config:
            raise ReproductionError(f"DNS acceptance recipe needs {key}")
    if str(config["name"]) != ACCEPT_RECIPE_NAME:
        raise ReproductionError(f"the full acceptance dataset name is {ACCEPT_RECIPE_NAME}")
    if str(config.get("acceptance")) != "deterministic_case_stream_v1":
        raise ReproductionError("acceptance must be deterministic_case_stream_v1")
    if config.get("source_scan") != "full":
        raise ReproductionError("the 3000 h acceptance recipe uses source_scan full")
    if str(config.get("claim_label") or "") != "espnet_public_se_recipe_reproduction":
        raise ReproductionError("the 3000 h acceptance recipe uses espnet_public_se_recipe_reproduction")
    if config.get("clean_limit") is not None or config.get("noise_limit") is not None or config.get("num_files") is not None:
        raise ReproductionError("the full acceptance recipe does not take clean_limit, noise_limit, or num_files")
    if float(config["total_hours"]) != 3000 or float(config["audio_seconds"]) != AUDIO_SECONDS:
        raise ReproductionError("the full DNS acceptance recipe is 3000 hours of 30-second mixtures")
    if str(config.get("include_non_16k_noise")) != "all":
        raise ReproductionError("the full acceptance recipe includes every mono non-16 kHz noise file")
    if int(config["plan_shard"]) < 1 or int(config["workers"]) < 1 or int(config["audit_realizations"]) < 1:
        raise ReproductionError("plan_shard, workers, and audit_realizations must be positive")
    if int(config["shard_size"]) < 1:
        raise ReproductionError("shard_size must be positive")
    for key, expected in (
        ("snr_lower", SNR_LOWER),
        ("snr_upper", SNR_UPPER),
        ("target_level_lower", LEVEL_LOWER),
        ("target_level_upper", LEVEL_UPPER),
    ):
        if key in config and int(config[key]) != expected:
            raise ReproductionError(f"{key} must be the TF-Locoformer DNS bound {expected}")
    if "silence_seconds" in config and float(config["silence_seconds"]) != 0.2:
        raise ReproductionError("DNS silence is 0.2 seconds")
    if not config.get("include_test", True):
        raise ReproductionError("DNS datasets require the official no_reverb test slot")
    counts = config.get("benchmark_counts")
    if counts is not None and (not isinstance(counts, list) or any(int(item) < 2 for item in counts)):
        raise ReproductionError("benchmark_counts must be a list of spans of at least 2")


def _prepare_accept_recipe(
    config: dict[str, Any],
    *,
    corpus: Path,
    staging: Path,
    resource_db: Path,
) -> dict[str, Any]:
    listed = _list_accept_recipe_sources(resource_db, corpus)
    cache_path = staging / "workspaces" / "dns2020-accept-recipe" / "hashes.jsonl"
    hashed = time.perf_counter()
    digest_of = _hashes(
        [row["path"] for row in listed["clean"] + listed["noise"] + listed["test_clean"] + listed["test_noisy"]],
        cache=cache_path,
        workers=int(config.get("workers") or 8),
        seed_caches=(staging / "workspaces" / "dns2020" / "hashes.jsonl",),
    )
    print(f"hashed {len(digest_of)} files in {time.perf_counter() - hashed:.1f}s", flush=True)
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
        name=ACCEPT_SPEECH_NAME,
        role="speech",
        note="All 16 kHz clean files for the 3000 h acceptance recipe, sorted then shuffled with random.Random(seed).",
    )
    noise = material_set(
        noise_assets,
        name=ACCEPT_NOISE_NAME,
        role="noise",
        note=(
            "All mono noise files, including non-16 kHz files. Those are resampled at plan and "
            "render time with librosa 1.0.0 soxr_hq. Source files are not rewritten."
        ),
    )
    definition = build_definition(
        seed=int(config["seed"]),
        num_files=RECIPE_FILES,
        clean_count=len(clean_names),
        noise_count=len(noise_names),
        order_sha256=order_sha256(clean_names, noise_names),
        inputs={"speech": speech["material_set_id"], "noise": noise["material_set_id"]},
        shard_size=int(config["shard_size"]),
        audio_seconds=AUDIO_SECONDS,
        total_hours=3000.0,
        excluded_non_16k_noise=int(listed["excluded_noise"]),
        acceptance=str(config["acceptance"]),
    )
    return {
        "definition": definition,
        "clean_names": clean_names,
        "noise_names": noise_names,
        "clean_paths": {row["name"]: row["path"] for row in listed["clean"]},
        "noise_paths": {row["name"]: row["path"] for row in listed["noise"]},
        "clean_assets": clean_assets,
        "noise_assets": noise_assets,
        "test_clean": test_clean,
        "test_noisy": test_noisy,
        "speech": speech,
        "noise": noise,
        "listed": listed,
        "summary": {
            "recipe_name": ACCEPT_RECIPE_NAME,
            "recipe_files": RECIPE_FILES,
            "recipe_build_id": definition["build_id"],
            "clean_files": len(clean_names),
            "noise_files": len(noise_names),
            "noise_16k": int(listed["noise_16k"]),
            "non_16k_noise": int(listed["non_16k_noise"]),
            "non_16k_rates": listed["non_16k_rates"],
            "excluded_noise": int(listed["excluded_noise"]),
            "test_pairs": len(test_clean),
            "speech_material_set_id": speech["material_set_id"],
            "noise_material_set_id": noise["material_set_id"],
        },
    }


def _list_accept_recipe_sources(resource_db: Path, corpus: Path) -> dict[str, Any]:
    rows, other = _catalog_rows(resource_db, corpus)
    clean = [row for row in rows if row["subset"] == "clean"]
    noise16 = [row for row in rows if row["subset"] == "noise"]
    non16 = []
    excluded = 0
    rates: Counter[str] = Counter()
    for row in rows:
        if row["subset"] != "noise_excluded":
            continue
        rates[str(int(row["sample_rate"]))] += 1
        if int(row["channels"]) != 1 or int(row["sample_rate"]) == SAMPLE_RATE:
            excluded += 1
            continue
        kept = dict(row)
        kept["subset"] = "noise"
        non16.append(kept)
    test_clean = [row for row in rows if row["subset"] == "test_no_reverb_clean"]
    test_noisy = [row for row in rows if row["subset"] == "test_no_reverb_noisy"]
    actual = {
        "clean": len(clean),
        "noise_16k": len(noise16),
        "non_16k": len(non16),
        "excluded": excluded,
        "rates": dict(rates),
        "test_clean": len(test_clean),
        "test_noisy": len(test_noisy),
    }
    expected_rates = {"44100": 94, "48000": 88, "96000": 6}
    if (
        actual["clean"] != 65348
        or actual["noise_16k"] != 65114
        or actual["non_16k"] != 188
        or actual["excluded"] != 0
        or actual["test_clean"] != 150
        or actual["test_noisy"] != 150
        or dict(rates) != expected_rates
    ):
        raise ReproductionError(f"DNS catalog does not match the P7 full-pool counts: {actual}")
    return {
        "clean": clean,
        "noise": noise16 + non16,
        "noise_16k": len(noise16),
        "non_16k_noise": len(non16),
        "non_16k_rates": dict(rates),
        "excluded_noise": excluded,
        "test_clean": test_clean,
        "test_noisy": test_noisy,
        "other_counts": other,
    }


def _plan_accept_shards(
    config: dict[str, Any],
    prepared: dict[str, Any],
    identity: dict[str, Any],
    shard_dir: Path,
    missing: list[tuple[int, int]],
) -> None:
    workers = int(config["workers"])
    payload = {
        "definition": prepared["definition"],
        "clean_names": list(prepared["clean_names"]),
        "noise_names": list(prepared["noise_names"]),
        "clean_paths": {name: str(path) for name, path in prepared["clean_paths"].items()},
        "noise_paths": {name: str(path) for name, path in prepared["noise_paths"].items()},
        "identity": identity,
        "directory": str(shard_dir),
    }
    if workers == 1 or len(missing) == 1:
        results = [_run_accept_shard(item, payload) for item in missing]
    else:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        results = []
        with ProcessPoolExecutor(
            max_workers=min(workers, len(missing)),
            mp_context=context,
            initializer=_init_accept_shard,
            initargs=(payload,),
        ) as pool:
            futures = [pool.submit(_run_accept_shard, item) for item in missing]
            for done, future in enumerate(as_completed(futures), start=1):
                item = future.result()
                results.append(item)
                if done % 10 == 0 or done == len(futures) or not item["ok"]:
                    print(
                        f"shard\t{done}\t{len(futures)}\t{item['start']}\t{item['stop']}\tok={item['ok']}",
                        flush=True,
                    )
    failed = [item for item in results if not item["ok"]]
    if failed:
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir.parent / "failures.json").write_text(
            json.dumps(failed, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ReproductionError(
            f"{len(failed)} acceptance shards failed. First: gen_index range "
            f"{failed[0]['start']}:{failed[0]['stop']} {failed[0]['error']}"
        )


def _init_accept_shard(payload: dict[str, Any]) -> None:
    global _SHARD_STATE
    _SHARD_STATE = payload


def _run_accept_shard(bounds: tuple[int, int], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    state = payload if payload is not None else _SHARD_STATE
    start, stop = int(bounds[0]), int(bounds[1])
    started = time.perf_counter()
    cpu_before = _cpu_seconds()
    read_before = _proc_read_bytes()
    try:
        clean_paths = {name: Path(path) for name, path in state["clean_paths"].items()}
        noise_paths = {name: Path(path) for name, path in state["noise_paths"].items()}
        with profile_acceptance() as profile:
            documents, totals = plan_accepted_span(
                state["definition"],
                state["clean_names"],
                clean_paths,
                state["noise_names"],
                noise_paths,
                start=start,
                stop=stop,
                workers=1,
            )
        write_shard(Path(state["directory"]), state["identity"], start, stop, list(zip(documents, _stats_rows(totals), strict=True)))
    except Exception as exc:
        return {
            "ok": False,
            "start": start,
            "stop": stop,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
        }
    return {
        "ok": True,
        "start": start,
        "stop": stop,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": _cpu_seconds() - cpu_before,
        "read_bytes": _proc_read_bytes() - read_before,
        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "profile": profile.as_dict(),
        "max_retries": max(int(value) for value in totals["outer_attempts"]),
    }


def _stats_rows(totals: dict[str, Any]) -> list[dict[str, int]]:
    rows = totals.get("case_stats")
    if not isinstance(rows, list) or len(rows) != len(totals["outer_attempts"]):
        raise ReproductionError("acceptance span did not return one stats row per case")
    return [{key: int(value) for key, value in row.items()} for row in rows]


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


def _hashes(
    paths: list[Path],
    *,
    cache: Path,
    workers: int,
    seed_caches: tuple[Path, ...] = (),
) -> dict[Path, str]:
    known = _read_cache(cache)
    for extra in seed_caches:
        for key, value in _read_cache(extra).items():
            known.setdefault(key, value)
    missing = [path for path in paths if (str(path), path.stat().st_size) not in known]
    found: dict[tuple[str, int], str] = dict(known)
    if missing:
        workers = max(1, workers)

        def one(path: Path) -> tuple[Path, int, str]:
            return path, path.stat().st_size, sha256_file(path)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, size, file_digest in pool.map(one, missing):
                found[(str(path), size)] = file_digest
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
    from src.experiments.data.db.registry import read_graph

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


def _register_accept_recipe(
    config: dict[str, Any],
    prepared: dict[str, Any],
    documents: list[dict[str, Any]],
    stats: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    workspace: Path,
    corpus: Path,
    staging: Path,
    database: Path,
    snapshots: Path,
    resource_db: Path,
    generated: Path,
    code: dict[str, Any],
) -> dict[str, Any]:
    train_docs, valid_docs, split_stats = split_train_valid(documents)
    if len(train_docs) != 324000 or len(valid_docs) != 36000:
        raise ReproductionError(f"acceptance split is {len(train_docs)}/{len(valid_docs)}, expected 324000/36000")
    summary["split"] = split_stats
    speech_paths = paths_by_name(prepared["clean_assets"], corpus)
    noise_paths = paths_by_name(prepared["noise_assets"], corpus)
    checked = time.perf_counter()
    report = audit_sample(
        prepared["definition"],
        documents,
        speech=speech_paths,
        noise=noise_paths,
        render_limit=min(int(config["audit_realizations"]), len(documents)),
    )
    if report["clean_activity_rejected"] or report["post_mix_clipped"] or report["source_clipped_pieces"]:
        raise ReproductionError(f"accepted DNS sample still fails an official check: {report}")
    summary["audit_seconds"] = time.perf_counter() - checked
    summary["audit"] = _audit_public(report)
    wave_started = time.perf_counter()
    probe_noise = {
        row["name"]: row["path"]
        for row in prepared["listed"]["noise"]
        if int(row["sample_rate"]) != SAMPLE_RATE
    }
    summary["waveform_audit"] = _waveform_audit(
        documents, stats, speech_paths, noise_paths, corpus, probe_noise=probe_noise
    )
    summary["waveform_seconds"] = time.perf_counter() - wave_started
    before = _dataset_rows(database)
    protected = {
        "resource_mtime_ns": resource_db.stat().st_mtime_ns,
        "generated_mtime_ns": _tree_mtime_ns(generated),
        "derived_mtime_ns": _tree_mtime_ns(ROOT / "data" / "derived"),
        "snapshot_names": sorted(path.name for path in snapshots.iterdir() if path.is_dir()) if snapshots.is_dir() else [],
    }
    test_set, test_cases, test_protocol = _load_shared_test(database)
    parent, ondemand = protocol_pair()
    if digest(_project_protocol(test_protocol)) != digest(_project_protocol(ondemand)):
        raise ReproductionError("registered no_reverb cases do not use the current P7 on-demand protocol")
    if digest(_project_protocol(_load_protocol(database, parent["protocol_id"]))) != digest(_project_protocol(parent)):
        raise ReproductionError("registered DNS formula protocol does not match the current builder")
    graph = _reuse_registered_assets(
        assemble_accept_graph(
            name=ACCEPT_RECIPE_NAME,
            claim=str(config["claim_label"]),
            definition=prepared["definition"],
            documents=documents,
            report=report,
            speech=prepared["speech"],
            noise=prepared["noise"],
            source_assets=prepared["clean_assets"] + prepared["noise_assets"] + prepared["test_clean"] + prepared["test_noisy"],
            test_cases=test_cases,
            resource_db=str(resource_db) if resource_db.is_file() else None,
            full_recipe=True,
            test_set=test_set,
            extra_protocols=[test_protocol],
        ),
        database,
    )
    dataset = graph.datasets[0]
    if dataset["name"] != ACCEPT_RECIPE_NAME or int(dataset["selection_uses_test"]) != 0:
        raise ReproductionError("acceptance dataset identity or test-selection flag is wrong")
    summary["dataset_id"] = dataset["dataset_id"]
    summary["selection_uses_test"] = dataset["selection_uses_test"]
    summary["leak"] = leak_report(
        dataset,
        index_rows(graph.case_sets, "case_set_id", "case_sets"),
        index_rows(graph.materials, "material_set_id", "materials"),
        index_rows(graph.cases, "case_id", "cases"),
        index_rows(graph.assets, "asset_id", "assets"),
        resource_db=str(resource_db) if resource_db.is_file() else None,
    )["status"]
    registered = time.perf_counter()
    write_graph(staging, graph)
    register_staging(staging, database, resource_db=str(resource_db) if resource_db.is_file() else None)
    summary["register_seconds"] = time.perf_counter() - registered
    snapped = time.perf_counter()
    summary["snapshot_hash"] = export_snapshot(
        database,
        f"{ACCEPT_RECIPE_NAME}@1",
        snapshots,
        corpora=corpus,
        generated=generated,
        derived=ROOT / "data" / "derived",
        provenance_extra={
            "worktree_dirty": bool(code["worktree_dirty"]),
            "generator_code_sha256": code["generator_code_sha256"],
            "porcelain_sha256": code["porcelain_sha256"],
        },
    )
    summary["snapshot_seconds"] = time.perf_counter() - snapped
    after = _dataset_rows(database)
    for key, row in before.items():
        if after.get(key) != row:
            raise ReproductionError(f"existing dataset {key} changed during acceptance registration")
    if resource_db.stat().st_mtime_ns != protected["resource_mtime_ns"]:
        raise ReproductionError("resource.duckdb was modified")
    if _tree_mtime_ns(generated) != protected["generated_mtime_ns"]:
        raise ReproductionError("data/generated changed during acceptance registration")
    if _tree_mtime_ns(ROOT / "data" / "derived") != protected["derived_mtime_ns"]:
        raise ReproductionError("data/derived changed during acceptance registration")
    snapshot_names = sorted(path.name for path in snapshots.iterdir() if path.is_dir())
    added = set(snapshot_names) - set(protected["snapshot_names"])
    if added - {dataset["dataset_id"]}:
        raise ReproductionError(f"unexpected snapshot directories: {sorted(added)}")
    if dataset["dataset_id"] not in snapshot_names:
        raise ReproductionError("acceptance snapshot was not written")
    label = f"{ACCEPT_RECIPE_NAME}@1"
    if label not in before and set(after) - set(before) != {label}:
        raise ReproductionError(f"dataset rows added: {sorted(set(after) - set(before))}")
    if label in before and set(after) != set(before):
        raise ReproductionError("re-registration changed the dataset name set")
    summary["snapshot_bytes"] = _tree_bytes(snapshots / dataset["dataset_id"])
    summary["plan_bytes"] = _tree_bytes(workspace)
    summary["split_counts"] = {"train": len(train_docs), "valid": len(valid_docs), "test": len(test_cases)}
    (workspace / "register-stats.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def _benchmark_accept_recipe(
    config: dict[str, Any],
    prepared: dict[str, Any],
    workspace: Path,
    corpus: Path,
) -> dict[str, Any]:
    counts = [int(item) for item in (config.get("benchmark_counts") or [100, 1000])]
    worker_sets = [int(item) for item in (config.get("benchmark_workers") or [1, 2, 4])]
    total = int(prepared["definition"]["params"]["num_files"])
    reports = []
    for count in counts:
        if count < 2 or count > total:
            raise ReproductionError(f"benchmark span {count} is outside 2:{total}")
        digests: dict[int, list[str]] = {}
        runs = []
        for workers in worker_sets:
            wall = time.perf_counter()
            cpu = _cpu_seconds()
            read_bytes = _proc_read_bytes()
            profile = None
            if workers == 1:
                with profile_acceptance() as profile:
                    documents, totals = plan_accepted_span(
                        prepared["definition"],
                        prepared["clean_names"],
                        prepared["clean_paths"],
                        prepared["noise_names"],
                        prepared["noise_paths"],
                        start=0,
                        stop=count,
                        workers=1,
                    )
            else:
                documents, totals = plan_accepted_span(
                    prepared["definition"],
                    prepared["clean_names"],
                    prepared["clean_paths"],
                    prepared["noise_names"],
                    prepared["noise_paths"],
                    start=0,
                    stop=count,
                    workers=workers,
                )
            digests[workers] = [digest(_stored(item)) for item in documents]
            attempts = [int(value) for value in totals["outer_attempts"]]
            runs.append(
                {
                    "workers": workers,
                    "count": count,
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": _cpu_seconds() - cpu,
                    "read_bytes": _proc_read_bytes() - read_bytes,
                    "maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "children_maxrss_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                    "profile": None if profile is None else profile.as_dict(),
                    "max_retries": max(attempts),
                    "mean_retries": sum(attempts) / len(attempts),
                    "attempt_histogram": {
                        str(value): attempts.count(value) for value in sorted(set(attempts))
                    },
                }
            )
            print(
                f"benchmark count={count} workers={workers} wall={runs[-1]['wall_seconds']:.1f}s",
                flush=True,
            )
        baseline = digests[worker_sets[0]]
        if any(digests[item] != baseline for item in worker_sets):
            raise ReproductionError(f"worker count changed the first {count} acceptance documents")
        report = {"count": count, "documents_match": True, "runs": runs}
        if count == counts[0]:
            official = _official_numeric_compare(
                documents,
                list(range(min(8, len(documents)))),
                prepared["clean_paths"],
                prepared["noise_paths"],
                corpus,
            )
            if official.get("status") != "PASS":
                raise ReproductionError(f"official numeric compare did not pass: {official}")
            report["official"] = official
            non16 = {
                row["name"]: row["path"]
                for row in prepared["listed"]["noise"]
                if int(row["sample_rate"]) != SAMPLE_RATE
            }
            report["resample"] = _resample_probe(non16, corpus)
        reports.append(report)
    return {
        "benchmarks": reports,
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "workspace": str(workspace),
    }


def _waveform_audit(
    documents: list[dict[str, Any]],
    stats: list[dict[str, Any]],
    speech: dict[str, Path],
    noise: dict[str, Path],
    corpus: Path,
    *,
    probe_noise: dict[str, Path] | None = None,
) -> dict[str, Any]:
    chosen = _audit_indexes(documents, stats)
    rows = []
    for gen_index in chosen:
        measured = measure_realization(documents[gen_index], {"speech": speech, "noise": noise})
        rendered = render_realization(documents[gen_index], {"speech": speech, "noise": noise})
        pcm_error = max(
            float(np.max(np.abs(rendered[name] * 32768.0 - np.rint(rendered[name] * 32768.0))))
            for name in ("clean", "noise", "noisy")
        )
        finite = all(bool(np.isfinite(rendered[name]).all()) for name in ("clean", "noise", "noisy"))
        if (
            measured["clean_activity_rejected"]
            or measured["source_clipped_pieces"]
            or measured["post_mix_clipped"]
            or int(measured["length"]) != 480000
            or not finite
            or pcm_error > 1e-6
        ):
            raise ReproductionError(f"gen_index {gen_index} waveform audit failed: {measured}")
        rows.append(
            {
                "gen_index": gen_index,
                "clean_activity": measured["clean_activity"],
                "snr_error_db": measured["snr_error_db"],
                "pcm_error": pcm_error,
            }
        )
    errors = [abs(float(row["snr_error_db"])) for row in rows]
    official = _official_numeric_compare(documents, chosen[:8], speech, noise, corpus)
    if official.get("status") != "PASS":
        raise ReproductionError(f"official numeric compare did not pass: {official}")
    return {
        "rendered": len(rows),
        "gen_indexes": chosen,
        "clean_activity_min": min(float(row["clean_activity"]) for row in rows),
        "max_snr_error_db": max(errors),
        "max_pcm_error": max(float(row["pcm_error"]) for row in rows),
        "official": official,
        "resample": _resample_probe(probe_noise if probe_noise is not None else noise, corpus),
    }


def _audit_indexes(documents: list[dict[str, Any]], stats: list[dict[str, Any]]) -> list[int]:
    ordered = sorted(range(len(documents)), key=lambda index: str(documents[index]["draws"]["utt_id"]))
    train_count = len(ordered) * 9 // 10
    groups = {
        "train": ordered[:train_count],
        "valid": ordered[train_count:],
        "retries_0": [],
        "retries_some": [],
        "non16": [],
        "multi": [],
        "short": [],
        "source_clip": [],
    }
    snr_min = min(int(item["draws"]["snr_db"]) for item in documents)
    level_min = min(int(item["draws"]["target_level_dbfs"]) for item in documents)
    snr_max = max(int(item["draws"]["snr_db"]) for item in documents)
    level_max = max(int(item["draws"]["target_level_dbfs"]) for item in documents)
    edges = {"snr_min": [], "snr_max": [], "level_min": [], "level_max": []}
    retry_pairs = sorted(
        ((int(stats[index]["retries"]), index) for index in range(len(documents))),
        reverse=True,
    )
    for index, document in enumerate(documents):
        retries = int(stats[index]["retries"])
        groups["retries_0" if retries == 0 else "retries_some"].append(index)
        pieces = [piece for piece in document["sources"]["speech"] + document["sources"]["noise"] if "name" in piece]
        if any(int(piece.get("source_rate", SAMPLE_RATE)) != SAMPLE_RATE for piece in pieces):
            groups["non16"].append(index)
        speech_n = sum(1 for piece in document["sources"]["speech"] if "name" in piece)
        noise_n = sum(1 for piece in document["sources"]["noise"] if "name" in piece)
        if speech_n > 1 or noise_n > 1:
            groups["multi"].append(index)
        if any(int(piece["length"]) < 16000 for piece in pieces):
            groups["short"].append(index)
        if int(stats[index]["clean_clip_skips"]) or int(stats[index]["noise_clip_skips"]):
            groups["source_clip"].append(index)
        if int(document["draws"]["snr_db"]) == snr_min:
            edges["snr_min"].append(index)
        if int(document["draws"]["snr_db"]) == snr_max:
            edges["snr_max"].append(index)
        if int(document["draws"]["target_level_dbfs"]) == level_min:
            edges["level_min"].append(index)
        if int(document["draws"]["target_level_dbfs"]) == level_max:
            edges["level_max"].append(index)
    chosen: list[int] = []
    seen: set[int] = set()
    for name, limit in (
        ("train", 80),
        ("valid", 80),
        ("non16", 80),
        ("retries_0", 40),
        ("retries_some", 40),
        ("multi", 40),
        ("short", 40),
        ("source_clip", 40),
    ):
        for index in _spread_indexes(groups[name], limit):
            if index not in seen:
                chosen.append(index)
                seen.add(index)
    for index in [item[1] for item in retry_pairs[:20]]:
        if index not in seen:
            chosen.append(index)
            seen.add(index)
    for indexes in edges.values():
        for index in _spread_indexes(indexes, 20):
            if index not in seen:
                chosen.append(index)
                seen.add(index)
    if not groups["non16"]:
        raise ReproductionError("waveform audit found no non-16 kHz noise case")
    return chosen


def _spread_indexes(indexes: list[int], limit: int) -> list[int]:
    if not indexes or limit < 1:
        return []
    if len(indexes) <= limit:
        return list(indexes)
    return [indexes[int(round(offset * (len(indexes) - 1) / (limit - 1)))] for offset in range(limit)]


def _official_numeric_compare(
    documents: list[dict[str, Any]],
    indexes: list[int],
    speech: dict[str, Path],
    noise: dict[str, Path],
    corpus: Path,
) -> dict[str, Any]:
    audiolib_path = corpus / "DNS2020" / "audiolib.py"
    if not audiolib_path.is_file():
        return {"status": "NOT_TESTED", "reason": "DNS audiolib.py is not in the corpus checkout"}
    from src.experiments.data.build.dns2020 import _render_pieces

    spec = importlib.util.spec_from_file_location("dns_challenge_audiolib_p10a1", audiolib_path)
    if spec is None or spec.loader is None:
        raise ReproductionError(f"cannot load {audiolib_path}")
    audiolib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audiolib)
    activity_error = 0.0
    mixer_error = 0.0
    clip_mismatch = 0
    for gen_index in indexes:
        document = documents[gen_index]
        clean = _render_pieces(document["sources"]["speech"], speech, allow_resample=True)
        noise_wave = _render_pieces(document["sources"]["noise"], noise, allow_resample=True)
        activity_error = max(
            activity_error,
            abs(float(activitydetector(clean)) - float(audiolib.activitydetector(clean))),
        )
        if bool(is_clipped(clean)) != bool(audiolib.is_clipped(clean)):
            clip_mismatch += 1
        level = int(document["draws"]["target_level_dbfs"])
        snr = int(document["draws"]["snr_db"])
        saved = np.random.randint
        np.random.randint = lambda low, high=None, size=None, level=level: level
        try:
            official = audiolib.snr_mixer(
                {"cfg": {}, "target_level_lower": LEVEL_LOWER, "target_level_upper": LEVEL_UPPER},
                clean.copy(),
                noise_wave.copy(),
                snr,
            )
        finally:
            np.random.randint = saved
        ours = snr_mixer(clean, noise_wave, snr, level)
        for left, right in zip(official, ours, strict=True):
            mixer_error = max(mixer_error, float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
    if activity_error > 0.0 or clip_mismatch or mixer_error > 1e-4:
        raise ReproductionError(
            f"official DNS functions diverged: activity {activity_error} mixer {mixer_error} clips {clip_mismatch}"
        )
    return {
        "status": "PASS",
        "cases": len(indexes),
        "max_activity_abs_error": activity_error,
        "max_mixer_abs_error": mixer_error,
        "clip_mismatches": clip_mismatch,
        "official_librosa_version": "UNKNOWN",
    }


def _resample_probe(noise: dict[str, Path], corpus: Path) -> dict[str, Any]:
    del corpus
    found: dict[int, Path] = {}
    for path in noise.values():
        info = sf.info(path)
        rate = int(info.samplerate)
        if rate != SAMPLE_RATE and rate not in found:
            found[rate] = path
        if len(found) == 3:
            break
    rows = []
    kaiser = _kaiser_resampler()
    for rate, path in sorted(found.items()):
        frames = min(int(sf.info(path).frames), int(rate * 2))
        audio, got = sf.read(path, start=0, stop=frames, dtype="float64", always_2d=True)
        mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
        if int(got) != rate:
            raise ReproductionError(f"{path} probe rate is {got}")
        converted = resample_to_16k(mono, rate)
        scipy_count = int(round(mono.shape[0] * SAMPLE_RATE / rate))
        import scipy.signal

        scipy_wave = np.asarray(scipy.signal.resample(mono, scipy_count), dtype=np.float64)
        overlap = min(converted.shape[0], scipy_wave.shape[0])
        row = {
            "source_rate": rate,
            "source_samples": int(mono.shape[0]),
            "soxr_hq_samples": int(converted.shape[0]),
            "scipy_resample_samples": int(scipy_wave.shape[0]),
            "soxr_vs_scipy_max_abs": float(np.max(np.abs(converted[:overlap] - scipy_wave[:overlap]))),
            "kaiser_best": {
                "status": "NOT_TESTED",
                "reason": (
                    "resampy from .venvs/asr imports numba, and that numba rejects the SE "
                    "interpreter's NumPy 2.5. The comparison is not run inside this process."
                ),
            },
        }
        if kaiser is not None:
            kaiser_wave = np.asarray(kaiser(mono, rate, SAMPLE_RATE), dtype=np.float64)
            overlap = min(converted.shape[0], kaiser_wave.shape[0])
            row["kaiser_best"] = {
                "samples": int(kaiser_wave.shape[0]),
                "max_abs_vs_soxr_hq": float(np.max(np.abs(converted[:overlap] - kaiser_wave[:overlap]))),
            }
        rows.append(row)
    if len(rows) < 3:
        raise ReproductionError(f"resample probe found {len(rows)} source rates, expected 44100, 48000, and 96000")
    return {
        "official_librosa_version": "UNKNOWN",
        "implementation": "librosa_1.0.0_soxr_hq",
        "files": rows,
    }


def _kaiser_resampler():
    try:
        import resampy
    except ImportError:
        site = ROOT / ".venvs" / "asr" / "lib" / "python3.12" / "site-packages"
        if not site.is_dir():
            return None
        sys.path.insert(0, str(site))
        try:
            import resampy
        except ImportError:
            return None
        finally:
            if sys.path and sys.path[0] == str(site):
                sys.path.pop(0)
    def convert(audio, source_rate, target_rate):
        return resampy.resample(np.asarray(audio, dtype=np.float64), source_rate, target_rate, filter="kaiser_best")
    return convert


def _assert_recipe_protocols(database: Path, parent: dict[str, Any], child: dict[str, Any]) -> None:
    """Keep the 150 test pairs on the P7 on-demand protocol and match the formula parent."""
    if child.get("parent_protocol_id") != parent["protocol_id"]:
        raise ReproductionError("acceptance protocol is not a child of the DNS formula protocol")
    stored_parent = _load_protocol(database, parent["protocol_id"])
    if digest(_project_protocol(stored_parent)) != digest(_project_protocol(parent)):
        raise ReproductionError("registered DNS formula protocol does not match the current builder")
    _case_set, _cases, test_protocol = _load_shared_test(database)
    formula, ondemand = protocol_pair()
    if formula["protocol_id"] != parent["protocol_id"]:
        raise ReproductionError("acceptance and length-plan protocols do not share one formula parent")
    if digest(_project_protocol(test_protocol)) != digest(_project_protocol(ondemand)):
        raise ReproductionError("registered no_reverb cases do not use the current P7 on-demand protocol")
    if test_protocol["protocol_id"] == child["protocol_id"]:
        raise ReproductionError("no_reverb test cases must stay on the P7 on-demand protocol")


def _load_shared_test(database: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    case_sets = _fetch_rows(database, "case_sets", "name = ? AND version = 1", [P7_TEST_SET_NAME])
    if len(case_sets) != 1:
        raise ReproductionError(f"expected one {P7_TEST_SET_NAME} case set")
    case_set = case_sets[0]
    items = _fetch_rows(database, "case_set_items", "case_set_id = ?", [case_set["case_set_id"]])
    items = sorted(items, key=lambda item: int(item["position"]))
    if len(items) != 150 or [int(item["position"]) for item in items] != list(range(150)):
        raise ReproductionError("the registered no_reverb test set is not 150 ordered pairs")
    case_set = dict(case_set)
    case_set["items"] = [{"position": int(item["position"]), "case_id": item["case_id"]} for item in items]
    cases = _fetch_rows(database, "cases", "case_id IN (" + ",".join("?" for _ in items) + ")", [item["case_id"] for item in items])
    by_id = {row["case_id"]: row for row in cases}
    ordered = [by_id[item["case_id"]] for item in items]
    protocol = _load_protocol(database, ordered[0]["protocol_id"])
    return case_set, ordered, protocol


def _load_protocol(database: Path, protocol_id: str) -> dict[str, Any]:
    rows = _fetch_rows(database, "protocols", "protocol_id = ?", [protocol_id])
    if len(rows) != 1:
        raise ReproductionError(f"protocol {protocol_id} is not registered once")
    return rows[0]


def _fetch_rows(database: Path, table: str, where: str, params: list[Any]) -> list[dict[str, Any]]:
    import duckdb

    from src.experiments.data.db.io import normalize_row
    from src.experiments.data.db.schema import TABLES

    spec = TABLES[table]
    connection = duckdb.connect(str(database), read_only=True)
    try:
        cursor = connection.execute(
            f"SELECT {', '.join(spec.columns)} FROM repro.{spec.name} WHERE {where}",
            params,
        )
        return [normalize_row(dict(zip(spec.columns, record, strict=True)), spec) for record in cursor.fetchall()]
    finally:
        connection.close()


def _project_protocol(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol_id",
        "name",
        "version",
        "kind",
        "algorithm",
        "algorithm_version",
        "rule_json",
        "official_ref_json",
        "deviation_class",
        "parent_protocol_id",
        "deviations_json",
    )
    return {key: row[key] for key in keys}


def _dataset_rows(database: Path) -> dict[str, dict[str, Any]]:
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            "SELECT name, version, dataset_id, protocol_id, declared_status FROM repro.datasets ORDER BY name, version"
        ).fetchall()
    finally:
        connection.close()
    return {
        f"{name}@{version}": {
            "dataset_id": dataset_id,
            "protocol_id": protocol_id,
            "declared_status": declared_status,
        }
        for name, version, dataset_id, protocol_id, declared_status in rows
    }


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime)


def _proc_read_bytes() -> int:
    try:
        text = Path("/proc/self/io").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def _package_versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("numpy", "soundfile", "librosa", "soxr", "scipy"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def _tree_mtime_ns(path: Path) -> int | None:
    if not path.exists():
        return None
    newest = path.stat().st_mtime_ns
    if path.is_file():
        return newest
    for item in path.rglob("*"):
        if item.is_file():
            newest = max(newest, item.stat().st_mtime_ns)
    return newest


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total
