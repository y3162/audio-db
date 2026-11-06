"""Plan, render, and register an MS-SNSD build described by a YAML file.

The full 30 h render is a separate build id from the scaled-down smoke build.
This module refuses to register a build whose verify report is not PASS.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

from src.reproduction.build.ms_snsd import (
    build_definition,
    ordered_paths,
    plan_realizations,
    verify_build,
    write_build,
)
from src.reproduction.db.registry import register_staging
from src.reproduction.db.snapshot import export_snapshot
from src.reproduction.db.staging import write_graph
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
from src.reproduction.prepare.ms_snsd import (
    assemble_smoke_graph,
    collect_sources,
    material_set,
    write_shapes,
)


def select_names(
    directory: Path,
    *,
    limit: int | None,
    minimum: int | None,
    maximum: int | None,
) -> list[str]:
    chosen = []
    for path in sorted(path for path in directory.glob("*.wav") if path.is_file()):
        info = sf.info(path)
        frames = int(info.frames)
        if minimum is not None and frames < minimum:
            continue
        if maximum is not None and frames > maximum:
            continue
        chosen.append(path.name)
        if limit is not None and len(chosen) >= limit:
            break
    if not chosen:
        raise ReproductionError(f"no wavs matched the length filter in {directory}")
    return chosen


def load_build_config(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ReproductionError(f"{path} must be a mapping")
    return document


def run_config(
    config_path: Path,
    step: str,
    *,
    corpus: Path = DEFAULT_CORPORA / "MS-SNSD",
    audio_root: Path = DEFAULT_GENERATED,
    staging: Path = DEFAULT_STAGING,
    database: Path = DEFAULT_DB,
    snapshots: Path = DEFAULT_SNAPSHOTS,
    resource_db: Path = DEFAULT_RESOURCE_DB,
) -> dict[str, Any]:
    if step not in ("plan", "generate", "register"):
        raise ReproductionError("step must be plan, generate, or register")
    config = load_build_config(config_path)
    claim = str(config.get("claim_label") or "official_recipe_scaled_down")
    if claim not in ("official_recipe_scaled_down", "espnet_public_se_recipe_reproduction"):
        raise ReproductionError(f"unsupported MS-SNSD claim_label: {claim}")
    started = time.perf_counter()
    train = _prepare_side(config, "train", corpus)
    test = _prepare_side(config, "test", corpus)
    summary = {
        "name": config["name"],
        "train_realizations": len(train["documents"]),
        "test_realizations": len(test["documents"]),
        "train_build_id": train["definition"]["build_id"],
        "test_build_id": test["definition"]["build_id"],
        "plan_seconds": time.perf_counter() - started,
    }
    print(f"planned train={summary['train_realizations']} test={summary['test_realizations']} in {summary['plan_seconds']:.1f}s", flush=True)
    workspace = staging / "workspaces" / "ms_snsd"
    _write_plan(workspace, train)
    _write_plan(workspace, test)
    if step == "plan":
        return summary
    generated = time.perf_counter()
    train_written = write_build(
        train["definition"], train["documents"],
        audio_root=audio_root, staging_root=workspace,
        clean_dir=corpus / "clean_train", noise_dir=corpus / "noise_train",
    )
    test_written = write_build(
        test["definition"], test["documents"],
        audio_root=audio_root, staging_root=workspace,
        clean_dir=corpus / "clean_test", noise_dir=corpus / "noise_test",
    )
    train_report = verify_build(
        train["definition"], train["documents"],
        audio_dir=train_written["audio_dir"], staging_dir=train_written["staging_dir"],
        clean_dir=corpus / "clean_train", noise_dir=corpus / "noise_train",
    )
    test_report = verify_build(
        test["definition"], test["documents"],
        audio_dir=test_written["audio_dir"], staging_dir=test_written["staging_dir"],
        clean_dir=corpus / "clean_test", noise_dir=corpus / "noise_test",
    )
    summary["generate_seconds"] = time.perf_counter() - generated
    print(f"generated in {summary['generate_seconds']:.1f}s", flush=True)
    summary["train_verify"] = {key: train_report[key] for key in ("files", "clipped_samples", "peak_abs", "status")}
    summary["test_verify"] = {key: test_report[key] for key in ("files", "clipped_samples", "peak_abs", "status")}
    if step == "generate":
        return summary
    graph = assemble_smoke_graph(
        train_definition=train["definition"],
        train_documents=train["documents"],
        train_manifests=_manifests(train_written["staging_dir"]),
        train_verify=train_report,
        test_definition=test["definition"],
        test_documents=test["documents"],
        test_manifests=_manifests(test_written["staging_dir"]),
        test_verify=test_report,
        source_assets=train["clean_assets"] + train["noise_assets"] + test["clean_assets"] + test["noise_assets"],
        train_clean=train["clean_set"],
        train_noise=train["noise_set"],
        test_clean=test["clean_set"],
        test_noise=test["noise_set"],
        dataset_name=str(config["name"]),
        resource_db=str(resource_db) if resource_db.is_file() else None,
        claim_label=claim,
    )
    write_graph(staging, graph)
    register_staging(staging, database, resource_db=str(resource_db) if resource_db.is_file() else None)
    dataset_id = graph.datasets[0]["dataset_id"]
    snapshot_hash = export_snapshot(
        database,
        f"{config['name']}@1",
        snapshots,
        corpora=DEFAULT_CORPORA,
        generated=audio_root,
        derived=ROOT / "data" / "derived",
    )
    _write_split_shapes(workspace / "shapes", graph)
    summary["dataset_id"] = dataset_id
    summary["snapshot_hash"] = snapshot_hash
    summary["selection_uses_test"] = graph.datasets[0]["selection_uses_test"]
    return summary


def _prepare_side(config: dict[str, Any], split: str, corpus: Path) -> dict[str, Any]:
    print(f"preparing {split}", flush=True)
    side = config[split]
    clean_dir = corpus / f"clean_{split}"
    noise_dir = corpus / f"noise_{split}"
    clean_pool = select_names(
        clean_dir,
        limit=side.get("clean_limit"),
        minimum=side.get("clean_min_frames"),
        maximum=side.get("clean_max_frames"),
    )
    noise_pool = select_names(
        noise_dir,
        limit=side.get("noise_limit"),
        minimum=side.get("noise_min_frames"),
        maximum=side.get("noise_max_frames"),
    )
    clean_names = ordered_paths(
        [clean_dir / name for name in clean_pool], seed=int(config["seed"]), split=split,
    )
    noise_names = ordered_paths(
        [noise_dir / name for name in noise_pool], seed=int(config["seed"]), split=split,
    )
    prefix = "train" if split == "train" else "test"
    tag = str(config["name"]).replace("tf_gridnet.ms_snsd.", "ms_snsd.")
    clean_assets, clean_frames = collect_sources(
        clean_dir, clean_names, rel_prefix=f"MS-SNSD/clean_{split}", subset=f"clean_{split}",
    )
    noise_assets, noise_frames = collect_sources(
        noise_dir, noise_names, rel_prefix=f"MS-SNSD/noise_{split}", subset=f"noise_{split}",
    )
    clean_set = material_set(
        clean_assets, name=f"{tag}.clean_{split}", role="speech", split_role=prefix,
        note="length-filtered MS-SNSD files, then one seeded shuffle",
    )
    noise_set = material_set(
        noise_assets, name=f"{tag}.noise_{split}", role="noise", split_role=prefix,
        note="length-filtered MS-SNSD files, then one seeded shuffle",
    )
    definition = build_definition(
        seed=int(config["seed"]),
        split=split,
        clean_names=clean_names,
        noise_names=noise_names,
        total_hours=float(side["total_hours"]),
        max_clips=side.get("max_clips"),
        shard_size=int(config["shard_size"]),
        inputs={"speech": clean_set["material_set_id"], "noise": noise_set["material_set_id"]},
    )
    documents = plan_realizations(definition, clean_frames, noise_frames)
    return {
        "definition": definition,
        "documents": documents,
        "clean_assets": clean_assets,
        "noise_assets": noise_assets,
        "clean_set": clean_set,
        "noise_set": noise_set,
    }


def _write_plan(workspace: Path, side: dict[str, Any]) -> None:
    definition = side["definition"]
    directory = workspace / "plans" / definition["build_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "build.json").write_text(
        json.dumps(definition, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    lines = "".join(json.dumps(item, sort_keys=True) + "\n" for item in side["documents"])
    (directory / "plan.jsonl").write_text(lines, encoding="utf-8")


def _manifests(staging_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((staging_dir / "shards").glob("shard-*.json"))
    ]


def _write_split_shapes(root: Path, graph) -> None:
    cases = {row["case_id"]: row for row in graph.cases}
    for case_set in graph.case_sets:
        rows = [cases[item["case_id"]] for item in case_set["items"]]
        paths = {}
        for row in rows:
            noisy = next(asset for asset in graph.assets if asset["asset_id"] == row["noisy_asset_id"])
            paths[row["utterance_key"]] = noisy["rel_path"]
        write_shapes(root / case_set["split_role"], rows, paths)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.reproduction.prepare.ms_snsd_run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", required=True, choices=("plan", "generate", "register"))
    args = parser.parse_args(argv)
    summary = run_config(Path(args.config), args.step)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
