"""Run one prepare step, or every step in order.

Steps are diagnose, lists, resolve, register, snapshot, and verify.
``all`` runs them in that order. A missing corpus file stops with
``PreparationBlocked`` before staging or a database is created.
Registration points cases at the 48 kHz sources. Training resamples them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.recipe import sha256_file
from src.reproduction.db.registry import initialize_database, read_graph, register_staging
from src.reproduction.db.snapshot import export_snapshot
from src.reproduction.errors import PreparationBlocked, ReproductionError
from src.reproduction.paths import ROOT
from src.reproduction.prepare.assemble import (
    build_voicebank_graph,
    load_dataset_specs,
    load_layout,
    load_protocols,
    make_corpus_asset,
)
from src.reproduction.prepare.collect import AudioFile, collect_audio, missing_audio, read_catalog
from src.reproduction.prepare.compare import compare_sources
from src.reproduction.prepare.lists import load_manifest
from src.reproduction.snapshot_reader import load_snapshot

STEPS = ("diagnose", "lists", "resolve", "register", "snapshot", "verify")


@dataclass
class PreparePaths:
    database: Path
    staging: Path
    snapshots: Path
    resource_db: Path | None
    corpora: Path
    derived: Path
    repo_root: Path = ROOT
    config_root: Path = ROOT / "configs" / "data"
    workers: int = 8


def execute(step: str, paths: PreparePaths) -> int:
    if step not in STEPS and step != "all":
        raise ReproductionError(f"unknown prepare step: {step}")
    selected = STEPS if step == "all" else (step,)
    state: dict[str, Any] | None = None
    for name in selected:
        if name in ("diagnose", "lists", "resolve", "register") and state is None:
            state = _resolve(paths, write=False)
        if name == "diagnose":
            _print_resolution(state)
        elif name == "lists":
            _print_lists(state)
        elif name == "resolve":
            _print_resolution(state)
        elif name == "register":
            _register(paths, state)
        elif name == "snapshot":
            _snapshot(paths)
        elif name == "verify":
            _verify(paths)
    return 0


def _resolve(paths: PreparePaths, *, write: bool) -> dict[str, Any]:
    """Read lists and audio. ``write`` is unused and must stay false."""
    del write
    manifest_path = paths.config_root / "lists" / "manifest.yaml"
    if not manifest_path.is_file():
        raise PreparationBlocked(f"list manifest is missing: {manifest_path}")
    lists = load_manifest(manifest_path, paths.repo_root)
    for model in ("mp_senet", "mossformergan"):
        if model not in lists:
            raise PreparationBlocked(f"list manifest has no {model} entry")
    _require_same_ids(lists)
    layout = load_layout(paths.config_root)
    train_ids = lists["mp_senet"]["train"]["ids"]
    test_ids = lists["mp_senet"]["test"]["ids"]
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ReproductionError(f"train and test lists overlap: {overlap[:5]}")
    if not paths.corpora.is_dir():
        raise PreparationBlocked(f"corpus directory is missing: {paths.corpora}")
    # Existence is checked before headers are read and before the resource database is opened.
    missing = missing_audio(corpora=paths.corpora, layout=layout, train_ids=train_ids, test_ids=test_ids)
    if missing:
        preview = ", ".join(missing[:5])
        raise PreparationBlocked(
            f"{len(missing)} VoiceBank files are missing under {paths.corpora}. Examples: {preview}"
        )
    catalog = None
    source_db_sha256 = None
    if paths.resource_db is not None:
        if not paths.resource_db.is_file():
            raise PreparationBlocked(f"resource database is missing: {paths.resource_db}")
        before = sha256_file(paths.resource_db)
        subsets = [
            layout["train_clean_subset"],
            layout["train_noisy_subset"],
            layout["test_clean_subset"],
            layout["test_noisy_subset"],
        ]
        catalog = read_catalog(paths.resource_db, layout["corpus"], subsets)
        audio = collect_audio(
            corpora=paths.corpora,
            layout=layout,
            train_ids=train_ids,
            test_ids=test_ids,
            catalog=catalog,
        )
        after = sha256_file(paths.resource_db)
        if after != before:
            raise ReproductionError("read-only catalog query changed resource.duckdb")
        source_db_sha256 = before
    else:
        audio = collect_audio(
            corpora=paths.corpora,
            layout=layout,
            train_ids=train_ids,
            test_ids=test_ids,
            catalog=None,
        )
    return {
        "lists": lists,
        "layout": layout,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "audio": audio,
        "source_db_sha256": source_db_sha256,
    }


def _register(paths: PreparePaths, state: dict[str, Any]) -> None:
    if not paths.database.is_file():
        initialize_database(paths.database)
    graph = _align_existing_assets(_graph(paths, state), paths.database)
    from src.reproduction.db.staging import write_graph

    write_graph(paths.staging, graph)
    resource = None if paths.resource_db is None else str(paths.resource_db)
    counts = register_staging(paths.staging, paths.database, resource_db=resource)
    print("register\t" + "\t".join(f"{name}={count}" for name, count in sorted(counts.items()) if count), flush=True)


def _snapshot(paths: PreparePaths) -> None:
    specs = load_dataset_specs(paths.config_root)
    for spec in specs:
        allow_unknown = spec["profile"] == "leakage_safe"
        digest = export_snapshot(
            paths.database,
            f"{spec['name']}@{spec['version']}",
            paths.snapshots,
            allow_unknown_leak_check=allow_unknown,
            corpora=paths.corpora,
            derived=paths.derived,
        )
        print(f"snapshot\t{spec['name']}@{spec['version']}\t{digest}\tallow_unknown={allow_unknown}", flush=True)


def _verify(paths: PreparePaths) -> None:
    before = sha256_file(paths.resource_db) if paths.resource_db is not None and paths.resource_db.is_file() else None
    specs = load_dataset_specs(paths.config_root)
    graph = read_graph(paths.database)
    datasets = {row["name"]: row for row in graph.datasets}
    report_rows = []
    for spec in specs:
        dataset = datasets.get(spec["name"])
        if dataset is None:
            raise ReproductionError(f"dataset is not registered: {spec['name']}")
        loaded = load_snapshot(paths.snapshots / dataset["dataset_id"])
        counts = _split_counts(loaded)
        for slot, expected in spec["expect"].items():
            if counts[slot] != expected:
                raise ReproductionError(f"{spec['name']} {slot} snapshot has {counts[slot]}, expected {expected}")
        again = export_snapshot(
            paths.database,
            f"{spec['name']}@{spec['version']}",
            paths.snapshots,
            allow_unknown_leak_check=spec["profile"] == "leakage_safe",
            corpora=paths.corpora,
            derived=paths.derived,
        )
        if again != loaded.snapshot_hash:
            raise ReproductionError(f"{spec['name']} snapshot hash changed on reload")
        audits = [
            row
            for row in graph.audits
            if row["subject_kind"] == "dataset" and row["subject_id"] == dataset["dataset_id"]
        ]
        report_rows.append(
            {
                "name": spec["name"],
                "version": spec["version"],
                "dataset_id": dataset["dataset_id"],
                "profile": dataset["profile"],
                "selection_uses_test": bool(dataset["selection_uses_test"]),
                "counts": counts,
                "snapshot_hash": loaded.snapshot_hash,
                "allow_unknown_leak_check": bool(loaded.manifest["provenance"]["allow_unknown_leak_check"]),
                "audits": {row["audit_kind"]: row["status"] for row in audits},
            }
        )
        print(
            f"verify\t{spec['name']}\t{counts['train']}/{counts['valid']}/{counts['test']}\t{loaded.snapshot_hash}",
            flush=True,
        )
    comparison = _comparison(paths)
    identity = _fifty_six_note(paths)
    if before is not None and sha256_file(paths.resource_db) != before:
        raise ReproductionError("verification changed resource.duckdb")
    report = {"datasets": report_rows, "librosa_comparison": comparison, "fifty_six_speaker": identity}
    destination = paths.staging / "voicebank-ondemand-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report\t{destination}", flush=True)


def _graph(paths: PreparePaths, state: dict[str, Any]):
    layout = state["layout"]
    cache = _hash_cache(paths.database)
    grouped: dict[str, dict[str, AudioFile]] = {}
    for record in state["audio"]:
        grouped.setdefault(record.utterance_id, {})[record.role] = record
    utterances = {}
    checked = 0
    for utterance_id, sides in grouped.items():
        clean = sides["clean"]
        noisy = sides["noisy"]
        utterances[utterance_id] = {
            "utterance_id": utterance_id,
            "speaker_id": clean.speaker_id,
            "transcript": clean.transcript,
            "clean": {"corpus": _corpus_asset(paths, clean, cache, verify=checked < 8)},
            "noisy": {"corpus": _corpus_asset(paths, noisy, cache, verify=checked < 8)},
        }
        checked += 1
    return build_voicebank_graph(
        protocols=load_protocols(paths.config_root),
        datasets=load_dataset_specs(paths.config_root),
        list_meta=state["lists"],
        train_ids=state["train_ids"],
        test_ids=state["test_ids"],
        holdout=tuple(layout["holdout_speakers"]),
        utterances=utterances,
        catalog_source_kind="resource_duckdb" if paths.resource_db is not None else "staging_parquet",
        source_db_sha256=state["source_db_sha256"],
        resource_db=None if paths.resource_db is None else str(paths.resource_db),
    )


def _align_existing_assets(graph, database: Path):
    """Keep corpus rows that are already registered.

    A new catalog id on an unchanged file would be a different asset row.
    Registration would then roll back. The existing catalog stays the source.
    """
    if not database.is_file():
        return graph
    from src.reproduction.db.registry import read_graph

    existing = {row["asset_id"]: row for row in read_graph(database).assets}
    if not existing:
        return graph
    for asset in graph.assets:
        previous = existing.get(asset["asset_id"])
        if previous is None:
            continue
        asset["catalog_id"] = previous["catalog_id"]
        if asset != previous:
            raise ReproductionError(
                f"corpus asset {asset['asset_id']} does not match the registered row"
            )
    used = {asset["catalog_id"] for asset in graph.assets}
    graph.catalogs = [row for row in graph.catalogs if row["catalog_id"] in used]
    graph.catalog_rows = {key: value for key, value in graph.catalog_rows.items() if key in used}
    return graph


def _hash_cache(database: Path) -> dict[str, dict[str, Any]]:
    if not database.is_file():
        return {}
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        try:
            rows = connection.execute(
            """
            SELECT rel_path, sha256, frames, size_bytes, sample_rate
            FROM repro.assets
            WHERE storage_root = 'corpora' AND corpus = 'VoiceBank-DEMAND'
            """
            ).fetchall()
        except duckdb.Error:
            return {}
    finally:
        connection.close()
    return {
        rel: {"sha256": digest, "frames": int(frames), "size_bytes": int(size), "sample_rate": int(rate)}
        for rel, digest, frames, size, rate in rows
    }


def _corpus_asset(paths: PreparePaths, record: AudioFile, cache: dict[str, dict[str, Any]], *, verify: bool) -> dict:
    cached = cache.get(record.rel_path)
    source = paths.corpora / record.rel_path
    if (
        cached
        and cached["frames"] == record.frames
        and cached["size_bytes"] == record.size_bytes
        and cached["sample_rate"] == record.sample_rate
        and not verify
    ):
        digest = str(cached["sha256"])
    else:
        digest = sha256_file(source)
        if cached and digest != cached["sha256"]:
            raise ReproductionError(f"stored sha256 does not match {source}")
    return make_corpus_asset(
        rel_path=record.rel_path,
        subset=record.subset,
        speaker_id=record.speaker_id,
        utterance_id=record.utterance_id,
        sample_rate=record.sample_rate,
        frames=record.frames,
        channels=record.channels,
        size_bytes=record.size_bytes,
        sha256=digest,
        role=record.role,
    )


def _comparison(paths: PreparePaths) -> dict[str, Any]:
    layout = load_layout(paths.config_root)
    lists = load_manifest(paths.config_root / "lists" / "manifest.yaml", paths.repo_root)
    train_ids = lists["mp_senet"]["train"]["ids"]
    test_ids = lists["mp_senet"]["test"]["ids"]
    chosen = []
    for split, ids, clean_key, noisy_key in (
        ("train", train_ids, "train_clean_subset", "train_noisy_subset"),
        ("test", test_ids, "test_clean_subset", "test_noisy_subset"),
    ):
        picks = {ids[0], ids[-1]}
        lengths = []
        for utterance_id in ids:
            source = paths.corpora / layout["corpus"] / layout[clean_key] / f"{utterance_id}.wav"
            lengths.append((source.stat().st_size, utterance_id))
        lengths.sort()
        picks.add(lengths[0][1])
        picks.add(lengths[-1][1])
        for utterance_id in sorted(picks):
            for key in (clean_key, noisy_key):
                subset = layout[key]
                source = paths.corpora / layout["corpus"] / subset / f"{utterance_id}.wav"
                derived = paths.derived / "VoiceBank-DEMAND" / "soxr_hq_16000" / subset / f"{utterance_id}.wav"
                chosen.append((source, derived if derived.is_file() else None))
    return compare_sources(chosen)


def _fifty_six_note(paths: PreparePaths) -> dict[str, Any]:
    left = paths.corpora / "VoiceBank-DEMAND" / "clean_trainset_28spk_wav" / "p226_001.wav"
    right = paths.corpora / "VoiceBank-DEMAND" / "clean_trainset_56spk_wav" / "p226_001.wav"
    if not left.is_file() or not right.is_file():
        return {"checked": False}
    return {
        "checked": True,
        "file": "p226_001.wav",
        "same_bytes": sha256_file(left) == sha256_file(right),
        "note": "56-speaker files are a different path and are not registered. Byte identity was checked for this one file only.",
    }


def _split_counts(loaded) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    sets: dict[str, str] = {}
    for binding in loaded.dataset["bindings"]:
        sets[binding["case_set_id"]] = binding["slot"].split(".", 1)[0]
    for item in loaded.case_set_items:
        counts[sets[item["case_set_id"]]] += 1
    return counts


def _require_same_ids(lists: dict[str, dict]) -> None:
    left = lists["mp_senet"]
    right = lists["mossformergan"]
    for side in ("train", "test"):
        if left[side]["ids"] != right[side]["ids"]:
            raise ReproductionError(
                f"MP-SENet and MossFormerGAN {side} ids differ; refusing to invent a merged list"
            )


def _print_lists(state: dict[str, Any]) -> None:
    for model, spec in sorted(state["lists"].items()):
        print(
            f"lists\t{model}\ttrain={len(spec['train']['ids'])}\ttest={len(spec['test']['ids'])}\t{spec['commit']}",
            flush=True,
        )


def _print_resolution(state: dict[str, Any]) -> None:
    _print_lists(state)
    print(
        f"resolve\taudio={len(state['audio'])}\ttrain={len(state['train_ids'])}\ttest={len(state['test_ids'])}",
        flush=True,
    )
