"""Resumable manifests for the full DNS acceptance plan.

A shard is metadata: the accepted spans, SNR, level, retry counts, and the
content hash. Waveforms are not written. A later run reuses a shard only when
the build, protocol, material order, and generator source hash still match.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.experiments.data.build.dns2020 import (
    GENERATOR,
    GENERATOR_VERSION,
    RENDERER_ACCEPT,
    RETRY_ACCEPT,
    _stored,
)
from src.experiments.data.db.staging import git_commit
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest
from src.experiments.data.paths import ROOT

GENERATOR_FILES = (
    "src/experiments/data/build/dns2020.py",
    "src/experiments/data/mix/dns_ondemand.py",
    "src/experiments/data/sampling/seeds.py",
    "src/experiments/data/prepare/dns2020.py",
    "src/experiments/data/prepare/dns_accept_store.py",
    "src/experiments/data/prepare/dns_run.py",
)
IDENTITY_FIELDS = (
    "build_id",
    "order_sha256",
    "protocol_id",
    "generator",
    "generator_version",
    "retry_policy",
    "renderer",
    "seed",
    "num_files",
    "plan_shard",
    "generator_code_sha256",
)


def generator_code_identity(root: Path = ROOT) -> dict[str, Any]:
    """Hash the generator sources. A dirty worktree is not identified by HEAD alone."""
    files = []
    for rel in GENERATOR_FILES:
        path = root / rel
        if not path.is_file():
            raise ReproductionError(f"generator source is missing: {path}")
        raw = path.read_bytes()
        files.append({"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    code = digest(files)
    porcelain = _porcelain(root)
    return {
        "generator_code_sha256": code,
        "files": files,
        "git_commit": git_commit(),
        "worktree_dirty": bool(porcelain.strip()),
        "porcelain_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
    }


def plan_identity(
    definition: dict[str, Any],
    *,
    protocol_id: str,
    plan_shard: int,
    code_sha256: str,
) -> dict[str, Any]:
    params = definition["params"]
    if params.get("retry_policy") != RETRY_ACCEPT or params.get("renderer") != RENDERER_ACCEPT:
        raise ReproductionError("acceptance shards require the P7.1 retry policy and renderer")
    if int(definition["generator_version"]) != GENERATOR_VERSION or definition["generator"] != GENERATOR:
        raise ReproductionError("acceptance shards require dns2020_synth version 1")
    identity = {
        "build_id": str(definition["build_id"]),
        "order_sha256": str(params["order_sha256"]),
        "protocol_id": str(protocol_id),
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "retry_policy": RETRY_ACCEPT,
        "renderer": RENDERER_ACCEPT,
        "seed": int(definition["seed"]),
        "num_files": int(params["num_files"]),
        "plan_shard": int(plan_shard),
        "generator_code_sha256": str(code_sha256),
    }
    if identity["plan_shard"] < 1:
        raise ReproductionError("plan_shard must be positive")
    return identity


def shard_ranges(total: int, plan_shard: int) -> list[tuple[int, int]]:
    if total < 1 or plan_shard < 1:
        raise ReproductionError("shard grid needs a positive count and shard size")
    return [(start, min(total, start + plan_shard)) for start in range(0, total, plan_shard)]


def shard_path(directory: Path, start: int, stop: int) -> Path:
    return directory / f"{int(start):07d}-{int(stop):07d}.jsonl"


def write_shard(
    directory: Path,
    identity: dict[str, Any],
    start: int,
    stop: int,
    rows: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> Path:
    """Write one contiguous gen_index range. The same bytes are a no-op."""
    start = int(start)
    stop = int(stop)
    if len(rows) != stop - start:
        raise ReproductionError(f"shard {start}:{stop} has {len(rows)} rows")
    encoded = []
    for offset, (document, stats) in enumerate(rows):
        gen_index = int(document["draws"]["gen_index"])
        if gen_index != start + offset:
            raise ReproductionError(f"shard {start}:{stop} row {offset} is gen_index {gen_index}")
        stored = _stored(document)
        encoded.append(
            {
                "kind": "case",
                "gen_index": gen_index,
                "content_sha256": digest(stored),
                "document": stored,
                "stats": {key: int(value) if key != "retries" else int(value) for key, value in stats.items()},
            }
        )
    header = {"kind": "shard", "start": start, "stop": stop, "count": stop - start, **identity}
    text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in [header, *encoded])
    directory.mkdir(parents=True, exist_ok=True)
    path = shard_path(directory, start, stop)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ReproductionError(f"acceptance shard already exists with different content: {path}")
        return path
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)
    return path


def read_shard(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReproductionError(f"cannot read acceptance shard {path}: {exc}") from exc
    if len(lines) < 2:
        raise ReproductionError(f"acceptance shard {path} is incomplete")
    try:
        header = json.loads(lines[0])
        rows = [json.loads(line) for line in lines[1:] if line]
    except json.JSONDecodeError as exc:
        raise ReproductionError(f"acceptance shard {path} is not JSON") from exc
    if header.get("kind") != "shard":
        raise ReproductionError(f"acceptance shard {path} has no header")
    start = int(header["start"])
    stop = int(header["stop"])
    if len(rows) != stop - start:
        raise ReproductionError(f"acceptance shard {path} count does not match {start}:{stop}")
    for offset, row in enumerate(rows):
        if row.get("kind") != "case" or int(row["gen_index"]) != start + offset:
            raise ReproductionError(f"acceptance shard {path} breaks gen_index order at {offset}")
        document = row["document"]
        if digest(_stored(document)) != row["content_sha256"]:
            raise ReproductionError(f"acceptance shard {path} gen_index {row['gen_index']} hash does not match")
    return header, rows


def missing_ranges(directory: Path, identity: dict[str, Any], total: int) -> list[tuple[int, int]]:
    """Return unfinished grid ranges. A mismatched shard is an error, not a silent redo."""
    expected = shard_ranges(total, int(identity["plan_shard"]))
    if not directory.is_dir():
        return expected
    found: dict[tuple[int, int], Path] = {}
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.endswith(".partial"):
            continue
        header, _rows = read_shard(path)
        _require_identity(header, identity)
        key = (int(header["start"]), int(header["stop"]))
        if key in found:
            raise ReproductionError(f"duplicate acceptance shard range {key[0]}:{key[1]}")
        found[key] = path
    expected_set = set(expected)
    extra = sorted(set(found) - expected_set)
    if extra:
        raise ReproductionError(f"acceptance shard grid does not match plan_shard: {extra[:3]}")
    return [item for item in expected if item not in found]


def load_documents(directory: Path, identity: dict[str, Any], total: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load every case in gen_index order. Missing or duplicate indexes are errors."""
    documents: list[dict[str, Any] | None] = [None] * int(total)
    stats: list[dict[str, Any] | None] = [None] * int(total)
    if not directory.is_dir():
        raise ReproductionError(f"acceptance shard directory is missing: {directory}")
    for path in sorted(directory.glob("*.jsonl")):
        header, rows = read_shard(path)
        _require_identity(header, identity)
        for row in rows:
            gen_index = int(row["gen_index"])
            if gen_index < 0 or gen_index >= total:
                raise ReproductionError(f"gen_index {gen_index} is outside 0:{total}")
            if documents[gen_index] is not None:
                raise ReproductionError(f"duplicate gen_index {gen_index}")
            documents[gen_index] = row["document"]
            stats[gen_index] = row["stats"]
    missing = [index for index, document in enumerate(documents) if document is None]
    if missing:
        raise ReproductionError(
            f"acceptance plan is missing {len(missing)} gen_index values, first {missing[:5]}"
        )
    return [item for item in documents if item is not None], [item for item in stats if item is not None]


def summarize_cases(
    documents: Sequence[dict[str, Any]],
    stats: Sequence[dict[str, Any]],
    *,
    clean_names: Iterable[str],
    noise_names: Iterable[str],
) -> dict[str, Any]:
    """Metadata audit. This does not render waveforms."""
    if len(documents) != len(stats):
        raise ReproductionError("acceptance summary rows and stats differ in length")
    clean = set(clean_names)
    noise = set(noise_names)
    seen: set[int] = set()
    uttids: set[str] = set()
    histogram: dict[str, int] = {}
    snr: dict[str, int] = {}
    level: dict[str, int] = {}
    max_retries = 0
    non16_cases = 0
    non16_pieces = 0
    multi_clean = 0
    multi_noise = 0
    clean_clip_cases = 0
    noise_clip_cases = 0
    activity_min = None
    short_pieces = 0
    for document, row_stats in zip(documents, stats, strict=True):
        gen_index = int(document["draws"]["gen_index"])
        if gen_index in seen:
            raise ReproductionError(f"duplicate gen_index {gen_index}")
        seen.add(gen_index)
        if int(document["output"]["length"]) != 480000:
            raise ReproductionError(f"gen_index {gen_index} length is {document['output']['length']}")
        value = int(document["draws"]["snr_db"])
        level_value = int(document["draws"]["target_level_dbfs"])
        if value < -5 or value >= 15:
            raise ReproductionError(f"gen_index {gen_index} SNR {value} is outside -5..14")
        if level_value < -35 or level_value >= -15:
            raise ReproductionError(f"gen_index {gen_index} level {level_value} is outside -35..-16")
        retries = int(document["draws"]["retries"])
        if retries < 0 or retries >= 50:
            raise ReproductionError(f"gen_index {gen_index} retries {retries} is outside 0..49")
        if retries != int(row_stats["retries"]):
            raise ReproductionError(f"gen_index {gen_index} retry count does not match its stats")
        histogram[str(retries)] = histogram.get(str(retries), 0) + 1
        snr[str(value)] = snr.get(str(value), 0) + 1
        level[str(level_value)] = level.get(str(level_value), 0) + 1
        max_retries = max(max_retries, retries)
        uttid = str(document["draws"]["utt_id"])
        if uttid in uttids:
            raise ReproductionError(f"duplicate DNS uttid {uttid}")
        uttids.add(uttid)
        speech_files = [piece for piece in document["sources"]["speech"] if "name" in piece]
        noise_files = [piece for piece in document["sources"]["noise"] if "name" in piece]
        if not speech_files or not noise_files:
            raise ReproductionError(f"gen_index {gen_index} is missing a speech or noise piece")
        if len(speech_files) > 1:
            multi_clean += 1
        if len(noise_files) > 1:
            multi_noise += 1
        speech_ids = {id(piece) for piece in speech_files}
        case_non16 = 0
        for piece in speech_files + noise_files:
            name = str(piece["name"])
            pool = clean if id(piece) in speech_ids else noise
            if name not in pool:
                raise ReproductionError(f"gen_index {gen_index} uses unknown material {name}")
            if int(piece["length"]) < 1:
                raise ReproductionError(f"gen_index {gen_index} has an empty piece")
            if int(piece["length"]) < 16000:
                short_pieces += 1
            if int(piece.get("source_rate", 16000)) != 16000:
                case_non16 += 1
        if case_non16:
            non16_cases += 1
            non16_pieces += case_non16
        if int(row_stats["clean_clip_skips"]):
            clean_clip_cases += 1
        if int(row_stats["noise_clip_skips"]):
            noise_clip_cases += 1
        activity = document["draws"].get("clean_activity")
        if activity is None or not float(activity) > 0.6:
            raise ReproductionError(f"gen_index {gen_index} clean activity is not above 0.6")
        activity_min = float(activity) if activity_min is None else min(activity_min, float(activity))
    if seen != set(range(len(documents))):
        raise ReproductionError("acceptance gen_index values are not 0..count-1")
    return {
        "cases": len(documents),
        "failed_cases": 0,
        "max_retries": max_retries,
        "attempt_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "snr_counts": dict(sorted(snr.items(), key=lambda item: int(item[0]))),
        "level_counts": dict(sorted(level.items(), key=lambda item: int(item[0]))),
        "non16_cases": non16_cases,
        "non16_pieces": non16_pieces,
        "multi_clean_cases": multi_clean,
        "multi_noise_cases": multi_noise,
        "clean_clip_cases": clean_clip_cases,
        "noise_clip_cases": noise_clip_cases,
        "short_piece_count": short_pieces,
        "clean_activity_min": activity_min,
        "uttids": len(uttids),
    }


def _require_identity(header: dict[str, Any], identity: dict[str, Any]) -> None:
    for key in IDENTITY_FIELDS:
        if header.get(key) != identity[key]:
            raise ReproductionError(
                f"acceptance shard {key} is {header.get(key)}, plan identity is {identity[key]}"
            )


def _porcelain(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
