"""Register the shared VoiceBank+DEMAND SNR-shift recipe.

One realization per utterance is stored. ``snr_shift_db`` is a protocol field,
so another SNR does not need another recipe. Rendering reads VCTK and the
noise asset only. Published clean and noisy files are a comparison reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.recipe import sha256_file
from src.experiments.data.db.audit import audits_for
from src.experiments.data.db.graph import Graph
from src.experiments.data.db.registry import initialize_database, register_staging
from src.experiments.data.db.snapshot import export_snapshot
from src.experiments.data.db.staging import load_graph, write_graph
from src.experiments.data.db.validate import validate_graph
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import (
    VOICEBANK_SNR_MIX_V1,
    audit_id_for,
    build_id_for,
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
from src.experiments.data.mix.voicebank_snr import (
    ALGORITHM,
    SAMPLE_RATE,
    STEPS,
    clean_scale_for,
    component_snr_db,
    decimal_string,
    noise_scale,
    official_pcm16,
    p56_snr_db,
    parse_decimal,
    pcm16_round_nearest_even,
    pcm_delta_stats,
    read_span,
    render_components,
)
from src.experiments.data.paths import DATA, DEFAULT_CORPORA, DEFAULT_SNAPSHOTS, ROOT

GENERATOR = "voicebank_snr_shift"
GENERATOR_VERSION = 1
DB_PATH = DATA / "db" / "reproduction-voicebank-snr-v1.duckdb"
STAGING = DATA / "staging" / "reproduction-voicebank-snr-v1"
EXPECTED_COUNTS = {"train": 10700, "valid": 872, "test": 824}
HOLDOUT_SPEAKERS = frozenset({"p286", "p287"})
TEST_SPEAKERS = frozenset({"p232", "p257"})
CORPUS_PREFIX = "data/corpora/"
OFFICIAL_REF = {
    "document": "VoiceBank-DEMAND",
    "path": "data/corpora/VoiceBank-DEMAND",
    "section": "published pairs are an inverse-analysis reference, not a generation input",
}
DEVIATIONS = [
    {
        "id": "published_mixer_unrecovered",
        "class": "research_improvement",
        "text": (
            "The published mixer source was not recovered. Gains, noise intervals, and the "
            "peak steps are the measured candidate. PCM16 uses round-to-nearest-even and is "
            "not claimed to be the publisher's quantizer."
        ),
    },
    {
        "id": "official_test_reference",
        "class": "research_improvement",
        "text": (
            "Official test clean and noisy files were used while fitting and checking the recipe. "
            "Training does not read them, and test is not a checkpoint-selection split."
        ),
    },
    {
        "id": "train_valid_noise_overlap",
        "class": "research_improvement",
        "text": (
            "Train and validation speakers are disjoint. Their noise files and noise intervals "
            "overlap. That overlap is not speech leakage and is not independence."
        ),
    },
    {
        "id": "fifty_six_speaker_excluded",
        "class": "research_improvement",
        "text": "The 56-speaker VoiceBank+DEMAND release is not part of this dataset.",
    },
]


def register_voicebank_snr(recipes: str | Path | None = None, *, workers: int | None = None) -> dict[str, Any]:
    """Measure from ``recipes`` when given, then register the staging graph and export snapshots."""

    if recipes is not None:
        build_from_recipes(Path(recipes), workers=workers)
    elif not (STAGING / "INDEX.json").is_file():
        raise ReproductionError(f"staging is missing: {STAGING}")
    initialize_database(DB_PATH)
    counts = register_staging(STAGING, DB_PATH)
    graph = load_graph(STAGING)
    snapshots = {}
    for name in ("voicebank.snr_shift.no", "voicebank.snr_shift.minus_15db"):
        snapshots[name] = export_snapshot(DB_PATH, f"{name}@1", DEFAULT_SNAPSHOTS, corpora=DEFAULT_CORPORA)
    return {"inserted": counts, "snapshots": snapshots, "datasets": _dataset_index(graph)}


def build_from_recipes(recipes: Path, *, workers: int | None = None) -> dict[str, Any]:
    """Read a prior recipe table, compare every utterance, and write staging."""

    rows = _read_recipes(recipes)
    assert_population(rows)
    source_sha = sha256_file(recipes)
    assets = _index_sources(rows)
    _check_headers(assets)
    build_id = _provisional_build_id(assets, source_sha)
    specs = [_worker_spec(row, assets, build_id) for row in rows]
    measured = _measure_all(specs, workers=workers)
    by_id = {row["utterance_id"]: row for row in rows}
    records = []
    verification_rows = []
    for spec in specs:
        result = measured[spec["utterance_id"]]
        record = _record(by_id[spec["utterance_id"]], assets, result)
        records.append(record)
        verification_rows.append(_verification_row(record, result))
    _assign_indexes(records)
    report = _aggregate(records, verification_rows, source_recipes_sha256=source_sha)
    _write_verification(verification_rows, report)
    _write_record_cache(records)
    if report["unaligned"] or report["nonfinite"] or not report["deterministic"]:
        raise ReproductionError(
            "voicebank measurement cannot be registered: "
            f"unaligned={report['unaligned']} nonfinite={report['nonfinite']} deterministic={report['deterministic']}"
        )
    graph = build_voicebank_snr_graph(records, verification_report=report, source_recipes_sha256=source_sha)
    validate_graph(graph)
    write_graph(STAGING, graph)
    cache = STAGING / "verification" / "records.jsonl"
    if cache.is_file():
        cache.unlink()
    return report


def assert_population(rows: Sequence[dict[str, Any]]) -> None:
    counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    seen = set()
    for row in rows:
        utterance = str(row["utterance_id"])
        if utterance in seen:
            raise ReproductionError(f"duplicate utterance {utterance}")
        seen.add(utterance)
        counts[row["split"]] += 1
    if counts != EXPECTED_COUNTS or len(rows) != sum(EXPECTED_COUNTS.values()):
        raise ReproductionError(f"voicebank population is {counts}, expected {EXPECTED_COUNTS}")


def build_voicebank_snr_graph(
    records: Sequence[dict[str, Any]],
    *,
    verification_report: dict[str, Any],
    source_recipes_sha256: str,
) -> Graph:
    """Build the shared-recipe graph. SNR shift is not copied into the realizations."""

    ordered = _assign_indexes(list(records))
    catalog, catalog_rows, assets = _assets_from_records(ordered)
    materials = _materials(ordered)
    build = _build(materials, source_recipes_sha256)
    published, no_shift, minus = _protocols()
    documents = {
        record["utterance_id"]: realization_document(record, build_id=build["build_id"], gen_index=int(record["gen_index"]))
        for record in ordered
    }
    cases = [
        _case(documents[record["utterance_id"]], protocol_id=no_shift["protocol_id"])
        for record in ordered
    ]
    case_sets = _case_sets(ordered, cases)
    datasets = [
        _blocked_dataset(published),
        _ready_dataset("voicebank.snr_shift.no", no_shift, case_sets, materials, shift=0),
        _ready_dataset("voicebank.snr_shift.minus_15db", minus, case_sets, materials, shift=-15),
    ]
    report = dict(verification_report)
    report["recipe_sha256"] = _recipe_sha256(documents, ordered)
    report["source_recipes_sha256"] = source_recipes_sha256
    audits = [
        _distribution_audit(dataset["dataset_id"], report)
        for dataset in datasets
        if dataset["declared_status"] == "ready"
    ]
    return Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: catalog_rows},
        assets=list(assets.values()),
        materials=list(materials.values()),
        protocols=[published, no_shift, minus],
        cases=cases,
        case_sets=list(case_sets.values()),
        datasets=datasets,
        builds=[build],
        verified={
            build["build_id"]: {
                "build_id": build["build_id"],
                "status": "PASS",
                "delivery": "on_demand",
                "wav_files": 0,
                "shard_ids": ["recipes"],
                "realizations": len(cases),
            }
        },
        audits=audits,
    )


def realization_document(record: dict[str, Any], *, build_id: str, gen_index: int) -> dict[str, Any]:
    speech = record["speech"]
    noise = record["noise"]
    length = int(speech["end"]) - int(speech["start"])
    return {
        "realization_version": 1,
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "sample_rate": SAMPLE_RATE,
        "sources": {
            "speech": {
                "asset": speech["asset_id"],
                "sha256": speech["sha256"],
                "channel": int(speech["channel"]),
                "start": int(speech["start"]),
                "end": int(speech["end"]),
            },
            "noise": {
                "asset": noise["asset_id"],
                "sha256": noise["sha256"],
                "channel": int(noise["channel"]),
                "start": int(noise["start"]),
                "end": int(noise["end"]),
                "kind": noise["kind"],
                "name": noise["name"],
            },
        },
        "draws": {
            "gen_index": int(gen_index),
            "build_id": build_id,
            "utterance_id": record["utterance_id"],
            "split": record["split"],
            "speaker_id": record["speaker_id"],
            "official_snr_db": record["official_snr_db"],
            "clean_scale": record["clean_scale"],
            "clean_gain_g": record["clean_gain_g"],
            "noise_gain_before_peak_beta": record["noise_gain_before_peak_beta"],
            "noise_gain_after_peak": record["noise_gain_after_peak"],
            "peak_steps": int(record["peak_steps"]),
            "target_peak_pcm": 16384,
            "silence": {
                "mode": "contiguous_slice",
                "source_frames": int(record["silence"]["source_frames"]),
                "leading_removed": int(record["silence"]["leading_removed"]),
                "trailing_removed": int(record["silence"]["trailing_removed"]),
            },
        },
        "steps": list(STEPS),
        "output": {"length": length, "teacher": "dry_clean"},
    }


def published_pair_paths(split: str, utterance_id: str) -> tuple[str, str]:
    """Published files used only by the comparison. Generation does not call this."""

    if split == "test":
        clean_dir, noisy_dir = "clean_testset_wav", "noisy_testset_wav"
    else:
        clean_dir, noisy_dir = "clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"
    base = f"{CORPUS_PREFIX}VoiceBank-DEMAND"
    return f"{base}/{clean_dir}/{utterance_id}.wav", f"{base}/{noisy_dir}/{utterance_id}.wav"


def normalize_recipe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Accept either numeric fields or the decimal strings from the DEMAND rows."""

    utterance = str(row["utterance_id"])
    speaker = utterance.split("_", 1)[0]
    recipe_split = str(row["split"])
    if recipe_split == "test":
        split = "test"
        if speaker not in TEST_SPEAKERS:
            raise ReproductionError(f"{utterance} is in the test split but not a test speaker")
    elif recipe_split == "train":
        if speaker in TEST_SPEAKERS:
            raise ReproductionError(f"test speaker {utterance} is in the train recipe split")
        split = "valid" if speaker in HOLDOUT_SPEAKERS else "train"
    else:
        raise ReproductionError(f"{utterance} has an unknown split {recipe_split}")
    noise_name = str(row["noise_type"])
    if noise_name in {"babble", "ssn"}:
        kind = noise_name
    else:
        kind = "demand"
    channel = _as_int(row["noise_channel_1based"], "noise_channel_1based") - 1
    if channel < 0:
        raise ReproductionError(f"{utterance} noise channel is not 1-based")
    start = _as_int(row["noise_start_48k"], "noise_start_48k")
    end = _as_int(row["noise_end_exclusive_48k"], "noise_end_exclusive_48k")
    offset = _as_int(row["vctk_alignment_offset"], "vctk_alignment_offset")
    if end <= start or offset < 0:
        raise ReproductionError(f"{utterance} has an empty or negative slice")
    expected_clean, expected_noisy = published_pair_paths(split, utterance)
    if "clean_path" in row and str(row["clean_path"]) != expected_clean:
        raise ReproductionError(f"{utterance} published clean path is not {expected_clean}")
    if "noisy_path" in row and str(row["noisy_path"]) != expected_noisy:
        raise ReproductionError(f"{utterance} published noisy path is not {expected_noisy}")
    return {
        "utterance_id": utterance,
        "speaker_id": speaker,
        "split": split,
        "noise_name": noise_name,
        "noise_kind": kind,
        "speech_path": _corpus_path(str(row["source_vctk"]), utterance),
        "speech_sha256": str(row["source_vctk_sha256"]),
        "speech_start": offset,
        "noise_path": _corpus_path(str(row["noise_path"]), utterance),
        "noise_sha256": str(row["noise_sha256"]),
        "noise_channel": channel,
        "noise_start": start,
        "noise_end": end,
        "official_snr_db": _snr(row["official_log_snr_db"]),
        "clean_gain_g": decimal_string(float(row["clean_gain_g"])),
        "noise_gain_before_peak_beta": decimal_string(float(row["noise_gain_before_peak_beta"])),
        "noise_gain_after_peak": decimal_string(float(row["noise_gain_after_peak"])),
        "peak_steps": _as_int(row["peak_steps"], "peak_steps"),
        "official_clean": expected_clean,
        "official_noisy": expected_noisy,
    }


def _read_recipes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReproductionError(f"recipe table is missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(normalize_recipe_row(json.loads(line)))
            except (TypeError, ValueError, KeyError, ReproductionError) as exc:
                raise ReproductionError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _index_sources(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        for kind, path_key, sha_key in (
            ("speech", "speech_path", "speech_sha256"),
            ("noise", "noise_path", "noise_sha256"),
        ):
            rel = row[path_key]
            asset_id = f"corpora/{rel[len(CORPUS_PREFIX):]}"
            current = found.get(asset_id)
            if current is None:
                found[asset_id] = {"asset_id": asset_id, "rel_path": rel[len(CORPUS_PREFIX):], "sha256": row[sha_key], "role": kind}
            elif current["sha256"] != row[sha_key]:
                raise ReproductionError(f"{rel} has two sha256 values")
    return found


def _check_headers(assets: dict[str, dict[str, Any]]) -> None:
    import soundfile as sf

    for index, asset in enumerate(assets.values(), 1):
        path = ROOT / CORPUS_PREFIX / asset["rel_path"]
        if not path.is_file():
            raise ReproductionError(f"source file is missing: {path}")
        digest = sha256_file(path)
        if digest != asset["sha256"]:
            raise ReproductionError(f"sha256 mismatch for {path}")
        info = sf.info(str(path))
        if int(info.samplerate) != SAMPLE_RATE or int(info.channels) < 1 or int(info.frames) < 1:
            raise ReproductionError(f"{path} is not 48 kHz audio")
        if asset["role"] == "speech" and int(info.channels) != 1:
            raise ReproductionError(f"{path} is not mono")
        asset["frames"] = int(info.frames)
        asset["channels"] = int(info.channels)
        asset["size_bytes"] = path.stat().st_size
        asset["sample_format"] = str(info.subtype or "")
        asset["corpus"] = asset["rel_path"].split("/", 1)[0]
        asset["subset"] = asset["rel_path"].split("/")[1] if asset["rel_path"].count("/") else asset["corpus"]
        if index % 2000 == 0 or index == len(assets):
            print(f"hashed\t{index}\t{len(assets)}", flush=True)


def _provisional_build_id(assets: dict[str, dict[str, Any]], source_sha: str) -> str:
    """The build id depends on the source files, not on the realizations."""

    materials = {}
    for asset in assets.values():
        materials.setdefault(asset["corpus"], []).append(asset["sha256"])
    inputs = {name: digest(sorted(shas)) for name, shas in sorted(materials.items())}
    return build_id_for(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        params={"recipe": "voicebank_snr_shift_v1", "sample_rate": SAMPLE_RATE, "source_recipes_sha256": source_sha},
        seed=0,
        inputs=inputs,
    )


def _worker_spec(row: dict[str, Any], assets: dict[str, dict[str, Any]], build_id: str) -> dict[str, Any]:
    speech = assets[f"corpora/{row['speech_path'][len(CORPUS_PREFIX):]}"]
    noise = assets[f"corpora/{row['noise_path'][len(CORPUS_PREFIX):]}"]
    end = row["speech_start"] + (row["noise_end"] - row["noise_start"])
    if end > int(speech["frames"]) or row["noise_end"] > int(noise["frames"]):
        raise ReproductionError(f"{row['utterance_id']} slice ends past the source")
    if row["noise_channel"] >= int(noise["channels"]):
        raise ReproductionError(f"{row['utterance_id']} noise channel is outside the file")
    return {
        "utterance_id": row["utterance_id"],
        "speaker_id": row["speaker_id"],
        "split": row["split"],
        "build_id": build_id,
        "speech_path": str(ROOT / row["speech_path"]),
        "speech_asset": speech["asset_id"],
        "speech_sha256": speech["sha256"],
        "speech_channel": 0,
        "speech_start": row["speech_start"],
        "speech_end": end,
        "speech_frames": int(speech["frames"]),
        "noise_path": str(ROOT / row["noise_path"]),
        "noise_asset": noise["asset_id"],
        "noise_sha256": noise["sha256"],
        "noise_channel": row["noise_channel"],
        "noise_start": row["noise_start"],
        "noise_end": row["noise_end"],
        "noise_kind": row["noise_kind"],
        "noise_name": row["noise_name"],
        "official_snr_db": row["official_snr_db"],
        "clean_gain_g": row["clean_gain_g"],
        "noise_gain_before_peak_beta": row["noise_gain_before_peak_beta"],
        "noise_gain_after_peak": row["noise_gain_after_peak"],
        "peak_steps": row["peak_steps"],
        "official_clean": str(ROOT / row["official_clean"]),
        "official_noisy": str(ROOT / row["official_noisy"]),
    }


def _measure_all(specs: Sequence[dict[str, Any]], *, workers: int | None) -> dict[str, dict[str, Any]]:
    count = int(workers if workers is not None else os.environ.get("AUDIO_DB_SNR_WORKERS", "8"))
    if count < 1:
        raise ReproductionError("worker count must be positive")
    found: dict[str, dict[str, Any]] = {}
    if count == 1:
        _warm_level()
        for index, spec in enumerate(specs, 1):
            found[spec["utterance_id"]] = _measure_one(spec)
            if index % 200 == 0 or index == len(specs):
                print(f"measured\t{index}\t{len(specs)}", flush=True)
        return found
    with ProcessPoolExecutor(max_workers=count, initializer=_warm_level) as pool:
        for index, result in enumerate(pool.map(_measure_one, specs, chunksize=8), 1):
            found[result["utterance_id"]] = result
            if index % 200 == 0 or index == len(specs):
                print(f"measured\t{index}\t{len(specs)}", flush=True)
    if len(found) != len(specs):
        raise ReproductionError("measurement did not return every utterance")
    return found


def _warm_level() -> None:
    from src.experiments.data.mix.voicebank_snr import active_speech_level

    active_speech_level(np.zeros(2048, dtype=np.float64), SAMPLE_RATE, 16)


def _measure_one(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        return _measure_one_unchecked(spec)
    except Exception as exc:
        raise RuntimeError(f"{spec.get('utterance_id')}: {exc}") from exc


def _measure_one_unchecked(spec: dict[str, Any]) -> dict[str, Any]:
    speech = read_span(
        Path(spec["speech_path"]),
        channel=int(spec["speech_channel"]),
        start=int(spec["speech_start"]),
        end=int(spec["speech_end"]),
    )
    noise = read_span(
        Path(spec["noise_path"]),
        channel=int(spec["noise_channel"]),
        start=int(spec["noise_start"]),
        end=int(spec["noise_end"]),
    )
    scale = clean_scale_for(speech)
    record = {
        "utterance_id": spec["utterance_id"],
        "speaker_id": spec["speaker_id"],
        "split": spec["split"],
        "official_snr_db": spec["official_snr_db"],
        "clean_scale": scale,
        "clean_gain_g": spec["clean_gain_g"],
        "noise_gain_before_peak_beta": spec["noise_gain_before_peak_beta"],
        "noise_gain_after_peak": spec["noise_gain_after_peak"],
        "peak_steps": spec["peak_steps"],
        "silence": {
            "source_frames": spec["speech_frames"],
            "leading_removed": spec["speech_start"],
            "trailing_removed": int(spec["speech_frames"]) - int(spec["speech_end"]),
        },
        "speech": {
            "asset_id": spec["speech_asset"],
            "sha256": spec["speech_sha256"],
            "channel": spec["speech_channel"],
            "start": spec["speech_start"],
            "end": spec["speech_end"],
        },
        "noise": {
            "asset_id": spec["noise_asset"],
            "sha256": spec["noise_sha256"],
            "channel": spec["noise_channel"],
            "start": spec["noise_start"],
            "end": spec["noise_end"],
            "kind": spec["noise_kind"],
            "name": spec["noise_name"],
        },
    }
    document = realization_document(record, build_id=spec["build_id"], gen_index=0)
    clean, noisy = render_components(document, speech, noise, snr_shift_db=0)
    again_clean, again_noisy = render_components(document, speech, noise, snr_shift_db=0)
    shifted_clean, shifted_noisy = render_components(document, speech, noise, snr_shift_db=-15)
    deterministic = bool(np.array_equal(clean, again_clean) and np.array_equal(noisy, again_noisy))
    clean_pcm, noisy_pcm = official_pcm16(document, speech, noise)
    official_clean = _read_published_pcm(Path(spec["official_clean"]))
    official_noisy = _read_published_pcm(Path(spec["official_noisy"]))
    clean_stats = pcm_delta_stats(clean_pcm, official_clean)
    noisy_stats = pcm_delta_stats(noisy_pcm, official_noisy)
    gain = parse_decimal(spec["clean_gain_g"], "clean_gain_g")
    after = parse_decimal(spec["noise_gain_after_peak"], "noise_gain_after_peak")
    if clean_stats.get("aligned") and noisy_stats.get("aligned"):
        official_float = official_clean.astype(np.float64) / np.float64(32768.0)
        diagnostic = pcm16_round_nearest_even(gain * official_float + after * noise)
        diagnostic_stats = pcm_delta_stats(diagnostic, official_noisy)
    else:
        diagnostic_stats = {"aligned": False, "actual_samples": int(clean_pcm.size), "reference_samples": int(official_clean.size)}
    noise_term = after * noise
    shifted_term = after * noise_scale(-15) * noise
    return {
        "utterance_id": spec["utterance_id"],
        "clean_scale": scale,
        "deterministic": deterministic,
        "nonfinite": not (np.isfinite(clean).all() and np.isfinite(noisy).all() and np.isfinite(shifted_noisy).all()),
        "clean": _public_stats(clean_stats),
        "noisy": _public_stats(noisy_stats),
        "diagnostic_noisy": _public_stats(diagnostic_stats),
        "histograms": {
            "clean": clean_stats.get("histogram", {}),
            "noisy": noisy_stats.get("histogram", {}),
            "diagnostic_noisy": diagnostic_stats.get("histogram", {}),
        },
        "p56_shift0_db": _finite(p56_snr_db(clean, noise_term)),
        "component_shift0_db": _finite(component_snr_db(clean, noise_term)),
        "p56_shift_m15_db": _finite(p56_snr_db(shifted_clean, shifted_term)),
        "component_shift_m15_db": _finite(component_snr_db(shifted_clean, shifted_term)),
        "peak_abs_shift0": float(np.max(np.abs(noisy))),
        "peak_abs_m15": float(np.max(np.abs(shifted_noisy))),
        "clipped_samples_m15": int(np.count_nonzero(np.abs(shifted_noisy) >= 1.0)),
    }


def _read_published_pcm(path: Path) -> np.ndarray:
    import soundfile as sf

    if not path.is_file():
        raise ReproductionError(f"published reference is missing: {path}")
    audio, rate = sf.read(str(path), dtype="int16", always_2d=True)
    if int(rate) != SAMPLE_RATE or int(audio.shape[1]) != 1:
        raise ReproductionError(f"{path} is not mono 48 kHz PCM")
    return np.ascontiguousarray(audio[:, 0])


def _record(row: dict[str, Any], assets: dict[str, dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    speech = assets[f"corpora/{row['speech_path'][len(CORPUS_PREFIX):]}"]
    noise = assets[f"corpora/{row['noise_path'][len(CORPUS_PREFIX):]}"]
    end = row["speech_start"] + (row["noise_end"] - row["noise_start"])
    return {
        "utterance_id": row["utterance_id"],
        "speaker_id": row["speaker_id"],
        "split": row["split"],
        "official_snr_db": row["official_snr_db"],
        "clean_scale": result["clean_scale"],
        "clean_gain_g": row["clean_gain_g"],
        "noise_gain_before_peak_beta": row["noise_gain_before_peak_beta"],
        "noise_gain_after_peak": row["noise_gain_after_peak"],
        "peak_steps": row["peak_steps"],
        "silence": {
            "source_frames": int(speech["frames"]),
            "leading_removed": row["speech_start"],
            "trailing_removed": int(speech["frames"]) - end,
        },
        "speech": _asset_view(speech, channel=0, start=row["speech_start"], end=end, speaker_id=row["speaker_id"], utterance_key=row["utterance_id"]),
        "noise": _asset_view(noise, channel=row["noise_channel"], start=row["noise_start"], end=row["noise_end"], speaker_id=None, utterance_key=Path(noise["rel_path"]).stem, kind=row["noise_kind"], name=row["noise_name"]),
    }


def _asset_view(asset: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "asset_id": asset["asset_id"],
        "rel_path": asset["rel_path"],
        "sha256": asset["sha256"],
        "frames": int(asset["frames"]),
        "channels": int(asset["channels"]),
        "size_bytes": int(asset["size_bytes"]),
        "sample_format": asset["sample_format"],
        "corpus": asset["corpus"],
        "subset": asset["subset"],
        **extra,
    }


def _verification_row(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    p56_0 = result["p56_shift0_db"]
    p56_m = result["p56_shift_m15_db"]
    component_0 = result["component_shift0_db"]
    component_m = result["component_shift_m15_db"]
    return {
        "utterance_id": record["utterance_id"],
        "split": record["split"],
        "speaker_id": record["speaker_id"],
        "noise_kind": record["noise"]["kind"],
        "noise_name": record["noise"]["name"],
        "official_snr_db": record["official_snr_db"],
        "peak_steps": int(record["peak_steps"]),
        "clean_gain_g": record["clean_gain_g"],
        "clean": result["clean"],
        "noisy": result["noisy"],
        "diagnostic_noisy": result["diagnostic_noisy"],
        "p56_shift0_db": p56_0,
        "p56_error_db": None if p56_0 is None else p56_0 - float(record["official_snr_db"]),
        "component_shift0_db": component_0,
        "p56_shift_m15_db": p56_m,
        "p56_delta_db": None if p56_0 is None or p56_m is None else p56_m - p56_0,
        "component_delta_db": None if component_0 is None or component_m is None else component_m - component_0,
        "peak_abs_shift0": result["peak_abs_shift0"],
        "peak_abs_m15": result["peak_abs_m15"],
        "clipped_samples_m15": result["clipped_samples_m15"],
        "nonfinite": result["nonfinite"],
        "deterministic": result["deterministic"],
        "histograms": result["histograms"],
    }


def _assign_indexes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"train": 0, "valid": 1, "test": 2}
    ordered = sorted(records, key=lambda row: (order[row["split"]], row["utterance_id"]))
    for index, record in enumerate(ordered):
        record["gen_index"] = index
    return ordered


def _aggregate(records: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], *, source_recipes_sha256: str) -> dict[str, Any]:
    histograms: dict[str, dict[str, int]] = {"clean": {}, "noisy": {}, "diagnostic_noisy": {}}
    groups: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        stream: {
            "split": defaultdict(_bucket),
            "noise_name": defaultdict(_bucket),
            "official_snr_db": defaultdict(_bucket),
            "peak": defaultdict(_bucket),
        }
        for stream in ("clean", "noisy", "diagnostic_noisy")
    }
    p56_errors = []
    p56_deltas = []
    component_deltas = []
    unaligned = 0
    nonfinite = 0
    deterministic = True
    clipped_utterances = 0
    clipped_samples = 0
    peak_m15 = 0.0
    g_other = 0
    clean_utterance_exact = 0
    noisy_utterance_exact = 0
    for record, row in zip(records, rows):
        deterministic = deterministic and bool(row["deterministic"])
        if row["nonfinite"]:
            nonfinite += 1
        if row["clipped_samples_m15"]:
            clipped_utterances += 1
            clipped_samples += int(row["clipped_samples_m15"])
        peak_m15 = max(peak_m15, float(row["peak_abs_m15"]))
        if record["clean_gain_g"] != "1":
            g_other += 1
        if any(not row[stream].get("aligned") for stream in ("clean", "noisy")):
            unaligned += 1
        else:
            if int(row["clean"]["exact"]) == int(row["clean"]["samples"]):
                clean_utterance_exact += 1
            if int(row["noisy"]["exact"]) == int(row["noisy"]["samples"]):
                noisy_utterance_exact += 1
        for stream in ("clean", "noisy", "diagnostic_noisy"):
            stats = row[stream]
            hist = row.get("histograms", {}).get(stream, {})
            _merge_hist(histograms[stream], hist)
            if not stats.get("aligned"):
                continue
            peak_key = "g_eq_1" if record["clean_gain_g"] == "1" else "g_ne_1"
            for name, key in (
                ("split", record["split"]),
                ("noise_name", record["noise"]["name"]),
                ("official_snr_db", str(record["official_snr_db"])),
                ("peak", peak_key),
            ):
                _add_bucket(groups[stream][name][key], stats)
        if row["p56_error_db"] is not None:
            p56_errors.append(float(row["p56_error_db"]))
        if row["p56_delta_db"] is not None:
            p56_deltas.append(float(row["p56_delta_db"]))
        if row["component_delta_db"] is not None:
            component_deltas.append(float(row["component_delta_db"]))
    bit_exact = unaligned == 0 and clean_utterance_exact == len(rows) and noisy_utterance_exact == len(rows)
    return {
        "status": "PASS" if bit_exact and not nonfinite and deterministic else "WARN",
        "bit_exact": bit_exact,
        "utterances": len(rows),
        "clean_utterance_exact": clean_utterance_exact,
        "noisy_utterance_exact": noisy_utterance_exact,
        "unaligned": unaligned,
        "nonfinite": nonfinite,
        "deterministic": deterministic,
        "g_ne_1": g_other,
        "groups": {
            stream: {name: {key: dict(value) for key, value in grouped.items()} for name, grouped in by_name.items()}
            for stream, by_name in groups.items()
        },
        "pcm_histogram": histograms,
        "p56_error_shift0": _error_summary(p56_errors, target=0.0),
        "p56_delta_m15": _error_summary(p56_deltas, target=-15.0),
        "component_delta_m15": _error_summary(component_deltas, target=-15.0),
        "peak_abs_m15_max": decimal_string(peak_m15) if rows else "0",
        "clipped_utterances_m15": clipped_utterances,
        "clipped_samples_m15": clipped_samples,
        "source_recipes_sha256": source_recipes_sha256,
        "verification_jsonl_sha256": "",
    }


def _write_record_cache(records: Sequence[dict[str, Any]]) -> None:
    """Keep the measured rows if graph validation fails, so the audio pass is not repeated."""

    directory = STAGING / "verification"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "records.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _write_verification(rows: Sequence[dict[str, Any]], report: dict[str, Any]) -> None:
    directory = STAGING / "verification"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "utterances.jsonl"
    digest_value = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["utterance_id"]):
            body = {key: value for key, value in row.items() if key != "histograms"}
            line = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
            handle.write(line)
            digest_value.update(line.encode("utf-8"))
    report["verification_jsonl_sha256"] = digest_value.hexdigest()
    summary = directory / "summary.json"
    summary.write_text(json.dumps(_jsonable(report), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _assets_from_records(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    assets: dict[str, dict[str, Any]] = {}
    for record in records:
        for name in ("speech", "noise"):
            source = record[name]
            current = assets.get(source["asset_id"])
            row = {
                "asset_id": source["asset_id"],
                "storage_root": "corpora",
                "rel_path": source["rel_path"],
                "corpus": source["corpus"],
                "subset": source["subset"],
                "speaker_id": source.get("speaker_id"),
                "utterance_key": source.get("utterance_key"),
                "sample_rate": SAMPLE_RATE,
                "frames": int(source["frames"]),
                "channels": int(source["channels"]),
                "sample_format": source["sample_format"],
                "size_bytes": int(source["size_bytes"]),
                "sha256": source["sha256"],
                "integrity": "sha256",
                "catalog_id": None,
                "derivation_id": None,
                "derived_from": None,
                "build_id": None,
                "extra_json": {},
            }
            if current is None:
                assets[source["asset_id"]] = row
            elif current["sha256"] != row["sha256"] or current["frames"] != row["frames"]:
                raise ReproductionError(f"asset {source['asset_id']} metadata disagrees across utterances")
    catalog_rows = [
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
        for asset in sorted(assets.values(), key=lambda item: item["rel_path"])
    ]
    counts: dict[str, int] = {}
    for row in catalog_rows:
        counts[row["corpus"]] = counts.get(row["corpus"], 0) + 1
    catalog = {
        "catalog_id": catalog_id_for(catalog_rows),
        "source_kind": "staging_parquet",
        "corpora_json": counts,
        "row_count": len(catalog_rows),
        "source_db_sha256": None,
    }
    for asset in assets.values():
        asset["catalog_id"] = catalog["catalog_id"]
    return catalog, catalog_rows, assets


def _materials(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        for role, source in (("speech", record["speech"]), ("noise", record["noise"])):
            key = (record["split"], role)
            if source["asset_id"] in seen[key]:
                continue
            seen[key].add(source["asset_id"])
            grouped[key].append(source)
    materials = {}
    for (split, role), sources in sorted(grouped.items()):
        ordered = sorted(sources, key=lambda item: item["asset_id"])
        name = f"voicebank.snr_shift.{split}.{role}"
        rule = {"rule": "explicit"}
        keys = [material_members_key(item["asset_id"], item["sha256"], item["size_bytes"]) for item in ordered]
        seconds = 0.0
        for item in ordered:
            seconds += item["frames"] / SAMPLE_RATE
        provenance = {
            "source": ordered[0]["corpus"],
            "path": f"{CORPUS_PREFIX}{ordered[0]['corpus']}",
            "sha256": digest([item["sha256"] for item in ordered]),
            "note": f"{split} {role} files used by the shared VoiceBank SNR recipe",
        }
        body = {
            "name": name,
            "version": 1,
            "role": role,
            "split_role": split,
            "origin": "official",
            "selection_rule_json": rule,
            "provenance_json": provenance,
            "member_count": len(ordered),
            "total_seconds": float(f"{seconds:.6f}"),
            "list_sha256": digest(keys),
            "integrity_min": weakest_integrity(["sha256"] * len(ordered)),
            "members": [
                {"position": index, "asset_id": item["asset_id"], "group_key": None, "attrs_json": None}
                for index, item in enumerate(ordered)
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
        materials[body["material_set_id"]] = body
    return materials


def _build(materials: dict[str, dict[str, Any]], source_sha: str) -> dict[str, Any]:
    inputs = {}
    for material in materials.values():
        inputs[f"{material['split_role']}.{material['role']}"] = material["material_set_id"]
    params = {
        "recipe": "voicebank_snr_shift_v1",
        "sample_rate": SAMPLE_RATE,
        "peak_factors": list(VOICEBANK_SNR_MIX_V1["peak_factors"]),
        "quantization": VOICEBANK_SNR_MIX_V1["official_quantization"],
        "source_recipes_sha256": source_sha,
    }
    return {
        "build_id": build_id_for(
            generator=GENERATOR,
            generator_version=GENERATOR_VERSION,
            params=params,
            seed=0,
            inputs=inputs,
        ),
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "params_json": params,
        "seed": 0,
        "inputs_json": inputs,
        "shard_size": 12396,
    }


def _protocols() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    published = _protocol("voicebank.snr_shift.published", 0, "official_effective", None, [])
    no_shift = _protocol("voicebank.snr_shift.no", 0, "research_improvement", published["protocol_id"], DEVIATIONS)
    minus = _protocol("voicebank.snr_shift.minus_15db", -15, "research_improvement", published["protocol_id"], DEVIATIONS)
    return published, no_shift, minus


def _protocol(name: str, shift: int, deviation: str, parent: str | None, deviations: list[dict[str, Any]]) -> dict[str, Any]:
    rule = {
        "sample_rate": SAMPLE_RATE,
        "teacher": "dry_clean",
        "delivery": "on_demand",
        "renderer": "voicebank_snr_shift_v1",
        "snr_shift_db": int(shift),
        "mix": dict(VOICEBANK_SNR_MIX_V1),
    }
    return {
        "protocol_id": protocol_id_for(
            algorithm=ALGORITHM,
            algorithm_version=1,
            kind="pregenerated",
            rule=rule,
            deviation_class=deviation,
            parent=parent,
        ),
        "name": name,
        "version": 1,
        "kind": "pregenerated",
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "rule_json": rule,
        "official_ref_json": dict(OFFICIAL_REF),
        "deviation_class": deviation,
        "parent_protocol_id": parent,
        "deviations_json": [dict(item) for item in deviations],
    }


def _case(document: dict[str, Any], *, protocol_id: str) -> dict[str, Any]:
    row = {
        "case_id": digest(document),
        "doc_kind": "realization_v1",
        "doc_json": document,
        "case_role": "generated",
        "protocol_id": protocol_id,
        "clean_asset_id": None,
        "noisy_asset_id": None,
        "noise_asset_id": None,
        "teacher_kind": "dry_clean",
        "sample_rate": SAMPLE_RATE,
        "frames": int(document["output"]["length"]),
        "speaker_id": document["draws"]["speaker_id"],
        "utterance_key": document["draws"]["utterance_id"],
        "transcript": None,
        "build_id": document["draws"]["build_id"],
        "gen_index": int(document["draws"]["gen_index"]),
    }
    row["content_hash"] = case_content_hash(row)
    return row


def _case_sets(records: Sequence[dict[str, Any]], cases: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {row["utterance_key"]: row for row in cases}
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for record in records:
        grouped[record["split"]].append(by_id[record["utterance_id"]])
    found = {}
    notes = {
        "train": "26 speakers. Sorted by utterance id. Test is not a selection split.",
        "valid": "Holdout speakers p286 and p287. Noise intervals overlap train and are not independent.",
        "test": "Official test speakers p232 and p257. Used as an inverse-analysis reference, not for checkpoint selection.",
    }
    for split, rows in grouped.items():
        name = f"voicebank.snr_shift.{split}"
        rule = {"rule": "explicit"}
        case_ids = [row["case_id"] for row in rows]
        body = {
            "name": name,
            "version": 1,
            "split_role": split,
            "origin": "derived_from_official",
            "selection_rule_json": rule,
            "provenance_json": {
                "source": "VoiceBank-DEMAND",
                "path": "data/corpora/VCTK",
                "note": notes[split],
            },
            "member_count": len(case_ids),
            "list_sha256": digest(case_ids),
            "items": [{"position": index, "case_id": case_id} for index, case_id in enumerate(case_ids)],
        }
        body["case_set_id"] = case_set_id_for(name=name, version=1, split_role=split, rule=rule, case_ids=case_ids)
        found[split] = body
    return found


def _blocked_dataset(protocol: dict[str, Any]) -> dict[str, Any]:
    dataset = {
        "name": "voicebank.snr_shift.published",
        "version": 1,
        "protocol_id": protocol["protocol_id"],
        "profile": "official_faithful",
        "material_origin": "derived_from_official",
        "claim_label": "official_reproduction",
        "declared_status": "blocked",
        "blocking_json": [
            {
                "item": "published mixer source",
                "reason": "The publisher's generator was not recovered. Ready datasets are research reconstructions, not an official reproduction.",
            }
        ],
        "selection_uses_test": False,
        "selection_note": "Blocked. No cases are bound, so this row cannot select a checkpoint.",
        "deviations_json": [dict(item) for item in DEVIATIONS],
        "bindings": [],
    }
    dataset["dataset_id"] = dataset_id_for(
        protocol_id=dataset["protocol_id"],
        bindings=[],
        profile=dataset["profile"],
        material_origin=dataset["material_origin"],
        claim_label=dataset["claim_label"],
        selection_uses_test=False,
        declared_status="blocked",
    )
    return dataset


def _ready_dataset(
    name: str,
    protocol: dict[str, Any],
    case_sets: dict[str, dict[str, Any]],
    materials: dict[str, dict[str, Any]],
    *,
    shift: int,
) -> dict[str, Any]:
    by_slot = {f"{row['split_role']}.{row['role']}": row["material_set_id"] for row in materials.values()}
    bindings = []
    for split in ("train", "valid", "test"):
        bindings.append({"slot": f"{split}.cases", "case_set_id": case_sets[split]["case_set_id"], "material_set_id": None})
        for role in ("speech", "noise"):
            bindings.append(
                {
                    "slot": f"{split}.{role}",
                    "case_set_id": None,
                    "material_set_id": by_slot[f"{split}.{role}"],
                }
            )
    note = (
        f"snr_shift_db is {shift}. Test is excluded from checkpoint selection. "
        "Official test pairs were an inverse-analysis reference. "
        "Train and validation noise intervals are not independent."
    )
    dataset = {
        "name": name,
        "version": 1,
        "protocol_id": protocol["protocol_id"],
        "profile": "official_faithful",
        "material_origin": "derived_from_official",
        "claim_label": "research_checkpoint_selection",
        "declared_status": "ready",
        "blocking_json": [],
        "selection_uses_test": False,
        "selection_note": note,
        "deviations_json": [dict(item) for item in DEVIATIONS],
        "bindings": bindings,
    }
    dataset["dataset_id"] = dataset_id_for(
        protocol_id=dataset["protocol_id"],
        bindings=bindings,
        profile=dataset["profile"],
        material_origin=dataset["material_origin"],
        claim_label=dataset["claim_label"],
        selection_uses_test=False,
        declared_status="ready",
    )
    return dataset


def _distribution_audit(dataset_id: str, report: dict[str, Any]) -> dict[str, Any]:
    body = _jsonable(report)
    return {
        "audit_id": audit_id_for(subject_kind="dataset", subject_id=dataset_id, audit_kind="distribution_check", report=body),
        "subject_kind": "dataset",
        "subject_id": dataset_id,
        "audit_kind": "distribution_check",
        "status": body["status"],
        "report_json": body,
    }


def _recipe_sha256(documents: dict[str, dict[str, Any]], records: Sequence[dict[str, Any]]) -> str:
    from src.experiments.data.ids import canonical_text

    digest_value = hashlib.sha256()
    for record in records:
        digest_value.update(canonical_text(documents[record["utterance_id"]]).encode("utf-8"))
        digest_value.update(b"\n")
    return digest_value.hexdigest()


def _dataset_index(graph: Graph) -> dict[str, str]:
    return {row["name"]: row["dataset_id"] for row in graph.datasets}


def _bucket() -> dict[str, int]:
    return {"utterances": 0, "samples": 0, "exact": 0, "max_abs_lsb": 0, "utterance_exact": 0}


def _add_bucket(bucket: dict[str, int], stats: dict[str, Any]) -> None:
    bucket["utterances"] += 1
    bucket["samples"] += int(stats["samples"])
    bucket["exact"] += int(stats["exact"])
    bucket["max_abs_lsb"] = max(int(bucket["max_abs_lsb"]), int(stats["max_abs_lsb"]))
    if int(stats["exact"]) == int(stats["samples"]):
        bucket["utterance_exact"] = int(bucket.get("utterance_exact", 0)) + 1


def _merge_hist(total: dict[str, int], part: dict[str, int]) -> None:
    for key, count in part.items():
        total[str(key)] = int(total.get(str(key), 0)) + int(count)


def _error_summary(values: Sequence[float], *, target: float) -> dict[str, Any]:
    if not values:
        return {"count": 0, "missing": 0, "max_abs_error": None, "within_0_01_db": 0}
    errors = [value - target for value in values]
    return {
        "count": len(values),
        "max_abs_error": decimal_string(max(abs(value) for value in errors)),
        "within_1e_6_db": sum(1 for value in errors if abs(value) <= 1e-6),
        "within_0_01_db": sum(1 for value in errors if abs(value) <= 0.01),
    }


def _public_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if not stats.get("aligned"):
        return {"aligned": False, "actual_samples": int(stats.get("actual_samples", 0)), "reference_samples": int(stats.get("reference_samples", 0))}
    samples = int(stats["samples"])
    return {
        "aligned": True,
        "samples": samples,
        "exact": int(stats["exact"]),
        "max_abs_lsb": int(stats["max_abs_lsb"]),
        "min_lsb": int(stats["min_lsb"]),
        "max_lsb": int(stats["max_lsb"]),
        "mae_lsb": (int(stats["sum_abs_lsb"]) / samples) if samples else 0.0,
        "rmse_lsb": math.sqrt(int(stats["sum_sq_lsb"]) / samples) if samples else 0.0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.6f}")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ReproductionError(f"verification report has {type(value).__name__}")


def _finite(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _corpus_path(path: str, utterance: str) -> str:
    if not path.startswith(CORPUS_PREFIX) or ".." in path.split("/"):
        raise ReproductionError(f"{utterance} path is outside data/corpora: {path}")
    return path


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise ReproductionError(f"{where} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        number = float(value)
        if number.is_integer():
            return int(number)
    raise ReproductionError(f"{where} is not an integer")


def _snr(value: Any) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise ReproductionError("official SNR is not finite")
    if number.is_integer():
        return int(number)
    return number


def _prepare_audits(graph: Graph) -> list[dict[str, Any]]:
    """Structural audits, used by tests that register without the production database."""

    return audits_for(graph, [row["dataset_id"] for row in graph.datasets], None)
