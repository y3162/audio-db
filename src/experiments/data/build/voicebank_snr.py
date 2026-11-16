"""Build VoiceBank SNR-shift datasets from the git recipe.

``configs/data/voicebank_snr/v1.jsonl`` is the only per-utterance recipe. Each line
is one of the 12,396 utterances. ``snr_shift_db`` is not in that file. It is
the protocol value of the dataset being registered, so every SNR shares the
same cases.

The build id still records ``source_recipes_sha256`` of the measurement input
that produced these gains (``9502da7b…``). That digest is not the hash of the
JSONL. Replacing it would mint a new build id and new case ids for the same
spans and gains. Registration reads VCTK, DEMAND48, Babble, and SSN. It does
not read published VoiceBank+DEMAND pairs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

from src.data.recipe import sha256_file
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import (
    VOICEBANK_SNR_MIX_V1,
    build_id_for,
    canonical_text,
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
    decimal_string,
    parse_decimal,
    peak_product,
)
from src.experiments.data.paths import DATA, DEFAULT_CORPORA, DEFAULT_SNAPSHOTS, ROOT

GENERATOR = "voicebank_snr_shift"
GENERATOR_VERSION = 1
RECIPE_VERSION = 1
RECIPE_PATH = ROOT / "configs" / "data" / "voicebank_snr" / "v1.jsonl"
# SHA-256 of the measurement input whose parameters were exported into the
# JSONL. It is an input of the published build id, not a hash of the JSONL.
MEASURED_SOURCE_SHA256 = "9502da7b24b81bed04781c1e66726945dfcb3024045111df8bbb5839e0d01c14"
DB_PATH = DATA / "db" / "reproduction-voicebank-snr-v1.duckdb"
STAGING = DATA / "staging" / "reproduction-voicebank-snr-v1"
EXPECTED_COUNTS = {"train": 10700, "valid": 872, "test": 824}
CANONICAL_SHIFTS = (0, -15, -20)
HOLDOUT_SPEAKERS = frozenset({"p286", "p287"})
TEST_SPEAKERS = frozenset({"p232", "p257"})
SPLIT_RANK = {"train": 0, "valid": 1, "test": 2}
NOISE_CORPORA = frozenset({"DEMAND48", "Babble", "SSN"})
CORPUS_PREFIX = "data/corpora/"
SHARD_SIZE = 12396
HEADER = Callable[[str, str], dict[str, Any]]
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
_RECIPE_KEYS = {
    "recipe_version",
    "utterance_id",
    "speaker_id",
    "split",
    "position",
    "sample_rate",
    "official_snr_db",
    "peak_steps",
    "target_peak_pcm",
    "output_length",
    "clean",
    "noise",
}
_CLEAN_KEYS = {
    "corpus",
    "rel_path",
    "sha256",
    "channel",
    "start",
    "end",
    "scale",
    "gain_g",
    "silence",
}
_NOISE_KEYS = {
    "corpus",
    "rel_path",
    "sha256",
    "channel",
    "start",
    "end",
    "kind",
    "name",
    "gain_before_peak_beta",
    "gain_after_peak",
}
_SILENCE_KEYS = {"mode", "source_frames", "leading_removed", "trailing_removed"}
_HEX = set("0123456789abcdef")
_SHIFT_TEXT = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")


def canonical_shift(value: Any) -> int | float:
    """Integer shifts stay integers. ``-15.0`` and ``-15`` are the same shift."""

    if isinstance(value, bool):
        raise ReproductionError("snr_shift_db must be a finite number")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReproductionError("snr_shift_db is not finite")
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        text = value.strip()
        if not _SHIFT_TEXT.fullmatch(text):
            raise ReproductionError(f"snr_shift_db is not a number: {value}")
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return canonical_shift(float(text))
    raise ReproductionError("snr_shift_db must be a finite number")


def shift_token(shift: int | float) -> str:
    """Dataset-name suffix. Zero is ``no``. The sign is never omitted."""

    number = canonical_shift(shift)
    if number == 0:
        return "no"
    if isinstance(number, int):
        magnitude = str(abs(number))
    else:
        text = decimal_string(float(number))
        if text.startswith("-"):
            text = text[1:]
        magnitude = text.replace(".", "p").replace("+", "")
    prefix = "minus" if float(number) < 0 else "plus"
    return f"{prefix}_{magnitude}db"


def dataset_name(shift: int | float) -> str:
    return f"voicebank.snr_shift.{shift_token(shift)}"


def parse_shifts(value: str | None) -> list[int | float]:
    """Empty means the canonical 0, −15, and −20 dB datasets."""

    if value is None or not str(value).strip():
        return list(CANONICAL_SHIFTS)
    found: list[int | float] = []
    seen: set[int | float] = set()
    for part in str(value).split(","):
        if not part.strip():
            raise ReproductionError("snr_shift_db contains an empty item")
        shift = canonical_shift(part)
        if shift in seen:
            raise ReproductionError(f"duplicate snr_shift_db {shift}")
        seen.add(shift)
        found.append(shift)
    if not found:
        raise ReproductionError("snr_shift_db is empty")
    names = [dataset_name(shift) for shift in found]
    if len(names) != len(set(names)):
        raise ReproductionError("snr shifts map to the same dataset name")
    return found


def dump_recipe_line(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def load_recipes(path: str | Path, *, check_population: bool = True) -> list[dict[str, Any]]:
    """Read one canonical JSONL. Line order is the case order."""

    path = Path(path)
    if not path.is_file():
        raise ReproductionError(f"voicebank recipe is missing: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if line != line.rstrip("\n") + "\n":
                raise ReproductionError(f"{path}:{line_number}: recipe line must end in one newline")
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReproductionError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ReproductionError(f"{path}:{line_number}: recipe line must be an object")
            if dump_recipe_line(parsed) != line.rstrip("\n"):
                raise ReproductionError(f"{path}:{line_number}: recipe line is not canonical JSON")
            try:
                row = _recipe_row(parsed)
            except ReproductionError as exc:
                raise ReproductionError(f"{path}:{line_number}: {exc}") from exc
            if row["utterance_id"] in seen:
                raise ReproductionError(f"duplicate utterance {row['utterance_id']}")
            seen.add(row["utterance_id"])
            rows.append(row)
    _check_order(rows)
    if check_population:
        assert_population(rows)
    return rows


def assert_population(rows: Sequence[dict[str, Any]]) -> None:
    counts = {"train": 0, "valid": 0, "test": 0}
    seen: set[str] = set()
    for row in rows:
        utterance = str(row["utterance_id"])
        if utterance in seen:
            raise ReproductionError(f"duplicate utterance {utterance}")
        seen.add(utterance)
        split = row["split"]
        if split not in counts:
            raise ReproductionError(f"{utterance} has an unknown split {split}")
        counts[split] += 1
    if counts != EXPECTED_COUNTS or len(rows) != sum(EXPECTED_COUNTS.values()):
        raise ReproductionError(f"voicebank population is {counts}, expected {EXPECTED_COUNTS}")


def concatenated_realization_sha256(documents: Sequence[dict[str, Any]]) -> str:
    """SHA-256 of canonical realizations in case order.

    Case order is train, validation, test, and utterance id inside each split.
    That is the published digest of the v1 realizations.
    """

    ordered = sorted(documents, key=lambda document: int(document["draws"]["gen_index"]))
    digest_value = hashlib.sha256()
    for document in ordered:
        digest_value.update(canonical_text(document).encode("utf-8"))
        digest_value.update(b"\n")
    return digest_value.hexdigest()


def records_from_recipes(
    rows: Sequence[dict[str, Any]],
    header_for: HEADER,
) -> list[dict[str, Any]]:
    """Attach source-file headers. ``header_for`` checks the stored SHA-256."""

    cache: dict[str, dict[str, Any]] = {}
    records = []
    for row in rows:
        speech = _asset_record(row["clean"], header_for, cache, role="speech", speaker=row["speaker_id"], utterance=row["utterance_id"])
        noise = _asset_record(
            row["noise"],
            header_for,
            cache,
            role="noise",
            speaker=None,
            utterance=Path(row["noise"]["rel_path"]).stem,
            kind=row["noise"]["kind"],
            name=row["noise"]["name"],
        )
        if int(speech["frames"]) != int(row["silence"]["source_frames"]):
            raise ReproductionError(
                f"{row['utterance_id']} speech frames are {speech['frames']}, recipe says {row['silence']['source_frames']}"
            )
        if int(row["speech_end"]) > int(speech["frames"]) or int(row["noise_end"]) > int(noise["frames"]):
            raise ReproductionError(f"{row['utterance_id']} slice ends past the source")
        if int(row["noise_channel"]) >= int(noise["channels"]):
            raise ReproductionError(f"{row['utterance_id']} noise channel is outside the file")
        if speech["channels"] != 1:
            raise ReproductionError(f"{row['utterance_id']} speech file is not mono")
        records.append(
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "official_snr_db": row["official_snr_db"],
                "clean_scale": row["clean_scale"],
                "clean_gain_g": row["clean_gain_g"],
                "noise_gain_before_peak_beta": row["noise_gain_before_peak_beta"],
                "noise_gain_after_peak": row["noise_gain_after_peak"],
                "peak_steps": row["peak_steps"],
                "silence": dict(row["silence"]),
                "speech": speech,
                "noise": noise,
            }
        )
    return records


def file_header(corpora: str | Path, rel_path: str, expected_sha256: str) -> dict[str, Any]:
    """Read one source header and require the recipe SHA-256."""

    path = Path(corpora) / rel_path
    if not path.is_file():
        raise ReproductionError(f"source file is missing: {path}")
    if "VoiceBank-DEMAND" in path.parts:
        raise ReproductionError("voicebank reconstruction must not read a published clean or noisy pair")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ReproductionError(f"sha256 mismatch for {path}")
    return _sound_header(path)


def hash_headers(
    rows: Sequence[dict[str, Any]],
    corpora: str | Path,
    *,
    workers: int | None = None,
) -> HEADER:
    """Hash each source once, then return a header lookup."""

    corpora = Path(corpora)
    wanted: dict[str, str] = {}
    for row in rows:
        for source in (row["clean"], row["noise"]):
            previous = wanted.get(source["rel_path"])
            if previous is None:
                wanted[source["rel_path"]] = source["sha256"]
            elif previous != source["sha256"]:
                raise ReproductionError(f"{source['rel_path']} has two sha256 values")
    paths = [corpora / rel for rel in sorted(wanted)]
    for path in paths:
        if not path.is_file():
            raise ReproductionError(f"source file is missing: {path}")
        if "VoiceBank-DEMAND" in path.parts:
            raise ReproductionError("voicebank reconstruction must not read a published clean or noisy pair")
    count = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    if count < 1:
        raise ReproductionError("worker count must be positive")
    hashed = _hash_paths(paths, count)
    headers = {path.relative_to(corpora).as_posix(): {**_sound_header(path), "sha256": digest_value} for path, digest_value in hashed.items()}
    for rel, expected in wanted.items():
        if headers[rel]["sha256"] != expected:
            raise ReproductionError(f"sha256 mismatch for {corpora / rel}")

    def lookup(rel_path: str, expected_sha256: str) -> dict[str, Any]:
        header = headers.get(rel_path)
        if header is None or header["sha256"] != expected_sha256:
            raise ReproductionError(f"sha256 mismatch for {rel_path}")
        return header

    return lookup


def build_voicebank_snr_graph(
    records: Sequence[dict[str, Any]],
    shifts: Sequence[int | float] = CANONICAL_SHIFTS,
) -> Any:
    """Shared cases, one protocol and dataset per shift, plus the blocked parent."""

    from src.experiments.data.db.graph import Graph

    requested = [canonical_shift(shift) for shift in shifts]
    if not requested:
        raise ReproductionError("at least one snr_shift_db is required")
    ordered = _assign_indexes(list(records))
    catalog, catalog_rows, assets = _assets_from_records(ordered)
    materials = _materials(ordered)
    build = _build(materials)
    protocols = _protocols(requested)
    anchor = next(protocol["protocol_id"] for protocol in protocols if protocol["name"] == "voicebank.snr_shift.no")
    requested_names = {dataset_name(shift) for shift in requested}
    ready = [
        protocol
        for protocol in protocols
        if protocol["name"] in requested_names and protocol["deviation_class"] == "research_improvement"
    ]
    documents = {
        record["utterance_id"]: realization_document(
            record,
            build_id=build["build_id"],
            gen_index=int(record["gen_index"]),
        )
        for record in ordered
    }
    cases = [_case(documents[record["utterance_id"]], protocol_id=anchor) for record in ordered]
    case_sets = _case_sets(ordered, cases)
    datasets = [_blocked_dataset(protocols[0])]
    datasets.extend(_ready_dataset(protocol, case_sets, materials) for protocol in ready)
    return Graph(
        catalogs=[catalog],
        catalog_rows={catalog["catalog_id"]: catalog_rows},
        assets=list(assets.values()),
        materials=list(materials.values()),
        protocols=protocols,
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


def register_voicebank_snr(
    *,
    recipes: str | Path | None = None,
    shifts: str | Sequence[int | float] | None = None,
    database: str | Path | None = None,
    staging: str | Path | None = None,
    snapshots: str | Path | None = None,
    corpora: str | Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Register the requested shifts and export a snapshot for each ready dataset."""

    from src.experiments.data.db.registry import initialize_database, register_staging
    from src.experiments.data.db.snapshot import export_snapshot
    from src.experiments.data.db.staging import write_graph
    from src.experiments.data.db.validate import validate_graph

    if shifts is None or shifts == "":
        selected = list(CANONICAL_SHIFTS)
    elif isinstance(shifts, str):
        selected = parse_shifts(shifts)
    else:
        selected = parse_shifts(",".join(str(item) for item in shifts))
    rows = load_recipes(recipes or RECIPE_PATH)
    header_for = hash_headers(rows, corpora or DEFAULT_CORPORA, workers=workers)
    records = records_from_recipes(rows, header_for)
    graph = build_voicebank_snr_graph(records, selected)
    validate_graph(graph)
    staging_path = Path(staging) if staging is not None else STAGING
    database_path = Path(database) if database is not None else DB_PATH
    snapshot_root = Path(snapshots) if snapshots is not None else DEFAULT_SNAPSHOTS
    write_graph(staging_path, graph)
    initialize_database(database_path)
    counts = register_staging(staging_path, database_path)
    exported = {}
    for dataset in graph.datasets:
        if dataset["declared_status"] != "ready":
            continue
        name = f"{dataset['name']}@{dataset['version']}"
        exported[name] = export_snapshot(database_path, name, snapshot_root, corpora=corpora or DEFAULT_CORPORA)
    return {
        "inserted": counts,
        "snapshots": exported,
        "datasets": {row["name"]: row["dataset_id"] for row in graph.datasets},
        "recipe": str(recipes or RECIPE_PATH),
        "shifts": selected,
    }


def stub_headers(rows: Sequence[dict[str, Any]]) -> HEADER:
    """Headers large enough for identity checks. They are not file metadata."""

    frames: dict[str, int] = {}
    channels: dict[str, int] = {}
    for row in rows:
        speech = row["clean"]["rel_path"]
        frames[speech] = max(frames.get(speech, 0), int(row["silence"]["source_frames"]))
        channels[speech] = 1
        noise = row["noise"]["rel_path"]
        frames[noise] = max(frames.get(noise, 0), int(row["noise"]["end"]))
        channels[noise] = max(channels.get(noise, 1), int(row["noise"]["channel"]) + 1)

    def lookup(rel_path: str, expected_sha256: str) -> dict[str, Any]:
        return {
            "frames": frames[rel_path],
            "channels": channels[rel_path],
            "size_bytes": 1,
            "sample_format": "PCM_16",
            "sha256": expected_sha256,
        }

    return lookup


def _recipe_row(row: dict[str, Any]) -> dict[str, Any]:
    extra = set(row) - _RECIPE_KEYS
    missing = _RECIPE_KEYS - set(row)
    if extra or missing or "snr_shift_db" in row:
        raise ReproductionError(f"recipe fields are wrong (missing {sorted(missing)}, extra {sorted(extra)})")
    if row["recipe_version"] != RECIPE_VERSION or isinstance(row["recipe_version"], bool):
        raise ReproductionError("recipe_version must be 1")
    if row["sample_rate"] != SAMPLE_RATE or isinstance(row["sample_rate"], bool):
        raise ReproductionError("sample_rate must be 48000")
    if row["target_peak_pcm"] != 16384 or isinstance(row["target_peak_pcm"], bool):
        raise ReproductionError("target_peak_pcm must be 16384")
    utterance = _text(row["utterance_id"], "utterance_id")
    speaker = _text(row["speaker_id"], "speaker_id")
    if not utterance.startswith(speaker + "_"):
        raise ReproductionError("utterance_id must start with the speaker id")
    split = row["split"]
    if split not in SPLIT_RANK:
        raise ReproductionError(f"unknown split {split}")
    _speaker_split(speaker, split, utterance)
    position = _non_negative(row["position"], "position")
    output_length = _positive(row["output_length"], "output_length")
    steps = _non_negative(row["peak_steps"], "peak_steps")
    peak_product(steps)
    official = _snr(row["official_snr_db"])
    clean = _source_row(row["clean"], _CLEAN_KEYS, "clean")
    noise = _source_row(row["noise"], _NOISE_KEYS, "noise")
    if clean["corpus"] != "VCTK":
        raise ReproductionError("clean corpus must be VCTK")
    if noise["corpus"] not in NOISE_CORPORA:
        raise ReproductionError("noise corpus must be DEMAND48, Babble, or SSN")
    if noise["kind"] not in {"demand", "babble", "ssn"}:
        raise ReproductionError("noise kind is unknown")
    if noise["kind"] in {"babble", "ssn"} and noise["name"] != noise["kind"]:
        raise ReproductionError("babble and ssn noise names must match the kind")
    if noise["kind"] == "demand" and noise["name"] in {"babble", "ssn"}:
        raise ReproductionError("demand noise cannot be named babble or ssn")
    silence = row["clean"]["silence"]
    if not isinstance(silence, dict) or set(silence) != _SILENCE_KEYS:
        raise ReproductionError("silence fields are wrong")
    if silence["mode"] != "contiguous_slice":
        raise ReproductionError("silence mode must be contiguous_slice")
    source_frames = _non_negative(silence["source_frames"], "silence.source_frames")
    leading = _non_negative(silence["leading_removed"], "silence.leading_removed")
    trailing = _non_negative(silence["trailing_removed"], "silence.trailing_removed")
    if clean["end"] - clean["start"] != output_length or noise["end"] - noise["start"] != output_length:
        raise ReproductionError("clean and noise slices must match output_length")
    if leading != clean["start"] or trailing != source_frames - clean["end"]:
        raise ReproductionError("silence removal does not match the clean slice")
    if source_frames < clean["end"]:
        raise ReproductionError("clean slice ends past the source")
    scale = parse_decimal(clean["scale"], "clean.scale")
    gain = parse_decimal(clean["gain_g"], "clean.gain_g")
    beta = parse_decimal(noise["gain_before_peak_beta"], "noise.gain_before_peak_beta")
    after = parse_decimal(noise["gain_after_peak"], "noise.gain_after_peak")
    if scale <= 0 or gain <= 0 or beta <= 0 or after <= 0:
        raise ReproductionError("scale and gains must be positive")
    expected_g = 1.0 / peak_product(steps)
    if abs(gain - expected_g) > 1e-9 * max(1.0, expected_g):
        raise ReproductionError("clean.gain_g does not match peak_steps")
    if abs(after - gain * beta) > 1e-8 * max(1.0, abs(after)):
        raise ReproductionError("noise.gain_after_peak does not match gain_g * gain_before_peak_beta")
    return {
        "utterance_id": utterance,
        "speaker_id": speaker,
        "split": split,
        "position": position,
        "official_snr_db": official,
        "clean_scale": clean["scale"],
        "clean_gain_g": clean["gain_g"],
        "noise_gain_before_peak_beta": noise["gain_before_peak_beta"],
        "noise_gain_after_peak": noise["gain_after_peak"],
        "peak_steps": steps,
        "output_length": output_length,
        "speech_end": clean["end"],
        "noise_end": noise["end"],
        "noise_channel": noise["channel"],
        "silence": {
            "mode": "contiguous_slice",
            "source_frames": source_frames,
            "leading_removed": leading,
            "trailing_removed": trailing,
        },
        "clean": clean,
        "noise": noise,
    }


def _source_row(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReproductionError(f"{where} must be an object")
    extra = set(value) - keys
    missing = keys - set(value)
    if extra or missing:
        raise ReproductionError(f"{where} fields are wrong (missing {sorted(missing)}, extra {sorted(extra)})")
    corpus = _text(value["corpus"], f"{where}.corpus")
    rel_path = _rel_path(value["rel_path"], f"{where}.rel_path")
    if rel_path.split("/", 1)[0] != corpus:
        raise ReproductionError(f"{where}.corpus does not match the relative path")
    sha = value["sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(char not in _HEX for char in sha):
        raise ReproductionError(f"{where}.sha256 must be 64 lowercase hex characters")
    start = _non_negative(value["start"], f"{where}.start")
    end = _non_negative(value["end"], f"{where}.end")
    if end <= start:
        raise ReproductionError(f"{where} slice is empty")
    channel = _non_negative(value["channel"], f"{where}.channel")
    found = {
        "corpus": corpus,
        "rel_path": rel_path,
        "sha256": sha,
        "channel": channel,
        "start": start,
        "end": end,
    }
    if where == "clean":
        found["scale"] = value["scale"]
        found["gain_g"] = value["gain_g"]
    else:
        found["kind"] = _text(value["kind"], "noise.kind")
        found["name"] = _text(value["name"], "noise.name")
        found["gain_before_peak_beta"] = value["gain_before_peak_beta"]
        found["gain_after_peak"] = value["gain_after_peak"]
    return found


def _check_order(rows: Sequence[dict[str, Any]]) -> None:
    expected = sorted(rows, key=lambda row: (SPLIT_RANK[row["split"]], row["utterance_id"]))
    if [row["utterance_id"] for row in rows] != [row["utterance_id"] for row in expected]:
        raise ReproductionError("recipe lines must be ordered by split, then utterance id")
    for index, row in enumerate(rows):
        if row["position"] != index:
            raise ReproductionError(f"{row['utterance_id']} position is {row['position']}, expected {index}")


def _asset_record(
    source: dict[str, Any],
    header_for: HEADER,
    cache: dict[str, dict[str, Any]],
    *,
    role: str,
    speaker: str | None,
    utterance: str,
    kind: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    rel_path = source["rel_path"]
    header = cache.get(rel_path)
    if header is None:
        header = header_for(rel_path, source["sha256"])
        if "sample_rate" in header and int(header["sample_rate"]) != SAMPLE_RATE:
            raise ReproductionError(f"{rel_path} is not 48 kHz")
        cache[rel_path] = header
    if header["sha256"] != source["sha256"]:
        raise ReproductionError(f"{rel_path} has two sha256 values")
    row = {
        "asset_id": f"corpora/{rel_path}",
        "rel_path": rel_path,
        "sha256": source["sha256"],
        "frames": int(header["frames"]),
        "channels": int(header["channels"]),
        "size_bytes": int(header["size_bytes"]),
        "sample_format": str(header["sample_format"]),
        "corpus": source["corpus"],
        "subset": rel_path.split("/")[1] if rel_path.count("/") else source["corpus"],
        "channel": int(source["channel"]),
        "start": int(source["start"]),
        "end": int(source["end"]),
        "speaker_id": speaker,
        "utterance_key": utterance,
    }
    if role == "noise":
        row["kind"] = kind
        row["name"] = name
    return row


def _sound_header(path: Path) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(path))
    if int(info.samplerate) != SAMPLE_RATE or int(info.channels) < 1 or int(info.frames) < 1:
        raise ReproductionError(f"{path} is not 48 kHz audio")
    return {
        "frames": int(info.frames),
        "channels": int(info.channels),
        "size_bytes": path.stat().st_size,
        "sample_format": str(info.subtype or ""),
        "sample_rate": int(info.samplerate),
    }


def _hash_paths(paths: Sequence[Path], workers: int) -> dict[Path, str]:
    if workers == 1 or len(paths) < 2:
        return {path: sha256_file(path) for path in paths}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        hashed = list(pool.map(sha256_file, paths, chunksize=8))
    if len(hashed) != len(paths):
        raise ReproductionError("source hashing did not return every file")
    return dict(zip(paths, hashed, strict=True))


def _assign_indexes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (SPLIT_RANK[row["split"]], row["utterance_id"]))
    for index, record in enumerate(ordered):
        record["gen_index"] = index
    return ordered


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


def _build(materials: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inputs = {}
    for material in materials.values():
        inputs[f"{material['split_role']}.{material['role']}"] = material["material_set_id"]
    params = {
        "recipe": "voicebank_snr_shift_v1",
        "sample_rate": SAMPLE_RATE,
        "peak_factors": list(VOICEBANK_SNR_MIX_V1["peak_factors"]),
        "quantization": VOICEBANK_SNR_MIX_V1["official_quantization"],
        "source_recipes_sha256": MEASURED_SOURCE_SHA256,
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
        "shard_size": SHARD_SIZE,
    }


def _protocols(shifts: Sequence[int | float]) -> list[dict[str, Any]]:
    """Shift 0 is always present. Case rows use that protocol."""

    selected: list[int | float] = []
    seen: set[int | float] = set()
    for shift in (0, *shifts):
        number = canonical_shift(shift)
        if number in seen:
            continue
        seen.add(number)
        selected.append(number)
    names = [dataset_name(shift) for shift in selected]
    if len(names) != len(set(names)):
        raise ReproductionError("snr shifts map to the same dataset name")
    published = _protocol("voicebank.snr_shift.published", 0, "official_effective", None, [])
    ready = [
        _protocol(dataset_name(shift), shift, "research_improvement", published["protocol_id"], DEVIATIONS)
        for shift in selected
    ]
    return [published, *ready]


def _protocol(
    name: str,
    shift: int | float,
    deviation: str,
    parent: str | None,
    deviations: list[dict[str, Any]],
) -> dict[str, Any]:
    rule = {
        "sample_rate": SAMPLE_RATE,
        "teacher": "dry_clean",
        "delivery": "on_demand",
        "renderer": "voicebank_snr_shift_v1",
        "snr_shift_db": shift,
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
    protocol: dict[str, Any],
    case_sets: dict[str, dict[str, Any]],
    materials: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shift = protocol["rule_json"]["snr_shift_db"]
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
        "name": protocol["name"],
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


def _speaker_split(speaker: str, split: str, utterance: str) -> None:
    if split == "test":
        if speaker not in TEST_SPEAKERS:
            raise ReproductionError(f"{utterance} is in the test split but not a test speaker")
        return
    if speaker in TEST_SPEAKERS:
        raise ReproductionError(f"test speaker {utterance} is not in the test split")
    if split == "valid" and speaker not in HOLDOUT_SPEAKERS:
        raise ReproductionError(f"{utterance} is in the validation split but not a holdout speaker")
    if split == "train" and speaker in HOLDOUT_SPEAKERS:
        raise ReproductionError(f"holdout speaker {utterance} is in the train split")


def _snr(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReproductionError("official_snr_db must be a finite number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReproductionError("official_snr_db is not finite")
        if value.is_integer():
            raise ReproductionError("integral official_snr_db must be a JSON integer")
        return value
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReproductionError(f"{where} must be a string")
    return value


def _rel_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value or ".." in value.split("/"):
        raise ReproductionError(f"{where} must be a relative path inside a corpus")
    return value


def _non_negative(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReproductionError(f"{where} must be a non-negative integer")
    return value


def _positive(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReproductionError(f"{where} must be a positive integer")
    return value
