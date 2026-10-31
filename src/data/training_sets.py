"""Register a small LibriSpeech+DEMAND smoke set for SE training.

This is not the VoiceBank-DEMAND protocol. VoiceBank+DEMAND wavs are not
linked under ``data/corpora``. The smoke set uses LibriSpeech train-clean-100
and dev-clean, which do not share speakers, mixed with disjoint DEMAND scenes.
LibriSpeech test-clean and CHiME3 et05 are not read.

The production database ``data/db/resource.duckdb`` is opened read-only.
Rows are written to ``data/staging/training`` and then to
``data/db/training.duckdb``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.recipe import canonical_recipe, case_content_hash, case_id_for, json_hash, sha256_file
from src.data.staging import register_staging
from src.evaluation.audio.render import prepare_noise_start

PROTOCOL = "libri-demand-smoke-v1"
TRAIN_SET = "libri_demand_smoke_train"
VALID_SET = "libri_demand_smoke_valid"
TRAIN_SCENES = ("OOFFICE", "OMEETING", "OHALLWAY", "DLIVING")
VALID_SCENES = ("NPARK", "NFIELD")
SNRS = (0.0, 5.0, 10.0, 15.0)
SAMPLE_RATE = 16000
MAX_SECONDS = 2.0
N_TRAIN = 8
N_VALID = 4


def plan_pairs(speech: list[dict[str, Any]], noise: list[dict[str, Any]], *, role: str, count: int) -> list[dict[str, Any]]:
    """Pair the first ``count`` speeches with noises, one utterance per speaker."""
    chosen = _one_per_speaker(speech)[:count]
    if len(chosen) < count:
        raise ValueError(f"{role} needs {count} speeches, found {len(chosen)}")
    if len(noise) < 1:
        raise ValueError(f"{role} needs at least one noise file")
    pairs = []
    for index, item in enumerate(chosen):
        pairs.append(
            {
                "speech": item,
                "noise": noise[index % len(noise)],
                "snr_db": float(SNRS[index % len(SNRS)]),
                "role": role,
                "index": index,
            }
        )
    return pairs


def assert_disjoint(train: list[dict[str, Any]], valid: list[dict[str, Any]]) -> None:
    train_speech = {item["speech"]["path"] for item in train}
    valid_speech = {item["speech"]["path"] for item in valid}
    train_noise = {item["noise"]["path"] for item in train}
    valid_noise = {item["noise"]["path"] for item in valid}
    train_speakers = {item["speech"].get("speaker_id") for item in train}
    valid_speakers = {item["speech"].get("speaker_id") for item in valid}
    if train_speech & valid_speech:
        raise ValueError("a clean utterance is in both train and validation")
    if train_noise & valid_noise:
        raise ValueError("a noise file is in both train and validation")
    if train_speakers & valid_speakers:
        raise ValueError("a speaker is in both train and validation")
    for item in train + valid:
        path = item["speech"]["path"].casefold()
        subset = str(item["speech"].get("subset", "")).casefold()
        if "test-clean" in path or subset == "test-clean" or "et05" in path or subset == "et05":
            raise ValueError(f"held-out split in the smoke set: {item['speech']['path']}")


def build_case(pair: dict[str, Any], *, corpora_root: Path, database_parent: Path) -> dict[str, Any]:
    speech = pair["speech"]
    noise = pair["noise"]
    speech_file = corpora_root / speech["path"]
    noise_file = corpora_root / noise["path"]
    seed = _seed(pair["role"], pair["index"], speech["path"])
    start = prepare_noise_start(speech_file, noise_file, SAMPLE_RATE, seed, MAX_SECONDS)
    recipe = canonical_recipe(
        {
            "recipe_version": 1,
            "sources": {
                "clean": {"path": _recipe_path(database_parent, speech_file), "sha256": sha256_file(speech_file)},
                "noise": {"path": _recipe_path(database_parent, noise_file), "sha256": sha256_file(noise_file)},
            },
            "operations": [
                {
                    "type": "additive_noise",
                    "version": 1,
                    "source": "noise",
                    "snr_db": pair["snr_db"],
                    "noise_start": int(start),
                    "noise_start_unit": "sample",
                    "seed": seed,
                },
                {"type": "peak_normalize", "version": 1, "limit": 0.99},
            ],
            "output": {"sample_rate": SAMPLE_RATE, "reference": "dry_clean", "max_seconds": MAX_SECONDS},
        }
    )
    metadata = {
        "corpus": "LibriSpeech",
        "subset": speech["subset"],
        "protocol": PROTOCOL,
        "official_voicebank_demand": False,
        "noise_scene": noise.get("subset"),
        "noise_path": noise["path"],
    }
    body = {
        "kind": "synthetic",
        "dataset": "LibriSpeech",
        "split": speech["subset"],
        "sample_rate": SAMPLE_RATE,
        "transcript": speech.get("transcript"),
        "speaker_id": None if speech.get("speaker_id") is None else str(speech["speaker_id"]),
        "recipe": recipe,
        "metadata": metadata,
    }
    return {
        "case_id": case_id_for(recipe),
        "recipe_json": json.dumps(recipe, ensure_ascii=False, sort_keys=True),
        "content_hash": case_content_hash(body),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        **{key: body[key] for key in ("kind", "dataset", "split", "sample_rate", "transcript", "speaker_id")},
    }


def write_staging(directory: Path, cases: list[dict[str, Any]], sets: list[tuple[str, str, list[str]]]) -> None:
    import duckdb

    directory.mkdir(parents=True, exist_ok=True)
    evaluation = directory / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE cases (
                case_id VARCHAR, kind VARCHAR, dataset VARCHAR, split VARCHAR,
                sample_rate INTEGER, transcript VARCHAR, speaker_id VARCHAR,
                recipe_json VARCHAR, content_hash VARCHAR, metadata VARCHAR
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["case_id"],
                    item["kind"],
                    item["dataset"],
                    item["split"],
                    item["sample_rate"],
                    item["transcript"],
                    item["speaker_id"],
                    item["recipe_json"],
                    item["content_hash"],
                    item["metadata"],
                )
                for item in cases
            ],
        )
        connection.execute(
            "COPY cases TO ? (FORMAT PARQUET)",
            [str(evaluation / "cases.parquet")],
        )
        connection.execute("CREATE TABLE sets (name VARCHAR, description VARCHAR, content_hash VARCHAR)")
        connection.execute("CREATE TABLE items (set_name VARCHAR, position INTEGER, case_id VARCHAR)")
        for name, description, case_ids in sets:
            digest = json_hash({"case_ids": case_ids, "description": description})
            connection.execute("INSERT INTO sets VALUES (?, ?, ?)", [name, description, digest])
            for position, case_id in enumerate(case_ids):
                connection.execute("INSERT INTO items VALUES (?, ?, ?)", [name, position, case_id])
        connection.execute("COPY sets TO ? (FORMAT PARQUET)", [str(evaluation / "sets.parquet")])
        connection.execute("COPY items TO ? (FORMAT PARQUET)", [str(evaluation / "set_items.parquet")])
    finally:
        connection.close()
    metadata = {
        "staging_version": 1,
        "files": {
            "cases": "evaluation/cases.parquet",
            "sets": "evaluation/sets.parquet",
            "set_items": "evaluation/set_items.parquet",
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def register_smoke(
    *,
    source_database: Path,
    destination: Path,
    staging: Path,
    corpora_root: Path,
) -> dict[str, Any]:
    if source_database.name != "resource.duckdb":
        raise ValueError("speech and noise rows are read from resource.duckdb")
    if destination.resolve() == source_database.resolve():
        raise ValueError("refusing to write the smoke set into resource.duckdb")
    import duckdb

    connection = duckdb.connect(str(source_database), read_only=True)
    try:
        train_speech = _speech(connection, "train-clean-100")
        valid_speech = _speech(connection, "dev-clean")
        train_noise = _noise(connection, TRAIN_SCENES)
        valid_noise = _noise(connection, VALID_SCENES)
    finally:
        connection.close()
    train = plan_pairs(train_speech, train_noise, role="train", count=N_TRAIN)
    valid = plan_pairs(valid_speech, valid_noise, role="validation", count=N_VALID)
    assert_disjoint(train, valid)
    cases = [
        build_case(pair, corpora_root=corpora_root, database_parent=destination.parent)
        for pair in train + valid
    ]
    train_ids = [item["case_id"] for item in cases[:N_TRAIN]]
    valid_ids = [item["case_id"] for item in cases[N_TRAIN:]]
    description_train = (
        "LibriSpeech train-clean-100 mixed with DEMAND office/domestic scenes at 0/5/10/15 dB. "
        "Smoke set for SE training. Not VoiceBank-DEMAND and not an evaluation set."
    )
    description_valid = (
        "LibriSpeech dev-clean mixed with DEMAND park/field noise at 0/5/10/15 dB. "
        "Validation smoke set. Speakers, utterances, and noise files are disjoint from the training smoke set. "
        "Not used as a test set."
    )
    write_staging(
        staging,
        cases,
        [
            (TRAIN_SET, description_train, train_ids),
            (VALID_SET, description_valid, valid_ids),
        ],
    )
    counts = register_staging(staging, destination)
    report = {
        "protocol": PROTOCOL,
        "train_set": TRAIN_SET,
        "validation_set": VALID_SET,
        "train_cases": N_TRAIN,
        "validation_cases": N_VALID,
        "train_speakers": sorted({str(item["speech"]["speaker_id"]) for item in train}),
        "validation_speakers": sorted({str(item["speech"]["speaker_id"]) for item in valid}),
        "train_noise": sorted({item["noise"]["path"] for item in train}),
        "validation_noise": sorted({item["noise"]["path"] for item in valid}),
        "snrs": list(SNRS),
        "registered": counts,
        "voicebank_demand_linked": (corpora_root / "VoiceBank-DEMAND").exists(),
    }
    (staging / "leak_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _speech(connection, subset: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT path, speaker_id, transcript, subset, frames
        FROM corpora.audio_files
        WHERE corpus = 'LibriSpeech' AND subset = ? AND sample_rate = 16000 AND frames >= 32000
        ORDER BY speaker_id, path
        """,
        [subset],
    ).fetchall()
    return [
        {"path": row[0], "speaker_id": row[1], "transcript": row[2], "subset": row[3], "frames": int(row[4])}
        for row in rows
    ]


def _noise(connection, scenes: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for scene in scenes:
        rows.extend(
            connection.execute(
                """
                SELECT path, subset
                FROM corpora.audio_files
                WHERE corpus = 'DEMAND' AND subset = ? AND sample_rate = 16000 AND file_name = 'ch01.wav'
                ORDER BY path
                """,
                [scene],
            ).fetchall()
        )
    found = {row[1]: {"path": row[0], "subset": row[1]} for row in rows}
    missing = [scene for scene in scenes if scene not in found]
    if missing:
        raise ValueError(f"missing DEMAND scenes: {missing}")
    return [found[scene] for scene in scenes]


def _one_per_speaker(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = []
    seen = set()
    for row in rows:
        speaker = row.get("speaker_id")
        if speaker in seen:
            continue
        seen.add(speaker)
        chosen.append(row)
    return chosen


def _seed(role: str, index: int, path: str) -> int:
    digest = hashlib.sha256(f"{PROTOCOL}|{role}|{index}|{path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _recipe_path(database_parent: Path, audio: Path) -> str:
    relative = Path(os_relpath(audio, database_parent))
    return relative.as_posix()


def os_relpath(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report = register_smoke(
        source_database=root / "data/db/resource.duckdb",
        destination=root / "data/db/training.duckdb",
        staging=root / "data/staging/training",
        corpora_root=root / "data/corpora",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
