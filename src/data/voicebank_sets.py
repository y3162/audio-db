"""VoiceBank-DEMAND 28-speaker train/validation recipes.

Validation speakers are p286 and p287 (872 files). The remaining 10,700
training files do not share speakers with the official 824-file test set.
Cases store the paired noisy file and use the clean file as dry_clean.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.audio import read_wav_stream_info
from src.data.recipe import canonical_recipe, case_content_hash, case_id_for, sha256_file
from src.data.staging import register_staging
from src.data.training_sets import write_staging

VALID_SPEAKERS = frozenset({"p286", "p287"})
TRAIN_SET = "voicebank_28_train"
VALID_SET = "voicebank_28_valid"


def build(corpora_root: Path, staging: Path, database: Path) -> dict[str, int]:
    root = corpora_root / "VoiceBank-DEMAND"
    clean = root / "clean_trainset_28spk_wav"
    noisy = root / "noisy_trainset_28spk_wav"
    names = sorted(path.name for path in clean.glob("*.wav"))
    noisy_names = {path.name for path in noisy.glob("*.wav")}
    if names != sorted(noisy_names):
        raise ValueError("clean and noisy 28-speaker filenames differ")
    test_speakers = {path.name.split("_", 1)[0] for path in (root / "clean_testset_wav").glob("*.wav")}
    train_speakers = {name.split("_", 1)[0] for name in names if name.split("_", 1)[0] not in VALID_SPEAKERS}
    if test_speakers & train_speakers or test_speakers & VALID_SPEAKERS:
        raise ValueError("train or validation speakers overlap the official test speakers")
    cases: list[dict] = []
    roles: dict[str, list[str]] = {"train": [], "valid": []}
    parent = database.parent
    for name in names:
        speaker = name.split("_", 1)[0]
        role = "valid" if speaker in VALID_SPEAKERS else "train"
        clean_path = clean / name
        noisy_path = noisy / name
        clean_info = read_wav_stream_info(clean_path)
        noisy_info = read_wav_stream_info(noisy_path)
        if clean_info != noisy_info:
            raise ValueError(f"shape mismatch: {name}")
        recipe = canonical_recipe(
            {
                "recipe_version": 1,
                "sources": {
                    "clean": {"path": _rel(parent, clean_path), "sha256": sha256_file(clean_path)},
                    "noisy": {"path": _rel(parent, noisy_path), "sha256": sha256_file(noisy_path)},
                },
                "operations": [],
                "output": {"sample_rate": clean_info[0], "reference": "dry_clean"},
            }
        )
        metadata = {
            "corpus": "VoiceBank-DEMAND",
            "original_filename": name,
            "pairing": "matching filename",
            "validation_speakers": sorted(VALID_SPEAKERS),
        }
        body = {
            "kind": "paired",
            "dataset": "VoiceBank-DEMAND",
            "split": role,
            "sample_rate": clean_info[0],
            "transcript": None,
            "speaker_id": speaker,
            "recipe": recipe,
            "metadata": metadata,
        }
        item = {
            "case_id": case_id_for(recipe),
            "recipe_json": json.dumps(recipe, sort_keys=True),
            "content_hash": case_content_hash(body),
            "metadata": json.dumps(metadata, sort_keys=True),
            **{key: body[key] for key in ("kind", "dataset", "split", "sample_rate", "transcript", "speaker_id")},
        }
        cases.append(item)
        roles[role].append(item["case_id"])
    if len(roles["train"]) != 10700 or len(roles["valid"]) != 872:
        raise ValueError(f"unexpected split sizes train={len(roles['train'])} valid={len(roles['valid'])}")
    write_staging(
        staging,
        cases,
        [
            (TRAIN_SET, "VoiceBank-DEMAND 28-speaker train, excluding p286 and p287.", roles["train"]),
            (VALID_SET, "VoiceBank-DEMAND validation speakers p286 and p287.", roles["valid"]),
        ],
    )
    register_staging(staging, database)
    return {"train": len(roles["train"]), "valid": len(roles["valid"])}


def _rel(parent: Path, path: Path) -> str:
    return Path(os_rel(path, parent)).as_posix()


def os_rel(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)
