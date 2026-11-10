"""Resolve pinned utterance ids to VoiceBank files and catalog rows.

This module only reads. A missing audio file raises ``PreparationBlocked``
before any derived file or database is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.audio import read_wav_stream_info
from src.experiments.data.errors import PreparationBlocked, ReproductionError
from src.experiments.data.prepare.lists import speaker_of


@dataclass(frozen=True)
class AudioFile:
    utterance_id: str
    speaker_id: str
    split: str
    role: str
    subset: str
    rel_path: str
    sample_rate: int
    frames: int
    channels: int
    size_bytes: int
    transcript: str | None


def missing_audio(
    *,
    corpora: Path,
    layout: dict[str, Any],
    train_ids: list[str],
    test_ids: list[str],
) -> list[str]:
    missing: list[str] = []
    for split, ids in (("train", train_ids), ("test", test_ids)):
        for utterance_id in ids:
            for role in ("clean", "noisy"):
                subset = layout[f"{split}_{role}_subset"]
                rel_path = f"{layout['corpus']}/{subset}/{utterance_id}.wav"
                if not (corpora / rel_path).is_file():
                    missing.append(rel_path)
    return missing


def collect_audio(
    *,
    corpora: Path,
    layout: dict[str, Any],
    train_ids: list[str],
    test_ids: list[str],
    catalog: dict[str, dict[str, Any]] | None,
) -> list[AudioFile]:
    missing: list[str] = []
    found: list[AudioFile] = []
    seen: set[str] = set()
    for split, ids in (("train", train_ids), ("test", test_ids)):
        text_subset = layout[f"{split}_text_subset"]
        for utterance_id in ids:
            speaker_id = speaker_of(utterance_id)
            transcript = _transcript(corpora, layout["corpus"], text_subset, utterance_id)
            for role in ("clean", "noisy"):
                subset = layout[f"{split}_{role}_subset"]
                rel_path = f"{layout['corpus']}/{subset}/{utterance_id}.wav"
                if rel_path in seen:
                    raise ReproductionError(f"duplicate corpus path: {rel_path}")
                seen.add(rel_path)
                path = corpora / rel_path
                if not path.is_file():
                    missing.append(rel_path)
                    continue
                rate, frames, channels = read_wav_stream_info(path)
                record = AudioFile(
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    split=split,
                    role=role,
                    subset=subset,
                    rel_path=rel_path,
                    sample_rate=int(rate),
                    frames=int(frames),
                    channels=int(channels),
                    size_bytes=path.stat().st_size,
                    transcript=transcript if role == "clean" else None,
                )
                if catalog is not None:
                    _match_catalog(record, catalog.get(rel_path))
                _expect_layout(record, layout)
                found.append(record)
    if missing:
        preview = ", ".join(missing[:5])
        raise PreparationBlocked(
            f"{len(missing)} VoiceBank files are missing under {corpora}. Examples: {preview}"
        )
    return found


def read_catalog(resource_db: Path, corpus: str, subsets: list[str]) -> dict[str, dict[str, Any]]:
    import duckdb

    connection = duckdb.connect(str(resource_db), read_only=True)
    try:
        placeholders = ", ".join("?" for _ in subsets)
        rows = connection.execute(
            f"""
            SELECT path, corpus, subset, sample_rate, frames, channels, speaker_id, utterance_id
            FROM corpora.audio_files
            WHERE corpus = ? AND subset IN ({placeholders})
            """,
            [corpus, *subsets],
        ).fetchall()
    finally:
        connection.close()
    found = {}
    for path, row_corpus, subset, sample_rate, frames, channels, speaker_id, utterance_id in rows:
        found[path] = {
            "path": path,
            "corpus": row_corpus,
            "subset": subset,
            "sample_rate": int(sample_rate),
            "frames": int(frames),
            "channels": int(channels),
            "speaker_id": speaker_id,
            "utterance_id": utterance_id,
        }
    return found


def derived_rel(layout: dict[str, Any], subset: str, utterance_id: str) -> str:
    return f"{layout['derived_rel_root']}/{subset}/{utterance_id}.wav"


def _transcript(corpora: Path, corpus: str, subset: str, utterance_id: str) -> str | None:
    path = corpora / corpus / subset / f"{utterance_id}.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _match_catalog(record: AudioFile, row: dict[str, Any] | None) -> None:
    if row is None:
        raise ReproductionError(f"resource catalog has no row for {record.rel_path}")
    mismatches = []
    for field in ("sample_rate", "frames", "channels", "subset"):
        if row[field] != getattr(record, field):
            mismatches.append(f"{field} catalog={row[field]} file={getattr(record, field)}")
    if row.get("speaker_id") not in (None, record.speaker_id):
        mismatches.append(f"speaker catalog={row.get('speaker_id')} file={record.speaker_id}")
    if row.get("utterance_id") not in (None, record.utterance_id):
        mismatches.append(f"utterance catalog={row.get('utterance_id')} file={record.utterance_id}")
    if mismatches:
        raise ReproductionError(f"{record.rel_path} does not match the catalog: {'; '.join(mismatches)}")


def _expect_layout(record: AudioFile, layout: dict[str, Any]) -> None:
    if record.sample_rate != int(layout["source_rate"]):
        raise ReproductionError(
            f"{record.rel_path} sample rate is {record.sample_rate}, expected {layout['source_rate']}"
        )
    if record.channels != 1:
        raise ReproductionError(f"{record.rel_path} has {record.channels} channels")
    if record.frames < 1:
        raise ReproductionError(f"{record.rel_path} has no frames")
