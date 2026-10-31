"""Parse JSONL or JSON-array manifests for DuckDB import.

Evaluation does not read these files. Register them with EvaluationDB.import_manifest
or ``make db-init``, then load the named set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.errors import DatasetError
from src.evaluation.sample import AudioSample

PATH_FIELDS = ("audio_path", "clean_path", "noisy_path", "noise_path")
KNOWN_FIELDS = {
    "utterance_id",
    "dataset",
    "split",
    "audio_path",
    "sample_rate",
    "transcript",
    "clean_path",
    "noisy_path",
    "noise_path",
    "snr_db",
    "speaker_id",
    "metadata",
}


def load_manifest(path: str | Path, root: str | Path | None = None) -> list[AudioSample]:
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetError(f"manifest not found: {file_path}")
    base = Path(root) if root is not None else file_path.parent
    text = file_path.read_text(encoding="utf-8")
    rows = _rows(file_path, text)
    if not rows:
        raise DatasetError(f"manifest is empty: {file_path}")
    return [_sample_from_row(row, lineno, base) for lineno, row in rows]


def _rows(path: Path, text: str) -> list[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON manifest {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise DatasetError("a JSON manifest must be an array of objects")
        rows = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise DatasetError(f"manifest item {index} must be an object")
            rows.append((index, item))
        return rows
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON on {path}:{lineno}: {exc}") from exc
        if not isinstance(item, dict):
            raise DatasetError(f"manifest line {lineno} must be an object")
        rows.append((lineno, item))
    return rows


def _sample_from_row(row: dict[str, Any], lineno: int, root: Path) -> AudioSample:
    if "utterance_id" not in row or not isinstance(row["utterance_id"], str) or not row["utterance_id"]:
        raise DatasetError(f"manifest line {lineno} needs a non-empty utterance_id")
    if "dataset" not in row or not isinstance(row["dataset"], str) or not row["dataset"]:
        raise DatasetError(f"manifest line {lineno} needs a non-empty dataset")
    metadata = row.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise DatasetError(f"manifest line {lineno} metadata must be an object")
    extra = {key: value for key, value in row.items() if key not in KNOWN_FIELDS}
    metadata = {**metadata, **extra}
    sample_rate = row.get("sample_rate")
    if sample_rate is not None and (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)):
        raise DatasetError(f"manifest line {lineno} sample_rate must be an integer")
    snr_db = row.get("snr_db")
    if snr_db is not None and (isinstance(snr_db, bool) or not isinstance(snr_db, (int, float))):
        raise DatasetError(f"manifest line {lineno} snr_db must be a number")
    paths = {name: _resolve(row.get(name), root, lineno, name) for name in PATH_FIELDS}
    if not any(paths.values()):
        raise DatasetError(f"manifest line {lineno} needs at least one audio path")
    return AudioSample(
        utterance_id=row["utterance_id"],
        dataset=row["dataset"],
        split=row.get("split"),
        audio_path=paths["audio_path"],
        sample_rate=sample_rate,
        transcript=row.get("transcript"),
        clean_path=paths["clean_path"],
        noisy_path=paths["noisy_path"],
        noise_path=paths["noise_path"],
        snr_db=None if snr_db is None else float(snr_db),
        speaker_id=row.get("speaker_id"),
        metadata=metadata,
    )


def _resolve(value: Any, root: Path, lineno: int, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DatasetError(f"manifest line {lineno} {field_name} must be a path string")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.as_posix()
