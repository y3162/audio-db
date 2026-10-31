"""Read result records written by this pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_record(run_dir: Path, record_id: str) -> dict[str, Any] | None:
    payload = read_json(run_dir / "records" / f"{record_id}.json")
    if not isinstance(payload, dict):
        return None
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL {path}:{lineno}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"JSONL {path}:{lineno} must be an object")
        records.append(item)
    return records


def read_record_files(run_dir: Path) -> list[dict[str, Any]]:
    directory = run_dir / "records"
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        payload = read_json(path)
        if isinstance(payload, dict):
            records.append(payload)
    return records
