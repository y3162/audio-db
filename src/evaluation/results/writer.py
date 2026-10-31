"""Atomic result files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def write_record(run_dir: Path, record: dict[str, Any]) -> Path:
    path = run_dir / "records" / f"{record['record_id']}.json"
    atomic_write_json(path, record)
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    text = ("\n".join(lines) + "\n") if lines else ""
    atomic_write_text(path, text)
