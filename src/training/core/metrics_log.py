"""Append-only training log.

The file is the primary record of a run. DuckDB is not involved. Resuming a
checkpoint drops lines whose global step is later than that checkpoint so the
repeated steps are not stored twice. Lines without a global step, such as the
run header, stay.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def truncate_after(path: Path, global_step: int) -> None:
    """Keep the header and records at or before ``global_step``."""
    if not path.is_file():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = record.get("global_step")
        if step is None or int(step) <= global_step:
            kept.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    text = ("\n".join(kept) + "\n") if kept else ""
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
