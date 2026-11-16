"""Persist the resolved experiment document next to a run.

The checkpoint hash already identifies the resolved configuration. These files
keep the text and the parent chain so a later resume does not need the parent
YAML files to remain unchanged. Rank 0 is the only writer.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.config import ExperimentConfig, config_sha256, dump_resolved_yaml
from src.experiments.errors import CheckpointError

RECORD_FORMAT = "audio-db-run-record-v1"


def publish_run_record(
    directory: Path,
    config: ExperimentConfig,
    *,
    resume: bool,
    world_size: int,
) -> None:
    """Write ``config.yaml`` and append one launch to ``metadata.json``.

    The run id is chosen once. A later launch keeps it and appends history.
    When the resolved text changes, the previous text is copied under
    ``config_history/`` before replacement.
    """

    if not isinstance(resume, bool):
        raise CheckpointError("resume flag must be a bool")
    text = dump_resolved_yaml(config)
    digest = config_sha256(config)
    meta_path = directory / "metadata.json"
    config_path = directory / "config.yaml"
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    launch = {
        "batch": _recorded_batch(config, world_size),
        "config_sha256": digest,
        "extends_chain": list(config.extends_chain),
        "resume": resume,
        "source_hashes": list(config.source_hashes),
        "source_path": None if config.source_path is None else str(config.source_path),
        "started_at": started_at,
    }
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read run metadata {meta_path}: {exc}") from exc
        if not isinstance(loaded, dict) or loaded.get("format") != RECORD_FORMAT:
            raise CheckpointError(f"run metadata is invalid: {meta_path}")
        launches = loaded.get("launches")
        if not isinstance(launches, list) or not isinstance(loaded.get("run_id"), str):
            raise CheckpointError(f"run metadata is invalid: {meta_path}")
        previous = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
        index = len(launches)
        if previous and previous != text:
            history = directory / "config_history"
            history.mkdir(exist_ok=True)
            _atomic_text(history / f"{index:04d}.yaml", previous)
        launch["index"] = index
        launches.append(launch)
        loaded["config_sha256"] = digest
        loaded["extends_chain"] = list(config.extends_chain)
        loaded["source_hashes"] = list(config.source_hashes)
        meta = loaded
    else:
        launch["index"] = 0
        meta = {
            "config_sha256": digest,
            "extends_chain": list(config.extends_chain),
            "format": RECORD_FORMAT,
            "launches": [launch],
            "run_id": uuid.uuid4().hex,
            "schema": config.schema,
            "source_hashes": list(config.source_hashes),
            "source_path": None if config.source_path is None else str(config.source_path),
        }
    _atomic_text(config_path, text)
    _atomic_text(meta_path, json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _recorded_batch(config: ExperimentConfig, world_size: int) -> dict[str, int]:
    """The effective batch for this launch. Resume writes the same fields again."""

    from src.experiments.calc.batch import exact_accumulation

    trainer = config.components[config.entry]
    global_batch = int(trainer.kwargs["global_batch_size"])
    micro_batch = int(trainer.kwargs["micro_batch_size"])
    return {
        "accumulation_steps": exact_accumulation(global_batch, micro_batch, world_size),
        "global_batch_size": global_batch,
        "micro_batch_size": micro_batch,
        "world_size": int(world_size),
    }


def record_batch_probe(directory: Path, report: dict[str, object]) -> None:
    """Attach the preflight batch report to the latest launch. Rank 0 calls this."""

    meta_path = directory / "metadata.json"
    if not meta_path.is_file():
        raise CheckpointError(f"run metadata is missing: {meta_path}")
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read run metadata {meta_path}: {exc}") from exc
    launches = loaded.get("launches") if isinstance(loaded, dict) else None
    if not isinstance(launches, list) or not launches or not isinstance(launches[-1], dict):
        raise CheckpointError(f"run metadata is invalid: {meta_path}")
    launches[-1]["batch_probe"] = report
    _atomic_text(meta_path, json.dumps(loaded, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise CheckpointError(f"cannot write {path}: {exc}") from exc
