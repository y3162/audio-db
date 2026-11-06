"""Publish ``data_provenance.json`` beside training checkpoints.

Rank 0 is the only writer. The file is created exclusively, so a second
process cannot truncate it. Evaluation reads the JSON without this module.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from src.evaluation.envinfo import git_metadata
from src.evaluation.provenance import (
    FORMAT,
    PROVENANCE_NAME,
    SCHEMA_VERSION,
    canonical_bytes,
    content_sha256,
    parse_provenance_bytes,
    sidecar_path,
)
from src.training.core.checkpoint import assert_output_allowed
from src.training.errors import TrainingError

_IDENTITY_KEYS = (
    "run_id",
    "pipeline",
    "dataset_id",
    "snapshot_hash",
    "config_hash",
    "profile",
    "claim_label",
    "selection_uses_test",
    "model_name",
    "data_sha256",
)


def build_provenance(dataset: Any, config: dict[str, Any], process: Any) -> dict[str, Any]:
    """Data-lineage document for this run. It does not include the optimizer step."""
    model = config.get("model") or {}
    model_name = model.get("name") if isinstance(model, dict) else None
    if not isinstance(model_name, str) or not model_name:
        model_name = "unknown"
    world_size = int(process.world_size)
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    virtual = data_cfg.get("virtual_world_size")
    ddp_mode = "virtual" if virtual else ("single" if world_size <= 1 else "real")
    git_commit = git_metadata().get("commit") or "unknown"
    identity = _json_object(dataset.identity(), "dataset identity")
    parts = _parts(dataset)
    hashed = {
        "model_name": model_name,
        "seed": config.get("seed"),
        "loss": config.get("loss"),
        "world_size": world_size,
        "ddp_mode": ddp_mode,
        "identity": identity,
    }
    if virtual:
        hashed["virtual_world_size"] = int(virtual)
    config_hash = content_sha256(hashed)
    pipeline = parts.get("pipeline") or identity.get("pipeline") or "legacy"
    run_id = content_sha256(
        {
            "format": "audio-db-training-run-v1",
            "pipeline": pipeline,
            "model_name": model_name,
            "dataset_id": parts.get("dataset_id"),
            "snapshot_hash": parts.get("snapshot_hash"),
            "config_hash": config_hash,
            "profile": parts.get("profile"),
            "claim_label": parts.get("claim_label"),
            "selection_uses_test": parts.get("selection_uses_test"),
            "seed": config.get("seed"),
        }
    )
    data_sha256 = content_sha256({"materials": parts["materials"], "sets": parts["sets"]})
    limitations = list(parts.get("limitations") or [])
    if virtual:
        limitations.append(
            "virtual_world_size reproduces the rank data plan in one process; "
            "it is not a distributed gradient all-reduce"
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "run_id": run_id,
        "model_name": model_name,
        "pipeline": pipeline,
        "dataset_id": parts.get("dataset_id"),
        "dataset_name": parts.get("dataset_name"),
        "dataset_version": parts.get("dataset_version"),
        "snapshot_hash": parts.get("snapshot_hash"),
        "protocol_id": parts.get("protocol_id"),
        "profile": parts.get("profile"),
        "claim_label": parts.get("claim_label"),
        "selection_uses_test": parts.get("selection_uses_test"),
        "git_commit": git_commit,
        "config_hash": config_hash,
        "ddp_mode": ddp_mode,
        "world_size": world_size,
        "seed": config.get("seed"),
        "sets": parts["sets"],
        "materials": parts["materials"],
        "lineage_counts": parts.get("lineage_counts") or {},
        "limitations": limitations,
        "data_sha256": data_sha256,
        "dataset_identity": identity,
    }
    if virtual:
        document["virtual_world_size"] = int(virtual)
    return document


def publish_provenance(directory: Path, document: dict[str, Any], *, rank: int) -> bytes:
    """Write the run provenance once. The same identity keeps the existing bytes."""
    if rank != 0:
        raise TrainingError("data provenance is written by rank 0 only")
    path = directory / PROVENANCE_NAME
    assert_output_allowed(path)
    raw = canonical_bytes(document)
    directory.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return _existing_or_reject(path, document, raw)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return _existing_or_reject(path, document, raw)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    except Exception:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    os.close(fd)
    temporary.unlink(missing_ok=True)
    return raw


def write_provenance_sidecar(checkpoint: Path, raw: bytes) -> None:
    """Copy the run provenance next to one checkpoint file."""
    path = sidecar_path(checkpoint)
    assert_output_allowed(path)
    if path.is_file() and path.read_bytes() == raw:
        return
    if path.is_file():
        raise TrainingError(f"checkpoint provenance sidecar already differs: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise TrainingError(f"checkpoint provenance sidecar is being written: {path}") from exc
    try:
        os.write(fd, raw)
        os.fsync(fd)
    except Exception:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.replace(temporary, path)


def provenance_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def assert_checkpoint_provenance(payload: dict[str, Any], checkpoint: Path, dataset: Any) -> None:
    """Reject a checkpoint whose provenance file is missing, different, or for other data."""
    expected = payload.get("provenance_sha256")
    if expected is None:
        return
    if not isinstance(expected, str):
        raise TrainingError("checkpoint provenance_sha256 is invalid")
    directory_file = checkpoint.parent / str(payload.get("provenance_file") or PROVENANCE_NAME)
    files = [path for path in (sidecar_path(checkpoint), directory_file) if path.is_file()]
    if not files:
        raise TrainingError(f"checkpoint {checkpoint.name} is missing data provenance")
    digests = []
    document = None
    for path in files:
        try:
            raw = path.read_bytes()
            parsed = parse_provenance_bytes(raw)
        except (OSError, UnicodeError, ValueError) as exc:
            raise TrainingError(f"data provenance is unreadable: {path}") from exc
        except Exception as exc:
            raise TrainingError(f"data provenance is unreadable: {path}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise TrainingError(f"data provenance does not match {checkpoint.name}")
        digests.append(digest)
        document = parsed
    if len(set(digests)) != 1 or document is None:
        raise TrainingError(f"checkpoint provenance files disagree for {checkpoint.name}")
    identity = _json_object(dataset.identity(), "dataset identity")
    stored = document.get("dataset_identity")
    if stored != identity:
        raise TrainingError("checkpoint provenance does not match this dataset")


def _parts(dataset: Any) -> dict[str, Any]:
    producer = getattr(dataset, "provenance_parts", None)
    if callable(producer):
        parts = producer()
        if not isinstance(parts, dict):
            raise TrainingError("provenance_parts must return an object")
        parts.setdefault("pipeline", "reproduction")
        parts.setdefault("materials", [])
        parts.setdefault("sets", [])
        return parts
    identity = _json_object(dataset.identity(), "dataset identity")
    return {
        "pipeline": "legacy",
        "dataset_id": None,
        "dataset_name": identity.get("name"),
        "dataset_version": None,
        "snapshot_hash": None,
        "protocol_id": None,
        "profile": None,
        "claim_label": None,
        "selection_uses_test": None,
        "materials": [],
        "sets": [],
        "lineage_counts": {"source": 0, "derived": 0, "unknown": 0},
        "limitations": ["legacy checkpoint does not record content hashes or source audio"],
    }


def _existing_or_reject(path: Path, document: dict[str, Any], raw: bytes) -> bytes:
    try:
        existing = path.read_bytes()
        parsed = parse_provenance_bytes(existing)
    except (OSError, UnicodeError, ValueError) as exc:
        raise TrainingError(f"data provenance is unreadable: {path}") from exc
    except Exception as exc:
        raise TrainingError(f"data provenance is unreadable: {path}") from exc
    if existing == raw:
        return existing
    if _identity(parsed) == _identity(document):
        return existing
    raise TrainingError("data provenance does not match this run")


def _identity(document: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(document.get(key) for key in _IDENTITY_KEYS)


def _json_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError(f"{where} must be an object")
    try:
        canonical_bytes(value)
    except TypeError as exc:
        raise TrainingError(f"{where} is not JSON") from exc
    return value
