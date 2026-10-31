"""Stable hashes and per-utterance seeds.

Seeds come from SHA-256. Python's built-in hash() is not used, because it
changes across processes when hash randomization is enabled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.evaluation.errors import ConfigError

SCHEMA_VERSION = 1
MOCK_FINGERPRINT = "0" * 64

_FILE_HASHES: dict[str, str] = {}


def derive_seed(*parts: object) -> int:
    """Return a process-stable 31-bit seed from the given parts."""
    payload = "\0".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(blob.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_file_sha256(path: Path) -> str:
    """Hash a checkpoint once per size and mtime."""
    stat = path.stat()
    key = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    cached = _FILE_HASHES.get(key)
    if cached is None:
        cached = sha256_file(path)
        _FILE_HASHES[key] = cached
    return cached


def tensor_sha256(waveform: torch.Tensor | None) -> str | None:
    if waveform is None:
        return None
    data = waveform.detach().cpu().contiguous().to(torch.float32).numpy()
    return sha256_bytes(data.tobytes())


def checkpoint_files(model) -> dict[str, str]:
    if model.checkpoints:
        return dict(model.checkpoints)
    if model.checkpoint:
        return {"checkpoint": model.checkpoint}
    return {}


def checkpoint_fingerprint(model) -> dict[str, str]:
    """Content hashes for a model. Mock models have a constant fingerprint."""
    if model.name == "mock":
        return {"mock": MOCK_FINGERPRINT}
    files = checkpoint_files(model)
    if not files:
        raise ConfigError(f"checkpoint is required for {model.name}")
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for name, raw_path in sorted(files.items()):
        path = Path(raw_path)
        if path.is_dir():
            raise ConfigError(
                f"checkpoint for {model.name}.{name} is a directory: {path}. "
                "Pass the weight file, or a mapping of names to files."
            )
        if not path.is_file():
            missing.append(f"{name}={path}")
            continue
        hashes[name] = cached_file_sha256(path)
    if missing:
        raise ConfigError(
            f"checkpoint files not found for {model.name}: {', '.join(missing)}. "
            "The pipeline does not substitute other weights."
        )
    return hashes


def checkpoint_status(model) -> dict[str, Any]:
    if model.name == "mock":
        return {"ready": True, "paths": {}, "fingerprint": {"mock": MOCK_FINGERPRINT}}
    files = checkpoint_files(model)
    paths = {name: {"path": path, "exists": Path(path).is_file()} for name, path in sorted(files.items())}
    ready = bool(paths) and all(item["exists"] for item in paths.values())
    return {"ready": ready, "paths": paths, "fingerprint": None}


def canonical_snr(snr_db: float | None) -> float | None:
    if snr_db is None:
        return None
    return float(f"{float(snr_db):.6f}")


def make_slot(
    *,
    config_hash: str,
    torch_version: str,
    utterance_id: str,
    dataset: str,
    model_name: str,
    variant: str | None,
    checkpoint_sha256: dict[str, str],
    snr_db: float | None,
    sample_rate: int | None,
) -> dict[str, Any]:
    """Identity of a result slot. Audio content is checked separately."""
    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "torch": torch_version,
        "utterance_id": utterance_id,
        "dataset": dataset,
        "model": model_name,
        "variant": variant,
        "checkpoint_sha256": checkpoint_sha256,
        "snr_db": canonical_snr(snr_db),
        "sample_rate": None if sample_rate is None else int(sample_rate),
    }


def make_identity(
    slot: dict[str, Any],
    *,
    input_sha256: str | None,
    clean_sha256: str | None,
    enhance_seed: int | None,
    mix_seed: int | None,
) -> dict[str, Any]:
    identity = dict(slot)
    identity["input_sha256"] = input_sha256
    identity["clean_sha256"] = clean_sha256
    identity["enhance_seed"] = enhance_seed
    identity["mix_seed"] = mix_seed
    return identity


def record_id(slot: dict[str, Any]) -> str:
    return sha256_json(slot)[:32]
