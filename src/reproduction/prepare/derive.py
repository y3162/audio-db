"""16 kHz PCM_16 derivatives of 48 kHz VoiceBank audio.

The resampler is soxr HQ. That is the configured default, not a claim that the
official 16 kHz distribution used the same converter. An existing file is
reused when its stamp matches the source bytes and this derivation. A different
file at the same path is left in place and reported as an error.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.data.recipe import sha256_file
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import derivation_id_for

TARGET_RATE = 16000
SOURCE_RATE = 48000
QUALITY = "HQ"


def derivation_row() -> dict:
    spec = _spec()
    kind = "resample"
    return {
        "derivation_id": derivation_id_for(
            kind=kind,
            tool=spec["tool"],
            tool_version=spec["tool_version"],
            params=spec["params"],
        ),
        "kind": kind,
        "spec_json": spec,
    }


def derive_all(jobs: list[dict], *, workers: int) -> list[dict]:
    """Resample each job. ``jobs`` items have ``source`` and ``dest`` paths."""
    if workers < 1:
        raise ReproductionError("workers must be positive")
    if not jobs:
        return []
    derivation_id = derivation_row()["derivation_id"]
    payloads = [
        {"source": str(job["source"]), "dest": str(job["dest"]), "derivation_id": derivation_id}
        for job in jobs
    ]
    if workers == 1:
        results = []
        for index, item in enumerate(payloads, 1):
            results.append(_derive_one(item))
            if index % 500 == 0 or index == len(payloads):
                print(f"derive\t{index}\t{len(payloads)}", flush=True)
        return results
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_derive_one, payloads, chunksize=8), 1):
            results.append(result)
            if index % 500 == 0 or index == len(payloads):
                print(f"derive\t{index}\t{len(payloads)}", flush=True)
    return results


def stamp_path(wav_path: Path) -> Path:
    return wav_path.with_name(wav_path.name + ".stamp.json")


def read_stamp(wav_path: Path) -> dict | None:
    path = stamp_path(wav_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReproductionError(f"cannot read derivation stamp: {path}") from exc


def _derive_one(job: dict) -> dict:
    source = Path(job["source"])
    dest = Path(job["dest"])
    derivation_id = job["derivation_id"]
    source_sha = sha256_file(source)
    if dest.is_file():
        existing = _adopt_or_refuse(source, dest, source_sha, derivation_id)
        if existing is not None:
            return existing
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = _resample(source)
    payload = _stamp(source_sha, derivation_id, info, dest_sha="")
    partial_wav = dest.with_name(dest.name + ".partial")
    partial_stamp = stamp_path(dest).with_name(stamp_path(dest).name + ".partial")
    try:
        _write_wav(partial_wav, info["audio"], info["sample_rate"])
        output_sha = sha256_file(partial_wav)
        payload["output_sha256"] = output_sha
        payload["size_bytes"] = partial_wav.stat().st_size
        partial_stamp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial_wav, dest)
        os.replace(partial_stamp, stamp_path(dest))
    except Exception:
        for path in (partial_wav, partial_stamp):
            if path.exists():
                path.unlink()
        raise
    payload.pop("audio", None)
    return _public_stamp(payload, dest)


def _adopt_or_refuse(source: Path, dest: Path, source_sha: str, derivation_id: str) -> dict | None:
    actual = sha256_file(dest)
    stamp = read_stamp(dest)
    if _stamp_matches(stamp, source_sha, derivation_id, actual):
        return _public_stamp(stamp, dest)
    info = _resample(source)
    partial = dest.with_name(dest.name + ".check.partial")
    try:
        _write_wav(partial, info["audio"], info["sample_rate"])
        recomputed = sha256_file(partial)
    finally:
        if partial.exists():
            partial.unlink()
    if recomputed != actual:
        raise ReproductionError(
            f"refusing to overwrite {dest}: existing sha256 {actual} does not match this derivation"
        )
    payload = _stamp(source_sha, derivation_id, info, dest_sha=actual)
    payload["size_bytes"] = dest.stat().st_size
    _write_stamp(dest, payload)
    return _public_stamp(payload, dest)


def _resample(source: Path) -> dict:
    import soundfile as sf
    import soxr

    audio, rate = sf.read(source, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if audio.ndim != 1:
        raise ReproductionError(f"{source} is not mono")
    if int(rate) != SOURCE_RATE:
        raise ReproductionError(f"{source} sample rate is {rate}, expected {SOURCE_RATE}")
    converted = np.asarray(soxr.resample(audio, SOURCE_RATE, TARGET_RATE, quality=QUALITY), dtype=np.float32)
    converted = np.ascontiguousarray(converted)
    if converted.size < 1:
        raise ReproductionError(f"{source} resampled to an empty waveform")
    clipped = int(np.count_nonzero(np.abs(converted) > 1.0))
    return {
        "audio": np.clip(converted, -1.0, 1.0),
        "sample_rate": TARGET_RATE,
        "frames": int(converted.size),
        "channels": 1,
        "source_frames": int(audio.size),
        "source_rate": SOURCE_RATE,
        "clipped_samples": clipped,
    }


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(path, audio, sample_rate, subtype="PCM_16", format="WAV")


def _stamp(source_sha: str, derivation_id: str, info: dict, *, dest_sha: str) -> dict:
    return {
        "source_sha256": source_sha,
        "source_frames": info["source_frames"],
        "source_rate": info["source_rate"],
        "derivation_id": derivation_id,
        "output_sha256": dest_sha,
        "frames": info["frames"],
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "clipped_samples": info["clipped_samples"],
        "size_bytes": 0,
    }


def _stamp_matches(stamp: dict | None, source_sha: str, derivation_id: str, output_sha: str) -> bool:
    if not isinstance(stamp, dict):
        return False
    return (
        stamp.get("source_sha256") == source_sha
        and stamp.get("derivation_id") == derivation_id
        and stamp.get("output_sha256") == output_sha
    )


def _write_stamp(dest: Path, payload: dict) -> None:
    path = stamp_path(dest)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _public_stamp(stamp: dict, dest: Path) -> dict:
    body = dict(stamp)
    body.pop("audio", None)
    body["dest"] = str(dest)
    return body


def _spec() -> dict:
    import soundfile as sf
    import soxr

    return {
        "tool": "soxr",
        "tool_version": soxr.__version__,
        "params": {
            "quality": QUALITY,
            "source_rate": SOURCE_RATE,
            "target_rate": TARGET_RATE,
            "subtype": "PCM_16",
            "reader": "soundfile",
            "reader_version": sf.__version__,
        },
    }
