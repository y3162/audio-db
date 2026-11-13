"""Render the canonical VoiceBank SNR-shift recipe without writing audio."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from src.experiments.data.errors import ReproductionError


def render_voicebank_shift(row: Mapping[str, Any], root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy, reference) at 48 kHz, preserving the v1 float mixture."""
    if row.get("status") != "ready":
        raise ReproductionError("SNR-shift recipe is not ready")

    def path(field: str) -> Path:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ReproductionError(f"missing {field}")
        result = Path(value)
        if result.is_absolute() or ".." in result.parts:
            raise ReproductionError(f"invalid {field}")
        return root / result

    return render_voicebank_shift_files(row, path("clean_path"), path("noise_path"))


def render_voicebank_shift_files(row: Mapping[str, Any], clean_path: Path, noise_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Render a ready recipe using already-resolved, validated asset paths."""
    if row.get("status") != "ready":
        raise ReproductionError("SNR-shift recipe is not ready")

    def finite(field: str) -> float:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReproductionError(f"invalid {field}") from exc
        if not math.isfinite(value):
            raise ReproductionError(f"invalid {field}")
        return value

    def index(field: str) -> int:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ReproductionError(f"invalid {field}")
        try:
            return int(value)
        except ValueError as exc:
            raise ReproductionError(f"invalid {field}") from exc

    try:
        clean, rate = sf.read(clean_path, dtype="float64", always_2d=False)
        start, end = index("noise_start_48k"), index("noise_end_exclusive_48k")
        with sf.SoundFile(noise_path) as handle:
            if handle.samplerate != 48000 or handle.channels != 1 or not 0 <= start < end <= handle.frames:
                raise ReproductionError("invalid noise format or span")
            handle.seek(start)
            noise = handle.read(end - start, dtype="float64")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ReproductionError("invalid SNR-shift audio or span") from exc
    if rate != 48000 or clean.ndim != 1 or clean.size == 0 or noise.shape != clean.shape:
        raise ReproductionError("SNR-shift audio has invalid rate or shape")
    reference = finite("clean_gain_g") * clean
    noisy = reference + finite("shifted_noise_gain") * noise
    if not np.isfinite(reference).all() or not np.isfinite(noisy).all():
        raise ReproductionError("SNR-shift audio is not finite")
    return noisy, reference
