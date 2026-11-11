"""Compare derived 16 kHz audio with the installed librosa loader.

Official MP-SENet pins librosa 0.7.2, whose default resampler is kaiser_best.
That package is not installed here, so this comparison uses the installed
``librosa.load(..., sr=16000)`` and records that limit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compare_sources(pairs: list[tuple[Path, Path | None]]) -> dict:
    """Compare on-demand PCM_16 with librosa and, when present, a stored 16 kHz wav."""
    import librosa
    import soundfile as sf

    from src.experiments.data.prepare.resample import resample_file

    rows = []
    for source, derived in pairs:
        loaded, rate = librosa.load(source, sr=16000, mono=True)
        fresh = resample_file(source)
        loaded = np.asarray(loaded, dtype=np.float32)
        row = {
            "source": source.name,
            "librosa_rate": int(rate),
            "librosa_frames": int(loaded.size),
            "ondemand_frames": int(fresh.size),
            "length_delta_ondemand_minus_librosa": int(fresh.size - loaded.size),
            "max_abs_pcm_vs_librosa": _max_abs(fresh, loaded),
        }
        if derived is not None and derived.is_file():
            stored, stored_rate = sf.read(derived, dtype="float32", always_2d=False)
            stored = np.asarray(stored, dtype=np.float32)
            row["stored_rate"] = int(stored_rate)
            row["stored_frames"] = int(stored.size)
            row["max_abs_ondemand_vs_stored"] = _max_abs(fresh, stored)
        rows.append(row)
    stored_errors = [row["max_abs_ondemand_vs_stored"] for row in rows if "max_abs_ondemand_vs_stored" in row]
    return {
        "method": "on-demand soxr HQ PCM_16 versus librosa.load(sr=16000); stored wavs compared when present",
        "librosa_version": librosa.__version__,
        "rows": rows,
        "max_abs_pcm_vs_librosa": max(row["max_abs_pcm_vs_librosa"] for row in rows),
        "max_abs_ondemand_vs_stored": max(stored_errors) if stored_errors else None,
        "stored_files_compared": len(stored_errors),
    }


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    count = min(left.size, right.size)
    if count == 0:
        return float("inf")
    delta = np.max(np.abs(left[:count] - right[:count]))
    return float(delta)
