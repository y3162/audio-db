"""Compare derived 16 kHz audio with the installed librosa loader.

Official MP-SENet pins librosa 0.7.2, whose default resampler is kaiser_best.
That package is not installed here, so this comparison uses the installed
``librosa.load(..., sr=16000)`` and records that limit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compare_files(pairs: list[tuple[Path, Path]]) -> dict:
    import librosa
    import soundfile as sf
    import soxr

    rows = []
    for source, derived in pairs:
        loaded, rate = librosa.load(source, sr=16000, mono=True)
        stored, stored_rate = sf.read(derived, dtype="float32", always_2d=False)
        native, native_rate = sf.read(source, dtype="float32", always_2d=False)
        fresh = np.asarray(soxr.resample(native, int(native_rate), 16000, quality="HQ"), dtype=np.float32)
        stored = np.asarray(stored, dtype=np.float32)
        loaded = np.asarray(loaded, dtype=np.float32)
        rows.append(
            {
                "source": source.name,
                "native_rate": int(native_rate),
                "native_frames": int(native.size),
                "librosa_rate": int(rate),
                "librosa_frames": int(loaded.size),
                "soxr_frames": int(fresh.size),
                "stored_rate": int(stored_rate),
                "stored_frames": int(stored.size),
                "length_delta_stored_minus_librosa": int(stored.size - loaded.size),
                "length_delta_soxr_minus_librosa": int(fresh.size - loaded.size),
                "max_abs_soxr_float_vs_librosa": _max_abs(fresh, loaded),
                "max_abs_pcm_vs_librosa": _max_abs(stored, loaded),
            }
        )
    return {
        "method": "executed librosa.load(sr=16000) and soxr.resample quality=HQ",
        "librosa_version": librosa.__version__,
        "librosa_default_res_type": "soxr_hq",
        "soxr_version": soxr.__version__,
        "official_mp_senet_librosa": "0.7.2 default kaiser_best was not executed; resampy is not installed",
        "rows": rows,
        "max_length_delta_stored_minus_librosa": max(row["length_delta_stored_minus_librosa"] for row in rows),
        "min_length_delta_stored_minus_librosa": min(row["length_delta_stored_minus_librosa"] for row in rows),
        "max_abs_soxr_float_vs_librosa": max(row["max_abs_soxr_float_vs_librosa"] for row in rows),
        "max_abs_pcm_vs_librosa": max(row["max_abs_pcm_vs_librosa"] for row in rows),
    }


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    count = min(left.size, right.size)
    if count == 0:
        return float("inf")
    delta = np.max(np.abs(left[:count] - right[:count]))
    return float(delta)
