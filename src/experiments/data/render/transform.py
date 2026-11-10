"""Normalization and windowing.

These functions do not draw random numbers and do not resample. Crop start
positions are already in ``CropSpec``.
"""

from __future__ import annotations

import numpy as np

from src.experiments.data.errors import ReproductionError
from src.experiments.data.render.types import CropSpec

_TARGET_DBFS = -25.0
_EPS = 1e-6


def trim_to_shorter(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MP-SENet keeps ``min(len(clean), len(noisy))`` before normalization."""
    length = min(int(noisy.shape[0]), int(clean.shape[0]))
    if length < 1:
        raise ReproductionError("pair has no samples")
    return noisy[:length], clean[:length]


def noisy_rms_unit(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale both waveforms by the noisy RMS factor.

    ``norm_factor = sqrt(length / sum(noisy**2))``. The sum is float32, matching
    the official ``torch.sum`` on a float32 waveform.
    """
    noisy32 = np.asarray(noisy, dtype=np.float32)
    clean32 = np.asarray(clean, dtype=np.float32)
    if noisy32.shape != clean32.shape or noisy32.ndim != 1 or noisy32.size < 1:
        raise ReproductionError("noisy RMS normalization expects equal mono waveforms")
    energy = np.sum(noisy32 * noisy32, dtype=np.float32)
    if not np.isfinite(energy) or float(energy) <= 0.0:
        raise ReproductionError("noisy waveform has no energy")
    scale = np.sqrt(np.float32(noisy32.size) / energy)
    return (noisy32 * scale).astype(np.float32), (clean32 * scale).astype(np.float32)


def independent_audio_norm(wave: np.ndarray) -> np.ndarray:
    """Two-stage −25 dBFS normalization used by MossFormerGAN.

    The official ``audio_norm`` runs in float64 because ``soundfile.read``
    returns float64. An empty peak set (a constant waveform) is the official
    failure: the peak mean is undefined. This raises instead of emitting NaN.
    """
    values = np.asarray(wave, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ReproductionError("audio_norm expects a non-empty mono waveform")
    rms = np.sqrt(np.mean(values * values))
    scaled = values * (10 ** (_TARGET_DBFS / 20.0) / (rms + _EPS))
    power = scaled * scaled
    average = np.mean(power)
    peaks = power[power > average]
    if peaks.size == 0:
        raise ReproductionError(
            "independent_audio_norm_v1 peak set is empty; the official audio_norm mean is undefined"
        )
    peak_rms = np.sqrt(np.mean(peaks))
    normalized = scaled * (10 ** (_TARGET_DBFS / 20.0) / (peak_rms + _EPS))
    if not np.isfinite(normalized).all():
        raise ReproductionError("independent_audio_norm_v1 produced a non-finite waveform")
    return normalized.astype(np.float32)


def apply_window(
    noisy: np.ndarray,
    clean: np.ndarray,
    crop: CropSpec | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Slice a planned window and tail-pad when the utterance is short.

    Both official loaders copy a short utterance to the front of a zero buffer
    and leave the tail at zero. They do not repeat the head to fill the window.
    """
    if noisy.shape != clean.shape or noisy.ndim != 1:
        raise ReproductionError("windowing expects equal mono waveforms")
    if crop is None:
        return noisy.astype(np.float32, copy=False), clean.astype(np.float32, copy=False), int(noisy.shape[0])
    if crop.pad not in ("none", "tail_zero", "head_place"):
        raise ReproductionError(f"unknown pad mode {crop.pad}")
    stop = crop.start + crop.length
    if crop.start < 0 or crop.length < 1 or stop > int(noisy.shape[0]):
        raise ReproductionError(
            f"crop {crop.start}:{stop} is outside the {noisy.shape[0]}-sample waveform"
        )
    noisy_view = np.asarray(noisy[crop.start:stop], dtype=np.float32)
    clean_view = np.asarray(clean[crop.start:stop], dtype=np.float32)
    pad = crop.pad_to - int(noisy_view.shape[0])
    if pad < 0 or crop.pad_to < 1:
        raise ReproductionError("crop pad_to is shorter than the taken samples")
    valid = int(noisy_view.shape[0])
    if pad == 0:
        return noisy_view, clean_view, valid
    return (
        np.pad(noisy_view, (0, pad)).astype(np.float32),
        np.pad(clean_view, (0, pad)).astype(np.float32),
        valid,
    )
