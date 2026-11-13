"""Float32 16 kHz audio from 48 kHz VoiceBank, rendered on demand.

The converter is soxr HQ without additional clipping or quantization.
Nothing is written, and nothing is cached.
The sample count for 48 kHz to 16 kHz is ``(source_frames + 1) // 3``. That
matches soxr 1.1.0 HQ on every registered VoiceBank file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import derivation_id_for

SOURCE_RATE = 48000
TARGET_RATE = 16000
QUALITY = "HQ"
ORDER = (
    "read_float32",
    "soxr_hq",
    "protocol_normalize",
    "protocol_window",
)


def soxr_hq_frame_count(frames: int, source_rate: int = SOURCE_RATE, target_rate: int = TARGET_RATE) -> int:
    """Sample count of ``soxr.resample(..., quality='HQ')`` without the waveform."""
    if frames < 1:
        raise ReproductionError(f"cannot resample {frames} frames")
    if source_rate != SOURCE_RATE or target_rate != TARGET_RATE:
        raise ReproductionError(
            f"VoiceBank on-demand resampling is {SOURCE_RATE} to {TARGET_RATE}, not {source_rate} to {target_rate}"
        )
    return (int(frames) + 1) // 3


def resample_file(path: Path) -> np.ndarray:
    """Read one 48 kHz wav and return 16 kHz float32."""
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if audio.ndim != 1:
        raise ReproductionError(f"{path} is not mono")
    return resample_float32(audio, int(rate))


def resample_float32(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample mono 48 kHz audio to float32 at 16 kHz with soxr HQ."""
    import soxr

    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim != 1:
        raise ReproductionError("resample input must be mono")
    if int(source_rate) != SOURCE_RATE:
        raise ReproductionError(f"sample rate is {source_rate}, expected {SOURCE_RATE}")
    if wave.size < 1:
        raise ReproductionError("resample input is empty")
    if not np.isfinite(wave).all():
        raise ReproductionError("resample input is non-finite")
    converted = np.asarray(soxr.resample(wave, SOURCE_RATE, TARGET_RATE, quality=QUALITY), dtype=np.float32)
    converted = np.ascontiguousarray(converted)
    if converted.size != soxr_hq_frame_count(int(wave.size)):
        raise ReproductionError(
            f"soxr HQ length {converted.size} does not match the recorded formula "
            f"for {wave.size} source frames"
        )
    if not np.isfinite(converted).all():
        raise ReproductionError("resampled waveform is non-finite")
    return converted


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


def _spec() -> dict:
    import soxr

    return {
        "tool": "soxr",
        "tool_version": soxr.__version__,
        "params": {
            "quality": QUALITY,
            "source_rate": SOURCE_RATE,
            "target_rate": TARGET_RATE,
            "output_dtype": "float32",
            "length": "source_frames_plus_one_div_3",
            "order": list(ORDER),
            "cache": "none",
        },
    }
