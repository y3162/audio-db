"""16 kHz PCM_16 of 48 kHz VoiceBank audio, computed when a batch is rendered.

The converter is soxr HQ followed by the same PCM_16 rounding soundfile uses
when it writes and reads a wav. Nothing is written, and nothing is cached.
The sample count for 48 kHz to 16 kHz is ``(source_frames + 1) // 3``. That
matches soxr 1.1.0 HQ on every registered VoiceBank file.
"""

from __future__ import annotations

import io
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
    "clip_to_unit",
    "pcm16_roundtrip",
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
    """Read one 48 kHz wav and return 16 kHz PCM_16 as float32."""
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if audio.ndim != 1:
        raise ReproductionError(f"{path} is not mono")
    return resample_pcm16(audio, int(rate))


def resample_pcm16(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """soxr HQ, clip to [-1, 1], then a PCM_16 write/read with no file left behind."""
    import soundfile as sf
    import soxr

    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim != 1:
        raise ReproductionError("resample input must be mono")
    if int(source_rate) != SOURCE_RATE:
        raise ReproductionError(f"sample rate is {source_rate}, expected {SOURCE_RATE}")
    if wave.size < 1:
        raise ReproductionError("resample input is empty")
    converted = np.asarray(soxr.resample(wave, SOURCE_RATE, TARGET_RATE, quality=QUALITY), dtype=np.float32)
    converted = np.ascontiguousarray(np.clip(converted, -1.0, 1.0))
    if converted.size != soxr_hq_frame_count(int(wave.size)):
        raise ReproductionError(
            f"soxr HQ length {converted.size} does not match the recorded formula "
            f"for {wave.size} source frames"
        )
    if not np.isfinite(converted).all():
        raise ReproductionError("resampled waveform is non-finite")
    buffer = io.BytesIO()
    sf.write(buffer, converted, TARGET_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    decoded, rate = sf.read(buffer, dtype="float32", always_2d=False)
    if int(rate) != TARGET_RATE:
        raise ReproductionError(f"PCM_16 round trip returned {rate} Hz")
    decoded = np.ascontiguousarray(decoded, dtype=np.float32)
    if decoded.shape != converted.shape:
        raise ReproductionError("PCM_16 round trip changed the sample count")
    if not np.isfinite(decoded).all():
        raise ReproductionError("PCM_16 round trip is non-finite")
    return decoded


def resample_spec() -> dict:
    """Closed rule object stored on an on-demand fixed-pair protocol."""
    return {
        "tool": "soxr",
        "quality": QUALITY,
        "source_rate": SOURCE_RATE,
        "target_rate": TARGET_RATE,
        "quantization": "pcm16",
        "length": "source_frames_plus_one_div_3",
        "order": list(ORDER),
    }


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
            "length": "source_frames_plus_one_div_3",
            "order": list(ORDER),
            "cache": "none",
        },
    }
