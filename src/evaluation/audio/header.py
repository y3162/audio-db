"""Header probe for evaluation registration.

``probe_header`` is the only constructor of a ``VerifiedHeader``. Registration
reuses that object for the resolved path and still compares it with the
declared sample rate. A sample rate without this object is not verified.

WAV files use the stream reader with consistency checks. A data chunk that is
empty, not a whole number of frames, longer than the file, or followed by
bytes that are not chunks is rejected. Other suffixes still use ``torchaudio.info``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.sample import AudioSample

_TOKEN = object()
_ATTR = "_verified_headers"
_header_seconds = 0.0


@dataclass(frozen=True, slots=True)
class VerifiedHeader:
    path: str
    sample_rate: int
    channels: int
    frames: int
    token: object

    def __post_init__(self) -> None:
        if self.token is not _TOKEN:
            raise RuntimeError("call probe_header to verify an audio file")


def reset_header_seconds() -> None:
    global _header_seconds
    _header_seconds = 0.0


def header_seconds() -> float:
    return _header_seconds


def resolved_audio_path(path: Path) -> str:
    return os.path.normpath(path.resolve())


def probe_header(path: Path) -> VerifiedHeader:
    """Read rate, channels, and frames. Count the call.

    WAV is not accepted from an unverified declared rate. The stream reader
    must see a consistent data chunk. Other formats stay on torchaudio.
    """
    global _header_seconds
    started = time.perf_counter()
    try:
        sample_rate, frames, channels = _stream_info(path)
    finally:
        _header_seconds += time.perf_counter() - started
    return VerifiedHeader(
        path=resolved_audio_path(path),
        sample_rate=sample_rate,
        channels=channels,
        frames=frames,
        token=_TOKEN,
    )


def _stream_info(path: Path) -> tuple[int, int, int]:
    if path.suffix.lower() == ".wav":
        from src.data.audio import read_wav_stream_info

        return read_wav_stream_info(path, consistent=True)
    import torchaudio

    info = torchaudio.info(str(path))
    sample_rate = int(info.sample_rate)
    frames = int(info.num_frames)
    channels = int(info.num_channels)
    if frames < 1 or sample_rate < 1 or channels < 1:
        raise RuntimeError(f"audio has no frames: {path}")
    return sample_rate, frames, channels


def attach_verified_header(sample: AudioSample, header: VerifiedHeader) -> None:
    if not isinstance(header, VerifiedHeader) or header.token is not _TOKEN:
        raise RuntimeError("call probe_header to verify an audio file")
    book = getattr(sample, _ATTR, None)
    if book is None:
        book = {}
        setattr(sample, _ATTR, book)
    book[header.path] = header


def verified_header(sample: AudioSample, path: Path) -> VerifiedHeader | None:
    book = getattr(sample, _ATTR, None)
    if not isinstance(book, dict):
        return None
    try:
        found = book.get(resolved_audio_path(path))
    except OSError:
        return None
    if not isinstance(found, VerifiedHeader) or found.token is not _TOKEN:
        return None
    return found
