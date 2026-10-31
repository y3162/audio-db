"""Waveform loading and saving.

Decoding uses torchaudio. Stream metadata in src.data.audio is not duplicated.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchaudio


class AudioNotFound(FileNotFoundError):
    """The requested audio path does not exist."""


def load_waveform(path: str | Path, max_seconds: float | None = None) -> tuple[torch.Tensor, int]:
    """Load mono float32 audio. max_seconds keeps a prefix of a long file."""
    waveform, sample_rate, _span = load_waveform_span(path, max_seconds)
    return waveform, sample_rate


def load_waveform_span(
    path: str | Path,
    max_seconds: float | None = None,
) -> tuple[torch.Tensor, int, dict[str, int | float | bool | None]]:
    """Load audio and record whether the prefix is shorter than the file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise AudioNotFound(file_path)
    source_frames, source_rate = _source_frames(file_path)
    if max_seconds is None or source_rate is None:
        waveform, sample_rate = torchaudio.load(str(file_path))
    else:
        max_frames = max(1, int(max_seconds * source_rate))
        waveform, sample_rate = torchaudio.load(
            str(file_path),
            frame_offset=0,
            num_frames=max_frames,
        )
    mono = _mono(waveform, file_path)
    # The crop limit uses the file header rate. used_frames stay on the decoded rate.
    limit_rate = int(source_rate if source_rate is not None else sample_rate)
    span = audio_span(source_frames, int(mono.numel()), limit_rate, max_seconds)
    span["sample_rate"] = int(sample_rate)
    return mono, int(sample_rate), span


def save_waveform(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Atomically write 32-bit float WAV."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a .wav suffix. ffmpeg selects the muxer from the extension, so a
    # trailing ".tmp.<pid>" is rejected.
    temporary = file_path.with_name(f".{file_path.stem}.{os.getpid()}.wav")
    audio = waveform.detach().cpu().to(torch.float32).reshape(1, -1).contiguous()
    torchaudio.save(
        str(temporary),
        audio,
        sample_rate,
        encoding="PCM_F",
        bits_per_sample=32,
    )
    os.replace(temporary, file_path)


def as_mono(waveform: torch.Tensor, sample_rate: int, max_seconds: float | None = None) -> tuple[torch.Tensor, int]:
    wave, rate, _span = as_mono_span(waveform, sample_rate, max_seconds)
    return wave, rate


def as_mono_span(
    waveform: torch.Tensor,
    sample_rate: int,
    max_seconds: float | None = None,
) -> tuple[torch.Tensor, int, dict[str, int | float | bool | None]]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional or (channels, time)")
    wave = waveform.detach().cpu().to(torch.float32).contiguous()
    source_frames = int(wave.numel())
    if max_seconds is not None:
        wave = wave[: max(1, int(max_seconds * sample_rate))].contiguous()
    span = audio_span(source_frames, int(wave.numel()), int(sample_rate), max_seconds)
    return wave, int(sample_rate), span


def audio_span(
    source_frames: int | None,
    used_frames: int,
    sample_rate: int,
    max_seconds: float | None,
) -> dict[str, int | float | bool | None]:
    """Say whether the used prefix is shorter than the source utterance.

    A configured max_seconds is not itself proof of a crop. The source must
    be longer than that limit. Unknown source length with a limit is left
    unresolved so WER can refuse the pair.
    """
    truncated: bool | None
    if max_seconds is None:
        truncated = False if used_frames >= 1 else None
    elif source_frames is None or source_frames < 1 or sample_rate < 1:
        truncated = None
    else:
        limit = max(1, int(float(max_seconds) * int(sample_rate)))
        truncated = int(source_frames) > limit
    return {
        "source_frames": source_frames,
        "used_frames": used_frames,
        "sample_rate": sample_rate,
        "max_seconds": None if max_seconds is None else float(max_seconds),
        "truncated": truncated,
    }


def _source_frames(path: Path) -> tuple[int | None, int | None]:
    try:
        info = torchaudio.info(str(path))
    except Exception:
        return None, None
    frames = int(info.num_frames)
    rate = int(info.sample_rate)
    if frames < 1 or rate < 1:
        return None, None
    return frames, rate


def _mono(waveform: torch.Tensor, file_path: Path) -> torch.Tensor:
    if waveform.ndim != 2 or waveform.numel() == 0:
        raise ValueError(f"audio is empty or not shaped (channels, time): {file_path}")
    if waveform.shape[0] == 1:
        mono = waveform[0]
    else:
        mono = waveform.mean(dim=0)
    return mono.to(torch.float32).contiguous()
