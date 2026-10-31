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
    file_path = Path(path)
    if not file_path.is_file():
        raise AudioNotFound(file_path)
    if max_seconds is None:
        waveform, sample_rate = torchaudio.load(str(file_path))
    else:
        info = torchaudio.info(str(file_path))
        max_frames = max(1, int(max_seconds * info.sample_rate))
        waveform, sample_rate = torchaudio.load(
            str(file_path),
            frame_offset=0,
            num_frames=max_frames,
        )
    if waveform.ndim != 2 or waveform.numel() == 0:
        raise ValueError(f"audio is empty or not shaped (channels, time): {file_path}")
    if waveform.shape[0] == 1:
        mono = waveform[0]
    else:
        mono = waveform.mean(dim=0)
    mono = mono.to(torch.float32).contiguous()
    return mono, int(sample_rate)


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
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional or (channels, time)")
    wave = waveform.detach().cpu().to(torch.float32).contiguous()
    if max_seconds is not None:
        wave = wave[: max(1, int(max_seconds * sample_rate))].contiguous()
    return wave, int(sample_rate)
