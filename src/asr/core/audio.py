"""Mono waveform checks shared by ASR backends."""

from __future__ import annotations

import torch


def prepare_waveform(waveform: torch.Tensor, sample_rate: int, target_rate: int = 16000) -> torch.Tensor:
    """Return a finite float32 mono waveform at ``target_rate`` on CPU."""
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
        raise ValueError("waveform must be a one-dimensional torch.Tensor")
    if not waveform.is_floating_point():
        raise ValueError("waveform must contain floating-point samples")
    if waveform.numel() == 0:
        raise ValueError("waveform is empty")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform contains non-finite samples")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if isinstance(target_rate, bool) or not isinstance(target_rate, int) or target_rate <= 0:
        raise ValueError("target_rate must be a positive integer")
    audio = waveform.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if sample_rate == target_rate:
        return audio
    import torchaudio.functional as AF

    resampled = AF.resample(audio, sample_rate, target_rate)
    if resampled.numel() == 0:
        raise ValueError("waveform is empty after resampling")
    return resampled.contiguous()
