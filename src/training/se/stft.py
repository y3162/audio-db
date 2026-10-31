"""Magnitude and phase STFT used by MP-SENet training.

The formulas follow the official training code at commit 89932cfe
(``mag_pha_stft`` / ``mag_pha_istft``). This module does not import
``src/se/official``. The inference wrapper in ``src.se`` applies the same
transform inside ``enhance`` and is not used here, so training does not go
through ``inference_mode``.

Centered STFT is padded with an index reflection before ``center=False``.
That spectrum matches ``torch.stft(..., center=True, pad_mode="reflect")``.
The CUDA backward of ``reflection_pad1d`` has no deterministic implementation,
so the pad is not left to that kernel.
"""

from __future__ import annotations

import torch

from src.training.errors import TrainingError


def _reflect_pad(waveform: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return waveform
    length = int(waveform.shape[-1])
    if length <= pad:
        raise TrainingError(
            f"waveform length {length} cannot be reflect-padded by {pad} samples"
        )
    left = waveform[..., 1 : pad + 1].flip(-1)
    right = waveform[..., -pad - 1 : -1].flip(-1)
    return torch.cat((left, waveform, right), dim=-1)


def mag_pha_stft(
    waveform: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 1.0,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.hann_window(win_size, device=waveform.device, dtype=waveform.dtype)
    framed = _reflect_pad(waveform, n_fft // 2) if center else waveform
    spectrum = torch.stft(
        framed,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    real = torch.view_as_real(spectrum)
    magnitude = torch.sqrt(real.pow(2).sum(-1) + 1e-9)
    phase = torch.atan2(real[..., 1] + 1e-10, real[..., 0] + 1e-5)
    magnitude = magnitude.pow(compress_factor)
    compressed = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    return magnitude, phase, compressed


def mag_pha_istft(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 1.0,
    center: bool = True,
) -> torch.Tensor:
    magnitude = magnitude.pow(1.0 / compress_factor)
    spectrum = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    window = torch.hann_window(win_size, device=spectrum.device, dtype=spectrum.real.dtype)
    return torch.istft(
        spectrum,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
    )
