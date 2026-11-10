"""Magnitude and phase STFT used by MP-SENet training.

The formulas follow the official training code at commit 89932cfe
(``mag_pha_stft`` / ``mag_pha_istft``). This module does not import
``src/se/official``. The inference wrapper in ``src.se`` applies the same
transform inside ``enhance`` and is not used here, so training does not go
through ``inference_mode``.

MP-SENet uses the native centered STFT, matching the official backward. On
strictly deterministic CUDA it uses explicit reflection instead, since native
``reflection_pad1d`` backward has no deterministic implementation.
"""

from __future__ import annotations

import torch

from src.experiments.calc.errors import TrainingError


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


def centered_stft(
    waveform: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor,
    *,
    normalized: bool = False,
) -> torch.Tensor:
    """Hann or custom-window STFT matching ``center=True`` without ``reflection_pad1d``."""
    pad = n_fft // 2
    if int(waveform.shape[-1]) <= pad:
        waveform = torch.nn.functional.pad(waveform, (0, pad + 1 - int(waveform.shape[-1])))
    framed = _reflect_pad(waveform, pad)
    return torch.stft(
        framed,
        n_fft,
        hop_length=hop,
        win_length=int(window.shape[-1]),
        window=window,
        center=False,
        normalized=normalized,
        return_complex=True,
    )


def mag_pha_stft(
    waveform: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 1.0,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Build the Hann window on CPU. The same call on CUDA differs by about 1e-7,
    # which is enough to wrap a phase bin by 2π. The reference code does this.
    window = torch.hann_window(win_size, dtype=waveform.dtype).to(waveform.device)
    deterministic_cuda = waveform.is_cuda and torch.are_deterministic_algorithms_enabled()
    framed = _reflect_pad(waveform, n_fft // 2) if center and deterministic_cuda else waveform
    spectrum = torch.stft(
        framed,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center and not deterministic_cuda,
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


def semamba_mag_phase_stft(
    waveform: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 0.3,
    *,
    center: bool = True,
    addeps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SEMamba++ magnitude and phase.

    The dataset path uses ``abs`` and ``angle``. The consistency path adds
    ``1e-10`` inside the square root and both ``atan2`` arguments. Centering
    uses an index reflection so the values match ``center=True`` without the
    non-deterministic CUDA reflection backward. The Hann window is built on
    CPU and moved, matching the reference training code. A window created
    directly on CUDA differs by about 1e-7 and can flip phase bins by 2π.
    """
    window = torch.hann_window(win_size, dtype=waveform.dtype).to(waveform.device)
    framed = _reflect_pad(waveform, n_fft // 2) if center else waveform
    spectrum = torch.stft(
        framed,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        normalized=False,
        return_complex=True,
    )
    if addeps:
        epsilon = 1e-10
        magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + epsilon)
        phase = torch.atan2(spectrum.imag + epsilon, spectrum.real + epsilon)
    else:
        magnitude = spectrum.abs()
        phase = spectrum.angle()
    magnitude = magnitude.pow(compress_factor)
    compressed = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    return magnitude, phase, compressed


def semamba_mag_phase_istft(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 0.3,
    *,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse of ``semamba_mag_phase_stft`` using the same CPU-built Hann window."""
    magnitude = magnitude.pow(1.0 / compress_factor)
    spectrum = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    window = torch.hann_window(win_size, dtype=spectrum.real.dtype).to(spectrum.device)
    return torch.istft(
        spectrum,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        length=length,
    )


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
    window = torch.hann_window(win_size, dtype=spectrum.real.dtype).to(spectrum.device)
    return torch.istft(
        spectrum,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
    )
