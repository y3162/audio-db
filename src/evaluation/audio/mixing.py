"""Mix clean and noise at a requested SNR.

Protocol ``rms_v1`` defines SNR from mean-square power of the two components
after resampling and length alignment:

    SNR_dB = 10 * log10(mean(clean^2) / mean(scaled_noise^2))

Every sample, including silence inside the utterance, stays in the mean.
There is no voice-activity detector. A clean signal whose power is below
1e-12 is returned unchanged and marked silent. Noise with no energy is an
error, because a finite SNR is then undefined.

A common gain may be applied afterwards so the mixture peak is at most 0.99.
That gain does not change this SNR. Hard clipping can, and is recorded
separately. The achieved SNR is measured before hard clipping.

This is not the VoiceBank-DEMAND recipe and not the URGENT mixture protocol.
Active speech level (ITU-T P.56) is not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio.functional as AF

PROTOCOL_RMS_V1 = "rms_v1"
POWER_EPS = 1e-12
PEAK_LIMIT = 0.99


@dataclass(frozen=True)
class MixResult:
    noisy: torch.Tensor
    clean: torch.Tensor
    noise: torch.Tensor
    sample_rate: int
    snr_db_requested: float
    snr_db_achieved: float | None
    noise_start: int
    scale: float
    protocol: str
    peak_scaled: bool
    hard_clipped: bool
    clean_silent: bool


def signal_power(waveform: torch.Tensor) -> float:
    if waveform.numel() == 0:
        return 0.0
    data = waveform.detach().to(torch.float64)
    return float(data.square().mean())


def rms_snr_db(clean: torch.Tensor, noise: torch.Tensor) -> float:
    clean_power = signal_power(clean)
    noise_power = signal_power(noise)
    if clean_power < POWER_EPS or noise_power < POWER_EPS:
        raise ValueError("SNR is undefined when clean or noise power is zero")
    return 10.0 * math.log10(clean_power / noise_power)


def mix_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sample_rate: int,
    noise_sample_rate: int,
    snr_db: float,
    seed: int,
    *,
    protocol: str = PROTOCOL_RMS_V1,
    prevent_clip: bool = False,
    hard_clip: bool = False,
    noise_start: int | None = None,
) -> MixResult:
    if protocol != PROTOCOL_RMS_V1:
        raise ValueError(f"unknown mixing protocol: {protocol}")
    if isinstance(snr_db, bool) or not isinstance(snr_db, (int, float)) or not math.isfinite(float(snr_db)):
        raise ValueError(f"invalid SNR: {snr_db!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if sample_rate <= 0 or noise_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    clean_wave = _finite_mono(clean, "clean")
    noise_wave = _finite_mono(noise, "noise")
    snr_value = float(snr_db)
    if noise_sample_rate != sample_rate:
        noise_wave = AF.resample(noise_wave, noise_sample_rate, sample_rate)
        if noise_wave.numel() == 0:
            raise ValueError("noise is empty after resampling")
    aligned, start = align_noise(noise_wave, clean_wave.numel(), seed, start=noise_start)
    clean_power = signal_power(clean_wave)
    noise_power = signal_power(aligned)
    if clean_power < POWER_EPS:
        zeros = torch.zeros_like(clean_wave)
        return MixResult(
            noisy=clean_wave.clone(),
            clean=clean_wave.clone(),
            noise=zeros,
            sample_rate=sample_rate,
            snr_db_requested=snr_value,
            snr_db_achieved=None,
            noise_start=start,
            scale=0.0,
            protocol=protocol,
            peak_scaled=False,
            hard_clipped=False,
            clean_silent=True,
        )
    if noise_power < POWER_EPS:
        raise ValueError("noise has zero energy")
    scale = math.sqrt(clean_power / noise_power) / (10.0 ** (snr_value / 20.0))
    scaled_noise = aligned * scale
    noisy = clean_wave + scaled_noise
    clean_ref = clean_wave
    peak_scaled = False
    if prevent_clip:
        peak = float(noisy.detach().abs().max())
        if peak > PEAK_LIMIT and peak > 0.0:
            factor = PEAK_LIMIT / peak
            noisy = noisy * factor
            scaled_noise = scaled_noise * factor
            clean_ref = clean_wave * factor
            peak_scaled = True
    achieved = rms_snr_db(clean_ref, scaled_noise)
    hard_clipped = False
    if hard_clip:
        clipped = noisy.clamp(-1.0, 1.0)
        hard_clipped = bool(torch.any(clipped != noisy).item())
        noisy = clipped
    return MixResult(
        noisy=noisy.contiguous(),
        clean=clean_ref.contiguous(),
        noise=scaled_noise.contiguous(),
        sample_rate=sample_rate,
        snr_db_requested=snr_value,
        snr_db_achieved=achieved,
        noise_start=start,
        scale=scale,
        protocol=protocol,
        peak_scaled=peak_scaled,
        hard_clipped=hard_clipped,
        clean_silent=False,
    )


def crop_start(noise_length: int, target_length: int, seed: int) -> int:
    """Seeded crop into noise after the same tiling rule as ``align_noise``."""
    if target_length <= 0:
        raise ValueError("target length must be positive")
    if noise_length <= 0:
        raise ValueError("noise length must be positive")
    tiled = noise_length
    if tiled < target_length:
        repeats = (target_length + tiled - 1) // tiled
        tiled *= repeats
    max_start = tiled - target_length
    if max_start == 0:
        return 0
    generator = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFF))
    return int(generator.integers(0, max_start + 1))


def align_noise(
    noise: torch.Tensor,
    length: int,
    seed: int,
    *,
    start: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Tile when noise is short, then crop from a seeded or stored start."""
    if length <= 0:
        raise ValueError("target length must be positive")
    original = int(noise.numel())
    if original <= 0:
        raise ValueError("noise is empty")
    if start is None:
        start = crop_start(original, length, seed)
    elif isinstance(start, bool) or not isinstance(start, int):
        raise ValueError("noise_start must be an integer")
    if original < length:
        repeats = (length + original - 1) // original
        noise = noise.repeat(repeats)
    max_start = int(noise.numel() - length)
    if start < 0 or start > max_start:
        raise ValueError(f"noise_start {start} is outside 0..{max_start}")
    return noise[start : start + length].contiguous(), start


def _finite_mono(waveform: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional waveform")
    wave = waveform.detach().cpu().to(torch.float32).contiguous()
    if not torch.isfinite(wave).all():
        raise ValueError(f"{label} contains non-finite samples")
    return wave
