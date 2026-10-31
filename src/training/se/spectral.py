"""Spectral objectives used by the model-specific trainers.

These follow the published training losses (multi-resolution time-frequency
L1, compressed magnitude, log-mel L1, and least-squares GAN). They do not
import ``src/se/official``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.training.se.losses import phase_losses


def multi_resolution_l1(
    target: torch.Tensor,
    estimate: torch.Tensor,
    windows: tuple[int, ...] = (256, 512, 768, 1024),
    *,
    time_domain_weight: float = 0.5,
    normalize_variance: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """ESPnet ``mr_l1_tfd`` with reduction ``sum`` inside each utterance.

    The returned value is the mean over the batch. Each STFT uses a
    rectangular window (``window=None``) and a hop of half the window.
    """
    if target.shape != estimate.shape:
        raise ValueError(f"shape mismatch {tuple(target.shape)} vs {tuple(estimate.shape)}")
    reference = target.float()
    predicted = estimate.float()
    if normalize_variance:
        reference = reference / torch.sqrt(torch.var(reference, dim=-1, unbiased=False, keepdim=True) + eps)
        predicted = predicted / torch.sqrt(torch.var(predicted, dim=-1, unbiased=False, keepdim=True) + eps)
    scale = torch.sum(predicted * reference, dim=-1, keepdim=True) / (
        torch.sum(predicted.square(), dim=-1, keepdim=True) + eps
    )
    aligned = predicted * scale
    time_loss = (aligned - reference).abs().sum(dim=-1)
    spectral = torch.zeros_like(time_loss)
    for window in windows:
        hop = window // 2
        spectral = spectral + _rect_magnitude_l1(reference, aligned, window, hop)
    spectral = spectral / len(windows)
    per_item = time_loss * time_domain_weight + (1.0 - time_domain_weight) * spectral
    return per_item.mean()


def _rect_magnitude_l1(reference: torch.Tensor, estimate: torch.Tensor, window: int, hop: int) -> torch.Tensor:
    ref_mag = _rect_stft(reference, window, hop).abs()
    est_mag = _rect_stft(estimate, window, hop).abs()
    return (est_mag - ref_mag).abs().sum(dim=(-2, -1))


def _rect_stft(waveform: torch.Tensor, window: int, hop: int) -> torch.Tensor:
    rectangular = torch.ones(window, device=waveform.device, dtype=waveform.dtype)
    return torch.stft(
        waveform,
        n_fft=window,
        hop_length=hop,
        win_length=window,
        window=rectangular,
        center=True,
        return_complex=True,
    )


def log_mel_l1(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    *,
    n_mels: tuple[int, ...] = (40, 80, 160, 320),
    windows: tuple[int, ...] = (256, 512, 1024, 2048),
    clamp: float = 1e-5,
) -> torch.Tensor:
    """Mean L1 of log10 mel magnitudes. Hop is one quarter of each window."""
    if len(n_mels) != len(windows):
        raise ValueError("n_mels and windows must have the same length")
    total = reference.new_zeros(())
    for bands, window in zip(n_mels, windows, strict=True):
        hop = max(window // 4, 1)
        ref = _log_mel(reference, sample_rate, bands, window, hop, clamp)
        est = _log_mel(estimate, sample_rate, bands, window, hop, clamp)
        frames = min(ref.shape[-1], est.shape[-1])
        total = total + F.l1_loss(ref[..., :frames], est[..., :frames])
    return total / len(windows)


def _log_mel(waveform: torch.Tensor, sample_rate: int, n_mels: int, window: int, hop: int, clamp: float) -> torch.Tensor:
    padded = waveform
    if padded.shape[-1] < window:
        padded = F.pad(padded, (0, window - padded.shape[-1]))
    spec = torch.stft(
        padded,
        n_fft=window,
        hop_length=hop,
        win_length=window,
        window=torch.hann_window(window, device=padded.device, dtype=padded.dtype),
        center=True,
        return_complex=True,
    ).abs()
    mel_basis = _mel_basis(sample_rate, window, n_mels, spec.device, spec.dtype)
    mel = torch.matmul(mel_basis, spec.clamp_min(0))
    return torch.log10(mel.clamp_min(clamp))


def _mel_basis(sample_rate: int, n_fft: int, n_mels: int, device, dtype) -> torch.Tensor:
    bins = n_fft // 2 + 1
    low = _hz_to_mel(0.0)
    high = _hz_to_mel(sample_rate / 2)
    points = torch.linspace(low, high, n_mels + 2, device=device, dtype=dtype)
    hz = _mel_to_hz(points)
    fft_hz = torch.linspace(0, sample_rate / 2, bins, device=device, dtype=dtype)
    basis = torch.zeros(n_mels, bins, device=device, dtype=dtype)
    for index in range(n_mels):
        left, center, right = hz[index], hz[index + 1], hz[index + 2]
        rising = (fft_hz - left) / (center - left).clamp_min(1e-6)
        falling = (right - fft_hz) / (right - center).clamp_min(1e-6)
        basis[index] = torch.clamp(torch.minimum(rising, falling), min=0)
    return basis


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def compressed_complex_losses(
    clean_mag: torch.Tensor,
    generated_mag: torch.Tensor,
    clean_phase: torch.Tensor,
    generated_phase: torch.Tensor,
    clean_complex: torch.Tensor,
    generated_complex: torch.Tensor,
    recomputed_complex: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Magnitude MSE, anti-wrapped phase, and the doubled complex terms."""
    magnitude = F.mse_loss(clean_mag, generated_mag)
    instantaneous, group_delay, angular = phase_losses(clean_phase, generated_phase)
    phase = instantaneous + group_delay + angular
    complex_loss = F.mse_loss(clean_complex, generated_complex) * 2
    consistency = F.mse_loss(generated_complex, recomputed_complex) * 2
    return {"magnitude": magnitude, "phase": phase, "complex": complex_loss, "consistency": consistency}


def least_squares_discriminator(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return torch.mean((1 - real_score).square()) + torch.mean(fake_score.square())


def least_squares_generator(fake_score: torch.Tensor) -> torch.Tensor:
    return torch.mean((1 - fake_score).square())


def feature_matching(real_features: list[torch.Tensor], fake_features: list[torch.Tensor], scale: float = 2.0) -> torch.Tensor:
    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("feature lists must be non-empty and the same length")
    total = real_features[0].new_zeros(())
    for real, fake in zip(real_features, fake_features, strict=True):
        total = total + F.l1_loss(fake, real.detach())
    return total / len(real_features) * scale


def si_sdr_loss(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative scale-invariant SDR, averaged over the batch."""
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    target_energy = torch.sum(reference.square(), dim=-1, keepdim=True).clamp_min(eps)
    projection = dot * reference / target_energy
    noise = estimate - projection
    ratio = torch.sum(projection.square(), dim=-1) / torch.sum(noise.square(), dim=-1).clamp_min(eps)
    return -10.0 * torch.log10(ratio.clamp_min(eps)).mean()


def iam_mask(clean: torch.Tensor, noisy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return clean.abs() / noisy.abs().clamp_min(eps)
