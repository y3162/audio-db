"""Spectral objectives used by the model-specific trainers.

These follow the published training losses (multi-resolution time-frequency
L1, compressed magnitude, log-mel L1, and least-squares GAN). They do not
import ``src/se/official``.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

from src.training.se.losses import anti_wrapping, phase_losses
from src.training.se.stft import _reflect_pad, centered_stft


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
    # ESPnet computes a separate product per loss branch. Reusing one product
    # changes float32 backward accumulation and can change Adam updates for
    # near-zero gradients even when the loss and forward values are identical.
    time_loss = (predicted * scale - reference).abs().sum(dim=-1)
    spectral = torch.zeros_like(time_loss)
    for window in windows:
        hop = window // 2
        spectral = spectral + _rect_magnitude_l1(reference, predicted * scale, window, hop)
    per_item = time_loss * time_domain_weight + (1.0 - time_domain_weight) * spectral / len(windows)
    return per_item.mean()


def _rect_magnitude_l1(reference: torch.Tensor, estimate: torch.Tensor, window: int, hop: int) -> torch.Tensor:
    ref_mag = _rect_stft(reference, window, hop).abs()
    est_mag = _rect_stft(estimate, window, hop).abs()
    return (est_mag - ref_mag).abs().sum(dim=(-2, -1))


def _rect_stft(waveform: torch.Tensor, window: int, hop: int) -> torch.Tensor:
    """Rectangular STFT matching ESPnet ``Stft(window=None)``.

    ``torch.stft(..., window=None)`` and an explicit ``torch.ones`` window are
    the same on CPU. On CUDA they are not: a 256-point transform on
    Quadro RTX 5000 differed by about 1e-5, and that alone moved this
    multi-resolution loss by one float32 ulp (0.001953125) after the waveform
    itself matched. The pad stays explicit so the backward does not go through
    ``reflection_pad1d``.
    """
    pad = window // 2
    if int(waveform.shape[-1]) <= pad:
        waveform = F.pad(waveform, (0, pad + 1 - int(waveform.shape[-1])))
    framed = _reflect_pad(waveform, pad)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A window was not provided",
            category=UserWarning,
        )
        return torch.stft(
            framed,
            window,
            hop_length=hop,
            win_length=window,
            window=None,
            center=False,
            normalized=False,
            return_complex=True,
        )


def matrix_phase_losses(reference: torch.Tensor, generated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SEMamba++ phase terms.

    Group delay and angular frequency use a first-difference matrix with a
    final ``-1`` on the diagonal. That is wider than ``torch.diff`` by one bin.
    """
    frequency = _difference_matrix(int(reference.shape[-2]), reference)
    time = _difference_matrix(int(reference.shape[-1]), reference)
    group_reference = torch.matmul(reference.transpose(1, 2), frequency)
    group_generated = torch.matmul(generated.transpose(1, 2), frequency)
    angular_reference = torch.matmul(reference, time)
    angular_generated = torch.matmul(generated, time)
    return (
        anti_wrapping(reference - generated).mean(),
        anti_wrapping(group_reference - group_generated).mean(),
        anti_wrapping(angular_reference - angular_generated).mean(),
    )


def _difference_matrix(size: int, like: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(size, size, device=like.device, dtype=like.dtype)
    return torch.triu(ones, diagonal=1) - torch.triu(ones, diagonal=2) - torch.eye(size, device=like.device, dtype=like.dtype)


def compressed_complex_losses(
    clean_mag: torch.Tensor,
    generated_mag: torch.Tensor,
    clean_phase: torch.Tensor,
    generated_phase: torch.Tensor,
    clean_complex: torch.Tensor,
    generated_complex: torch.Tensor,
    recomputed_complex: torch.Tensor,
    *,
    phase_mode: str = "diff",
) -> dict[str, torch.Tensor]:
    """Magnitude MSE, anti-wrapped phase, and the doubled complex terms."""
    magnitude = F.mse_loss(clean_mag, generated_mag)
    phase_fn = matrix_phase_losses if phase_mode == "matrix" else phase_losses
    instantaneous, group_delay, angular = phase_fn(clean_phase, generated_phase)
    phase = instantaneous + group_delay + angular
    complex_loss = F.mse_loss(clean_complex, generated_complex) * 2
    consistency = F.mse_loss(generated_complex, recomputed_complex) * 2
    return {"magnitude": magnitude, "phase": phase, "complex": complex_loss, "consistency": consistency}


def least_squares_discriminator(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return torch.mean((1 - real_score).square()) + torch.mean(fake_score.square())


def least_squares_generator(fake_score: torch.Tensor) -> torch.Tensor:
    return torch.mean((1 - fake_score).square())


def feature_matching(
    real_features: list[torch.Tensor],
    fake_features: list[torch.Tensor],
    scale: float = 2.0,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sum or mean of per-map L1 distances, then ``scale``.

    SEMamba++ sums the map means and multiplies by 2. Averaging would change
    the loss coefficient by the number of maps.
    """
    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("feature lists must be non-empty and the same length")
    total = real_features[0].new_zeros(())
    for real, fake in zip(real_features, fake_features, strict=True):
        total = total + F.l1_loss(fake, real.detach())
    if reduction == "sum":
        return total * scale
    if reduction == "mean":
        return total / len(real_features) * scale
    raise ValueError(f"unsupported feature-matching reduction {reduction}")


def summed_slaney_log_mel(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    *,
    n_mels: tuple[int, ...] = (40, 80, 160, 320),
    windows: tuple[int, ...] = (256, 512, 1024, 2048),
    clamp: float = 1e-5,
) -> torch.Tensor:
    """Sum of mean L1 distances between log10 Slaney mel spectrograms.

    This is the multi-scale mel term used by SEMamba++: four windows, hop
    equal to one quarter of the window, Hann, centered STFT, log10, and no
    extra magnitude term. The four scales are added, not averaged.
    """
    if len(n_mels) != len(windows):
        raise ValueError("n_mels and windows must have the same length")
    if reference.ndim == 3:
        reference = reference.squeeze(1)
        estimate = estimate.squeeze(1)
    total = reference.new_zeros(())
    for bands, window in zip(n_mels, windows, strict=True):
        hop = window // 4
        ref = _slaney_log_mel(reference, sample_rate, bands, window, hop, clamp)
        est = _slaney_log_mel(estimate, sample_rate, bands, window, hop, clamp)
        frames = min(ref.shape[-1], est.shape[-1])
        total = total + F.l1_loss(ref[..., :frames], est[..., :frames])
    return total


def _slaney_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    window: int,
    hop: int,
    clamp: float,
) -> torch.Tensor:
    spec = centered_stft(
        waveform,
        window,
        hop,
        torch.hann_window(window, device=waveform.device, dtype=waveform.dtype),
    ).abs()
    basis = _slaney_basis(sample_rate, window, n_mels, spec.device, spec.dtype)
    mel = torch.matmul(basis, spec.clamp_min(0))
    return torch.log10(mel.clamp_min(clamp))


def _slaney_basis(sample_rate: int, n_fft: int, n_mels: int, device, dtype) -> torch.Tensor:
    bins = n_fft // 2 + 1
    fft_hz = torch.linspace(0, sample_rate / 2, bins, device=device, dtype=dtype)
    mel_min = _hz_to_mel_slaney(fft_hz.new_tensor(0.0))
    mel_max = _hz_to_mel_slaney(fft_hz.new_tensor(sample_rate / 2))
    points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device, dtype=dtype)
    hz = _mel_to_hz_slaney(points)
    weights = torch.zeros(n_mels, bins, device=device, dtype=dtype)
    gaps = hz[1:] - hz[:-1]
    for index in range(n_mels):
        lower = (fft_hz - hz[index]) / gaps[index].clamp_min(1e-12)
        upper = (hz[index + 2] - fft_hz) / gaps[index + 1].clamp_min(1e-12)
        weights[index] = torch.clamp(torch.minimum(lower, upper), min=0)
    scale = 2.0 / (hz[2:] - hz[:n_mels]).clamp_min(1e-12)
    return weights * scale.unsqueeze(1)


def _hz_to_mel_slaney(hz: torch.Tensor) -> torch.Tensor:
    spacing = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / spacing
    logstep = math.log(6.4) / 27.0
    linear = hz / spacing
    logarithmic = min_log_mel + torch.log(hz.clamp_min(min_log_hz) / min_log_hz) / logstep
    return torch.where(hz >= min_log_hz, logarithmic, linear)


def _mel_to_hz_slaney(mel: torch.Tensor) -> torch.Tensor:
    spacing = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / spacing
    logstep = math.log(6.4) / 27.0
    linear = mel * spacing
    logarithmic = min_log_hz * torch.exp(logstep * (mel - min_log_mel))
    return torch.where(mel >= min_log_mel, logarithmic, linear)


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
