"""MossFormerGAN generator terms. The comparison task keeps its own copy.

Coefficients are ri 0.1, magnitude 0.9, time 0.2, gan 0.05. The time term
reconstructs from the clean real part and the predicted imaginary part.
``metric_targets`` is looked up on this module so a PESQ stub can replace it
without touching the comparison task.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.calc.losses import metric_targets
from src.experiments.calc.stft import centered_stft

_RI = 0.1
_MAG = 0.9
_TIME = 0.2
_GAN = 0.05
OFFICIAL_ADAMW_WEIGHT_DECAY = 0.01

def _stft(waveform: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    # The released solver constructs the window on CPU before moving it.
    window = torch.hamming_window(win, periodic=False, dtype=waveform.dtype).to(waveform.device)
    # Same frames as center=True. The pad is an index reflection so CUDA
    # deterministic mode does not need reflection_pad1d_backward.
    return torch.view_as_real(centered_stft(waveform, n_fft, hop, window))

def _istft(spectrum: torch.Tensor, n_fft: int, hop: int, win: int, length: int) -> torch.Tensor:
    window = torch.hamming_window(win, periodic=False, dtype=spectrum.dtype).to(spectrum.device)
    return torch.istft(torch.view_as_complex(spectrum), n_fft, hop, win, window=window, center=False)

def _compress(spectrum: torch.Tensor) -> torch.Tensor:
    value = torch.complex(spectrum[..., 0], spectrum[..., 1])
    magnitude = value.abs().pow(0.3)
    return torch.stack((magnitude * torch.cos(value.angle()), magnitude * torch.sin(value.angle())), 1)

def _uncompress(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    value = torch.complex(real, imag)
    magnitude = value.abs().pow(1.0 / 0.3)
    phase = value.angle()
    return torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), -1)


def _generator_parts(self, module, discriminator, clean: torch.Tensor, noisy: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unweighted generator terms for ``loss_mossformergan_se_16k``.

    The time term reconstructs from the clean real part and the predicted
    imaginary part. The GAN term asks the discriminator to score the fake
    pair as 1. The weighted sum is applied by the caller.
    """
    scale = torch.sqrt(noisy.shape[-1] / (noisy.square().sum(dim=-1) + 1e-6)).unsqueeze(-1)
    normalized = noisy * scale
    real, imag, pred_mag, label_real, _label_imag, label_mag = self._forward(
        module, normalized, clean * scale
    )
    ones = torch.ones(clean.shape[0], device=clean.device)
    gan = F.mse_loss(discriminator(label_mag, pred_mag).flatten(), ones)
    loss_mag = F.mse_loss(pred_mag, label_mag)
    loss_ri = F.mse_loss(real, label_real) + F.mse_loss(imag, _label_imag)
    audio = _istft(_uncompress(label_real, imag).squeeze(1), self.n_fft, self.hop, self.win, clean.shape[-1])
    teacher = clean * scale
    length = min(audio.shape[-1], teacher.shape[-1])
    time_loss = (audio[..., :length] - teacher[..., :length]).abs().mean()
    return {
        "ri": loss_ri,
        "magnitude": loss_mag,
        "time": time_loss,
        "gan": gan,
        "audio": audio[..., :length],
        "teacher": teacher[..., :length],
        "label_mag": label_mag,
        "pred_mag": pred_mag,
        "predicted_real": real,
        "predicted_imag": imag,
        "label_real": label_real,
    }

def _generator_terms(self, module, discriminator, clean: torch.Tensor, noisy: torch.Tensor) -> dict[str, torch.Tensor]:
    """Generator objective. Coefficients stay in this function."""

    parts = self._generator_parts(module, discriminator, clean, noisy)
    generator_loss = _RI * parts["ri"] + _MAG * parts["magnitude"] + _TIME * parts["time"] + _GAN * parts["gan"]
    return {**parts, "generator": generator_loss}

def _forward(self, module, noisy: torch.Tensor, clean: torch.Tensor):
    noisy_spec = _compress(_stft(noisy, self.n_fft, self.hop, self.win)).permute(0, 1, 3, 2)
    produced = module(noisy_spec)
    real = produced["predicted_real"]
    imag = produced["predicted_imag"]
    real = real.permute(0, 1, 3, 2)
    imag = imag.permute(0, 1, 3, 2)
    pred_mag = torch.sqrt(real ** 2 + imag ** 2)
    label_spec = _compress(_stft(clean, self.n_fft, self.hop, self.win))
    label_real = label_spec[:, 0, :, :].unsqueeze(1)
    label_imag = label_spec[:, 1, :, :].unsqueeze(1)
    label_mag = torch.sqrt(label_real ** 2 + label_imag ** 2)
    return real, imag, pred_mag, label_real, label_imag, label_mag


generator_parts = _generator_parts
generator_terms = _generator_terms
forward_spectra = _forward
