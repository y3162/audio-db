"""SEMamba++ discriminators.

Independent implementation of the published training discriminators:

- multi-scale sub-band CQT: hops ``(512, 256, 256)``, 8 octaves, bins per
  octave ``(12, 24, 36)``, ``nnAudio`` ``CQT2010v2`` at twice the sample rate
- multi-resolution magnitude STFT: resolutions
  ``(1024, 120, 600)``, ``(2048, 240, 1200)``, ``(512, 50, 240)``

The CQT stack uses 128 channels. A configuration field of 32 exists in the
reference constructor and is not passed into the layer widths, so 128 is the
width that actually runs. Feature maps for the CQT stack are the activations
after each convolution except the final score. The STFT stack also keeps the
score map. Least-squares and feature-matching losses live in ``spectral.py``.

``nnAudio`` is MIT-licensed and is the published CQT. This module does not
import ``src/se/official``.
"""

from __future__ import annotations

import torch
from nnAudio.features.cqt import CQT2010v2
from torch import nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Resample

from src.training.errors import TrainingError
from src.training.se.stft import _reflect_pad

_CQT_HOPS = (512, 256, 256)
_CQT_OCTAVES = (8, 8, 8)
_CQT_BINS = (12, 24, 36)
_FILTERS = 128
_KERNEL = (3, 9)
_DILATIONS = (1, 2, 4)
_RESOLUTIONS = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))


def _pad2d(kernel: tuple[int, int], dilation: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    return (
        ((kernel[0] - 1) * dilation[0]) // 2,
        ((kernel[1] - 1) * dilation[1]) // 2,
    )


def _as_channel(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 2:
        return waveform.unsqueeze(1)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        return waveform
    raise TrainingError(f"discriminator expected [batch, time] or [batch, 1, time], got {tuple(waveform.shape)}")


class CQTSubbandDiscriminator(nn.Module):
    """One CQT scale: resample to 32 kHz, split octaves, then a conv stack."""

    def __init__(self, hop_length: int, n_octaves: int, bins_per_octave: int, sample_rate: int = 16000) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.resample = Resample(orig_freq=sample_rate, new_freq=sample_rate * 2)
        self.cqt_transform = CQT2010v2(
            sr=sample_rate * 2,
            hop_length=hop_length,
            n_bins=bins_per_octave * n_octaves,
            bins_per_octave=bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
            verbose=False,
        )
        self.conv_pres = nn.ModuleList(
            nn.Conv2d(2, 2, kernel_size=_KERNEL, padding=_pad2d(_KERNEL)) for _ in range(n_octaves)
        )
        # filters_scale is 1 and the cap is 1024, so every layer stays at 128.
        layers: list[nn.Module] = [
            nn.Conv2d(2, _FILTERS, kernel_size=_KERNEL, padding=_pad2d(_KERNEL))
        ]
        for dilation in _DILATIONS:
            layers.append(
                weight_norm(
                    nn.Conv2d(
                        _FILTERS,
                        _FILTERS,
                        kernel_size=_KERNEL,
                        stride=(1, 2),
                        dilation=(dilation, 1),
                        padding=_pad2d(_KERNEL, (dilation, 1)),
                    )
                )
            )
        layers.append(weight_norm(nn.Conv2d(_FILTERS, _FILTERS, kernel_size=(3, 3), padding=_pad2d((3, 3)))))
        self.convs = nn.ModuleList(layers)
        self.conv_post = weight_norm(nn.Conv2d(_FILTERS, 1, kernel_size=(3, 3), padding=_pad2d((3, 3))))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = self.resample(_as_channel(waveform))
        transform = self.cqt_transform(hidden)
        real = transform[:, :, :, 0].unsqueeze(1)
        imag = transform[:, :, :, 1].unsqueeze(1)
        spectrum = torch.cat((real, imag), dim=1).permute(0, 1, 3, 2)
        bands = []
        for index, layer in enumerate(self.conv_pres):
            start = index * self.bins_per_octave
            stop = start + self.bins_per_octave
            bands.append(layer(spectrum[:, :, :, start:stop]))
        hidden = torch.cat(bands, dim=-1)
        features: list[torch.Tensor] = []
        for layer in self.convs:
            hidden = self.activation(layer(hidden))
            features.append(hidden)
        return self.conv_post(hidden), features


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, sample_rate: int = 16000) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            CQTSubbandDiscriminator(hop, octaves, bins, sample_rate)
            for hop, octaves, bins in zip(_CQT_HOPS, _CQT_OCTAVES, _CQT_BINS, strict=True)
        )

    def forward(self, waveform: torch.Tensor, paired: torch.Tensor | None = None):
        scores = []
        features: list[torch.Tensor] = []
        paired_scores = []
        paired_features: list[torch.Tensor] = []
        for discriminator in self.discriminators:
            score, maps = discriminator(waveform)
            scores.append(score)
            features.extend(maps)
            if paired is not None:
                score, maps = discriminator(paired)
                paired_scores.append(score)
                paired_features.extend(maps)
        if paired is not None:
            return scores, paired_scores, features, paired_features
        return scores, features


class ResolutionDiscriminator(nn.Module):
    def __init__(self, resolution: tuple[int, int, int]) -> None:
        super().__init__()
        if len(resolution) != 3:
            raise TrainingError(f"resolution must be n_fft, hop, window, got {resolution}")
        self.resolution = tuple(int(item) for item in resolution)
        width = 32
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, width, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(width, width, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(width, width, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(width, width, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(width, width, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(width, 1, (3, 3), padding=(1, 1)))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        magnitude = self._magnitude(_as_channel(waveform)).unsqueeze(1)
        features: list[torch.Tensor] = []
        hidden = magnitude
        for layer in self.convs:
            hidden = self.activation(layer(hidden))
            features.append(hidden)
        score = self.conv_post(hidden)
        features.append(score)
        return score.flatten(1), features

    def _magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        n_fft, hop, window = self.resolution
        pad = (n_fft - hop) // 2
        if waveform.is_cuda and torch.are_deterministic_algorithms_enabled():
            framed = _reflect_pad(waveform.squeeze(1), pad)
        else:
            framed = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect").squeeze(1)
        spectrum = torch.stft(
            framed,
            n_fft=n_fft,
            hop_length=hop,
            win_length=window,
            window=torch.hann_window(window).to(framed.device),
            center=False,
            return_complex=True,
        )
        return torch.view_as_real(spectrum).norm(p=2, dim=-1)


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resolutions = _RESOLUTIONS
        self.discriminators = nn.ModuleList(ResolutionDiscriminator(resolution) for resolution in _RESOLUTIONS)

    def forward(self, waveform: torch.Tensor, paired: torch.Tensor | None = None):
        scores = []
        features: list[torch.Tensor] = []
        paired_scores = []
        paired_features: list[torch.Tensor] = []
        for discriminator in self.discriminators:
            score, maps = discriminator(waveform)
            scores.append(score)
            features.extend(maps)
            if paired is not None:
                score, maps = discriminator(paired)
                paired_scores.append(score)
                paired_features.extend(maps)
        if paired is not None:
            return scores, paired_scores, features, paired_features
        return scores, features
