"""Discriminators owned by the training package.

SEMamba++ has no license file in the reference tree, so this is an independent
multi-resolution and sub-band convolutional discriminator. It implements the
published least-squares objective, not a copy of that repository. MossFormerGAN
uses a convolutional score on the pair of compressed magnitudes. The released
discriminator also has a Conformer after a NISQA-style CNN; that block is not
reimplemented here and the difference is recorded in the training notes.
"""

from __future__ import annotations

import torch
from torch import nn


class _ScoreStack(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        widths = (channels, channels * 2, channels * 4)
        layers: list[nn.Module] = []
        in_ch = 1
        for width in widths:
            layers.append(nn.utils.spectral_norm(nn.Conv2d(in_ch, width, (3, 3), (2, 2), (1, 1))))
            layers.append(nn.LeakyReLU(0.2))
            in_ch = width
        self.layers = nn.ModuleList(layers)
        self.score = nn.utils.spectral_norm(nn.Conv2d(in_ch, 1, (3, 3), (1, 1), (1, 1)))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        hidden = value
        for layer in self.layers:
            hidden = layer(hidden)
            if isinstance(layer, nn.LeakyReLU):
                features.append(hidden)
        score = self.score(hidden)
        features.append(score)
        return score.mean(dim=(-2, -1)).squeeze(1), features


class MultiResolutionDiscriminator(nn.Module):
    """Waveform discriminator at several STFT window sizes."""

    def __init__(self, windows: tuple[int, ...] = (256, 512, 1024)) -> None:
        super().__init__()
        self.windows = windows
        self.stacks = nn.ModuleList(_ScoreStack(16) for _ in windows)

    def forward(self, waveform: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        scores = []
        features: list[torch.Tensor] = []
        for window, stack in zip(self.windows, self.stacks, strict=True):
            hop = window // 4
            spectrum = torch.stft(
                waveform,
                n_fft=window,
                hop_length=hop,
                win_length=window,
                window=torch.hann_window(window, device=waveform.device, dtype=waveform.dtype),
                center=True,
                return_complex=True,
            )
            magnitude = spectrum.abs().unsqueeze(1)
            score, feats = stack(magnitude)
            scores.append(score)
            features.extend(feats)
        return scores, features


class SubbandDiscriminator(nn.Module):
    """Four log-spaced bands of one STFT magnitude, scored separately."""

    def __init__(self, n_fft: int = 400, bands: int = 4) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.bands = bands
        self.stacks = nn.ModuleList(_ScoreStack(8) for _ in range(bands))

    def forward(self, waveform: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hop = self.n_fft // 4
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=hop,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=waveform.device, dtype=waveform.dtype),
            center=True,
            return_complex=True,
        ).abs()
        bins = spectrum.shape[-2]
        edges = [int(round(index * bins / self.bands)) for index in range(self.bands + 1)]
        scores = []
        features: list[torch.Tensor] = []
        for index, stack in enumerate(self.stacks):
            band = spectrum[:, edges[index] : max(edges[index + 1], edges[index] + 1)]
            score, feats = stack(band.unsqueeze(1))
            scores.append(score)
            features.extend(feats)
        return scores, features


class WaveformGANDiscriminator(nn.Module):
    """Multi-period waveform discriminator for vocoder stages.

    Periods follow the published HiFi-GAN / Vocos setup at a smaller width.
    """

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5), channels: int = 16) -> None:
        super().__init__()
        self.periods = periods
        self.heads = nn.ModuleList(_PeriodHead(period, channels) for period in periods)

    def forward(self, waveform: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        scores = []
        features: list[torch.Tensor] = []
        for head in self.heads:
            score, feats = head(waveform)
            scores.append(score)
            features.extend(feats)
        return scores, features


class _PeriodHead(nn.Module):
    def __init__(self, period: int, channels: int) -> None:
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                nn.utils.spectral_norm(nn.Conv2d(1, channels, (5, 1), (3, 1), (2, 0))),
                nn.utils.spectral_norm(nn.Conv2d(channels, channels * 2, (5, 1), (3, 1), (2, 0))),
                nn.utils.spectral_norm(nn.Conv2d(channels * 2, channels * 4, (5, 1), (3, 1), (2, 0))),
            ]
        )
        self.score = nn.utils.spectral_norm(nn.Conv2d(channels * 4, 1, (3, 1), (1, 1), (1, 0)))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        time = waveform.shape[-1]
        remainder = time % self.period
        if remainder:
            waveform = torch.nn.functional.pad(waveform, (0, self.period - remainder))
        frames = waveform.shape[-1] // self.period
        hidden = waveform.view(waveform.shape[0], 1, frames, self.period)
        features = []
        for conv in self.convs:
            hidden = self.act(conv(hidden))
            features.append(hidden)
        score = self.score(hidden)
        features.append(score)
        return score.mean(dim=(-2, -1)).squeeze(1), features


class MagnitudePairDiscriminator(nn.Module):
    """Score in (0, 1) from a reference magnitude and an estimate magnitude."""
    def __init__(self) -> None:
        super().__init__()
        self.pair = nn.Conv2d(2, 1, (1, 1))
        self.stack = _ScoreStack(16)

    def forward(self, reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        if reference.ndim == 3:
            reference = reference.unsqueeze(1)
        if estimate.ndim == 3:
            estimate = estimate.unsqueeze(1)
        hidden = self.pair(torch.cat((reference, estimate), dim=1))
        score, _features = self.stack(hidden)
        return torch.sigmoid(score)
