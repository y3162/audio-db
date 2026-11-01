"""Discriminators owned by the training package.

Legacy discriminator components. MossFormerGAN training uses the released
architecture in ``mossformer_discriminator`` instead.
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










class _FeatureScale(nn.Module):
    def __init__(self, in_dim: int, width: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.utils.weight_norm(nn.Conv1d(in_dim, width, 1)),
                nn.utils.weight_norm(nn.Conv1d(width, width, 7, stride=2, padding=3)),
                nn.utils.weight_norm(nn.Conv1d(width, width, 7, stride=2, padding=3)),
                nn.utils.weight_norm(nn.Conv1d(width, width, 7, stride=2, padding=3)),
                nn.utils.weight_norm(nn.Conv1d(width, width, 3, stride=1, padding=1)),
            ]
        )
        self.score = nn.utils.weight_norm(nn.Conv1d(width, 1, 3, padding=1))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = sequence.transpose(1, 2)
        features = []
        for conv in self.convs:
            hidden = self.act(conv(hidden))
            features.append(hidden)
        score = self.score(hidden)
        features.append(score)
        return score.flatten(1), features




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
