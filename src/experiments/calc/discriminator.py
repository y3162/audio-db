"""Metric discriminator used by MP-SENet training.

The architecture follows Yexin Lu's MIT-licensed MP-SENet
(https://github.com/yxlu-0102/MP-SENet, commit 89932cfe). The license text is
``src/se/models/mp_senet/LICENSE``. This module does not import
``src/se/official``.
"""

from __future__ import annotations

import torch
from torch import nn


class _LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * value)


class _GlobalMaxPool(nn.Module):
    """Global max with the same values as ``AdaptiveMaxPool2d(1)``.

    ``adaptive_max_pool2d_backward_cuda`` has no deterministic implementation.
    ``amax`` matches the forward values and has a deterministic backward.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.amax(dim=(-2, -1), keepdim=True)


class MetricDiscriminator(nn.Module):
    def __init__(self, dim: int = 16, in_channel: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),
            nn.utils.spectral_norm(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True),
            nn.PReLU(dim * 8),
            _GlobalMaxPool(),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim * 8, dim * 4)),
            nn.Dropout(0.3),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Linear(dim * 4, 1)),
            _LearnableSigmoid1d(1),
        )

    def forward(self, reference_magnitude: torch.Tensor, estimate_magnitude: torch.Tensor) -> torch.Tensor:
        paired = torch.stack((reference_magnitude, estimate_magnitude), dim=1)
        return self.layers(paired)
