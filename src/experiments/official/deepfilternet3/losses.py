"""YAML class path for the DeepFilterNet3 loss terms.

The implementations live in ``loss.py``. This module remains because
``tests/fixtures/configs/experiments/deepfilternet3.yaml`` and
``tests/experiments/official_loss_yaml.py`` name
``losses.DeepFilterNetSpectralLoss`` and ``losses.LocalSNRLoss``.
Remove it when those class paths name ``loss``.
"""

from src.experiments.official.deepfilternet3.loss import (
    DeepFilterNetSpectralLoss,
    LocalSNRLoss,
    official_deepfilternet_objective,
)

__all__ = [
    "DeepFilterNetSpectralLoss",
    "LocalSNRLoss",
    "official_deepfilternet_objective",
]
