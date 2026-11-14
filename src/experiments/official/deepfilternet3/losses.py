"""DeepFilterNet3 loss terms. STFT sizes and SNR limits come from the model.

The multi-resolution factors and the local-SNR window are loss parameters.
They are not copied from ``DFN3Params``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.calc.errors import TrainingError
from src.experiments.errors import BuildError
from src.experiments.official.deepfilternet3.reference import _local_snr_db, _multi_resolution_spectral
from src.experiments.training.objectives import WeightedObjective, require_tensor

OFFICIAL_FACTOR = 500.0
OFFICIAL_COMPLEX_FACTOR = 500.0
OFFICIAL_GAMMA = 0.3
OFFICIAL_FFTS = (256, 512, 1024, 2048)
OFFICIAL_MAGNITUDE_FLOOR = 1e-12
OFFICIAL_LSNR_MILLISECONDS = 20
OFFICIAL_LSNR_FLOOR = 1e-12
OFFICIAL_LSNR_WEIGHT = 1e-3


def official_deepfilternet_objective() -> WeightedObjective:
    """Published spectral weight 1 and local-SNR weight 1e-3."""

    spectral = DeepFilterNetSpectralLoss(
        "spectral",
        factor=OFFICIAL_FACTOR,
        complex_factor=OFFICIAL_COMPLEX_FACTOR,
        gamma=OFFICIAL_GAMMA,
        fft_sizes=OFFICIAL_FFTS,
        magnitude_floor=OFFICIAL_MAGNITUDE_FLOOR,
    )
    local = LocalSNRLoss(
        "local_snr",
        milliseconds=OFFICIAL_LSNR_MILLISECONDS,
        energy_floor=OFFICIAL_LSNR_FLOOR,
    )
    return WeightedObjective([spectral, local], {"spectral": 1.0, "local_snr": OFFICIAL_LSNR_WEIGHT})


class DeepFilterNetSpectralLoss(torch.nn.Module):
    """Official multi-resolution magnitude and complex spectral loss.

    Each resolution's two MSEs are scaled inside this term, in FFT order.
    The objective weight is separate and is 1 for the published recipe.
    """

    def __init__(
        self,
        name: str,
        factor: float,
        complex_factor: float,
        gamma: float,
        fft_sizes: Sequence[int],
        magnitude_floor: float,
    ) -> None:
        super().__init__()
        if not isinstance(name, str) or not name:
            raise BuildError("loss term name must be a non-empty string")
        self.name = name
        self.factor = _finite(factor, "factor")
        self.complex_factor = _finite(complex_factor, "complex_factor")
        self.gamma = _finite(gamma, "gamma")
        self.fft_sizes = _even_sizes(fft_sizes)
        self.magnitude_floor = _positive(magnitude_floor, "magnitude_floor")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        return _multi_resolution_spectral(
            require_tensor(context, "estimate", self.name),
            require_tensor(context, "target", self.name),
            factor=self.factor,
            complex_factor=self.complex_factor,
            gamma=self.gamma,
            ffts=self.fft_sizes,
            magnitude_floor=self.magnitude_floor,
        )


class LocalSNRLoss(torch.nn.Module):
    """MSE between the network local SNR and the waveform-derived target.

    ``milliseconds`` and ``energy_floor`` are this term's parameters. Sample
    rate, FFT, hop, and the clamp limits are read from the context so they
    stay the model's values.
    """

    def __init__(self, name: str, milliseconds: int, energy_floor: float) -> None:
        super().__init__()
        if not isinstance(name, str) or not name:
            raise BuildError("loss term name must be a non-empty string")
        if isinstance(milliseconds, bool) or not isinstance(milliseconds, int) or milliseconds < 1:
            raise BuildError("milliseconds must be an integer >= 1")
        self.name = name
        self.milliseconds = milliseconds
        self.energy_floor = _positive(energy_floor, "energy_floor")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        predicted = require_tensor(context, "predicted_lsnr", self.name)
        target = _local_snr_db(
            require_tensor(context, "clean_complex", self.name),
            require_tensor(context, "noise_complex", self.name),
            sample_rate=_context_int(context, "sample_rate", self.name),
            n_fft=_context_int(context, "n_fft", self.name),
            hop=_context_int(context, "hop", self.name),
            low=_context_float(context, "lsnr_low", self.name),
            high=_context_float(context, "lsnr_high", self.name),
            milliseconds=self.milliseconds,
            floor=self.energy_floor,
        )
        predicted = predicted.reshape(target.shape[0], -1)
        if predicted.shape[-1] != target.shape[-1]:
            raise TrainingError(
                f"local SNR length {predicted.shape[-1]} does not match the "
                f"{self.milliseconds} ms target {target.shape[-1]}"
            )
        return F.mse_loss(predicted, target)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise BuildError(f"{label} must be finite")
    return number


def _positive(value: object, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise BuildError(f"{label} must be positive")
    return number


def _even_sizes(values: object) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise BuildError("fft_sizes must be a non-empty sequence of integers")
    cleaned: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2 != 0:
            raise BuildError(f"fft size {value!r} must be an even integer >= 2")
        cleaned.append(value)
    return tuple(cleaned)


def _context_int(context: Mapping[str, Any], key: str, term: str) -> int:
    if key not in context:
        raise ValueError(f"loss term {term} requires context[{key!r}]")
    value = context[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"loss term {term} context[{key!r}] must be an integer")
    return value


def _context_float(context: Mapping[str, Any], key: str, term: str) -> float:
    if key not in context:
        raise ValueError(f"loss term {term} requires context[{key!r}]")
    value = context[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"loss term {term} context[{key!r}] must be a finite number")
    return float(value)
