"""Loss terms whose formulas live in Python and whose parameters come from YAML.

Each ``forward`` reads a named context. Missing keys are refused. These
modules do not own coefficients that belong to an objective, except for a
scale that is inside one published term (the complex MSE factor of 2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.calc.losses import discriminator_loss, phase_losses
from src.experiments.calc.spectral import (
    feature_matching,
    least_squares_discriminator,
    least_squares_generator,
    matrix_phase_losses,
    summed_slaney_log_mel,
)
from src.experiments.errors import BuildError
from src.experiments.training.objectives import (
    ObjectiveOutput,
    require_sequence,
    require_tensor,
)


def _name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError("loss term name must be a non-empty string")
    return value


def _key(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{label} must be a non-empty context key")
    return value


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


class ContextTerm(torch.nn.Module):
    """Return a scalar that the trainer has already computed.

    MossFormerGAN's RI, magnitude, time, and GAN terms share one STFT and one
    iSTFT. The trainer keeps that official reconstruction and this term only
    reads the resulting scalar. It is not a waveform L1.
    """

    def __init__(self, name: str, key: str | None = None) -> None:
        super().__init__()
        self.name = _name(name)
        self.key = self.name if key is None else _key(key, "key")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        value = require_tensor(context, self.key, self.name)
        if value.ndim != 0:
            raise ValueError(f"loss term {self.name} expected a scalar context[{self.key!r}]")
        return value


class MagnitudeMSE(torch.nn.Module):
    """``mse(target, estimate)`` with the published argument order."""

    def __init__(self, name: str, estimate: str, target: str) -> None:
        super().__init__()
        self.name = _name(name)
        self.estimate = _key(estimate, "estimate")
        self.target = _key(target, "target")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        return F.mse_loss(
            require_tensor(context, self.target, self.name),
            require_tensor(context, self.estimate, self.name),
        )


class PhaseLoss(torch.nn.Module):
    """Sum of instantaneous phase, group delay, and angular frequency.

    ``mode`` is ``diff`` (``torch.diff``) or ``matrix`` (SEMamba++ difference
    matrix). The three means are added before the objective weight.
    """

    def __init__(self, name: str, mode: str) -> None:
        super().__init__()
        self.name = _name(name)
        if mode not in {"diff", "matrix"}:
            raise BuildError("phase mode must be diff or matrix")
        self.mode = mode

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        reference = require_tensor(context, "clean_phase", self.name)
        generated = require_tensor(context, "generated_phase", self.name)
        phase_fn = matrix_phase_losses if self.mode == "matrix" else phase_losses
        instantaneous, group_delay, angular = phase_fn(reference, generated)
        return instantaneous + group_delay + angular


class ScaledComplexMSE(torch.nn.Module):
    """Complex or consistency MSE multiplied by an internal scale.

    The scale is the published factor inside the term. The objective applies
    its own coefficient afterwards.
    """

    def __init__(self, name: str, scale: float, estimate: str, target: str) -> None:
        super().__init__()
        self.name = _name(name)
        self.scale = _finite(scale, "scale")
        self.estimate = _key(estimate, "estimate")
        self.target = _key(target, "target")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        return F.mse_loss(
            require_tensor(context, self.target, self.name),
            require_tensor(context, self.estimate, self.name),
        ) * self.scale


class TimeL1(torch.nn.Module):
    """Mean L1 with the published argument order ``l1(target, estimate)``."""

    def __init__(self, name: str, estimate: str, target: str) -> None:
        super().__init__()
        self.name = _name(name)
        self.estimate = _key(estimate, "estimate")
        self.target = _key(target, "target")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        return F.l1_loss(
            require_tensor(context, self.target, self.name),
            require_tensor(context, self.estimate, self.name),
        )


class WaveformL1(torch.nn.Module):
    """Mean absolute error of context waveforms ``estimate`` and ``target``.

    This is a research term. It does not replace MossFormerGAN's time loss,
    which reconstructs from the clean real part and the predicted imaginary
    part before the mean is taken.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = _name(name)

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        estimate = require_tensor(context, "estimate", self.name)
        target = require_tensor(context, "target", self.name)
        if estimate.shape != target.shape:
            raise ValueError(
                f"loss term {self.name} shape mismatch {tuple(estimate.shape)} vs {tuple(target.shape)}"
            )
        return F.l1_loss(estimate, target)


class MetricMSE(torch.nn.Module):
    """Generator metric term: MSE of the discriminator score toward 1."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = _name(name)

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        metric = require_tensor(context, "metric", self.name)
        ones = require_tensor(context, "ones", self.name)
        return F.mse_loss(metric.flatten(), ones)


class MPSENetDiscriminatorObjective(torch.nn.Module):
    """Real scores toward 1, and fake scores toward PESQ when PESQ exists.

    A missing PESQ target omits the fake term and still returns the real term.
    That branch is the published control flow, not a coefficient.
    """

    def forward(self, context: Mapping[str, Any]) -> ObjectiveOutput:
        if "pesq_target" not in context:
            raise ValueError("loss term discriminator requires context['pesq_target']")
        loss = discriminator_loss(
            require_tensor(context, "real_score", "discriminator"),
            require_tensor(context, "fake_score", "discriminator"),
            require_tensor(context, "ones", "discriminator"),
            context["pesq_target"],
        )
        return ObjectiveOutput(loss, (("discriminator", loss, 1.0),))


class MossFormerDiscriminatorObjective(torch.nn.Module):
    """Real pair toward 1 plus fake pair toward normalized PESQ."""

    def forward(self, context: Mapping[str, Any]) -> ObjectiveOutput:
        real = require_tensor(context, "real_score", "discriminator")
        fake = require_tensor(context, "fake_score", "discriminator")
        ones = require_tensor(context, "ones", "discriminator")
        target = require_tensor(context, "pesq_target", "discriminator")
        loss = F.mse_loss(real.flatten(), ones) + F.mse_loss(fake.flatten(), target.to(fake.device))
        return ObjectiveOutput(loss, (("discriminator", loss, 1.0),))


class SummedLeastSquaresGenerator(torch.nn.Module):
    """Sum least-squares generator scores, first family then second."""

    def __init__(self, name: str, first: str, second: str) -> None:
        super().__init__()
        self.name = _name(name)
        self.first = _key(first, "first")
        self.second = _key(second, "second")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        first_scores = require_sequence(context, self.first, self.name)
        second_scores = require_sequence(context, self.second, self.name)
        return sum(least_squares_generator(score) for score in first_scores) + sum(
            least_squares_generator(score) for score in second_scores
        )


class SummedLeastSquaresDiscriminator(torch.nn.Module):
    """Sum least-squares discriminator pairs, first family then second."""

    def __init__(self, name: str, first_real: str, first_fake: str, second_real: str, second_fake: str) -> None:
        super().__init__()
        self.name = _name(name)
        self.first_real = _key(first_real, "first_real")
        self.first_fake = _key(first_fake, "first_fake")
        self.second_real = _key(second_real, "second_real")
        self.second_fake = _key(second_fake, "second_fake")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        first_real = require_sequence(context, self.first_real, self.name)
        first_fake = require_sequence(context, self.first_fake, self.name)
        second_real = require_sequence(context, self.second_real, self.name)
        second_fake = require_sequence(context, self.second_fake, self.name)
        loss_q = sum(
            least_squares_discriminator(real, fake)
            for real, fake in zip(first_real, first_fake, strict=True)
        )
        loss_r = sum(
            least_squares_discriminator(real, fake)
            for real, fake in zip(second_real, second_fake, strict=True)
        )
        return loss_q + loss_r


class PairedFeatureMatching(torch.nn.Module):
    """Add two feature-matching losses. ``scale`` is inside each call."""

    def __init__(
        self,
        name: str,
        scale: float,
        reduction: str,
        first_real: str,
        first_fake: str,
        second_real: str,
        second_fake: str,
    ) -> None:
        super().__init__()
        self.name = _name(name)
        self.scale = _finite(scale, "scale")
        if reduction not in {"sum", "mean"}:
            raise BuildError("feature matching reduction must be sum or mean")
        self.reduction = reduction
        self.first_real = _key(first_real, "first_real")
        self.first_fake = _key(first_fake, "first_fake")
        self.second_real = _key(second_real, "second_real")
        self.second_fake = _key(second_fake, "second_fake")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        first = feature_matching(
            list(require_sequence(context, self.first_real, self.name)),
            list(require_sequence(context, self.first_fake, self.name)),
            scale=self.scale,
            reduction=self.reduction,
        )
        second = feature_matching(
            list(require_sequence(context, self.second_real, self.name)),
            list(require_sequence(context, self.second_fake, self.name)),
            scale=self.scale,
            reduction=self.reduction,
        )
        return first + second


class SlaneyLogMelLoss(torch.nn.Module):
    """Sum of Slaney log-mel L1 scales. The sample rate comes from the context."""

    def __init__(self, name: str, n_mels: Sequence[int], windows: Sequence[int], clamp: float) -> None:
        super().__init__()
        self.name = _name(name)
        self.n_mels = _positive_ints(n_mels, "n_mels")
        self.windows = _positive_ints(windows, "windows")
        if len(self.n_mels) != len(self.windows):
            raise BuildError("n_mels and windows must have the same length")
        self.clamp = _positive(clamp, "clamp")

    def forward(self, context: Mapping[str, Any]) -> torch.Tensor:
        if "sample_rate" not in context:
            raise ValueError(f"loss term {self.name} requires context['sample_rate']")
        sample_rate = context["sample_rate"]
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise ValueError(f"loss term {self.name} sample_rate must be an integer")
        return summed_slaney_log_mel(
            require_tensor(context, "reference", self.name),
            require_tensor(context, "estimate", self.name),
            sample_rate,
            n_mels=self.n_mels,
            windows=self.windows,
            clamp=self.clamp,
        )


def _positive_ints(values: object, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise BuildError(f"{label} must be a non-empty sequence of integers")
    cleaned: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BuildError(f"{label} entries must be integers >= 1")
        cleaned.append(value)
    return tuple(cleaned)
