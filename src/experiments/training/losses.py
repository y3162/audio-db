"""Loss modules constructed by the experiment factory.

``MultiResolutionL1Loss`` calls ``src.experiments.calc.spectral.multi_resolution_l1``.
MS-SNSD keeps ``normalize_variance`` false. That flag is the URGENT loss
setting and is not the model's internal waveform standard-deviation scaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.experiments.errors import BuildError
from src.experiments.calc.spectral import multi_resolution_l1

OFFICIAL_WINDOWS = (256, 512, 768, 1024)


class MultiResolutionL1Loss(torch.nn.Module):
    """ESPnet ``mr_l1_tfd`` as a module. SI-SDR is not added.

    ``forward`` receives ``(estimate, target)`` so a trainer can call
    ``loss(prediction, batch.targets)``. The spectral function expects the
    target first.
    """

    def __init__(
        self,
        windows: Sequence[int] = OFFICIAL_WINDOWS,
        time_domain_weight: float = 0.5,
        normalize_variance: bool = False,
    ) -> None:
        super().__init__()
        if normalize_variance:
            raise BuildError(
                "MS-SNSD mr_l1_tfd does not use variance normalization; "
                "normalize_variance must be false"
            )
        if isinstance(time_domain_weight, bool) or not isinstance(time_domain_weight, (int, float)):
            raise BuildError("time_domain_weight must be a number")
        weight = float(time_domain_weight)
        if weight < 0.0 or weight > 1.0:
            raise BuildError("time_domain_weight must be between 0 and 1")
        if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence) or not windows:
            raise BuildError("windows must be a non-empty sequence of integers")
        cleaned: list[int] = []
        for window in windows:
            if isinstance(window, bool) or not isinstance(window, int) or window < 2 or window % 2 != 0:
                raise BuildError(f"window {window!r} must be an even integer >= 2")
            cleaned.append(window)
        self.windows = tuple(cleaned)
        self.time_domain_weight = weight
        self.normalize_variance = False

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return multi_resolution_l1(
            target,
            estimate,
            self.windows,
            time_domain_weight=self.time_domain_weight,
            normalize_variance=False,
        )
