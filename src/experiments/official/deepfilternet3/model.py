"""DeepFilterNet3 for the experiment framework.

The module is ``src.se.build_model("deepfilternet3")``. ``DfNet`` is not
reimplemented. ``DFStream`` stays outside this module: the trainer resets it
and builds ERB and deep-filter features. ``df_fc_a`` remains in the state
dict and is unused by ``forward``, so DDP needs ``find_unused_parameters``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.se import build_model
from src.se.models.deepfilternet3.model import DfNet
from src.experiments.calc.rng import seed_model_init


class DeepFilterNet3Model(torch.nn.Module):
    """One initialized ``DfNet``. Training calls ``forward``, not ``enhance``."""

    def __init__(
        self,
        *,
        seed: int,
        architecture: Mapping[str, Any] | None = None,
        variant: str = "deepfilternet3",
    ) -> None:
        super().__init__()
        if variant != "deepfilternet3":
            raise BuildError(f"DeepFilterNet3Model implements variant deepfilternet3, got {variant!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BuildError("seed must be an integer >= 0")
        if architecture is None:
            config: dict[str, Any] | None = None
        elif isinstance(architecture, Mapping):
            config = dict(architecture)
        else:
            raise BuildError("architecture must be a mapping")
        try:
            with seed_model_init(seed):
                generator = build_model("deepfilternet3", config, variant="deepfilternet3")
        except (TypeError, ValueError) as exc:
            raise BuildError(f"failed to build deepfilternet3: {exc}") from exc
        if not isinstance(generator, DfNet):
            raise BuildError(f"deepfilternet3 builder returned {type(generator).__name__}, expected DfNet")
        self.generator = generator
        self.variant = "deepfilternet3"
        self.seed = seed

    def forward(
        self,
        spec: torch.Tensor,
        feat_erb: torch.Tensor,
        feat_spec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return enhanced spectrum, mask, local SNR, and deep-filter coefficients."""

        enhanced, mask, lsnr, coefs = self.generator(spec, feat_erb, feat_spec)
        return enhanced, mask, lsnr, coefs

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        """Raw ``DfNet`` state, including the unused ``df_fc_a`` parameter."""

        exported: dict[str, torch.Tensor] = {}
        for name, value in self.generator.state_dict().items():
            if not torch.is_tensor(value):
                raise RuntimeFailure(f"parameter {name} is not a tensor")
            exported[name] = value.detach().cpu().clone()
        if not any("df_fc_a" in name for name in exported):
            raise RuntimeFailure("export is missing df_fc_a")
        return exported
