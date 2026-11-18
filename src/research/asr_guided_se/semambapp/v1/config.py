"""The three v1 switches. Projection width, the ASR id, and TDT's coefficient are fixed."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.experiments.errors import BuildError

GUIDANCE_DIM = 128
STFT_HOP = 100
ASR_HOP = 1280
TDT_LAMBDA = 1.0
MODEL_ID = "asr_guided_se.semambapp.v1"


@dataclass
class GuidanceV1:
    """YAML object for one ASR-guided SEMamba++ v1 run.

    ``encoder_layer`` is the only extra field. It is read when ASR features
    are on. The projection width, alignment, fusion site, and TDT coefficient
    are not fields. The five fields stay fixed after construction. The
    experiment builder may attach its component id.
    """

    asr_features_enabled: bool
    encoder_layer: int
    error_weight_enabled: bool
    error_weight_alpha: float
    tdt_loss_enabled: bool

    def __post_init__(self) -> None:
        _flag("asr_features_enabled", self.asr_features_enabled)
        _flag("error_weight_enabled", self.error_weight_enabled)
        _flag("tdt_loss_enabled", self.tdt_loss_enabled)
        if isinstance(self.encoder_layer, bool) or not isinstance(self.encoder_layer, int) or self.encoder_layer < 0:
            raise BuildError("encoder_layer must be an integer >= 0")
        if (
            isinstance(self.error_weight_alpha, bool)
            or not isinstance(self.error_weight_alpha, (int, float))
            or not math.isfinite(float(self.error_weight_alpha))
            or float(self.error_weight_alpha) < 0
        ):
            raise BuildError("error_weight_alpha must be a finite number >= 0")
        object.__setattr__(self, "error_weight_alpha", float(self.error_weight_alpha))

    def __setattr__(self, name: str, value: object) -> None:
        if name in getattr(self, "__dataclass_fields__", {}) and hasattr(self, name):
            raise BuildError(f"{name} is fixed")
        object.__setattr__(self, name, value)

    @property
    def uses_asr(self) -> bool:
        return self.asr_features_enabled or self.error_weight_enabled or self.tdt_loss_enabled

    @property
    def changes_loss(self) -> bool:
        return self.error_weight_enabled or self.tdt_loss_enabled


def _flag(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise BuildError(f"{name} must be a boolean")
