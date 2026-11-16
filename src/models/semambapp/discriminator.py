"""Import path for the published SEMamba++ discriminator networks.

The implementations live in ``model.py``. This module remains because tests
and the comparison task import ``discriminator.MultiResolutionDiscriminator``,
``MultiScaleSubbandCQTDiscriminator``, and ``ResolutionDiscriminator``.
Remove it when those imports name ``model``.
"""

from src.models.semambapp.model import (
    CQTSubbandDiscriminator,
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
    ResolutionDiscriminator,
)

__all__ = [
    "CQTSubbandDiscriminator",
    "MultiResolutionDiscriminator",
    "MultiScaleSubbandCQTDiscriminator",
    "ResolutionDiscriminator",
]
