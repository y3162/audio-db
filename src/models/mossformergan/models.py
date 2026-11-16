"""YAML class path for the MossFormerGAN generator and discriminator.

The implementations live in ``model.py``. This module remains because
``configs/reproduction/experiments/official/mossformergan_se_16k.yaml`` and
``tests/fixtures/configs/experiments/mossformergan_se_16k.yaml`` name
``models.MossFormerGANGenerator`` and ``models.MossFormerGANDiscriminator``.
Remove it when those class paths name ``model``.
"""

from src.models.mossformergan.model import (
    OFFICIAL_ARCHITECTURE,
    MossFormerGANDiscriminator,
    MossFormerGANGenerator,
    architecture_config,
    require_official_architecture,
)

__all__ = [
    "OFFICIAL_ARCHITECTURE",
    "MossFormerGANDiscriminator",
    "MossFormerGANGenerator",
    "architecture_config",
    "require_official_architecture",
]
