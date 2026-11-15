"""YAML class path for the MP-SENet generator and metric discriminator.

The classes are defined in ``model.py``. This module remains because the
experiment YAML addresses:

- ``src.experiments.official.mp_senet.models.MPSENetGANGenerator``
- ``src.experiments.official.mp_senet.models.MPSENetMetricDiscriminator``

Those paths are in ``configs/reproduction/experiments/official/mp_senet_gan.yaml``
and ``tests/fixtures/configs/experiments/mp_senet_gan.yaml``. Remove this module
when those documents name ``model`` instead of ``models``.
"""

from src.experiments.official.mp_senet.model import (
    OFFICIAL_ARCHITECTURE,
    MPSENetGANGenerator,
    MPSENetMetricDiscriminator,
    architecture_config,
    network_config,
    pack_forward,
    require_official_architecture,
)

__all__ = [
    "OFFICIAL_ARCHITECTURE",
    "MPSENetGANGenerator",
    "MPSENetMetricDiscriminator",
    "architecture_config",
    "network_config",
    "pack_forward",
    "require_official_architecture",
]
