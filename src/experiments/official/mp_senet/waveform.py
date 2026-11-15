"""YAML class path for the waveform adapter.

The class is defined in ``dataset.py``. This module remains because fixtures
and tests address ``src.experiments.official.mp_senet.waveform.MPSENetWaveformAdapter``
(``tests/fixtures/configs/experiments/mp_senet_l1.yaml``). Remove this module
when that class path names ``dataset`` instead.
"""

from src.experiments.official.mp_senet.dataset import MPSENetWaveformAdapter

__all__ = ["MPSENetWaveformAdapter"]
