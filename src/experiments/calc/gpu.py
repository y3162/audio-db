"""Map a visible CUDA index to the physical GPU index.

The training runtime does not refuse a device because another process is
using memory or utilization. ``CUDA_VISIBLE_DEVICES`` still selects which
devices are visible. Occupancy is recorded by the launcher.
"""

from __future__ import annotations

from src.evaluation.errors import ConfigError
from src.evaluation.gpu import physical_device_index
from src.experiments.calc.errors import TrainingError


def physical_index(local_rank: int) -> int:
    try:
        return physical_device_index(local_rank)
    except ConfigError as exc:
        raise TrainingError(str(exc)) from exc
