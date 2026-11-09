"""Experiment data supply.

``TensorPairData`` remains the in-memory implementation. Fixed-pair snapshots
use ``SnapshotDataModule`` and do not wrap the reproduction training classes.
"""

from __future__ import annotations

from src.experiments.data.fixed_pair import (
    FixedPairProtocol,
    MicroBatcher,
    OrderSampler,
    SnapshotDataModule,
    SnapshotSource,
)
from src.experiments.data.sample import Batch, ChunkSpan, Crop, SampleRef
from src.experiments.data.tensor import EpochView, TensorPairData

__all__ = [
    "Batch",
    "ChunkSpan",
    "Crop",
    "EpochView",
    "FixedPairProtocol",
    "MicroBatcher",
    "OrderSampler",
    "SampleRef",
    "SnapshotDataModule",
    "SnapshotSource",
    "TensorPairData",
]
