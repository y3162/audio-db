"""Experiment data supply.

``TensorPairData`` remains the in-memory implementation. Snapshot supply uses
``SnapshotDataModule`` and does not wrap the reproduction training classes.
"""

from __future__ import annotations

from src.experiments.data.chunk import ChunkProtocol, ChunkSampler
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
    "ChunkProtocol",
    "ChunkSampler",
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
