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
from src.experiments.data.online import Dfn3IndexSampler, Dfn3Protocol, SemambaProtocol
from src.experiments.data.sample import Batch, ChunkSpan, Crop, SampleRef
from src.experiments.data.tensor import EpochView, TensorPairData

__all__ = [
    "Batch",
    "ChunkProtocol",
    "ChunkSampler",
    "ChunkSpan",
    "Crop",
    "Dfn3IndexSampler",
    "Dfn3Protocol",
    "EpochView",
    "FixedPairProtocol",
    "MicroBatcher",
    "OrderSampler",
    "SampleRef",
    "SemambaProtocol",
    "SnapshotDataModule",
    "SnapshotSource",
    "TensorPairData",
]
