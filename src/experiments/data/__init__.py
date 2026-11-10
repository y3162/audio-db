"""Experiment data supply.

Importing this package does not open DuckDB and does not construct a trainer.
Database commands live in ``src.experiments.data.cli``. Snapshot training uses
``SnapshotDataModule``.
"""

from __future__ import annotations

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


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {"Batch", "ChunkSpan", "Crop", "SampleRef"}:
        from src.experiments.data.sample import Batch, ChunkSpan, Crop, SampleRef

        values = {"Batch": Batch, "ChunkSpan": ChunkSpan, "Crop": Crop, "SampleRef": SampleRef}
    elif name in {"ChunkProtocol", "ChunkSampler"}:
        from src.experiments.data.chunk import ChunkProtocol, ChunkSampler

        values = {"ChunkProtocol": ChunkProtocol, "ChunkSampler": ChunkSampler}
    elif name in {
        "FixedPairProtocol",
        "MicroBatcher",
        "OrderSampler",
        "SnapshotDataModule",
        "SnapshotSource",
    }:
        from src.experiments.data.fixed_pair import (
            FixedPairProtocol,
            MicroBatcher,
            OrderSampler,
            SnapshotDataModule,
            SnapshotSource,
        )

        values = {
            "FixedPairProtocol": FixedPairProtocol,
            "MicroBatcher": MicroBatcher,
            "OrderSampler": OrderSampler,
            "SnapshotDataModule": SnapshotDataModule,
            "SnapshotSource": SnapshotSource,
        }
    elif name in {"Dfn3IndexSampler", "Dfn3Protocol", "SemambaProtocol"}:
        from src.experiments.data.online import Dfn3IndexSampler, Dfn3Protocol, SemambaProtocol

        values = {
            "Dfn3IndexSampler": Dfn3IndexSampler,
            "Dfn3Protocol": Dfn3Protocol,
            "SemambaProtocol": SemambaProtocol,
        }
    else:
        from src.experiments.data.tensor import EpochView, TensorPairData

        values = {"EpochView": EpochView, "TensorPairData": TensorPairData}
    return values[name]
