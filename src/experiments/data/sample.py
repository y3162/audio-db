"""Values shared by snapshot data modules. They do not own audio bytes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Crop:
    start: int
    length: int
    pad_to: int


@dataclass(frozen=True)
class ChunkSpan:
    start: int
    stop: int


@dataclass(frozen=True)
class SampleRef:
    """One planned sample for one rank. The waveform is not stored here."""

    sample_id: str
    realization_id: str
    index: int
    case_id: str | None = None
    crop: Crop | None = None
    chunk: ChunkSpan | None = None


@dataclass(frozen=True)
class Batch:
    """One collated micro-batch.

    The first four fields stay positional so existing ``TensorPairData``
    callers keep working. Audio batches add lengths and per-sample ids.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    sample_id: str
    realization_id: str
    lengths: torch.Tensor | None = None
    sample_ids: tuple[str, ...] = ()
    realization_ids: tuple[str, ...] = ()
    teacher_kind: str | None = None
