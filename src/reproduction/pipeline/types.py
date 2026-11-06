"""Shared values passed between the planner, the renderer, and the trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class CropSpec:
    """A window chosen before any waveform is read.

    ``length`` is the number of real samples taken from the file.
    ``pad_to`` is the tensor length after tail zeros. ``pad`` is ``tail_zero``
    when the utterance is shorter than the window, and ``none`` when the crop
    already has ``pad_to`` samples.
    """

    start: int
    length: int
    pad_to: int
    pad: str


@dataclass(frozen=True)
class SampleRef:
    """One planned sample. It does not contain audio."""

    case_id: str | None
    index: int
    draw: tuple | None
    length: int | None
    crop: CropSpec | None
    chunk: None = None


@dataclass(frozen=True)
class MicroBatchPlan:
    samples: tuple[SampleRef, ...]


@dataclass(frozen=True)
class UpdateGroup:
    """One optimizer step: ``accumulation`` micro-batches."""

    micro_batches: tuple[MicroBatchPlan, ...]


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    world_size: int
    updates: int
    rank_groups: tuple[tuple[UpdateGroup, ...], ...]
    plan_hash: str
    order_policy: str
    stats: Mapping[str, Any]


@dataclass
class RenderedSample:
    noisy: np.ndarray
    clean: np.ndarray
    sample_rate: int
    valid_length: int
    case_id: str | None
    teacher_kind: str
    extras: dict[str, np.ndarray]


@dataclass
class Batch:
    """Task-facing batch. ``clean`` is the protocol teacher."""

    noisy: torch.Tensor
    clean: torch.Tensor
    lengths: torch.Tensor
    case_ids: tuple[str | None, ...]
    teacher_kind: str
    extras: dict[str, Any]

    def as_task_batch(self) -> dict[str, Any]:
        """Dict consumed by existing training tasks.

        Tasks read ``clean``, ``noisy``, and ``case_id``. ``lengths`` records
        the unpadded sample count; tail zeros stay in the waveform because
        the official losses do not mask them.
        """
        batch = {
            "noisy": self.noisy,
            "clean": self.clean,
            "lengths": self.lengths,
            "case_id": list(self.case_ids),
            "teacher_kind": self.teacher_kind,
        }
        batch.update(self.extras)
        return batch
