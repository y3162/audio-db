"""Stack rendered samples into the dict a training task already accepts."""

from __future__ import annotations

import numpy as np
import torch

from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.types import Batch, RenderedSample


def assemble(
    samples: list[RenderedSample],
    *,
    dataset_id: str,
    snapshot_hash: str,
) -> dict:
    if not samples:
        raise ReproductionError("cannot assemble an empty batch")
    teacher = samples[0].teacher_kind
    rate = samples[0].sample_rate
    if any(sample.teacher_kind != teacher or sample.sample_rate != rate for sample in samples):
        raise ReproductionError("a batch mixes teacher kinds or sample rates")
    width = max(int(sample.noisy.shape[0]) for sample in samples)
    if width < 1:
        raise ReproductionError("batch width is zero")
    noisy = np.stack([_pad(sample.noisy, width) for sample in samples]).astype(np.float32)
    clean = np.stack([_pad(sample.clean, width) for sample in samples]).astype(np.float32)
    if noisy.shape != clean.shape:
        raise ReproductionError("assembled clean and noisy shapes differ")
    batch = Batch(
        noisy=torch.from_numpy(noisy),
        clean=torch.from_numpy(clean),
        lengths=torch.tensor([int(sample.valid_length) for sample in samples], dtype=torch.int64),
        case_ids=tuple(sample.case_id for sample in samples),
        teacher_kind=teacher,
        extras={"dataset_id": dataset_id, "snapshot_hash": snapshot_hash},
    )
    task_batch = batch.as_task_batch()
    if task_batch["noisy"].shape != task_batch["clean"].shape:
        raise ReproductionError("task batch lost clean/noisy alignment")
    return task_batch


def _pad(wave: np.ndarray, width: int) -> np.ndarray:
    gap = width - int(wave.shape[0])
    if gap < 0 or wave.ndim != 1:
        raise ReproductionError("cannot pad a waveform that is already wider than the batch")
    if gap == 0:
        return np.asarray(wave, dtype=np.float32)
    return np.pad(np.asarray(wave, dtype=np.float32), (0, gap))
