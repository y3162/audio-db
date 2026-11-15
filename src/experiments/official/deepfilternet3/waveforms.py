"""Map a DeepFilterNet3 micro-batch onto ``component_losses`` argument order."""

from __future__ import annotations

import torch

from src.experiments.data import Batch


def clean_noisy(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean, then noisy.

    Snapshot collation stores the degraded waveform in ``Batch.inputs`` and
    the clean target in ``Batch.targets``. ``component_losses`` takes clean
    first. Passing the fields in declaration order trains and validates on the
    swapped pair, which changes the loss, the local-SNR target, and which
    checkpoint counts as best.
    """

    return batch.targets, batch.inputs
