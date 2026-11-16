"""Batch reduction for per-sample TDT NLL. The coefficient is applied by the objective."""

from __future__ import annotations

import torch


def mean_tdt(per_sample: torch.Tensor) -> torch.Tensor:
    """Mean over the micro-batch. A scalar is already reduced."""

    if per_sample.ndim == 0:
        return per_sample
    return per_sample.mean()
