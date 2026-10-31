"""STOI and optional ESTOI.

A missing clean reference is not a score of zero. A missing library is reported
and does not stop other metrics.
"""

from __future__ import annotations

from typing import Any

import torch
import torchaudio.functional as AF


def library_status() -> str | None:
    try:
        import pystoi  # noqa: F401
    except ImportError:
        return (
            "pystoi is not installed. Install it in .venvs/lab with "
            "uv pip install --python .venvs/lab/bin/python pystoi scipy"
        )
    return None


def score_stoi(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    *,
    extended: bool = False,
) -> dict[str, Any]:
    missing = library_status()
    if missing:
        return {"status": "unavailable", "reason": missing, "value": None, "extended": extended}
    from pystoi import stoi as stoi_fn

    try:
        reference, estimate, used_rate, resampled, trimmed = _prepare(reference, estimate, sample_rate)
        value = float(
            stoi_fn(
                reference.numpy().astype("float64"),
                estimate.numpy().astype("float64"),
                used_rate,
                extended=extended,
            )
        )
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "value": None,
            "extended": extended,
        }
    if value != value:  # NaN
        return {
            "status": "error",
            "reason": "STOI returned NaN",
            "value": None,
            "extended": extended,
            "sample_rate": used_rate,
            "resampled": resampled,
            "trimmed_samples": trimmed,
        }
    return {
        "status": "ok",
        "reason": None,
        "value": value,
        "extended": extended,
        "sample_rate": used_rate,
        "resampled": resampled,
        "trimmed_samples": trimmed,
    }


def _prepare(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, int]:
    reference = reference.detach().cpu().to(torch.float32).contiguous()
    estimate = estimate.detach().cpu().to(torch.float32).contiguous()
    used_rate = int(sample_rate)
    resampled = False
    if used_rate != 16000:
        reference = AF.resample(reference, used_rate, 16000)
        estimate = AF.resample(estimate, used_rate, 16000)
        used_rate = 16000
        resampled = True
    trimmed = abs(int(reference.numel() - estimate.numel()))
    length = min(int(reference.numel()), int(estimate.numel()))
    if length <= 0:
        raise ValueError("STOI received empty audio")
    return reference[:length].contiguous(), estimate[:length].contiguous(), used_rate, resampled, trimmed
