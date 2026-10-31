"""PESQ wideband/narrowband scoring.

A missing clean reference is not a score of zero. A missing library is reported
and does not stop other metrics.
"""

from __future__ import annotations

from typing import Any

import torch
import torchaudio.functional as AF


def library_status() -> str | None:
    try:
        import pesq  # noqa: F401
    except ImportError:
        return (
            "pesq is not installed. Install it in .venvs/lab with "
            "uv pip install --python .venvs/lab/bin/python pesq"
        )
    return None


def score_pesq(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    missing = library_status()
    if missing:
        return {"status": "unavailable", "reason": missing, "value": None}
    from pesq import pesq as pesq_fn

    try:
        reference, estimate, used_rate, resampled, trimmed = _prepare(reference, estimate, sample_rate)
        mode = "wb" if used_rate == 16000 else "nb"
        value = float(
            pesq_fn(
                used_rate,
                reference.numpy().astype("float64"),
                estimate.numpy().astype("float64"),
                mode,
            )
        )
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "value": None,
        }
    return {
        "status": "ok",
        "reason": None,
        "value": value,
        "mode": mode,
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
    resampled = False
    used_rate = int(sample_rate)
    if used_rate not in (8000, 16000):
        reference = AF.resample(reference, used_rate, 16000)
        estimate = AF.resample(estimate, used_rate, 16000)
        used_rate = 16000
        resampled = True
    trimmed = abs(int(reference.numel() - estimate.numel()))
    length = min(int(reference.numel()), int(estimate.numel()))
    if length <= 0:
        raise ValueError("PESQ received empty audio")
    return reference[:length].contiguous(), estimate[:length].contiguous(), used_rate, resampled, trimmed
