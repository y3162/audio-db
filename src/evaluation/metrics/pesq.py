"""PESQ wideband/narrowband scoring.

A missing clean reference is not a score of zero. A missing library is reported
and does not stop other metrics.

``pesq==0.0.4`` is the ITU-T P.862 / P.862.2 reference code. Wideband uses
``Downsample = 64``. The loader then adds ``2 * 75 * 64`` samples of search
padding, which is itself a multiple of 64, so the VAD length
``(N + 9600) // 64`` increases exactly when ``N`` crosses a multiple of 64.
In one observed failure the new bin changed the detected utterances and delay,
and the MOS jumped by multiple points. The guard compares the full score with
the last length before the preceding bin; these lengths can differ by up to 64
samples. A large difference may also vary gradually within the bin, without a
detected split or delay change. It is a conservative length-sensitivity flag,
not proof that an utterance split occurred. The raw score is not edited and the
delay is not forced to zero. Differences larger than ``SPLIT_ABS_TOL`` remain
``na`` because the full-length value's reliability has not been established.
"""

from __future__ import annotations

from typing import Any
import math

import torch
import torchaudio.functional as AF

# One-bin crop versus the raw score on 16 time-aligned VoiceBank pairs stayed
# within 0.014. The unaligned failure jumps by about 3.3. 0.1 sits in that gap.
SPLIT_ABS_TOL = 0.1
_DOWNSAMPLE = {8000: 32, 16000: 64}


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
        if not torch.isfinite(reference).all() or not torch.isfinite(estimate).all():
            raise ValueError("non-finite audio")
        if not torch.any(reference != 0) or not torch.any(estimate != 0):
            return {"status": "na", "reason": "zero_energy_audio", "value": None}
        mode = "wb" if used_rate == 16000 else "nb"
        value = float(
            pesq_fn(
                used_rate,
                reference.numpy().astype("float64"),
                estimate.numpy().astype("float64"),
                mode,
            )
        )
        if not math.isfinite(value):
            raise ValueError("PESQ returned a non-finite value")
        probe = previous_downsample_length(int(reference.numel()), _DOWNSAMPLE[used_rate])
        boundary_value = None
        if probe >= used_rate // 4:
            boundary_value = float(
                pesq_fn(
                    used_rate,
                    reference[:probe].numpy().astype("float64"),
                    estimate[:probe].numpy().astype("float64"),
                    mode,
                )
            )
            if not math.isfinite(boundary_value):
                raise ValueError("PESQ returned a non-finite boundary value")
            if abs(value - boundary_value) > SPLIT_ABS_TOL:
                return {
                    "status": "na",
                    "reason": "pesq_utterance_split_discontinuity",
                    "value": None,
                    "raw_value": value,
                    "boundary_value": boundary_value,
                    "boundary_length": probe,
                    "mode": mode,
                    "sample_rate": used_rate,
                    "resampled": resampled,
                    "trimmed_samples": trimmed,
                }
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
        "boundary_value": boundary_value,
        "mode": mode,
        "sample_rate": used_rate,
        "resampled": resampled,
        "trimmed_samples": trimmed,
    }


def previous_downsample_length(length: int, downsample: int) -> int:
    """Last length whose wideband or narrowband VAD bin is the previous one.

    Search padding is an integer number of downsample bins, so the bin index
    of an input of length ``N`` changes only when ``N`` is a multiple of
    ``downsample``.
    """
    if downsample < 1:
        raise ValueError("downsample must be positive")
    return (int(length) // downsample) * downsample - 1


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
