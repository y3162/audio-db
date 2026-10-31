"""Scale-invariant SDR. No extra package is required."""

from __future__ import annotations

import math
from typing import Any

import torch


def score_si_sdr(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().cpu().to(torch.float64).reshape(-1)
    estimate = estimate.detach().cpu().to(torch.float64).reshape(-1)
    length = min(int(reference.numel()), int(estimate.numel()))
    if length <= 0:
        return {"status": "error", "reason": "empty audio", "value": None, "trimmed_samples": 0}
    trimmed = abs(int(reference.numel() - estimate.numel()))
    reference = reference[:length]
    estimate = estimate[:length]
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    reference_energy = torch.dot(reference, reference)
    if float(reference_energy) < 1e-12:
        return {
            "status": "na",
            "reason": "reference_has_zero_energy",
            "value": None,
            "trimmed_samples": trimmed,
        }
    projection = torch.dot(estimate, reference) / reference_energy * reference
    noise = estimate - projection
    noise_energy = torch.dot(noise, noise)
    projection_energy = torch.dot(projection, projection)
    if float(noise_energy) < 1e-12 * float(projection_energy.clamp_min(1e-12)):
        return {
            "status": "ok",
            "reason": "numerically_infinite",
            "value": None,
            "trimmed_samples": trimmed,
        }
    value = 10.0 * math.log10(float(projection_energy) / float(noise_energy))
    if not math.isfinite(value):
        return {
            "status": "ok",
            "reason": "numerically_infinite",
            "value": None,
            "trimmed_samples": trimmed,
        }
    return {"status": "ok", "reason": None, "value": value, "trimmed_samples": trimmed}
