"""TF-Locoformer validation SI-SDR.

Training stores the negative SI-SDR loss. Validation stores this positive
value and does not put it on the backward path.
"""

from __future__ import annotations

import torch


def per_sample_si_snr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB for each batch row.

    The batch mean of this tensor is the negation of ``si_sdr_loss``. The
    training log stores that loss. Validation stores this positive value.
    """

    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    target_energy = torch.sum(reference.square(), dim=-1, keepdim=True).clamp_min(eps)
    projection = dot * reference / target_energy
    noise = estimate - projection
    ratio = torch.sum(projection.square(), dim=-1) / torch.sum(noise.square(), dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))
