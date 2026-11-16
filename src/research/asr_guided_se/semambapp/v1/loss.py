"""Error-weighted spectral terms and the unweighted TDT addition.

Error weighting off calls the injected objective, including mel's zero-weighted
second L1. TDT is added with coefficient 1, which does not multiply the tensor.
Adversarial, feature matching, and TDT are not frame-weighted.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from src.experiments.calc.losses import anti_wrapping
from src.experiments.calc.spectral import _difference_matrix, _slaney_log_mel
from src.experiments.errors import RuntimeFailure
from src.experiments.training.objectives import ObjectiveOutput, apply_weight, call_objective
from src.research.asr_guided_se.semambapp.v1.config import GuidanceV1

_SPECTRAL = frozenset({"magnitude", "phase", "complex", "consistency", "mel"})


def frame_weight_and_valid(
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    n_frames: int,
    hop: int,
    alpha: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``w = 1 + alpha * m``. A frame center ``j * hop`` is an error only inside the content."""

    if sample_error.ndim != 2:
        raise RuntimeFailure(f"sample_error must be [B, T], got {tuple(sample_error.shape)}")
    if content_length.ndim != 1 or content_length.shape[0] != sample_error.shape[0]:
        raise RuntimeFailure("content_length must be [B] aligned with sample_error")
    if n_frames < 1 or hop <= 0 or alpha < 0:
        raise RuntimeFailure(f"invalid frame weight n_frames={n_frames} hop={hop} alpha={alpha}")
    device = sample_error.device
    content = content_length.to(device=device, dtype=torch.long)
    centers = torch.arange(n_frames, device=device) * hop
    n_valid = torch.where(
        content > 0,
        torch.div(content, hop, rounding_mode="floor") + 1,
        torch.zeros_like(content),
    )
    frame_index = torch.arange(n_frames, device=device).view(1, -1)
    valid = frame_index < n_valid.view(-1, 1)
    inside = centers.view(1, -1) < content.view(-1, 1)
    in_tensor = centers < sample_error.shape[1]
    if sample_error.shape[1] == 0:
        belongs = torch.zeros(sample_error.shape[0], n_frames, dtype=torch.bool, device=device)
    else:
        index = centers.clamp(max=sample_error.shape[1] - 1)
        belongs = sample_error[:, index] & inside & in_tensor.view(1, -1) & valid
    weight = belongs.to(dtype=dtype) * float(alpha) + 1.0
    return weight, valid


def weighted_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor,
    time_dim: int,
) -> torch.Tensor:
    """``sum(valid * w * l) / sum(valid * w)``. Padding is in neither sum."""

    if values.shape[0] != weight.shape[0] or values.shape[time_dim] != weight.shape[1]:
        raise RuntimeFailure(
            f"time axis mismatch values {tuple(values.shape)} dim {time_dim} weight {tuple(weight.shape)}"
        )
    gate = weight.to(dtype=values.dtype) * valid.to(device=values.device, dtype=values.dtype)
    view = [1] * values.ndim
    view[0] = gate.shape[0]
    view[time_dim] = gate.shape[1]
    broadcast = gate.view(*view).expand_as(values)
    present = broadcast > 0
    masked = torch.where(present, values, torch.zeros_like(values))
    numer = (masked * broadcast).sum()
    denom = broadcast.sum()
    zero = numer * 0
    return torch.where(denom > 0, numer / denom.clamp_min(torch.finfo(values.dtype).tiny), zero)


def guided_generator_loss(
    objective: torch.nn.Module,
    context: Mapping[str, Any],
    *,
    guidance: GuidanceV1,
    sample_error: torch.Tensor | None,
    content_length: torch.Tensor | None,
    tdt: torch.Tensor | None,
    stft_hop: int,
) -> ObjectiveOutput:
    """Official objective, plus frame weights and TDT only when those switches are on."""

    if tdt is not None and (not isinstance(tdt, torch.Tensor) or tdt.ndim != 0):
        raise RuntimeFailure("TDT loss must be a scalar tensor")
    if not guidance.error_weight_enabled:
        result = call_objective(objective, context)
        if tdt is None:
            return result
        if "tdt" in result.terms:
            raise RuntimeFailure("generator objective already has a tdt term")
        added = apply_weight(tdt, 1.0)
        total = result.total + added
        return ObjectiveOutput(total, result.records + (("tdt", tdt, 1.0),))
    if sample_error is None or content_length is None:
        raise RuntimeFailure("error weighting requires a sample mask and content lengths")
    order = tuple(getattr(objective, "order", ()))
    weights = getattr(objective, "weights", None)
    if not order or not isinstance(weights, dict):
        raise RuntimeFailure("error weighting requires a weighted objective")
    total: torch.Tensor | None = None
    records: list[tuple[str, torch.Tensor, float]] = []
    spectral_weight: tuple[torch.Tensor, torch.Tensor] | None = None
    for name in order:
        module = objective.term(name)
        if name in _SPECTRAL:
            value = _spectral_term(
                name,
                module,
                context,
                sample_error,
                content_length,
                float(guidance.error_weight_alpha),
                stft_hop,
                spectral_weight,
            )
            if name != "mel":
                spectral_weight = _stft_weight(
                    context["clean_mag"],
                    sample_error,
                    content_length,
                    stft_hop,
                    float(guidance.error_weight_alpha),
                )
        else:
            value = module(context)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise RuntimeFailure(f"loss term {name} must return a scalar tensor")
        weight = float(weights[name])
        records.append((name, value, weight))
        scaled = apply_weight(value, weight)
        total = scaled if total is None else total + scaled
    if total is None:
        raise RuntimeFailure("generator objective has no terms")
    if tdt is not None:
        if "tdt" in weights:
            raise RuntimeFailure("generator objective already has a tdt term")
        added = apply_weight(tdt, 1.0)
        total = total + added
        records.append(("tdt", tdt, 1.0))
    return ObjectiveOutput(total, tuple(records))


def _stft_weight(
    like: torch.Tensor,
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    hop: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return frame_weight_and_valid(
        sample_error,
        content_length,
        int(like.shape[-1]),
        hop,
        alpha,
        like.dtype,
    )


def _spectral_term(
    name: str,
    module: torch.nn.Module,
    context: Mapping[str, Any],
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    alpha: float,
    stft_hop: int,
    cached: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if name == "mel":
        return _weighted_mel(module, context, sample_error, content_length, alpha)
    if cached is None:
        cached = _stft_weight(context["clean_mag"], sample_error, content_length, stft_hop, alpha)
    weight, valid = cached
    if name == "magnitude":
        difference = (context["clean_mag"] - context["generated_mag"]).square()
        return weighted_mean(difference, weight, valid, 2)
    if name == "phase":
        mode = getattr(module, "mode", None)
        if mode != "matrix":
            raise RuntimeFailure("error-weighted phase requires the SEMamba++ matrix mode")
        instantaneous, group_delay, angular = _phase_maps(context["clean_phase"], context["generated_phase"])
        return (
            weighted_mean(instantaneous, weight, valid, 2)
            + weighted_mean(group_delay, weight, valid, 1)
            + weighted_mean(angular, weight, valid, 2)
        )
    if name in {"complex", "consistency"}:
        scale = float(getattr(module, "scale"))
        if name == "complex":
            difference = (context["clean_complex"] - context["generated_complex"]).square()
        else:
            difference = (context["generated_complex"] - context["recomputed_complex"]).square()
        return weighted_mean(difference, weight, valid, 2) * scale
    raise RuntimeFailure(f"no weighted form for {name}")


def _phase_maps(
    reference: torch.Tensor,
    generated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frequency = _difference_matrix(int(reference.shape[-2]), reference)
    time = _difference_matrix(int(reference.shape[-1]), reference)
    group_reference = torch.matmul(reference.transpose(1, 2), frequency)
    group_generated = torch.matmul(generated.transpose(1, 2), frequency)
    angular_reference = torch.matmul(reference, time)
    angular_generated = torch.matmul(generated, time)
    return (
        anti_wrapping(reference - generated),
        anti_wrapping(group_reference - group_generated),
        anti_wrapping(angular_reference - angular_generated),
    )


def _weighted_mel(
    module: torch.nn.Module,
    context: Mapping[str, Any],
    sample_error: torch.Tensor,
    content_length: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    sample_rate = context["sample_rate"]
    reference = context["reference"]
    estimate = context["estimate"]
    if reference.ndim == 3:
        reference = reference.squeeze(1)
        estimate = estimate.squeeze(1)
    total = reference.new_zeros(())
    for bands, window in zip(module.n_mels, module.windows, strict=True):
        hop = int(window) // 4
        ref = _slaney_log_mel(reference, sample_rate, int(bands), int(window), hop, float(module.clamp))
        est = _slaney_log_mel(estimate, sample_rate, int(bands), int(window), hop, float(module.clamp))
        frames = min(ref.shape[-1], est.shape[-1])
        diff = (ref[..., :frames] - est[..., :frames]).abs()
        weight, valid = frame_weight_and_valid(
            sample_error, content_length, frames, hop, alpha, diff.dtype
        )
        total = total + weighted_mean(diff, weight, valid, -1)
    return total
