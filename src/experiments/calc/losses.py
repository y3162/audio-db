"""MP-SENet generator and metric-discriminator losses.

Weights match ``train.py`` at commit 89932cfe. The complex and consistency
terms are multiplied by 2 before the 0.1 weight, as in that file. A time L1
loss used by the foundation tests is separate and is not this objective.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Commit 89932cfe loss_gen_all.
MAGNITUDE_WEIGHT = 0.9
PHASE_WEIGHT = 0.3
COMPLEX_WEIGHT = 0.1
CONSISTENCY_WEIGHT = 0.1
METRIC_WEIGHT = 0.05
TIME_WEIGHT = 0.2
COMPLEX_SCALE = 2.0


def anti_wrapping(value: torch.Tensor) -> torch.Tensor:
    return torch.abs(value - torch.round(value / (2 * math.pi)) * (2 * math.pi))


def phase_losses(reference: torch.Tensor, generated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Instantaneous phase, group delay, and instantaneous angular frequency."""
    instantaneous = anti_wrapping(reference - generated).mean()
    group_delay = anti_wrapping(torch.diff(reference, dim=1) - torch.diff(generated, dim=1)).mean()
    angular = anti_wrapping(torch.diff(reference, dim=2) - torch.diff(generated, dim=2)).mean()
    return instantaneous, group_delay, angular


def generator_signal_losses(
    clean_mag: torch.Tensor,
    generated_mag: torch.Tensor,
    clean_phase: torch.Tensor,
    generated_phase: torch.Tensor,
    clean_complex: torch.Tensor,
    generated_complex: torch.Tensor,
    recomputed_complex: torch.Tensor,
    clean_audio: torch.Tensor,
    generated_audio: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Weighted signal losses, without the metric term.

    The sum is the generator objective minus ``metric * METRIC_WEIGHT``.
    A CUDA step can backpropagate this while PESQ is still running, then add
    the metric term after the discriminator update.
    """

    magnitude = F.mse_loss(clean_mag, generated_mag)
    instantaneous, group_delay, angular = phase_losses(clean_phase, generated_phase)
    phase = instantaneous + group_delay + angular
    complex_loss = F.mse_loss(clean_complex, generated_complex) * COMPLEX_SCALE
    consistency = F.mse_loss(generated_complex, recomputed_complex) * COMPLEX_SCALE
    time = F.l1_loss(clean_audio, generated_audio)
    signal = (
        magnitude * MAGNITUDE_WEIGHT
        + phase * PHASE_WEIGHT
        + complex_loss * COMPLEX_WEIGHT
        + consistency * CONSISTENCY_WEIGHT
        + time * TIME_WEIGHT
    )
    return {
        "signal": signal,
        "magnitude": magnitude,
        "phase": phase,
        "complex": complex_loss,
        "consistency": consistency,
        "time": time,
    }


def generator_losses(
    clean_mag: torch.Tensor,
    generated_mag: torch.Tensor,
    clean_phase: torch.Tensor,
    generated_phase: torch.Tensor,
    clean_complex: torch.Tensor,
    generated_complex: torch.Tensor,
    recomputed_complex: torch.Tensor,
    clean_audio: torch.Tensor,
    generated_audio: torch.Tensor,
    metric: torch.Tensor,
    ones: torch.Tensor,
) -> dict[str, torch.Tensor]:
    terms = generator_signal_losses(
        clean_mag,
        generated_mag,
        clean_phase,
        generated_phase,
        clean_complex,
        generated_complex,
        recomputed_complex,
        clean_audio,
        generated_audio,
    )
    metric_loss = F.mse_loss(metric.flatten(), ones)
    total = terms["signal"] + metric_loss * METRIC_WEIGHT
    return {
        "total": total,
        "magnitude": terms["magnitude"],
        "phase": terms["phase"],
        "complex": terms["complex"],
        "consistency": terms["consistency"],
        "metric": metric_loss,
        "time": terms["time"],
    }


def discriminator_loss(
    real_score: torch.Tensor,
    fake_score: torch.Tensor,
    ones: torch.Tensor,
    pesq_target: torch.Tensor | None,
) -> torch.Tensor:
    """Real pairs are trained toward 1. Fakes are trained toward normalized PESQ.

    When PESQ cannot be scored, the fake term is omitted. That matches the
    official branch which adds zero instead of a PESQ target. The real term
    still updates the discriminator.
    """
    loss = F.mse_loss(ones, real_score.flatten())
    if pesq_target is not None:
        loss = loss + F.mse_loss(pesq_target.to(fake_score.device), fake_score.flatten())
    return loss


def metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    """Normalized wideband PESQ, ``(score - 1) / 3.5``, or None if any item fails.

    Four or more rows are scored in a spawn process pool. Each score is the
    same ``pesq`` call as the serial path, and the row order is preserved.
    Fewer rows stay serial because the pool does not pay for itself.
    """
    if sample_rate != 16000:
        raise ValueError(f"MP-SENet metric PESQ is defined at 16 kHz, got {sample_rate}")
    clean_rows = clean.detach().cpu()
    enhanced_rows = enhanced.detach().cpu()
    pairs = [
        (sample_rate, reference.numpy(), estimate.numpy())
        for reference, estimate in zip(clean_rows, enhanced_rows, strict=True)
    ]
    if len(pairs) >= 4:
        scores = list(_pesq_pool().map(_pesq_wideband, pairs))
    else:
        scores = [_pesq_wideband(pair) for pair in pairs]
    if any(score < 0.0 for score in scores):
        return None
    return torch.tensor([(score - 1.0) / 3.5 for score in scores], dtype=torch.float32)


def _pesq_wideband(item: tuple) -> float:
    sample_rate, reference, estimate = item
    from pesq import pesq

    try:
        return float(pesq(sample_rate, reference, estimate, "wb"))
    except Exception:
        return -1.0


_PESQ_POOL = None


def _pesq_pool():
    global _PESQ_POOL
    if _PESQ_POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        _PESQ_POOL = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"))
    return _PESQ_POOL
