"""Numeric comparison helpers for official-vs-independent waveform checks."""
import math

import torch


def align(reference, actual):
    reference = reference.detach().float().reshape(-1).cpu()
    actual = actual.detach().float().reshape(-1).cpu()
    length = min(reference.numel(), actual.numel())
    return reference[:length], actual[:length], reference.numel(), actual.numel()


def si_sdr(reference, estimate):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    energy = reference.square().sum().clamp_min(1e-12)
    projection = (estimate * reference).sum() / energy * reference
    noise = estimate - projection
    ratio = projection.square().sum() / noise.square().sum().clamp_min(1e-12)
    return float(10.0 * torch.log10(ratio))


def lag(reference, estimate, max_lag=200):
    if reference.numel() < 8:
        return 0
    step = max(1, reference.numel() // 8000)
    left = reference[::step]
    right = estimate[::step]
    left = left - left.mean()
    right = right - right.mean()
    best, best_lag = -1e30, 0
    limit = min(max_lag, left.numel() // 4)
    for shift in range(-limit, limit + 1):
        if shift < 0:
            score = float((left[-shift:] * right[:shift]).sum())
        elif shift > 0:
            score = float((left[:-shift] * right[shift:]).sum())
        else:
            score = float((left * right).sum())
        if score > best:
            best, best_lag = score, shift
    return best_lag * step


def compare(reference, actual):
    left, right, left_len, right_len = align(reference, actual)
    if left.numel() == 0:
        return {'length_ref': left_len, 'length_act': right_len, 'max_abs': None}
    diff = (left - right).abs()
    denom = left.abs().mean().clamp_min(1e-8)
    centered_l = left - left.mean()
    centered_r = right - right.mean()
    corr_den = centered_l.norm() * centered_r.norm()
    correlation = float((centered_l * centered_r).sum() / corr_den) if float(corr_den) > 0 else 1.0
    return {
        'length_ref': left_len,
        'length_act': right_len,
        'compared': int(left.numel()),
        'max_abs': float(diff.max()),
        'mae': float(diff.mean()),
        'rmse': float(diff.square().mean().sqrt()),
        'relative_mae': float(diff.mean() / denom),
        'si_sdr': si_sdr(left, right),
        'correlation': correlation,
        'lag_samples': lag(left, right),
        'nan': bool((~torch.isfinite(left)).any() or (~torch.isfinite(right)).any()),
    }


def format_row(name, stats):
    if stats.get('max_abs') is None:
        return f'{name}: empty'
    sdr = stats['si_sdr']
    sdr_text = 'inf' if math.isinf(sdr) else f'{sdr:.3f}'
    return (
        f"{name}: max={stats['max_abs']:.3e} mae={stats['mae']:.3e} "
        f"rmse={stats['rmse']:.3e} rel={stats['relative_mae']:.3e} "
        f"si-sdr={sdr_text} corr={stats['correlation']:.6f} "
        f"lag={stats['lag_samples']} len={stats['length_ref']}/{stats['length_act']} "
        f"nan={stats['nan']}"
    )
