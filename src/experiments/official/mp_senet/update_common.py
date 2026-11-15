"""Shared pieces of one MP-SENet GAN update.

CPU and CUDA call these helpers. They do not choose which loss runs first,
and they do not synchronize ranks. The two update modules own that order.
"""

from __future__ import annotations

import torch

from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.runtime import ProcessRuntime

LOSS_NAMES = ("total", "magnitude", "phase", "complex", "consistency", "metric", "time")


def signal_context(parts: dict[str, torch.Tensor], clean: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the signal-term context from one generator forward."""

    return {
        "clean_audio": clean,
        "clean_complex": parts["clean_complex"],
        "clean_mag": parts["clean_mag"],
        "clean_phase": parts["clean_phase"],
        "generated_audio": parts["audio"],
        "generated_complex": parts["complex"],
        "generated_mag": parts["mag"],
        "generated_phase": parts["phase"],
        "recomputed_complex": parts["recomputed_complex"],
    }


def metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    """Look up PESQ targets on the losses module so tests can patch that function."""

    from src.experiments.calc import losses as loss_lib

    return loss_lib.metric_targets(clean, enhanced, sample_rate)


def inputs_ready(clean: torch.Tensor, noisy: torch.Tensor, n_fft: int) -> bool:
    """Accept a matching pair long enough for the STFT. CPU also requires finite values."""

    if clean.ndim != 2 or clean.shape != noisy.shape:
        return False
    if int(clean.shape[-1]) <= n_fft // 2:
        return False
    if clean.is_cuda:
        # A non-finite waveform makes the loss non-finite, which rejects the step.
        return True
    return bool(torch.isfinite(clean).all()) and bool(torch.isfinite(noisy).all())


def waveforms(batch: Batch, runtime: ProcessRuntime) -> tuple[torch.Tensor, torch.Tensor]:
    batch = runtime.move(batch)
    clean = batch.targets
    noisy = batch.inputs
    if clean.ndim != 2 or clean.shape != noisy.shape:
        raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(clean.shape)}")
    return clean, noisy


def backward_if_finite(
    runtime: ProcessRuntime,
    loss: torch.Tensor,
    what: str,
    *,
    retain_graph: bool = False,
) -> None:
    if runtime.device.type != "cuda":
        finite = bool(torch.isfinite(loss.detach()).all())
        if not runtime.agree(finite, policy="all"):
            raise RuntimeFailure(f"{what} is not finite on every rank")
    runtime.backward(loss, retain_graph=retain_graph)
