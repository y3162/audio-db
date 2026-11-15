"""One DeepFilterNet3 accumulation. The cosine row is already on the optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.deepfilternet3.training_step import component_losses
from src.experiments.official.deepfilternet3.waveforms import clean_noisy
from src.experiments.runtime import ProcessRuntime
from src.se.models.deepfilternet3.dsp import DFStream


@dataclass(frozen=True)
class DFStep:
    loss: float
    applied: bool
    sample_id: str


def run_accumulation(
    model: Any,
    optimizer: torch.optim.Optimizer,
    runtime: ProcessRuntime,
    view: Any,
    index: int,
    *,
    accumulation: int,
    grad_clip: float,
    component_id: str,
    stream: DFStream | None,
    objective: torch.nn.Module,
) -> DFStep:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total: torch.Tensor | None = None
    sample_ids: list[str] = []
    for offset in range(accumulation):
        sync = offset == accumulation - 1
        batches = view.batch(index + offset)
        sample_ids.extend(batch.sample_id for batch in batches)
        with runtime.sync_gradients([model], enabled=sync):
            with runtime.autocast():
                micro = forward_loss(stream, model, objective, batches, component_id, runtime)
            runtime.backward(micro / accumulation)
        detached = (micro / accumulation).detach()
        total = detached if total is None else total + detached
    runtime.clip(model, grad_clip, optimizer=optimizer)
    result = runtime.step(optimizer)
    runtime.end_step()
    if total is None or not bool(torch.isfinite(total).all()):
        raise RuntimeFailure(f"component {component_id}: loss is not finite")
    return DFStep(float(total), bool(result.applied), ",".join(sample_ids))


def forward_loss(
    stream: DFStream | None,
    model: Any,
    objective: torch.nn.Module,
    batches: list[Batch],
    component_id: str,
    runtime: ProcessRuntime,
) -> torch.Tensor:
    if not batches:
        raise RuntimeFailure(f"component {component_id}: empty micro-batch")
    total: torch.Tensor | None = None
    for batch in batches:
        batch = runtime.move(batch)
        clean, noisy = clean_noisy(batch)
        value = loss_tensor(stream, clean, noisy, model.execution(), objective)
        total = value if total is None else total + value
    assert total is not None
    return total / len(batches)


def loss_tensor(
    stream: DFStream | None,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    module: torch.nn.Module,
    objective: torch.nn.Module,
) -> torch.Tensor:
    if stream is None:
        raise RuntimeFailure("DeepFilterNet3 stream is not ready")
    try:
        parts = component_losses(stream, module, clean, noisy, objective)
    except TrainingError as exc:
        raise RuntimeFailure(str(exc)) from exc
    total = parts["total"]
    if not bool(torch.isfinite(total).all()):
        raise RuntimeFailure("deepfilternet loss is not finite")
    return total


def stream_for(model: Any, device: torch.device) -> DFStream:
    """Open a DFStream from the model's own sample rate and FFT sizes."""

    params = model.generator.params
    return DFStream(
        params.sr,
        params.fft_size,
        params.hop_size,
        params.nb_erb,
        params.nb_df,
        params.min_nb_freqs,
        params.norm_tau,
        device,
    )
