"""CUDA single-rank MP-SENet update with accumulation 1.

Order, after ``zero_grad`` and one generator forward:

1. Start PESQ on a side thread.
2. Backward the signal loss, keeping the generator graph.
3. ``drain_previous()``. The caller may flush the previous step's log here.
   It must not offer a checkpoint or touch research, selection, or progress.
4. Wait for PESQ, then discriminator backward and ``optimizer.step``.
5. Backward the metric loss and step the generator.

CPU ranks do not use this function. It does not save.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.lengths import require_discriminator_resolution
from src.experiments.official.mp_senet.models import pack_forward
from src.experiments.official.mp_senet.update_common import (
    backward_if_finite,
    inputs_ready,
    signal_context,
    waveforms,
)
from src.experiments.runtime import ProcessRuntime
from src.experiments.training.objectives import call_objective


def cuda_overlapped_step(
    update: Any,
    batch: Batch,
    runtime: ProcessRuntime,
    *,
    phase_hook: Callable[..., None],
    drain_previous: Callable[[], None],
    note_gradients_averaged: Callable[[], None],
) -> dict[str, float]:
    """PESQ runs while the signal backward uses the same generator graph."""

    phase_hook("zero_grad")
    update.optim_d.zero_grad(set_to_none=True)
    update.optim_g.zero_grad(set_to_none=True)
    clean, noisy = waveforms(batch, runtime)
    if not inputs_ready(clean, noisy, update.generator.unwrapped().n_fft):
        raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
    phase_hook("before_generator_forward")
    try:
        parts = pack_forward(update.generator(noisy, clean))
        require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
    except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
        raise RuntimeFailure(f"generator forward failed: {type(exc).__name__}: {exc}") from exc
    pending = _schedule_metric_targets(
        clean,
        parts["audio"],
        int(update.generator.unwrapped().architecture["sampling_rate"]),
    )
    ones = torch.ones(clean.shape[0], device=clean.device)
    signal = call_objective(update.signal_objective, signal_context(parts, clean))
    backward_if_finite(runtime, signal.total / update.accumulation, "generator signal loss", retain_graph=True)
    # Previous-step log I/O overlaps the rest of the PESQ thread.
    # The callback must not offer a checkpoint.
    drain_previous()
    try:
        targets = pending.result()
        real = update.discriminator(parts["clean_mag"], parts["clean_mag"])
        fake = update.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
        loss_d = call_objective(
            update.discriminator_objective,
            {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
        ).total / update.accumulation
    except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
        raise RuntimeFailure(f"discriminator forward failed: {type(exc).__name__}: {exc}") from exc
    backward_if_finite(runtime, loss_d, "discriminator loss")
    discriminator_result = runtime.step(update.optim_d)
    phase_hook("after_discriminator_step", [(clean, parts)])
    raw = update.discriminator.unwrapped().discriminator
    frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    try:
        for parameter in frozen:
            parameter.requires_grad_(False)
        try:
            metric = raw(parts["clean_mag"], parts["mag_hat"])
            metric_out = call_objective(update.metric_objective, {"metric": metric, "ones": ones})
        except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
            raise RuntimeFailure(f"generator loss failed: {type(exc).__name__}: {exc}") from exc
        backward_if_finite(runtime, metric_out.total / update.accumulation, "generator metric loss")
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
    note_gradients_averaged()
    runtime.average_gradients(update.generator)
    total = signal.total + metric_out.total
    logged = torch.stack(
        [
            signal.terms["complex"].detach(),
            signal.terms["consistency"].detach(),
            loss_d.detach() * update.accumulation,
            total.detach(),
            signal.terms["magnitude"].detach(),
            metric_out.terms["metric"].detach(),
            signal.terms["phase"].detach(),
            signal.terms["time"].detach(),
        ]
    )
    host = torch.empty(logged.shape, dtype=torch.float32, pin_memory=True)
    host.copy_(logged.detach().float(), non_blocking=True)
    generator_result = runtime.step(update.optim_g)
    runtime.end_step()
    runtime.broadcast_buffers(update.generator)
    runtime.broadcast_buffers(update.discriminator)
    if not discriminator_result.applied or not generator_result.applied:
        raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
    complex_v, consistency_v, disc_v, total_v, magnitude_v, metric_v, phase_v, time_v = (
        float(item) for item in host.tolist()
    )
    phase_hook("end_step")
    return {
        "complex": complex_v,
        "consistency": consistency_v,
        "discriminator": disc_v,
        "generator": total_v,
        "lr_d": float(update.optim_d.param_groups[0]["lr"]),
        "lr_g": float(update.optim_g.param_groups[0]["lr"]),
        "magnitude": magnitude_v,
        "metric": metric_v,
        "phase": phase_v,
        "sample_id": batch.sample_id,
        "time": time_v,
    }


def _schedule_metric_targets(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> Any:
    """Copy waveforms without blocking, then score PESQ on a side thread."""

    from src.experiments.calc.losses import metric_targets

    clean_cpu = torch.empty(clean.shape, dtype=clean.dtype, pin_memory=True)
    enhanced_cpu = torch.empty(enhanced.shape, dtype=enhanced.dtype, pin_memory=True)
    clean_cpu.copy_(clean.detach(), non_blocking=True)
    enhanced_cpu.copy_(enhanced.detach(), non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    box: dict[str, object] = {}
    done = threading.Event()

    def run() -> None:
        try:
            event.synchronize()
            box["value"] = metric_targets(clean_cpu, enhanced_cpu, sample_rate)
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run, name="mp-pesq", daemon=True)
    thread.start()

    class _Pending:
        def result(self) -> object:
            done.wait()
            if "error" in box:
                raise box["error"]
            return box.get("value")

    return _Pending()
