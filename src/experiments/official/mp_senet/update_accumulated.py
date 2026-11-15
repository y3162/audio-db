"""CPU and distributed MP-SENet update, and CUDA when accumulation is above 1.

Order for every micro-batch, then once for the window:

1. Generator forward with gradient sync held off. The graph stays.
2. Discriminator forward and backward. Sync is on only for the last batch.
3. Discriminator ``optimizer.step``.
4. Generator backward of the signal-plus-metric loss, using the updated
   discriminator and the stored generator graph.
5. ``average_gradients`` on the generator, then the generator step.

``sync_gradients`` wraps the forward and the backward that belongs to it.
This module chooses when that flag is on. It does not implement the reduction.
It does not save, and it does not flush a previous step's log.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.lengths import require_discriminator_resolution
from src.experiments.official.mp_senet.models import pack_forward
from src.experiments.official.mp_senet.update_common import (
    LOSS_NAMES,
    backward_if_finite,
    inputs_ready,
    metric_targets,
    signal_context,
    waveforms,
)
from src.experiments.runtime import ProcessRuntime
from src.experiments.training.objectives import call_objective


def accumulated_step(
    update: Any,
    batches: list[Batch],
    runtime: ProcessRuntime,
    *,
    phase_hook: Callable[..., None],
    note_generator_sync: Callable[[bool], None],
    note_gradients_averaged: Callable[[], None],
) -> dict[str, float]:
    """Discriminator micro-batches, one discriminator step, then the generator."""

    phase_hook("zero_grad")
    update.optim_d.zero_grad(set_to_none=True)
    update.optim_g.zero_grad(set_to_none=True)
    stored: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    discriminator_total = 0.0
    sample_ids: list[str] = []
    with runtime.sync_gradients([update.generator], enabled=False):
        for index, batch in enumerate(batches):
            last = index == len(batches) - 1
            clean, noisy = waveforms(batch, runtime)
            sample_ids.append(batch.sample_id)
            ready = inputs_ready(clean, noisy, update.generator.unwrapped().n_fft)
            if not runtime.agree(ready, policy="all"):
                raise RuntimeFailure("a rank rejected its micro-batch before the generator forward")
            phase_hook("before_generator_forward")
            error: str | None = None
            parts: dict[str, torch.Tensor] | None = None
            try:
                parts = pack_forward(update.generator(noisy, clean))
                require_discriminator_resolution(parts["clean_mag"].shape[-2], parts["clean_mag"].shape[-1])
            except (TrainingError, RuntimeFailure, RuntimeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors = runtime.gather_objects(error)
            if any(errors):
                raise RuntimeFailure("generator forward failed: " + "; ".join(item for item in errors if item))
            if parts is None:
                raise RuntimeFailure("generator forward produced no tensors")
            stored.append((clean, parts))
            ones = torch.ones(clean.shape[0], device=clean.device)
            with runtime.sync_gradients([update.discriminator], enabled=last):
                disc_error: str | None = None
                loss: torch.Tensor | None = None
                try:
                    real = update.discriminator(parts["clean_mag"], parts["clean_mag"])
                    fake = update.discriminator(parts["clean_mag"], parts["mag_hat"].detach())
                    targets = metric_targets(
                        clean, parts["audio"], int(update.generator.unwrapped().architecture["sampling_rate"])
                    )
                    loss = call_objective(
                        update.discriminator_objective,
                        {"real_score": real, "fake_score": fake, "ones": ones, "pesq_target": targets},
                    ).total / update.accumulation
                except (TrainingError, RuntimeError, ValueError) as exc:
                    disc_error = f"{type(exc).__name__}: {exc}"
                disc_errors = runtime.gather_objects(disc_error)
                if any(disc_errors):
                    raise RuntimeFailure(
                        "discriminator forward failed: " + "; ".join(item for item in disc_errors if item)
                    )
                if loss is None:
                    raise RuntimeFailure("discriminator loss is missing")
                backward_if_finite(runtime, loss, "discriminator loss")
                discriminator_total += float(loss.detach()) * update.accumulation
        discriminator_result = runtime.step(update.optim_d)
        phase_hook("after_discriminator_step", stored)
        raw = update.discriminator.unwrapped().discriminator
        frozen = [parameter for parameter in raw.parameters() if parameter.requires_grad]
        component_totals = {name: 0.0 for name in LOSS_NAMES}
        try:
            for parameter in frozen:
                parameter.requires_grad_(False)
            for clean, parts in stored:
                ones = torch.ones(clean.shape[0], device=clean.device)
                gen_error: str | None = None
                total: torch.Tensor | None = None
                try:
                    metric = raw(parts["clean_mag"], parts["mag_hat"])
                    signal = call_objective(update.signal_objective, signal_context(parts, clean))
                    metric_out = call_objective(update.metric_objective, {"metric": metric, "ones": ones})
                    total_tensor = signal.total + metric_out.total
                    losses = {
                        "total": total_tensor,
                        "magnitude": signal.terms["magnitude"],
                        "phase": signal.terms["phase"],
                        "complex": signal.terms["complex"],
                        "consistency": signal.terms["consistency"],
                        "metric": metric_out.terms["metric"],
                        "time": signal.terms["time"],
                    }
                    total = total_tensor / update.accumulation
                except (TrainingError, RuntimeError, ValueError) as exc:
                    gen_error = f"{type(exc).__name__}: {exc}"
                gen_errors = runtime.gather_objects(gen_error)
                if any(gen_errors):
                    raise RuntimeFailure(
                        "generator loss failed: " + "; ".join(item for item in gen_errors if item)
                    )
                if total is None:
                    raise RuntimeFailure("generator loss is missing")
                synced = update.generator.gradient_sync_enabled()
                if synced is not None:
                    note_generator_sync(synced)
                backward_if_finite(runtime, total, "generator loss")
                for name in LOSS_NAMES:
                    component_totals[name] += float(losses[name].detach())
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
        note_gradients_averaged()
        runtime.average_gradients(update.generator)
    generator_result = runtime.step(update.optim_g)
    runtime.end_step()
    runtime.broadcast_buffers(update.generator)
    runtime.broadcast_buffers(update.discriminator)
    if not discriminator_result.applied or not generator_result.applied:
        raise RuntimeFailure("MP-SENet GAN requires both optimizers to apply in one logical step")
    count = len(batches)
    phase_hook("end_step")
    return {
        "complex": component_totals["complex"] / count,
        "consistency": component_totals["consistency"] / count,
        "discriminator": discriminator_total / count,
        "generator": component_totals["total"] / count,
        "lr_d": float(update.optim_d.param_groups[0]["lr"]),
        "lr_g": float(update.optim_g.param_groups[0]["lr"]),
        "magnitude": component_totals["magnitude"] / count,
        "metric": component_totals["metric"] / count,
        "phase": component_totals["phase"] / count,
        "sample_id": ",".join(sample_ids),
        "time": component_totals["time"] / count,
    }
