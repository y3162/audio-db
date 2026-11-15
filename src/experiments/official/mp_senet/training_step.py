"""One MP-SENet GAN update.

``run`` picks one of two paths. CPU, more than one rank, and CUDA with
accumulation above 1 use :func:`~src.experiments.official.mp_senet.update_accumulated.accumulated_step`.
CUDA with one rank and one micro-batch uses
:func:`~src.experiments.official.mp_senet.update_cuda_overlap.cuda_overlapped_step`.

Both paths keep one generator forward until the discriminator has stepped.
The generator backward then uses that graph and the updated discriminator.
The CPU path backwards the discriminator first, then the sum of the signal
and metric losses. The CUDA path overlaps PESQ with the signal backward,
then steps the discriminator, then the metric term, then the generator.

This module does not own epochs, validation, schedulers, best selection,
research evaluation, the step journal, or checkpoints. It does not accept a
trainer. The caller passes the models, optimizers, objectives, and callables:

- ``phase_hook(phase, payload=None)`` at ``zero_grad``,
  ``before_generator_forward``, ``after_discriminator_step`` (the retained
  generator graphs), and ``end_step``. It must not save or advance progress.
- ``drain_previous()`` only on the CUDA overlap path, after the signal
  backward and before waiting for PESQ. The caller may flush the previous
  step's log there. It must not offer a checkpoint.
- ``note_generator_sync(enabled)`` when a distributed generator backward
  sees a sync flag. The CUDA overlap path does not call it.
- ``note_gradients_averaged()`` immediately before ``average_gradients``.

``run`` returns only after both optimizers have stepped and ``end_step`` has
returned. A failure leaves that pair unfinished. ``amp_fp16`` is refused
because the scaler would decide overflow per optimizer.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mp_senet.update_accumulated import accumulated_step
from src.experiments.official.mp_senet.update_cuda_overlap import cuda_overlapped_step
from src.experiments.runtime import PreparedModel, ProcessRuntime


def refuse_gan_amp(runtime: ProcessRuntime) -> None:
    """Refuse FP16 AMP. The two AdamW steps are not one scaler decision."""

    if getattr(runtime, "precision", "fp32") == "amp_fp16":
        raise RuntimeFailure(
            "MP-SENet official GAN refuses amp_fp16. "
            "The current scaler decides overflow per optimizer, so the discriminator "
            "step and the generator step are not one atomic update. FP32 is unchanged."
        )


class MPSENetUpdate:
    """Discriminator-then-generator step for the injected MP-SENet objectives."""

    def __init__(
        self,
        *,
        generator: PreparedModel,
        discriminator: PreparedModel,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        signal_objective: torch.nn.Module,
        metric_objective: torch.nn.Module,
        discriminator_objective: torch.nn.Module,
        accumulation: int,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.signal_objective = signal_objective
        self.metric_objective = metric_objective
        self.discriminator_objective = discriminator_objective
        self.accumulation = accumulation

    def run(
        self,
        batches: list[Batch],
        runtime: ProcessRuntime,
        *,
        phase_hook: Callable[..., None],
        drain_previous: Callable[[], None],
        note_generator_sync: Callable[[bool], None],
        note_gradients_averaged: Callable[[], None],
    ) -> dict[str, float]:
        """Apply one official pair, or raise before either optimizer is a safe point."""

        refuse_gan_amp(runtime)
        if len(batches) != self.accumulation:
            raise RuntimeFailure(
                f"official step received {len(batches)} micro-batches, accumulation is {self.accumulation}"
            )
        counts = runtime.gather_objects(len(batches))
        if any(item != len(batches) for item in counts):
            raise RuntimeFailure("ranks disagree on the number of micro-batches")
        self.generator.train()
        self.discriminator.train()
        if runtime.device.type == "cuda" and runtime.world_size == 1 and len(batches) == 1:
            return cuda_overlapped_step(
                self,
                batches[0],
                runtime,
                phase_hook=phase_hook,
                drain_previous=drain_previous,
                note_gradients_averaged=note_gradients_averaged,
            )
        return accumulated_step(
            self,
            batches,
            runtime,
            phase_hook=phase_hook,
            note_generator_sync=note_generator_sync,
            note_gradients_averaged=note_gradients_averaged,
        )
