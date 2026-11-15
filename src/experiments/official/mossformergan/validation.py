"""MossFormerGAN validation. Failed PESQ batches stay in the denominator."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.mossformergan.reference import reduce_validation, validation_report
from src.experiments.official.mossformergan.view import OfficialCall
from src.experiments.runtime import ProcessRuntime


def evaluate(
    runtime: ProcessRuntime,
    data: Any,
    generator: Any,
    discriminator: Any,
    *,
    trainer_epoch: int,
    n_fft: int,
    hop: int,
    win: int,
    sample_rate: int,
) -> dict[str, Any]:
    cursor = (int(data.epoch), int(data.update_index))
    rng = runtime.capture_rng()
    summary: dict[str, Any] | None = None
    try:
        produced = list(data.validation_epoch(trainer_epoch))
        aligned = align_validation(runtime, produced, discriminator)
        proxy = OfficialCall(
            n_fft=n_fft,
            hop=hop,
            win=win,
            sample_rate=sample_rate,
            generator=generator.execution(),
            discriminator=discriminator.execution(),
            device=runtime.device,
            validation_requires_all_ranks=_syncbn_active(discriminator),
        )
        reports: list[dict[str, Any]] = []
        for batch, alignment in aligned:
            if not isinstance(batch, Batch):
                raise RuntimeFailure("validation did not return a batch")
            moved = runtime.move(batch)
            payload = {"clean": moved.targets, "noisy": moved.inputs, "validation_alignment_pad": alignment}
            try:
                reports.append(validation_report(proxy, payload))
            except TrainingError as exc:
                raise RuntimeFailure(f"validation report failed: {exc}") from exc
        try:
            summary = reduce_validation(proxy, reports, collective=runtime.world_size > 1)
        except TrainingError as exc:
            raise RuntimeFailure(f"validation reduction failed: {exc}") from exc
    finally:
        runtime.restore_rng(rng)
        generator.train()
        discriminator.train()
    runtime.broadcast_buffers(generator)
    runtime.broadcast_buffers(discriminator)
    moved = (int(data.epoch), int(data.update_index))
    if moved != cursor:
        raise RuntimeFailure("validation advanced the training cursor")
    if not isinstance(summary, dict):
        raise RuntimeFailure("validation did not return a summary")
    return summary


def validation_proxy(runtime: ProcessRuntime, generator: Any, discriminator: Any, *, n_fft: int, hop: int, win: int, sample_rate: int) -> OfficialCall:
    return OfficialCall(
        n_fft=n_fft, hop=hop, win=win, sample_rate=sample_rate,
        generator=generator.execution(), discriminator=discriminator.execution(),
        device=runtime.device, validation_requires_all_ranks=_syncbn_active(discriminator),
    )


def align_validation(
    runtime: ProcessRuntime,
    batches: list[Batch],
    discriminator: Any,
) -> list[tuple[Batch, bool]]:
    if runtime.world_size == 1 or not _syncbn_active(discriminator):
        return [(batch, False) for batch in batches]
    counts = runtime.gather_objects(len(batches))
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in counts):
        raise RuntimeFailure("validation batch counts are not integers")
    if any(item == 0 for item in counts):
        rendered = ", ".join(str(item) for item in counts)
        raise RuntimeFailure(f"validation rank has no batch to repeat for SyncBatchNorm alignment: {rendered}")
    target = max(int(item) for item in counts)
    if all(int(item) == int(counts[0]) for item in counts):
        return [(batch, False) for batch in batches]
    padded = [(batch, False) for batch in batches]
    source = 0
    while len(padded) < target:
        padded.append((batches[source % len(batches)], True))
        source += 1
    return padded


def _syncbn_active(discriminator: Any) -> bool:
    return any(isinstance(module, torch.nn.SyncBatchNorm) for module in discriminator.unwrapped().modules())
