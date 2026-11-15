"""Epoch validation. Each utterance is scored alone and averaged with equal weight."""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.deepfilternet3.update import loss_tensor
from src.experiments.official.deepfilternet3.waveforms import clean_noisy
from src.experiments.runtime import ProcessRuntime
from src.experiments.trainers import _rank
from src.se.models.deepfilternet3.dsp import DFStream


def evaluate(
    runtime: ProcessRuntime,
    data: Any,
    model: Any,
    stream: DFStream | None,
    objective: torch.nn.Module,
    trainer_epoch: int,
    component_id: str,
) -> float:
    was_training = bool(model.training)
    cursor = (int(data.epoch), int(data.update_index))
    rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
    values: list[float] = []
    try:
        model.eval()
        if _rank(runtime) == 0:
            with torch.no_grad():
                for batch in data.validation_epoch(trainer_epoch):
                    if not isinstance(batch, Batch):
                        raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                    batch = runtime.move(batch)
                    clean, noisy = clean_noisy(batch)
                    for row in range(int(clean.shape[0])):
                        value = loss_tensor(
                            stream,
                            clean[row : row + 1],
                            noisy[row : row + 1],
                            model.unwrapped(),
                            objective,
                        )
                        if not bool(torch.isfinite(value).all()):
                            raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                        values.append(float(value.detach()))
        gathered = runtime.gather_objects(values)
    finally:
        restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
        if was_training:
            model.train()
        moved = (int(data.epoch), int(data.update_index))
        if moved != cursor:
            raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
    scores = next((item for item in gathered if item), None)
    if not scores:
        raise RuntimeFailure(f"component {component_id}: validation produced no utterances")
    return sum(scores) / len(scores)
