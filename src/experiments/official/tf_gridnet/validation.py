"""Official TF-GridNet validation.

Rank-local totals are gathered and divided by the sample count. The injected
objective supplies the scalar. This module does not step a scheduler, rank
n-best files, or move the training cursor.
"""

from __future__ import annotations

import torch

from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.training.objectives import call_objective


class GridNetValidator:
    """Evaluate one epoch with the injected objective."""

    def __init__(self, model: PreparedModel, objective: torch.nn.Module) -> None:
        self.model = model
        self.objective = objective

    def evaluate(self, data: object, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, float | int]:
        """Return the weighted loss. The training cursor and RNG are restored."""

        module = self.model
        was_training = bool(module.training)
        cursor = (int(data.epoch), int(data.update_index))
        rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
        local_sum = 0.0
        local_count = 0
        try:
            module.eval()
            with torch.no_grad():
                for batch in data.validation_epoch(trainer_epoch):
                    if not isinstance(batch, Batch):
                        raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                    batch = runtime.move(batch)
                    prediction = module.inference(batch.inputs)
                    value = _objective_total(self.objective, prediction, batch.targets)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise RuntimeFailure(f"component {component_id}: validation loss is not a scalar")
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                    weight = int(batch.inputs.shape[0])
                    local_sum += float(value.detach()) * weight
                    local_count += weight
            gathered = runtime.gather_objects({"count": local_count, "sum": local_sum})
            total = 0.0
            count = 0
            for item in gathered:
                total += float(item["sum"])
                count += int(item["count"])
            if count < 1:
                raise RuntimeFailure(f"component {component_id}: validation produced no batches")
            loss = total / count
        finally:
            restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
            if was_training:
                module.train()
            moved = (int(data.epoch), int(data.update_index))
            if moved != cursor:
                raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return {"loss": loss, "weight": count}


def _objective_total(objective: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the total of the injected objective."""

    try:
        result = call_objective(objective, {"estimate": prediction, "target": target})
    except ValueError as exc:
        raise RuntimeFailure(str(exc)) from exc
    return result.total
