"""Official TF-Locoformer validation.

The injected objective supplies ``valid.loss``. SI-SDR in dB is the negation
of the detached training monitor and is not added to that loss.
"""

from __future__ import annotations

import torch

from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.official.tf_locoformer.metrics import per_sample_si_snr
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.training.objectives import call_objective


class LocoformerValidator:
    """Evaluate one epoch with the injected objective and the SI-SDR monitor."""

    def __init__(self, model: PreparedModel, objective: torch.nn.Module) -> None:
        self.model = model
        self.objective = objective

    def evaluate(self, data: object, runtime: ProcessRuntime, trainer_epoch: int, component_id: str) -> dict[str, float | int]:
        module = self.model
        was_training = bool(module.training)
        cursor = (int(data.epoch), int(data.update_index))
        rng = capture_rank_rng(cuda=runtime.device.type == "cuda")
        local_loss_sum = 0.0
        local_loss_count = 0
        local_si_sum = 0.0
        local_si_count = 0
        try:
            module.eval()
            with torch.no_grad():
                for batch in data.validation_epoch(trainer_epoch):
                    if not isinstance(batch, Batch):
                        raise RuntimeFailure(f"component {component_id}: validation did not return a batch")
                    batch = runtime.move(batch)
                    try:
                        prediction = module.inference(batch.inputs)
                    except ValueError as exc:
                        raise RuntimeFailure(str(exc)) from exc
                    value = _objective_total(self.objective, prediction, batch.targets)
                    if not isinstance(value, torch.Tensor) or value.numel() != 1:
                        raise RuntimeFailure(f"component {component_id}: validation loss is not a scalar")
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeFailure(f"component {component_id}: validation loss is not finite")
                    scores = per_sample_si_snr(prediction, batch.targets)
                    weight = int(batch.inputs.shape[0])
                    if int(scores.numel()) != weight:
                        raise RuntimeFailure(f"component {component_id}: SI-SNR count does not match the batch")
                    if not bool(torch.isfinite(scores).all()):
                        raise RuntimeFailure(f"component {component_id}: validation SI-SNR is not finite")
                    local_loss_sum += float(value.detach()) * weight
                    local_loss_count += weight
                    local_si_sum += float(scores.detach().sum())
                    local_si_count += weight
            gathered = runtime.gather_objects(
                {
                    "loss_count": local_loss_count,
                    "loss_sum": local_loss_sum,
                    "si_count": local_si_count,
                    "si_sum": local_si_sum,
                }
            )
            loss_sum = 0.0
            loss_count = 0
            si_sum = 0.0
            si_count = 0
            for item in gathered:
                loss_sum += float(item["loss_sum"])
                loss_count += int(item["loss_count"])
                si_sum += float(item["si_sum"])
                si_count += int(item["si_count"])
            if loss_count < 1 or si_count != loss_count:
                raise RuntimeFailure(f"component {component_id}: validation produced no batches")
            loss = loss_sum / loss_count
            si_snr = si_sum / si_count
        finally:
            restore_rank_rng(rng, cuda=runtime.device.type == "cuda")
            if was_training:
                module.train()
            moved = (int(data.epoch), int(data.update_index))
            if moved != cursor:
                raise RuntimeFailure(f"component {component_id}: validation advanced the training cursor")
        return {"loss": loss, "si_snr": si_snr, "weight": loss_count}


def _objective_total(objective: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    try:
        result = call_objective(objective, {"estimate": prediction, "target": target})
    except ValueError as exc:
        raise RuntimeFailure(str(exc)) from exc
    return result.total
