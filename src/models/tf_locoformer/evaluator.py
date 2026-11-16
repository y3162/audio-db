"""TF-Locoformer validation, SI-SNR ranking, and n-best inference files.

The injected objective supplies the validation loss. Per-sample SI-SNR is the
second selection statistic. This module does not step the scheduler and does
not offer a checkpoint.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from src.experiments.calc.average import average_checkpoint_files, remove_symlink, replace_symlink
from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng
from src.experiments.data import Batch
from src.experiments.errors import RuntimeFailure
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.training.objectives import call_objective

# --- Validation. Full batches, injected objective total, mean SI-SNR. ---

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

# --- SI-SNR used by validation. It is not the training objective. ---

def per_sample_si_snr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB for each batch row.

    The batch mean of this tensor is the negation of ``si_sdr_loss``. The
    training log stores that loss. Validation stores this positive value.
    """

    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    target_energy = torch.sum(reference.square(), dim=-1, keepdim=True).clamp_min(eps)
    projection = dot * reference / target_energy
    noise = estimate - projection
    ratio = torch.sum(projection.square(), dim=-1) / torch.sum(noise.square(), dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))

# --- valid.si_snr is maximized. valid.loss ranking stays in TF-GridNet. ---

def rank_validation_si_snr(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Highest ``si_snr`` first. Equal scores keep the earlier epoch.

    Rows without ``si_snr`` are omitted. The value is SI-SDR in dB.
    """

    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise RuntimeFailure("keep_nbest must be a positive integer")
    ranked = [row for row in history if row.get("si_snr") is not None]
    ordered = sorted(ranked, key=lambda row: (-float(row["si_snr"]), int(row["epoch"])))
    return ordered[:keep]

# --- Official selection files. Rank 0 writes; every rank waits. ---

_EPOCH_FILE = re.compile(r"^(\d+)epoch\.pt$")


def write_selection(
    directory: Path,
    exported: dict[str, torch.Tensor],
    espnet_epoch: int,
    by_loss: list[dict[str, Any]],
    by_si: list[dict[str, Any]],
    keep: int,
    logical_step: int,
    validation_profile: str,
) -> None:
    """Write the epoch export, loss n-best, and SI-SNR best. Rank 0 only.

    Loss uses the ESPnet n-best average. SI-SNR keeps its own best file and
    protects its n-best epoch files. It does not write an average. Equal
    scores already resolved to the earlier epoch in the ranked lists.
    """

    if not by_loss:
        raise RuntimeFailure("n-best selection received an empty loss history")
    directory.mkdir(parents=True, exist_ok=True)
    _save_state(directory / f"{espnet_epoch}epoch.pt", exported)
    loss_best = by_loss[0]
    if int(loss_best["epoch"]) == int(espnet_epoch):
        _save_state(directory / "valid.loss.best.pt", exported)
        _write_json(
            directory / "valid.loss.best.pt.json",
            {
                "criterion": "valid.loss.min",
                "epoch": int(espnet_epoch),
                "global_step": int(logical_step),
                "metric": "loss",
                "validation_profile": validation_profile,
                "value": float(loss_best["loss"]),
            },
        )
    if by_si and int(by_si[0]["epoch"]) == int(espnet_epoch):
        _save_state(directory / "valid.si_snr.best.pt", exported)
        _write_json(
            directory / "valid.si_snr.best.pt.json",
            {
                "criterion": "valid.si_snr.max",
                "epoch": int(espnet_epoch),
                "global_step": int(logical_step),
                "metric": "si_snr",
                "sign": "si_sdr_db",
                "validation_profile": validation_profile,
                "value": float(by_si[0]["si_snr"]),
            },
        )
    kept = {int(row["epoch"]) for row in by_loss + by_si}
    for path in directory.glob("*epoch.pt"):
        match = _EPOCH_FILE.match(path.name)
        if match and int(match.group(1)) not in kept:
            path.unlink()
    best_name = f"{int(loss_best['epoch'])}epoch.pt"
    if keep > 1 and len(by_loss) >= keep:
        epochs = [int(row["epoch"]) for row in by_loss]
        averaged = average_checkpoint_files([directory / f"{epoch}epoch.pt" for epoch in epochs])
        average_name = f"valid.loss.ave_{keep}best.pt"
        _save_state(directory / average_name, averaged)
        _write_json(
            directory / f"{average_name}.json",
            {
                "average": "espnet_nbest_float_mean_int_sum",
                "criterion": "valid.loss.min",
                "epochs": epochs,
                "files": [f"{epoch}epoch.pt" for epoch in epochs],
                "global_step": int(logical_step),
                "metric": "loss",
                "validation_profile": validation_profile,
                "values": [float(row["loss"]) for row in by_loss],
            },
        )
        replace_symlink(directory / "valid.loss.ave.pt", average_name)
        # Left from the epochs before n-best existed. Its epoch file may be gone by now.
        remove_symlink(directory / "valid.loss.ave_1best.pt")
        return
    replace_symlink(directory / "valid.loss.ave_1best.pt", best_name)
    replace_symlink(directory / "valid.loss.ave.pt", "valid.loss.ave_1best.pt")


def _save_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
