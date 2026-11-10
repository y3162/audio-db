"""TF-GridNet: Trainer.run versus Task.train_update versus the ESPnet separator.

The 4096-sample batch-1 run is the condition already compared for the
component step. A following 48000-sample batch-1 step is closer to the
official chunk and is not the official batch of 4.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-continue"))

from src.experiments.checkpoint import CheckpointManager  # noqa: E402
from src.experiments.data import Batch  # noqa: E402
from src.experiments.losses import MultiResolutionL1Loss  # noqa: E402
from src.experiments.official.tf_gridnet.model import TFGridNetModel  # noqa: E402
from src.experiments.official.tf_gridnet.reference import TFGridNetTask  # noqa: E402
from src.experiments.official.tf_gridnet.trainer import TFGridNetTrainer  # noqa: E402
from src.experiments.runtime import PreparePolicy, ProcessRuntime  # noqa: E402
from tests.numerics.reference.compare_tf_gridnet import (  # noqa: E402
    ARCHITECTURE,
    CLIP,
    LR,
    BETAS,
    _adam,
    _enhance,
    _official_loss_class,
    _official_value,
    _pair,
    _step,
    _task_config,
)
from tests.numerics.reference.common import weight_report  # noqa: E402


class _Epoch:
    def __init__(self, batches: list[Batch]) -> None:
        self.updates = len(batches) + 4
        self._batches = batches

    def batch(self, index: int) -> list[Batch]:
        return [self._batches[index]]


class _Data:
    group_size = 1
    epoch = 0
    update_index = 0

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches

    def train_epoch(self, epoch: int, *, start_update: int):
        del epoch, start_update
        return _Epoch(self._batches)

    def validation_epoch(self, epoch: int):
        del epoch
        return iter(())


def _batch(noisy: torch.Tensor, clean: torch.Tensor, index: int) -> Batch:
    return Batch(inputs=noisy, targets=clean, sample_id=f"g{index}", realization_id=f"g{index}")


def _waves(length: int, count: int, device: torch.device):
    generator = torch.Generator().manual_seed(3)
    clean = torch.randn(count, length, generator=generator)
    noisy = clean * 0.5 + 0.05 * torch.randn(count, length, generator=generator)
    return clean.to(device), noisy.to(device)


def _trainer_run(device, initial, clean, noisy, steps: int) -> dict:
    policy = PreparePolicy(trainable=True, distribute="none")
    runtime = ProcessRuntime(device="cuda" if device.type == "cuda" else "cpu", precision="fp32", allow_busy_gpu=True)
    model = runtime.prepare_model(TFGridNetModel(seed=7, architecture=ARCHITECTURE), policy)
    model.state_module.generator.load_state_dict(initial, strict=True)
    optimizer = torch.optim.Adam(model.parameters, lr=LR, betas=BETAS, eps=1e-8, weight_decay=0.0)
    optimizer._experiment_component_id = "optim"  # type: ignore[attr-defined]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=1)
    batches = [_batch(noisy[index : index + 1], clean[index : index + 1], index) for index in range(steps)]
    trainer = TFGridNetTrainer(
        model,
        MultiResolutionL1Loss(),
        _Data(batches),
        optimizer,
        scheduler,
        max_logical_steps=steps,
        accumulation=1,
        grad_clip=CLIP,
        stop_after_logical_steps=steps,
        checkpoint_interval=10**9,
        max_epochs=1,
    )
    directory = OUT / f"gridnet-trainer-{clean.shape[-1]}"
    checkpoint = CheckpointManager(directory)
    checkpoint.directory.mkdir(parents=True, exist_ok=True)
    result = trainer.run(runtime, checkpoint)
    state = {key: value.detach().cpu().clone() for key, value in model.state_module.generator.state_dict().items()}
    runtime.shutdown()
    return {"stop_reason": result.stop_reason, "logical_step": result.logical_step, "state": state, "losses": list(trainer.losses)}


def _task_run(device, initial, clean, noisy, steps: int):
    task = TFGridNetTask(_task_config(str(device)), device)
    task.generator.load_state_dict(initial, strict=True)
    task.optim_g = _adam(task.generator)
    losses = []
    for index in range(steps):
        metrics = task.train_update(
            [{"clean": clean[index : index + 1], "noisy": noisy[index : index + 1]}],
            1,
        )
        losses.append(metrics["mr_l1_tfd"])
    return task.generator, losses


def _official_run(device, initial, clean, noisy, steps: int):
    official, independent = _pair(device)
    official.load_state_dict(initial, strict=True)
    independent.load_state_dict(initial, strict=True)
    criterion = _official_loss_class()(
        window_sz=[256, 512, 768, 1024],
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        normalize_variance=False,
        reduction="sum",
    ).to(device)
    optim_o = _adam(official)
    optim_i = _adam(independent)
    rows = []
    for index in range(steps):
        rows.append(
            _step(
                official,
                independent,
                optim_o,
                optim_i,
                criterion,
                clean[index : index + 1],
                noisy[index : index + 1],
            )
        )
    return official, rows


def _cpu_state(module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _gap(left: dict, right: dict) -> dict:
    worst = 0.0
    worst_name = ""
    for name, value in left.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        other = right[name]
        delta = float((value - other).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    official, _independent = _pair(device)
    initial = _cpu_state(official)
    clean, noisy = _waves(4096, 3, device)
    official_done, official_rows = _official_run(device, initial, clean, noisy, 3)
    task_done, task_losses = _task_run(device, initial, clean, noisy, 3)
    trainer = _trainer_run(device, initial, clean, noisy, 3)
    report = {
        "device": str(device),
        "official_commit": "0d136c5456d403cb2c0a4c13e76e8136e32e9b7a",
        "official_function": "espnet2 TFGridNet separator forward + TimeDomainL1 (mr_l1_tfd) via compare_tf_gridnet._step",
        "trainer_class": "TFGridNetTrainer.run",
        "condition_4096": {
            "samples": 4096,
            "batch_size": 1,
            "steps": 3,
            "official_rows": [
                {"loss_gap": row["loss_gap"], "grad": row["grad"]["max_abs"], "weight": row["weight"]["max_abs"]}
                for row in official_rows
            ],
            "task_losses": task_losses,
            "trainer_losses": trainer["losses"],
            "trainer_versus_task": _gap(trainer["state"], _cpu_state(task_done)),
            "official_versus_task": weight_report(official_done, task_done),
        },
    }
    try:
        clean_l, noisy_l = _waves(48000, 1, device)
        official_l, rows_l = _official_run(device, initial, clean_l, noisy_l, 1)
        task_l, losses_l = _task_run(device, initial, clean_l, noisy_l, 1)
        report["condition_48000_batch1"] = {
            "samples": 48000,
            "batch_size": 1,
            "official_batch_size": 4,
            "steps": 1,
            "official_versus_independent_loss_gap": rows_l[0]["loss_gap"],
            "official_versus_independent_grad": rows_l[0]["grad"]["max_abs"],
            "official_versus_independent_weight": rows_l[0]["weight"]["max_abs"],
            "task_loss": losses_l[0],
            "official_versus_task": weight_report(official_l, task_l),
        }
    except RuntimeError as exc:
        report["condition_48000_batch1"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
    path = OUT / "gridnet_trainer.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps(report["condition_4096"]["trainer_versus_task"], indent=2), flush=True)


if __name__ == "__main__":
    main()
