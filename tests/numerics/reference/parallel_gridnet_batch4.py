"""TF-GridNet at the official chunk: 48000 samples, batch 4, three steps.

Earlier notes called a length-48000 batch-1 slice a batch of 4. This script
feeds all four waveforms to one optimizer step. The official side is the
separator plus ``TimeDomainL1`` plus Adam, not ESPnet's epoch driver.
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

from src.experiments.data import Batch  # noqa: E402
from src.experiments.losses import MultiResolutionL1Loss  # noqa: E402
from src.experiments.official.tf_gridnet.model import TFGridNetModel  # noqa: E402
from src.experiments.official.tf_gridnet.trainer import TFGridNetTrainer  # noqa: E402
from src.experiments.runtime import PreparePolicy, ProcessRuntime  # noqa: E402
from tests.numerics.reference.compare_tf_gridnet import (  # noqa: E402
    ARCHITECTURE,
    BETAS,
    CLIP,
    LR,
    _adam,
    _official_loss_class,
    _pair,
    _step,
)
from tests.numerics.reference.parallel_gridnet_trainer import _cpu_state, _task_config  # noqa: E402
from src.experiments.official.tf_gridnet.reference import TFGridNetTask  # noqa: E402
from tests.numerics.reference.common import NoCheckpoints, weight_report  # noqa: E402


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

    def identity(self) -> dict[str, object]:
        return {"sample_rate": 16000, "teacher": "dry_clean"}


def _bank(steps: int, batch: int, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    clean = torch.randn(steps, batch, length, generator=generator)
    noisy = clean * 0.5 + 0.05 * torch.randn(steps, batch, length, generator=generator)
    return clean.to(device), noisy.to(device)


def _official_steps(device, initial, clean, noisy):
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
    for index in range(clean.shape[0]):
        rows.append(_step(official, independent, optim_o, optim_i, criterion, clean[index], noisy[index]))
    return official, independent, rows


def _task_steps(device, initial, clean, noisy):
    task = TFGridNetTask(_task_config(str(device)), device)
    task.generator.load_state_dict(initial, strict=True)
    task.optim_g = _adam(task.generator)
    losses = []
    for index in range(clean.shape[0]):
        metrics = task.train_update([{"clean": clean[index], "noisy": noisy[index]}], 1)
        losses.append(metrics["mr_l1_tfd"])
    return task.generator, losses


def _trainer_steps(device, initial, clean, noisy) -> dict:
    policy = PreparePolicy(trainable=True, distribute="none")
    # Official and task models in this process already occupy the GPU.
    runtime = ProcessRuntime(device="cuda" if device.type == "cuda" else "cpu", precision="fp32")
    model = runtime.prepare_model(TFGridNetModel(seed=7, architecture=ARCHITECTURE), policy)
    model.state_module.generator.load_state_dict(initial, strict=True)
    optimizer = torch.optim.Adam(model.parameters, lr=LR, betas=BETAS, eps=1e-8, weight_decay=0.0)
    optimizer._experiment_component_id = "optim"  # type: ignore[attr-defined]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=1)
    batches = [
        Batch(inputs=noisy[index].detach().cpu(), targets=clean[index].detach().cpu(), sample_id=f"b{index}", realization_id=f"b{index}")
        for index in range(clean.shape[0])
    ]
    trainer = TFGridNetTrainer(
        model,
        MultiResolutionL1Loss(),
        _Data(batches),
        optimizer,
        scheduler,
        max_logical_steps=clean.shape[0],
        accumulation=1,
        grad_clip=CLIP,
        stop_after_logical_steps=clean.shape[0],
        max_epochs=1,
    )
    directory = OUT / "gridnet-trainer-48000-b4"
    checkpoint = NoCheckpoints(directory)
    checkpoint.directory.mkdir(parents=True, exist_ok=True)
    result = trainer.run(runtime, checkpoint)
    state = _cpu_state(model.state_module.generator)
    losses = list(trainer.losses)
    runtime.shutdown()
    return {"stop_reason": result.stop_reason, "logical_step": result.logical_step, "state": state, "losses": losses}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    steps = int(os.environ.get("AUDIO_DB_GRID_STEPS", "3"))
    official, _independent = _pair(device)
    initial = _cpu_state(official)
    clean, noisy = _bank(steps, 4, 48000, device)
    official_done, independent_done, rows = _official_steps(device, initial, clean, noisy)
    task_done, task_losses = _task_steps(device, initial, clean, noisy)
    trainer = _trainer_steps(device, initial, clean, noisy)
    report = {
        "device": str(device),
        "official_commit": "0d136c5456d403cb2c0a4c13e76e8136e32e9b7a",
        "official_function": "espnet2 TFGridNet separator + TimeDomainL1 + Adam via compare_tf_gridnet._step",
        "not_called": "ESPnet epoch driver / AbsTask.train",
        "samples": 48000,
        "batch_size": 4,
        "steps": steps,
        "official_versus_independent": [
            {"loss_gap": row["loss_gap"], "grad": row["grad"]["max_abs"], "weight": row["weight"]["max_abs"], "worst_weight": row["weight"].get("worst", "")}
            for row in rows
        ],
        "task_losses": task_losses,
        "trainer_losses": trainer["losses"],
        "trainer_stop": trainer["stop_reason"],
        "official_versus_task": weight_report(official_done, task_done),
        "independent_versus_task": weight_report(independent_done, task_done),
        "trainer_versus_task": weight_report_state(trainer["state"], task_done),
    }
    path = OUT / "gridnet_48000_batch4_steps.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps({
        "rows": report["official_versus_independent"],
        "official_versus_task": report["official_versus_task"],
        "trainer_versus_task": report["trainer_versus_task"],
        "losses": {"task": task_losses, "trainer": trainer["losses"]},
    }, indent=2), flush=True)


def weight_report_state(state: dict, module: torch.nn.Module) -> dict:
    named = dict(module.state_dict())
    worst = 0.0
    worst_name = ""
    for name, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        delta = float((value.cpu() - named[name].detach().cpu()).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


if __name__ == "__main__":
    main()
