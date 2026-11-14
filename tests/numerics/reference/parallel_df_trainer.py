"""DeepFilterNet3 Trainer.run versus DeepFilterNetTask.train_update.

The official ``df.train.run_epoch`` comparison stays the one already recorded
with a feature stub. This run checks that the real trainer follows the task
on the same three synthetic waveforms and the same cosine table.
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
libdf_dir = os.environ.get("AUDIO_DB_LIBDF_PY", "/tmp/audio-db-official-numerics/libdf-py")
if libdf_dir not in sys.path:
    sys.path.insert(0, libdf_dir)

from src.experiments.data import Batch  # noqa: E402
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model  # noqa: E402
from src.experiments.official.deepfilternet3.reference import DeepFilterNetTask  # noqa: E402
from src.experiments.official.deepfilternet3.losses import official_deepfilternet_objective  # noqa: E402
from src.experiments.official.deepfilternet3.trainer import DeepFilterNet3Trainer  # noqa: E402
from src.experiments.runtime import PreparePolicy, ProcessRuntime  # noqa: E402
from tests.numerics.reference.compare_deepfilternet import _resample, _task_config  # noqa: E402
from tests.numerics.reference.common import NoCheckpoints, optim_report, weight_report  # noqa: E402


class _Epoch:
    def __init__(self, batches: list[Batch], updates: int) -> None:
        self.updates = updates
        self._batches = batches

    def batch(self, index: int) -> list[Batch]:
        return [self._batches[index]]


class _Data:
    group_size = 1
    epoch = 0
    update_index = 0

    def __init__(self, batches: list[Batch], updates: int) -> None:
        self._batches = batches
        self._updates = updates

    def train_epoch(self, epoch: int, *, start_update: int):
        del epoch, start_update
        return _Epoch(self._batches, self._updates)

    def validation_epoch(self, epoch: int):
        del epoch
        return iter(())

    def identity(self) -> dict[str, object]:
        return {"sample_rate": 48000, "teacher": "official_mixed_target"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    task = DeepFilterNetTask(_task_config(), device)
    held = {key: value.detach().cpu().clone() for key, value in task.generator.state_dict().items()}
    generator = torch.Generator().manual_seed(5)
    clean_bank = torch.randn(3, 1600, generator=generator)
    noisy_bank = clean_bank * 0.5 + 0.05 * torch.randn(3, 1600, generator=generator)
    batches = []
    task_losses = []
    for index in range(3):
        clean = _resample(clean_bank[index : index + 1]).to(device)
        noisy = _resample(noisy_bank[index : index + 1]).to(device)
        batches.append(Batch(inputs=noisy.cpu(), targets=clean.cpu(), sample_id=f"d{index}", realization_id=f"d{index}"))
        task_losses.append(task.train_update([{"clean": clean, "noisy": noisy}], 1)["deepfilter"])
    policy = PreparePolicy(trainable=True, distribute="none", find_unused_parameters=True)
    runtime = ProcessRuntime(device="cuda" if device.type == "cuda" else "cpu", precision="fp32")
    model = runtime.prepare_model(DeepFilterNet3Model(seed=7), policy)
    model.state_module.generator.load_state_dict(held, strict=True)
    optimizer = torch.optim.AdamW(model.parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-12, amsgrad=True)
    optimizer._experiment_component_id = "optim"  # type: ignore[attr-defined]
    # Eight updates match compare_deepfilternet.STEPS_PER_EPOCH. Stop after 3
    # so validation does not run and the cosine index matches the task.
    trainer = DeepFilterNet3Trainer(
        model,
        _Data(batches, 8),
        optimizer,
        objective=official_deepfilternet_objective(),
        max_logical_steps=3,
        accumulation=1,
        grad_clip=1.0,
        steps_per_epoch=8,
        stop_after_logical_steps=3,
    )
    directory = OUT / "df-trainer"
    checkpoint = NoCheckpoints(directory)
    checkpoint.directory.mkdir(parents=True, exist_ok=True)
    result = trainer.run(runtime, checkpoint)
    valid_batch = batches[0]
    trainer.data._valid = [valid_batch]

    def _validation_epoch(epoch: int):
        del epoch
        return iter(trainer.data._valid)

    trainer.data.validation_epoch = _validation_epoch
    validated = trainer._validate(runtime, 0, "trainer")
    module = model.state_module
    was_training = bool(module.training)
    module.eval()
    task.generator.eval()
    with torch.no_grad():
        correct_eval = float(
            trainer._loss_tensor(valid_batch.targets.to(device), valid_batch.inputs.to(device), module).detach()
        )
        swapped_eval = float(
            trainer._loss_tensor(valid_batch.inputs.to(device), valid_batch.targets.to(device), module).detach()
        )
    if was_training:
        module.train()
    task_valid = task.validation_report(
        {"clean": valid_batch.targets.to(device), "noisy": valid_batch.inputs.to(device)}
    )
    report = {
        "device": str(device),
        "official_commit": "d375b2d8309e0935d165700c91da9de862a99c31",
        "comparison": "DeepFilterNet3Trainer.run versus DeepFilterNetTask.train_update",
        "not_compared_here": "df.train.run_epoch was compared earlier with a libdfdata stub and is not re-claimed as the data loader",
        "steps": 3,
        "samples_at_16k_before_resample": 1600,
        "task_losses": task_losses,
        "trainer_losses": list(trainer.losses),
        "trainer_stop": result.stop_reason,
        "trainer_logical_step": result.logical_step,
        "weight": weight_report(model.state_module.generator, task.generator),
        "optim": optim_report(optimizer, task.optim_g),
        "validation": {
            "trainer_mean": validated,
            "trainer_eval_loss": correct_eval,
            "swapped_eval_loss": swapped_eval,
            "task_validation_report": task_valid["losses"]["generator"]["value"],
            "mode": "eval",
            "batch": "first training waveform, batch size 1, after the 3 updates",
        },
    }
    path = OUT / "df_trainer.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps({"weight": report["weight"], "losses": {"task": task_losses, "trainer": report["trainer_losses"]}}, indent=2), flush=True)
    runtime.shutdown()


if __name__ == "__main__":
    main()
