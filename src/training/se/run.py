"""Build a process, a model task, and a recipe dataset, then train."""

from __future__ import annotations

import os
from pathlib import Path

import torch.distributed as dist

from src.training.core.batch import plan_batches
from src.training.core.distributed import setup_process
from src.training.core.gpu import assert_gpu_idle
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from src.training.data.dataset import load_recipe_dataset
from src.training.errors import TrainingError
from src.training.se.config import load_training_config
from src.training.se.deepfilternet import DeepFilterNetTask
from src.training.se.mossformergan import MossFormerGANTask
from src.training.se.mp_senet import MPSENetTask
from src.training.se.semambapp import SEMambaTask
from src.training.se.stages import StagedTask
from src.training.se.tf_gridnet import TFGridNetTask
from src.training.se.waveform import WaveformTask


def build_task(config: dict, device):
    name = config["model"]["name"]
    loss = config["loss"]
    if name == "mp_senet":
        return MPSENetTask(config, device)
    if name == "tf_gridnet" and loss == "waveform_l1":
        return WaveformTask(config, device)
    if name == "tf_gridnet":
        return TFGridNetTask(config, device)
    if name == "semambapp":
        return SEMambaTask(config, device)
    if name == "mossformergan":
        return MossFormerGANTask(config, device)
    if name == "deepfilternet3":
        return DeepFilterNetTask(config, device)
    if name in ("unipase", "stupase"):
        return StagedTask(config, device)
    raise TrainingError(f"no training task for {name}")


def execute(config_path: str | Path, *, resume: str | Path | None = None) -> None:
    config = load_training_config(config_path)
    device_name = config["runtime"]["device"]
    if device_name == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        assert_gpu_idle(local_rank, allow_busy=config["runtime"]["allow_busy_gpu"])
    set_determinism(config["deterministic"], cuda=device_name == "cuda")
    process = setup_process(device_name)
    plan = plan_batches(
        config["global_batch_size"],
        config["micro_batch_size_per_gpu"],
        process.world_size,
    )
    if process.rank == 0:
        print(
            "batch plan "
            f"global={plan.global_batch_size} micro={plan.micro_batch_size} "
            f"world={plan.world_size} accumulation={plan.accumulation_steps}",
            flush=True,
        )
    task = build_task(config, process.device)
    data = config["data"]
    sample_rate = int(config["model"]["sampling_rate"])
    train = load_recipe_dataset(
        data["database"],
        data["train_set"],
        role="train",
        seed=config["seed"],
        sample_rate=sample_rate,
        segment_size=config["segment_size"],
        training=True,
    )
    validation = None
    if data["validation_set"]:
        validation = load_recipe_dataset(
            data["database"],
            data["validation_set"],
            role="validation",
            seed=config["seed"],
            sample_rate=sample_rate,
            segment_size=None,
            training=False,
        )
    trainer = Trainer(
        task,
        plan,
        process,
        max_steps=config["max_steps"],
        seed=config["seed"],
        checkpoint_dir=Path(config["checkpoint_dir"]),
        checkpoint_interval=config["checkpoint_interval"],
        validation_interval=config["validation_interval"],
        config=config,
    )
    trainer.fit(train, validation, resume=None if resume is None else Path(resume))
    if process.distributed and dist.is_initialized():
        dist.destroy_process_group()
