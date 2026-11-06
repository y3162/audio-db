"""Build a process, a model task, and a recipe dataset, then train."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import torch.distributed as dist

from src.training.core.batch import plan_batches
from src.training.core.batch_schedule import scheduled_batch_size
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
from src.training.se.tf_gridnet import TFGridNetTask
from src.training.se.tf_locoformer import TFLocoformerTask
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
    if name == "tf_locoformer":
        return TFLocoformerTask(config, device)
    raise TrainingError(f"no training task for {name}")


def require_on_demand(dataset: dict) -> None:
    """New training reads on-demand snapshots. Stored wav datasets are refused."""
    delivery = (dataset.get("rule_json") or {}).get("delivery")
    name = f"{dataset.get('name')}@{dataset.get('version')}"
    if delivery != "on_demand":
        raise TrainingError(
            f"dataset {name} is not delivery on_demand; "
            "stored wav and pre-resampled wav training is disabled"
        )


def experiment_lock(directory: Path):
    """Exclusive lock for one experiment directory. Same-process reentry succeeds."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise TrainingError(f"experiment is already running: {directory}") from exc
    return handle


def execute(config_path: str | Path, *, resume: str | Path | None = None,
            deadline: float | None = None, segment_steps: int | None = None,
            strict_resume: bool = False) -> Trainer:
    config = load_training_config(config_path)
    device_name = config["runtime"]["device"]
    if device_name == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        assert_gpu_idle(local_rank, allow_busy=config["runtime"]["allow_busy_gpu"])
    set_determinism(config["deterministic"], cuda=device_name == "cuda")
    process = setup_process(device_name)
    data = config["data"]
    plan_world = process.world_size
    virtual = data.get("virtual_world_size")
    if virtual is not None:
        if process.world_size != 1:
            raise TrainingError("data.virtual_world_size cannot be combined with a distributed process")
        if not config.get("chunk_iterator"):
            raise TrainingError("data.virtual_world_size requires chunk_iterator")
        plan_world = int(virtual)
    if config.get("batch_schedule"):
        first = scheduled_batch_size(
            0,
            int(config["batch_size_limit"]),
            [tuple(item) for item in config["batch_schedule"]],
        )
        if first > int(config["micro_batch_size_per_gpu"]):
            raise TrainingError(
                f"scheduled batch {first} exceeds micro_batch_size_per_gpu "
                f"{config['micro_batch_size_per_gpu']}"
            )
        plan = plan_batches(first, first, plan_world)
    else:
        plan = plan_batches(
            config["global_batch_size"],
            config["micro_batch_size_per_gpu"],
            plan_world,
        )
    if process.rank == 0:
        print(
            "batch plan "
            f"global={plan.global_batch_size} micro={plan.micro_batch_size} "
            f"world={plan.world_size} accumulation={plan.accumulation_steps}",
            flush=True,
        )
    task = build_task(config, process.device)
    sample_rate = int(config["model"]["sampling_rate"])
    if data.get("pipeline") == "reproduction" and config.get("chunk_iterator"):
        from src.reproduction.pipeline.pregenerated import open_chunk_datasets

        train, validation = open_chunk_datasets(
            dataset=data["dataset"],
            load_workers=data["load_workers"],
            verify_on_read=data["verify_on_read"],
        )
        require_on_demand(train.store.dataset)
        allowed = {
            ("tf_gridnet", "pregenerated.ms_snsd_tfgridnet"),
            ("tf_locoformer", "pregenerated.dns_tflocoformer"),
        }
        if (config["model"]["name"], train.algorithm) not in allowed:
            train.close()
            raise TrainingError(
                f"model {config['model']['name']} does not match pregenerated dataset {train.algorithm}"
            )
    elif data.get("pipeline") == "reproduction":
        from src.reproduction.pipeline.training_data import open_training_data

        train = open_training_data(
            dataset=data["dataset"],
            allow_test_selection=data["allow_test_selection"],
            seed=config["seed"],
            global_batch=plan.global_batch_size,
            micro_batch=plan.micro_batch_size,
            accumulation=plan.accumulation_steps,
            world_size=process.world_size,
            segment_size=config["segment_size"],
            model_sample_rate=sample_rate,
            load_workers=data["load_workers"],
            verify_on_read=data["verify_on_read"],
        )
        expected = {
            "mp_senet": "fixed_pair.mp_senet",
            "mossformergan": "fixed_pair.mossformergan_se_16k",
            "deepfilternet3": "online.dfn3",
            "semambapp": "online.semambapp_gsr",
        }.get(config["model"]["name"])
        require_on_demand(train.store.dataset)
        if expected != train.algorithm:
            train.close()
            raise TrainingError(
                f"model {config['model']['name']} does not match dataset algorithm {train.algorithm}"
            )
        if process.rank == 0 and train.batch_warning:
            print(train.batch_warning, flush=True)
        validation = None
    else:
        train = load_recipe_dataset(
            data["database"],
            data["train_set"],
            role="train",
            seed=config["seed"],
            sample_rate=sample_rate,
            segment_size=config["segment_size"],
            training=True,
            rms_normalize=data["rms_normalize"],
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
                rms_normalize=data["rms_normalize"],
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
    if strict_resume and resume is not None:
        from src.training.core.checkpoint import load_checkpoint
        saved = load_checkpoint(Path(resume), process.device)
        if {k: v for k, v in saved.get("config", {}).items() if k != "logging"} != {k: v for k, v in config.items() if k != "logging"}:
            raise TrainingError("checkpoint configuration does not match this experiment")
    lock = experiment_lock(Path(config["checkpoint_dir"])) if process.rank == 0 else None
    try:
        trainer.fit(train, validation, resume=None if resume is None else Path(resume),
                    deadline=deadline, segment_steps=segment_steps)
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
    if process.distributed and dist.is_initialized():
        dist.destroy_process_group()
    return trainer
