"""Rank-local RNG probe for MossFormerGAN 2 GPU checkpoints.

Not part of the CPU unittest suite. Writes only under /tmp/p10c-fix2b1.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(os.environ.get("P10C_OUT", "/tmp/p10c-fix2b1"))
PY_NOTES: list[dict] = []


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha(value) -> str:
    import torch

    tensor = value.detach().cpu().contiguous()
    return _sha(tensor.numpy().tobytes())


def _rng_view() -> dict:
    import random

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    view = {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "python": _sha(repr(random.getstate()).encode()),
        "numpy": _sha(numpy_state[1].tobytes() + str(numpy_state[2]).encode()),
        "torch": _tensor_sha(torch.random.get_rng_state()),
        "cuda_current": None,
        "cuda_all": [],
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        view["cuda_current"] = _tensor_sha(torch.cuda.get_rng_state())
        view["cuda_all"] = [_tensor_sha(torch.cuda.get_rng_state(index)) for index in range(torch.cuda.device_count())]
    return view


def _bn_view(task) -> list[dict]:
    import torch

    from src.training.core.distributed import unwrap

    rows = []
    for name, module in unwrap(task.discriminator).named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        rows.append(
            {
                "name": name,
                "running_mean": _tensor_sha(module.running_mean),
                "running_var": _tensor_sha(module.running_var),
                "num_batches_tracked": int(module.num_batches_tracked.detach().cpu()),
                "weight": _tensor_sha(module.weight),
                "bias": _tensor_sha(module.bias),
                "running_var_max": float(module.running_var.detach().float().max().cpu()),
            }
        )
    return rows


def _note(tag: str, task=None) -> None:
    row = {"tag": tag, **_rng_view()}
    if task is not None:
        row["bn"] = _bn_view(task)
    PY_NOTES.append(row)


def _write_notes(name: str) -> None:
    path = OUT / "probe" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PY_NOTES, indent=2), encoding="utf-8")


def _mechanism() -> None:
    import torch
    import torch.distributed as dist

    from src.training.core.distributed import setup_process
    from src.training.core.rng import seed_training_stream

    process = setup_process("cuda")
    _note("after_process_init")
    seed_training_stream(777, cuda=True)
    _note("after_seed_training_stream")
    device = process.device
    data = torch.ones(32, 128, device=device)
    for _ in range(4):
        data = torch.nn.functional.dropout(data, p=0.5, training=True)
    _note("after_dropout_on_current_device")
    payload = PY_NOTES[-1]
    gathered = [None for _ in range(process.world_size)]
    dist.all_gather_object(gathered, payload)
    if process.rank == 0:
        (OUT / "probe").mkdir(parents=True, exist_ok=True)
        (OUT / "probe" / "mechanism.json").write_text(json.dumps(gathered, indent=2), encoding="utf-8")
        rank0 = gathered[0]
        rank1 = gathered[1]
        print(
            json.dumps(
                {
                    "rank0_current": rank0["cuda_current"],
                    "rank1_current": rank1["cuda_current"],
                    "currents_equal": rank0["cuda_current"] == rank1["cuda_current"],
                    "rank0_slot0": rank0["cuda_all"][0],
                    "rank0_slot1": rank0["cuda_all"][1],
                    "rank0_slot1_equals_rank1_current": rank0["cuda_all"][1] == rank1["cuda_current"],
                    "rank0_slot0_equals_rank1_current": rank0["cuda_all"][0] == rank1["cuda_current"],
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


def _install_train_hooks() -> None:
    import torch

    from src.training.core.distributed import unwrap
    from src.training.se.mossformergan import MossFormerGANTask

    original = MossFormerGANTask.distribute

    def distribute(self, process):
        _note("before_distribute", self)
        original(self, process)
        _note("after_distribute", self)
        generator = unwrap(self.generator)
        discriminator = unwrap(self.discriminator)
        counts = {"g": 0, "d": 0, "gb": 0, "db": 0}

        def before_g(_module, _inputs):
            counts["g"] += 1
            if counts["g"] <= 3:
                _note(f"before_generator_forward_{counts['g']}", self)

        def after_g(_module, _inputs, _output):
            if counts["g"] <= 3:
                _note(f"after_generator_forward_{counts['g']}", self)

        def before_d(_module, _inputs):
            counts["d"] += 1
            if counts["d"] <= 3:
                _note(f"before_discriminator_forward_{counts['d']}", self)

        def after_d(_module, _inputs, _output):
            if counts["d"] <= 3:
                _note(f"after_discriminator_forward_{counts['d']}", self)

        def after_gb(_module, _gin, _gout):
            counts["gb"] += 1
            if counts["gb"] <= 2:
                _note(f"after_generator_backward_{counts['gb']}", self)

        def after_db(_module, _gin, _gout):
            counts["db"] += 1
            if counts["db"] <= 2:
                _note(f"after_discriminator_backward_{counts['db']}", self)

        generator.register_forward_pre_hook(before_g)
        generator.register_forward_hook(after_g)
        discriminator.register_forward_pre_hook(before_d)
        discriminator.register_forward_hook(after_d)
        generator.register_full_backward_hook(after_gb)
        discriminator.register_full_backward_hook(after_db)
        for label, optim in (("g", self.optim_g), ("d", self.optim_d)):
            step = optim.step

            def wrapped(*args, _step=step, _label=label, **kwargs):
                result = _step(*args, **kwargs)
                _note(f"after_optimizer_{_label}", self)
                return result

            optim.step = wrapped

    MossFormerGANTask.distribute = distribute

    from src.training.core import trainer as trainer_mod

    original_save = trainer_mod.Trainer._save

    def _save(self, *args, **kwargs):
        _note("before_checkpoint_capture", self.task)
        return original_save(self, *args, **kwargs)

    trainer_mod.Trainer._save = _save


def _train() -> None:
    import torch

    from src.training.se.run import execute

    _install_train_hooks()
    config = OUT / "probe" / "train.yaml"
    execute(config)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        import torch as torch_mod

        from src.training.core.checkpoint import load_checkpoint

        payload = load_checkpoint(OUT / "probe-train" / "step_00000001.pt", torch.device("cpu"))
        cuda = payload["rng"].get("cuda")
        saved = {
            "cuda_all": [_tensor_sha(item) for item in cuda] if cuda else [],
            "python": _sha(repr(payload["rng"]["python"]).encode()),
            "torch": _tensor_sha(payload["rng"]["torch"]),
        }
        (OUT / "probe" / "checkpoint_rng.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    _write_notes(f"train-rank{rank}.json")


def _resume_restore() -> None:
    import torch

    from src.training.core.checkpoint import load_checkpoint
    from src.training.core.distributed import setup_process
    from src.training.core.rng import restore_rng, set_determinism
    from src.training.se.config import load_training_config
    from src.training.se.run import build_task

    config = load_training_config(OUT / "probe" / "train.yaml")
    set_determinism(bool(config["deterministic"]), cuda=True)
    process = setup_process("cuda")
    task = build_task(config, process.device)
    _note("after_build_before_restore", task)
    payload = load_checkpoint(OUT / "probe-train" / "step_00000001.pt", process.device)
    task.load_state_dict(payload["task"])
    _note("after_load_state_before_rng", task)
    restore_rng(payload["rng"], cuda=True)
    _note("after_restore_rng", task)
    task.distribute(process)
    _note("after_distribute_restored", task)
    _write_notes(f"restore-rank{process.rank}.json")
    if process.distributed:
        import torch.distributed as dist

        dist.destroy_process_group()


def _write_train_config() -> None:
    import yaml

    config = {
        "seed": 777,
        "deterministic": False,
        "global_batch_size": 4,
        "micro_batch_size_per_gpu": 1,
        "required_world_size": 2,
        "experiment_profile": "full",
        "experiment_claim": "official_faithful",
        "max_steps": 350053,
        "segment_size": 32000,
        "loss": "official",
        "grad_clip": 10.0,
        "checkpoint_dir": str(OUT / "probe-train"),
        "checkpoint_interval": 1,
        "validation_interval": 2893,
        "best_checkpoint_start_epoch": 40,
        "verification_max_updates": 1,
        "optimizer": {"name": "adamw", "lr": 0.0005, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "mossformergan",
            "variant": "se_16k",
            "sampling_rate": 16000,
            "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
        },
        "data": {
            "pipeline": "reproduction",
            "dataset": "mossformergan.vb28.ondemand.official@1",
            "allow_test_selection": True,
            "load_workers": 4,
            "prefetch": True,
            "verify_on_read": False,
        },
        "runtime": {"device": "cuda", "allow_busy_gpu": False},
    }
    path = OUT / "probe" / "train.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "write-config":
        _write_train_config()
    elif phase == "mechanism":
        _mechanism()
    elif phase == "train":
        _train()
    elif phase == "restore":
        _resume_restore()
    else:
        raise SystemExit(f"unknown phase {phase}")
