"""One-step reproduction smoke for MP-SENet and MossFormerGAN.

This is not a discovered unit test. It uses the P2 snapshots and the real
training tasks. Validation inside the trainer is left off so the run does not
score the whole official test split; one validation batch is scored directly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from src.reproduction.paths import ROOT
from src.training.se.config import load_training_config
from src.training.se.run import execute


def main() -> None:
    if len(sys.argv) == 3:
        root = Path(tempfile.mkdtemp(prefix="p3-smoke-"))
        try:
            payload = _model_smoke(root, sys.argv[1], sys.argv[2])
            print(json.dumps(_plain(payload)))
        finally:
            shutil.rmtree(root, ignore_errors=True)
        return
    device = os.environ.get("P3_SMOKE_DEVICE", "")
    if not device:
        device = "cuda" if _gpu_count() else "cpu"
    configs = [
        "configs/training/reproduction/mp_senet_vb28_official.yaml",
        "configs/training/reproduction/mossformergan_vb28_official.yaml",
    ]
    runs = []
    for config in configs:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), config, device],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            print(completed.stdout, file=sys.stderr)
            raise SystemExit(completed.returncode)
        line = completed.stdout.strip().splitlines()[-1]
        runs.append(json.loads(line))
    print(json.dumps({"device": device, "gpu_count": _gpu_count(), "runs": runs}, indent=2))


def _model_smoke(root: Path, config_path: str, device: str) -> dict:
    source = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    source["max_steps"] = 2
    source["checkpoint_interval"] = 0
    source["validation_interval"] = 0
    source["checkpoint_dir"] = str(root / "ckpt")
    source["runtime"]["device"] = device
    source["runtime"]["allow_busy_gpu"] = False
    source["data"]["load_workers"] = 2
    source["data"]["prefetch"] = True
    path = root / "config.yaml"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    first = execute(path, segment_steps=1)
    # The checkpointed model is still allocated in this process. The resume
    # below is that same run, so the idle check would see our own memory.
    source["runtime"]["allow_busy_gpu"] = True
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    checkpoint = root / "ckpt" / "step_00000001.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint for {config_path}")
    plan_hash = None
    from src.training.core.checkpoint import load_checkpoint
    import torch

    saved = load_checkpoint(checkpoint, torch.device("cpu"))
    plan_hash = saved["data_state"]["plan_hash"]
    dataset_id = saved["data_state"]["dataset_id"]
    snapshot_hash = saved["data_state"]["snapshot_hash"]
    # A second process-equivalent plan from the same pure function.
    from src.reproduction.pipeline.training_data import open_training_data

    probe = open_training_data(
        dataset=source["data"]["dataset"],
        allow_test_selection=source["data"]["allow_test_selection"],
        seed=source["seed"],
        global_batch=saved["batch_plan"]["global_batch_size"],
        micro_batch=saved["batch_plan"]["micro_batch_size"],
        accumulation=saved["batch_plan"]["accumulation_steps"],
        world_size=saved["batch_plan"]["world_size"],
        segment_size=source["segment_size"],
        model_sample_rate=source["model"]["sampling_rate"],
        load_workers=1,
    )
    try:
        recomputed = probe.plan_for(0).plan_hash
        validation = next(probe.validation(rank=0, world_size=1, epoch=0))
    finally:
        probe.close()
    from src.training.se.run import build_task

    # The first execute destroyed nothing on world size 1. Build a fresh task
    # for one validation batch so the smoke does not score every utterance.
    config = load_training_config(path)
    process_device = torch.device(device)
    task = build_task(config, process_device)
    task.load_state_dict(saved["task"])
    report = task.validation_report(validation)
    resumed = execute(path, resume=checkpoint)
    return {
        "config": config_path,
        "model": source["model"]["name"],
        "device": device,
        "first_stop": first.stop_reason,
        "resumed_stop": resumed.stop_reason,
        "resumed_step": resumed.global_step,
        "dataset_id": dataset_id,
        "snapshot_hash": snapshot_hash,
        "plan_hash": plan_hash,
        "recomputed_plan_hash": recomputed,
        "plan_hash_match": plan_hash == recomputed,
        "validation_case": validation["case_id"],
        "validation_length": int(validation["clean"].shape[-1]),
        "validation_losses": report["losses"],
        "world_size": saved["batch_plan"]["world_size"],
    }


def _gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
