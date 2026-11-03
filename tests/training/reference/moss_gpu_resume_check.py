"""Load one MossFormerGAN checkpoint twice and continue from the same state.

The four-process resume path copies saved records, so those records alone do
not prove the process restored tensors. This script rebuilds the worker, loads
the checkpoint, and hashes the restored tensors before the next update.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.core.rng import capture_rng, restore_rng  # noqa: E402
from tests.training.four_process.compare import (  # noqa: E402
    _batch,
    _load_modules,
    digest,
    rng_parts,
    seed_probe,
)
from tests.training.four_process.adapters import ADAPTERS  # noqa: E402


def _moments(optimizers):
    found = {}
    for name, opt in optimizers.items():
        rows = []
        for param, state in opt.state.items():
            rows.append({
                "exp_avg": state.get("exp_avg"),
                "exp_avg_sq": state.get("exp_avg_sq"),
                "step": state.get("step"),
            })
        found[name] = rows
    return found


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="continuous four-process directory")
    parser.add_argument("--worker", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fork-step", type=int, default=2)
    parser.add_argument("--extra-steps", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    cfg = json.loads((args.run / "config.json").read_text())
    fork = args.run / args.worker / f"fork-step-{args.fork_step}.pt"
    device = torch.device(cfg["device"])
    cuda = device.type == "cuda"
    implementation = args.worker.split("_")[0]
    adapter = ADAPTERS[cfg["model"]]
    if hasattr(adapter, "configure"):
        adapter.configure(cfg)
    seed_probe(cfg["seed"], False)
    modules = adapter.build(implementation, device)
    prepared = torch.load(args.run / "input.pt", map_location="cpu", weights_only=False)
    _load_modules(modules, prepared["initial"])
    optimizers = adapter.optimizers(modules, 0.001)
    schedulers = adapter.schedulers(optimizers)
    seed_probe(cfg["seed"], cuda)
    checkpoint = torch.load(fork, map_location=device, weights_only=False)
    _load_modules(modules, checkpoint["modules"])
    for key, opt in optimizers.items():
        opt.load_state_dict(checkpoint["optimizers"][key])
    for key, sched in schedulers.items():
        sched.load_state_dict(checkpoint["schedulers"][key])
    restore_rng(checkpoint["rng"], cuda=cuda)
    restored_rng = capture_rng(cuda=cuda)
    pre = {
        "parameters_equal": all(
            torch.equal(module.state_dict()[name].detach().cpu(), value.detach().cpu())
            for name, module in modules.items()
            for name, value in checkpoint["modules"][name].items()
        ),
        "optimizer_moments_equal": digest(_moments(optimizers)) == digest(_moments_from_checkpoint(checkpoint)),
        "scheduler_equal": digest({key: sched.state_dict() for key, sched in schedulers.items()}) == digest(checkpoint["schedulers"]),
        "python_rng": digest(restored_rng["python"]) == digest(checkpoint["rng"]["python"]),
        "numpy_rng": digest(restored_rng["numpy"]) == digest(checkpoint["rng"]["numpy"]),
        "torch_rng": digest(restored_rng["torch"]) == digest(checkpoint["rng"]["torch"]),
        "cuda_rng": (digest(restored_rng["cuda"]) == digest(checkpoint["rng"]["cuda"])) if cuda else None,
        "global_step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "accumulation_remainder": 0,
        "accumulation_note": "the checkpoint is written after train_step finishes both micro-batches and both optimizer steps",
    }
    position = int(checkpoint["data_position"])
    records = []
    for _ in range(args.extra_steps):
        clean, noisy, items = _batch(args.run, cfg, position, None)
        clean, noisy = clean.to(device), noisy.to(device)
        step = adapter.train_step(implementation, modules, optimizers, schedulers, clean, noisy, position, None)
        loss = float(step.losses["generator"].detach())
        disc = step.losses.get("discriminator")
        records.append({
            "step": position + 1,
            "generator_loss": loss,
            "discriminator_loss": None if disc is None else float(disc.detach()),
            "finite": bool(torch.isfinite(step.losses["generator"]).all()),
            "input_hash": {"clean": digest(clean.detach().cpu()), "noisy": digest(noisy.detach().cpu())},
            "case_id": [item["case_id"] for item in items],
            "model_hash": digest({name: module.state_dict() for name, module in modules.items()}),
            "optimizer_hash": digest({key: opt.state_dict() for key, opt in optimizers.items()}),
            "rng_parts": rng_parts(cuda),
        })
        position += 1
    payload = {"worker": args.worker, "fork": str(fork), "pre": pre, "after": records}
    (args.output / "resume.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not all(pre[key] for key in ("parameters_equal", "optimizer_moments_equal", "scheduler_equal", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")):
        raise SystemExit("pre-update state did not match the checkpoint")
    if not all(row["finite"] for row in records):
        raise SystemExit("nonfinite loss")
    print(args.output / "resume.json")


def _moments_from_checkpoint(checkpoint):
    found = {}
    for name, state in checkpoint["optimizers"].items():
        rows = []
        for slot in state.get("state", {}).values():
            rows.append({
                "exp_avg": slot.get("exp_avg"),
                "exp_avg_sq": slot.get("exp_avg_sq"),
                "step": slot.get("step"),
            })
        found[name] = rows
    return found


if __name__ == "__main__":
    main()
