"""Short real-GPU preflight for TF-GridNet and SEMamba++.

Writes only under --output. Does not use the full-training checkpoint directories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml

from src.training.core.distributed import unwrap
from src.training.se.run import execute
from src.training.se.semambapp import SEMambaTask
from src.training.se.tf_gridnet import TFGridNetTask


def write_config(source: Path, dest: Path, checkpoint_dir: Path, steps: int,
                 max_epochs: int | None) -> None:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["checkpoint_dir"] = str(checkpoint_dir)
    raw["max_steps"] = steps
    raw["checkpoint_interval"] = 0
    raw["validation_interval"] = 0
    if max_epochs is not None and "chunk_iterator" in raw:
        raw["chunk_iterator"]["max_epochs"] = max_epochs
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def instrument(times: list[float]) -> None:
    for cls in (TFGridNetTask, SEMambaTask):
        original = cls.train_update

        def wrapped(self, batches, accumulation, _original=original):
            start = time.perf_counter()
            metrics = _original(self, batches, accumulation)
            times.append(time.perf_counter() - start)
            modules = [self.generator]
            for name in ("discriminator_mrd", "discriminator_band"):
                module = getattr(self, name, None)
                if module is not None:
                    modules.append(module)
            finite = all(
                torch.isfinite(parameter.detach()).all()
                for module in modules
                for parameter in unwrap(module).parameters()
            )
            if not finite:
                raise RuntimeError("non-finite parameter")
            return metrics

        cls.train_update = wrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--segment-steps", type=int)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.yaml"
    write_config(args.config, config_path, args.output / "checkpoints", args.steps, args.max_epochs)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    times: list[float] = []
    instrument(times)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = execute(
        config_path,
        resume=args.resume,
        segment_steps=args.segment_steps,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "stop_reason": trainer.stop_reason,
        "global_step": trainer.global_step,
        "checkpoint": trainer.last_checkpoint,
        "elapsed_seconds": round(elapsed, 3),
        "step_seconds": [round(item, 4) for item in times],
        "step_seconds_median": round(sorted(times)[len(times) // 2], 4) if times else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
