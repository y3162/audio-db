"""Show that a resumed step starts from the continuous checkpoint.

The fork file is the continuous checkpoint copied at --fork-step. Resume loads
that file, keeps its records, then appends new steps. Post-update hashes may
differ on CUDA. This script checks the saved pre-update records and does not
train.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.numerics.four_process.compare import digest  # noqa: E402


def _records(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = payload["records"]
    del payload
    return rows


def _pair(continuous: Path, resumed: Path, fork_step: int):
    source = torch.load(continuous / "official_a" / f"fork-step-{fork_step}.pt", map_location="cpu", weights_only=False)
    source_modules = digest(source["modules"])
    source_optimizers = digest(source["optimizers"])
    source_schedulers = digest(source["schedulers"])
    recorded_rng = {"cpu": source["rng"]["torch"], "cuda": source["rng"].get("cuda")}
    source_rng = digest(recorded_rng)
    del source
    report = {"workers": {}}
    for worker in ("official_a", "official_b", "independent_a", "independent_b"):
        before = _records(continuous / worker / "result.pt")
        after = _records(resumed / worker / "result.pt")
        pre = before[fork_step - 1]
        resumed_pre = after[fork_step - 1]
        post = before[fork_step]
        resumed_post = after[fork_step]
        report["workers"][worker] = {
            "pre_model": pre["model_hash"] == resumed_pre["model_hash"],
            "pre_optimizer": pre["optimizer_hash"] == resumed_pre["optimizer_hash"],
            "pre_scheduler": pre["scheduler_hash"] == resumed_pre["scheduler_hash"],
            "pre_rng_parts": pre["rng_parts"] == resumed_pre["rng_parts"],
            "pre_input": pre["input_hash"] == resumed_pre["input_hash"],
            "pre_step": pre["step"] == resumed_pre["step"] == fork_step,
            "post_model_equal": post["model_hash"] == resumed_post["model_hash"],
            "post_optimizer_equal": post["optimizer_hash"] == resumed_post["optimizer_hash"],
            "post_rng_parts_equal": post["rng_parts"] == resumed_post["rng_parts"],
            "post_input_equal": post["input_hash"] == resumed_post["input_hash"],
            "post_scheduler_equal": post["scheduler_hash"] == resumed_post["scheduler_hash"],
        }
    # The copied checkpoint is worker-specific. Compare official_a only to its own file.
    official_pre = _records(continuous / "official_a" / "result.pt")[fork_step - 1]
    report["official_a_fork_file"] = {
        "modules_match_record": source_modules == official_pre["model_hash"],
        "optimizers_match_record": source_optimizers == official_pre["optimizer_hash"],
        "schedulers_match_record": source_schedulers == official_pre["scheduler_hash"],
        "rng_match_record": source_rng == official_pre["rng_hash"],
    }
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    pairs = {
        "mp_senet_fixed_seed17": (
            ROOT / "data/results/training/final-accept-mp-fixed-gpu-seed17-20261103",
            ROOT / "data/results/training/final-accept-mp-fixed-gpu-seed17-20261103-fork/final-accept-mp-fixed-gpu-seed17-20261103",
        ),
        "deepfilternet3": (
            ROOT / "data/results/training/final-accept-deepfilternet3-gpu-resume-source-20261103",
            ROOT / "data/results/training/final-accept-deepfilternet3-gpu-resume-source-20261103-fork/final-accept-deepfilternet3-gpu-resume-source-20261103",
        ),
        "semambapp_stride_fixed": (
            ROOT / "data/results/training/final-accept-semamba-fixed-resume-source-20261103",
            ROOT / "data/results/training/final-accept-semamba-fixed-resume-source-20261103-fork/final-accept-semamba-fixed-resume-source-20261103",
        ),
    }
    found = {}
    for name, (continuous, resumed) in pairs.items():
        if not continuous.is_dir() or not resumed.is_dir():
            found[name] = {"missing": True, "continuous": continuous.is_dir(), "resumed": resumed.is_dir()}
            continue
        found[name] = _pair(continuous, resumed, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(found, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
