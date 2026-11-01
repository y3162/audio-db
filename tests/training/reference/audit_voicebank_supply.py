"""Audit the real VoiceBank-DEMAND training set through RecipeDataset.

CPU only. Does not load SEMamba++. Writes a new directory and does not modify
the production database.

    .venvs/lab/bin/python tests/training/reference/audit_voicebank_supply.py \
        data/results/training/semamba-supply-20261101
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from src.training.core.batch import plan_batches
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from src.training.data.dataset import load_recipe_dataset
from src.training.data.sampler import epoch_indices
from src.training.errors import TrainingError
from tests.training.supply import AuditedDataset, SupplyTask
from tests.training.support import cpu_process, tiny_config

DB = ROOT / "data/db/semamba-voicebank-train-20261101-compat.duckdb"
TRAIN = "voicebank_28_train"
VALID = "voicebank_28_valid"


def main() -> None:
    destination = Path(sys.argv[1])
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    if not DB.is_file():
        raise SystemExit(f"missing training database {DB}")
    destination.mkdir(parents=True)
    set_determinism(True, cuda=False)
    train = load_recipe_dataset(DB, TRAIN, role="train", seed=1234, sample_rate=16000, segment_size=8000, training=True)
    valid = load_recipe_dataset(DB, VALID, role="validation", seed=1234, sample_rate=16000, segment_size=None, training=False)
    plans = {
        "gpu1_micro8": plan_batches(8, 8, 1),
        "gpu1_micro2_accum4": plan_batches(8, 2, 1),
        "gpu2_micro2_accum2": plan_batches(8, 2, 2),
    }
    groups = {name: _case_groups(train, plan, seed=1234, steps=5) for name, plan in plans.items()}
    full = [tuple(step) for step in groups["gpu1_micro8"]]
    accum = [tuple(step) for step in groups["gpu1_micro2_accum4"]]
    ddp_sets = [set(step) for step in groups["gpu2_micro2_accum2"]]
    epoch_report = _epoch_report(train, seed=1234)
    changed = _case_groups(train, plans["gpu1_micro8"], seed=1235, steps=1)
    audited = AuditedDataset(train)
    task = SupplyTask()
    config = tiny_config(destination / "ckpt", max_steps=2, seed=1234, global_batch_size=8, micro_batch_size_per_gpu=8)
    Trainer(
        task,
        plans["gpu1_micro8"],
        cpu_process(),
        max_steps=2,
        seed=1234,
        checkpoint_dir=destination / "ckpt",
        checkpoint_interval=0,
        validation_interval=0,
        config=config,
    ).fit(audited)
    rendered = [row for record in task.records for row in record["rows"]]
    report = {
        "database": str(DB),
        "train_set": TRAIN,
        "train_size": len(train),
        "validation_set": VALID,
        "validation_size": len(valid),
        "same_order_micro8_and_micro2": full == accum,
        "ddp_set_matches_single_process": [set(step) == other for step, other in zip(full, ddp_sets)],
        "ddp_order_differs": [list(step) != list(other) for step, other in zip(groups["gpu1_micro8"], groups["gpu2_micro2_accum2"])],
        "other_seed_changes_first_batch": groups["gpu1_micro8"][0] != changed[0],
        "epoch": epoch_report,
        "unsupported_world": _unsupported(),
        "rendered_steps": 2,
        "rendered_rows": len(rendered),
        "rng_draws": audited.rng_draws,
        "recipe_matches": all(row["recipe_matches_case"] for row in rendered),
        "pairs_aligned": all(row["pair_aligned"] for row in rendered),
        "clean_noisy_distinct": all(row["clean_sha256"] != row["noisy_sha256"] for row in rendered),
        "rows": rendered,
    }
    (destination / "supply.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "rows"}, indent=2))
    if not all(
        [
            report["train_size"] == 10700,
            report["validation_size"] == 872,
            report["same_order_micro8_and_micro2"],
            all(report["ddp_set_matches_single_process"]),
            report["other_seed_changes_first_batch"],
            report["rng_draws"] == 0,
            report["recipe_matches"],
            report["pairs_aligned"],
            report["epoch"]["duplicate_ids"] == 0,
        ]
    ):
        raise SystemExit("voicebank supply audit failed")


def _case_groups(dataset, plan, *, seed: int, steps: int) -> list[list[str]]:
    from tests.training.supply import optimizer_groups

    grouped = optimizer_groups(len(dataset), seed, plan, steps=steps)
    return [[dataset.examples[index].case_id for rank in step for index in rank] for step in grouped]


def _epoch_report(dataset, *, seed: int) -> dict:
    order = epoch_indices(len(dataset), seed, 0, 1, 0)
    usable = (len(order) // 8) * 8
    used = order[:usable]
    return {
        "size": len(order),
        "used": len(used),
        "dropped": len(order) - len(used),
        "duplicate_ids": len(used) - len(set(used)),
    }


def _unsupported() -> str:
    try:
        plan_batches(8, 8, 3)
    except TrainingError as exc:
        return str(exc)
    return "not rejected"


if __name__ == "__main__":
    main()
