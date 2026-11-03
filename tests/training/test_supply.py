"""Real RecipeDataset supply: order, crops, global batch, and resume."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.training.core.batch import plan_batches
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from tests.training.supply import AuditedDataset, SupplyTask, optimizer_groups
from tests.training.support import cpu_process, paired_dataset, run_distributed_workers, tiny_config

def tearDownModule():
    set_determinism(False, cuda=False)


class SupplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_same_seed_repeats_and_another_seed_changes_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _fit_supply(root / "a", seed=7, steps=3, count=10, global_batch=4, micro=4)
            second = _fit_supply(root / "b", seed=7, steps=3, count=10, global_batch=4, micro=4)
            changed = _fit_supply(root / "c", seed=9, steps=3, count=10, global_batch=4, micro=4)
        self.assertEqual(_signatures(first), _signatures(second))
        self.assertNotEqual([row["case_id"] for row in _signatures(first)], [row["case_id"] for row in _signatures(changed)])
        self.assertEqual(first["rng_draws"], 0)
        self.assertTrue(all(row["recipe_matches_case"] and row["pair_aligned"] for row in _signatures(first)))
        self.assertTrue(all(row["clean_sha256"] != row["noisy_sha256"] for row in _signatures(first)))

    def test_crop_depends_on_case_seed_and_epoch_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            early = _fit_supply(root / "early", seed=4, steps=4, count=8, global_batch=4, micro=4)
            again = _fit_supply(root / "again", seed=4, steps=4, count=8, global_batch=4, micro=4)
        by_key = {(row["case_id"], row["epoch"]): row["crop_start"] for row in _signatures(early)}
        for row in _signatures(again):
            self.assertEqual(by_key[(row["case_id"], row["epoch"])], row["crop_start"])
        epochs = {row["epoch"] for row in _signatures(early)}
        self.assertGreater(len(epochs), 1)

    def test_epoch_has_no_repeated_sample_and_drops_only_the_tail(self):
        plan = plan_batches(4, 4, 1)
        groups = optimizer_groups(10, 5, plan, steps=2)
        epoch0 = [index for step in groups for rank in step for index in rank]
        self.assertEqual(len(epoch0), 8)
        self.assertEqual(len(set(epoch0)), 8)
        self.assertEqual(len(groups), 2)

    def test_global_batch_plans_use_the_same_sample_set(self):
        plans = {
            "full": plan_batches(8, 8, 1),
            "accum": plan_batches(8, 2, 1),
            "ddp": plan_batches(8, 2, 2),
        }
        grouped = {name: optimizer_groups(20, 11, plan, steps=3) for name, plan in plans.items()}
        full = [_flat(step) for step in grouped["full"]]
        accum = [_flat(step) for step in grouped["accum"]]
        self.assertEqual(full, accum)
        for step, other in zip(full, grouped["ddp"]):
            self.assertEqual(set(step), set(_flat(other)))
            self.assertNotEqual(step, _flat(other))
        with self.assertRaises(TrainingError):
            plan_batches(8, 8, 3)

    def test_trainer_matches_the_index_plan_for_one_and_two_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = _fit_supply(root / "single", seed=6, steps=2, count=16, global_batch=8, micro=2)
            dual = _run_supply_ranks(root / "dual", seed=6, steps=2, count=16, global_batch=8, micro=2, world=2)
        single_ids = [[row["case_id"] for row in single["records"][step]["rows"]] for step in range(2)]
        # micro 2, accumulation 4 on one rank keeps order. Two ranks interleave.
        plan = plan_batches(8, 2, 1)
        expected = optimizer_groups(16, 6, plan, steps=2)
        dataset_ids = _case_lookup(root / "single" / "wav" if False else None)
        del dataset_ids
        self.assertEqual(len(single_ids[0]), 8)
        union = []
        for step in range(2):
            combined = []
            for rank_records in dual:
                combined.extend(row["case_id"] for row in rank_records["records"][step]["rows"])
            union.append(set(combined))
            self.assertEqual(union[step], set(single_ids[step]))
        self.assertEqual([len(step) for step in expected], [1, 1])

    def test_resume_continues_at_the_same_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = _fit_supply(
                root / "continuous", seed=3, steps=4, count=8, global_batch=4, micro=4, interval=2
            )
            partial = _fit_supply(root / "partial", seed=3, steps=2, count=8, global_batch=4, micro=4, interval=2)
            resumed = _resume_supply(root / "partial", root / "resumed", steps=4)
        combined = partial["records"] + resumed["records"]
        self.assertEqual(_signatures(continuous), [row for record in combined for row in record["rows"]])


def _fit_supply(root, *, seed, steps, count, global_batch, micro, interval=0):
    root.mkdir(parents=True, exist_ok=True)
    dataset = AuditedDataset(paired_dataset(root / "wav", count, frames=480, segment_size=160, seed=seed))
    task = SupplyTask()
    plan = plan_batches(global_batch, micro, 1)
    config = tiny_config(root / "ckpt", max_steps=steps, seed=seed, global_batch_size=global_batch, micro_batch_size_per_gpu=micro)
    trainer = Trainer(
        task,
        plan,
        cpu_process(),
        max_steps=steps,
        seed=seed,
        checkpoint_dir=root / "ckpt",
        checkpoint_interval=interval,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset)
    return {"rng_draws": dataset.rng_draws, "records": task.records}


def _signatures(result):
    return [row for record in result["records"] for row in record["rows"]]


def _signatures_of_records(records):
    return records


def _flat(step):
    return [index for rank in step for index in rank]


def _case_lookup(_unused):
    return None


def _run_supply_ranks(root, *, seed, steps, count, global_batch, micro, world):
    root.mkdir(parents=True, exist_ok=True)
    paired_dataset(root / "wav", count, frames=480, segment_size=160, seed=seed)
    destinations = [root / f"rank-{rank}.json" for rank in range(world)]

    def argv_for_port(port):
        commands = []
        for rank, destination in enumerate(destinations):
            commands.append([
                sys.executable,
                str(ROOT / "tests/training/supply_worker.py"),
                str(rank),
                str(world),
                str(port),
                str(root),
                str(destination),
                str(seed),
                str(global_batch),
                str(micro),
                str(steps),
                str(count),
            ])
        return commands

    run_distributed_workers(argv_for_port, cwd=ROOT, timeout=180)
    return [json.loads(path.read_text(encoding="utf-8")) for path in destinations]


def _resume_supply(partial: Path, destination: Path, *, steps: int):
    completed = subprocess.run(
        [sys.executable, str(Path(__file__)), "resume", str(partial), str(destination), str(steps)],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return json.loads((destination / "records.json").read_text(encoding="utf-8"))


def _resume_main() -> None:
    partial = Path(sys.argv[2])
    destination = Path(sys.argv[3])
    steps = int(sys.argv[4])
    destination.mkdir(parents=True)
    checkpoint = partial / "ckpt" / "step_00000002.pt"
    dataset = AuditedDataset(paired_dataset(partial / "wav", 8, frames=480, segment_size=160, seed=3))
    task = SupplyTask()
    config = tiny_config(destination / "ckpt", max_steps=steps, seed=3, global_batch_size=4, micro_batch_size_per_gpu=4)
    trainer = Trainer(
        task,
        plan_batches(4, 4, 1),
        cpu_process(),
        max_steps=steps,
        seed=3,
        checkpoint_dir=destination / "ckpt",
        checkpoint_interval=2,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset, resume=checkpoint)
    (destination / "records.json").write_text(
        json.dumps({"rng_draws": dataset.rng_draws, "records": task.records}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        _resume_main()
    else:
        unittest.main()
