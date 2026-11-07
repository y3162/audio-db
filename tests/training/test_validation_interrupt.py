"""A killed validation is rerun from the first batch and does not double-step."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.training.core.batch import plan_batches
from src.training.core.distributed import ProcessInfo
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from tests.training.support import cpu_process


class _Rows:
    def __init__(self, n: int, name: str):
        self.n = n
        self.name = name
        self.load_workers = 1

    def __len__(self):
        return self.n

    def set_epoch(self, epoch):
        self.epoch = epoch

    def identity(self):
        return {"name": self.name, "role": "train", "size": self.n}

    def collate(self, indices):
        return {"case_id": [f"{self.name}-{index}" for index in indices]}


class _Task:
    def __init__(self, *, fail_at: int = 0, skip_first: bool = False):
        self.fail_at = fail_at
        self.skip_first = skip_first
        self.optimizer_updates = 0
        self.scheduler_steps = 0
        self.validations = 0
        self.updates = 0
        self.last_skip = None

    def distribute(self, process):
        del process

    def train_mode(self):
        return None

    def eval_mode(self):
        return None

    def learning_rates(self):
        return {"generator": 0.1}

    def state_dict(self):
        return {
            "optimizer_updates": self.optimizer_updates,
            "scheduler_steps": self.scheduler_steps,
        }

    def load_state_dict(self, state):
        self.optimizer_updates = int(state["optimizer_updates"])
        self.scheduler_steps = int(state["scheduler_steps"])

    def export_for_inference(self):
        return {"weight": torch.tensor([float(self.optimizer_updates)])}

    def generator_state_dict(self):
        return self.export_for_inference()

    def train_update(self, batches, accumulation):
        del batches, accumulation
        self.updates += 1
        if self.skip_first and self.updates == 1:
            self.last_skip = {"reason": "pesq", "case": "first"}
            return {"loss": 1.0, "update_applied": 0.0}
        self.optimizer_updates += 1
        return {"loss": 0.5, "update_applied": 1.0}

    def step_schedulers(self, validation=None):
        del validation
        self.scheduler_steps += 1

    def validation_report(self, batch):
        del batch
        self.validations += 1
        if self.fail_at and self.validations >= self.fail_at:
            raise RuntimeError("validation killed")
        metric = {"status": "ok", "value": 1.25, "reason": None, "count": 1}
        return {
            "losses": {"generator": {"status": "ok", "value": 0.5, "reason": None}},
            "pesq": metric,
            "stoi": {"status": "na", "value": None, "reason": "unused", "count": 0},
        }


def _trainer(root: Path, task, *, steps: int, validation_interval: int) -> Trainer:
    config = {
        "seed": 1,
        "loss": "official",
        "model": {"name": "mossformergan"},
        "best_checkpoint_start_epoch": 0,
        "data": {"load_workers": 1, "prefetch": False},
    }
    return Trainer(
        task,
        plan_batches(1, 1, 1),
        cpu_process(),
        max_steps=steps,
        seed=1,
        checkpoint_dir=root,
        checkpoint_interval=0,
        validation_interval=validation_interval,
        config=config,
    )


class ValidationInterruptTests(unittest.TestCase):
    def test_pending_resume_does_not_double_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _Rows(1, "train")
            valid = _Rows(2, "valid")
            first = _Task(fail_at=2)
            with self.assertRaisesRegex(RuntimeError, "validation killed"):
                _trainer(root, first, steps=1, validation_interval=1).fit(train, valid)
            self.assertEqual(first.optimizer_updates, 1)
            self.assertEqual(first.scheduler_steps, 0)
            self.assertFalse((root / "best_generator.pt").exists())
            pending = torch.load(root / "step_00000001.pt", map_location="cpu", weights_only=False)
            self.assertEqual(pending["validation_pending"]["kind"], "recipe")
            self.assertEqual(pending["task"]["optimizer_updates"], 1)
            self.assertEqual(pending["task"]["scheduler_steps"], 0)
            with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "type": "validation",
                    "global_step": 1,
                    "pesq": {"status": "ok", "value": 999, "reason": None, "count": 1},
                    "best_checkpoint": "injected",
                }) + "\n")
            (root / "best_generator.json").write_text(json.dumps({"pesq": 999}), encoding="utf-8")
            second = _Task(fail_at=0)
            trainer = _trainer(root, second, steps=2, validation_interval=1)
            trainer.fit(train, valid, resume=root / "step_00000001.pt")
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(second.optimizer_updates, 2)
            self.assertEqual(second.scheduler_steps, 2)
            self.assertEqual(second.validations, 4)
            completed = torch.load(root / "step_00000001.pt", map_location="cpu", weights_only=False)
            self.assertNotIn("validation_pending", completed)
            self.assertEqual(completed["task"]["optimizer_updates"], 1)
            self.assertEqual(completed["task"]["scheduler_steps"], 1)
            best = json.loads((root / "best_generator.json").read_text(encoding="utf-8"))
            self.assertEqual(best["pesq"], 1.25)
            records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            validations = [row for row in records if row.get("type") == "validation"]
            self.assertEqual([row["global_step"] for row in validations], [1, 2])
            self.assertTrue(all(row["pesq"]["value"] == 1.25 for row in validations))
            self.assertTrue(all(row.get("validation_committed") for row in validations))

    def test_pesq_skip_does_not_increment_or_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = _Task(skip_first=True)
            trainer = _trainer(root, task, steps=1, validation_interval=1)
            trainer.fit(_Rows(2, "train"), _Rows(1, "valid"))
            self.assertEqual(task.updates, 2)
            self.assertEqual(task.optimizer_updates, 1)
            self.assertEqual(trainer.global_step, 1)
            self.assertEqual(task.scheduler_steps, 1)
            self.assertEqual(task.validations, 1)
            records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            skips = [row for row in records if row.get("type") == "pesq_skip"]
            self.assertEqual(len(skips), 1)
            self.assertEqual(skips[0]["last_skip"]["reason"], "pesq")
            self.assertEqual(skips[0]["global_step"], 0)
            self.assertEqual([row["global_step"] for row in records if row.get("type") == "train"], [1])

    def test_syncbn_count_mismatch_refuses_before_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = _Task()
            task.validation_requires_all_ranks = True
            config = {"seed": 1, "loss": "official", "model": {"name": "mossformergan"}, "best_checkpoint_start_epoch": 0}
            trainer = Trainer(
                task,
                plan_batches(1, 1, 1),
                ProcessInfo(rank=0, world_size=2, local_rank=0, device=torch.device("cpu"), distributed=True),
                max_steps=1,
                seed=1,
                checkpoint_dir=root,
                checkpoint_interval=0,
                validation_interval=1,
                config=config,
            )

            def all_gather(gathered, local):
                gathered[0].copy_(local)
                gathered[1].fill_(local.item() + 3)

            with patch("src.training.core.trainer.dist.is_initialized", return_value=True), patch(
                "src.training.core.trainer.dist.all_gather", side_effect=all_gather
            ):
                with self.assertRaisesRegex(TrainingError, "validation batch counts differ"):
                    trainer._validate(_Rows(2, "valid"), 0)
            self.assertEqual(task.validations, 0)


if __name__ == "__main__":
    unittest.main()
