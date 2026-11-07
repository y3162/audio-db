"""A PESQ skip must not move MossFormer plateau off the data-epoch end."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer
from src.training.se.mossformergan import MossFormerGANTask
from tests.training.support import cpu_process


class _Stream:
    def __init__(self, render) -> None:
        self.updates = 2
        self.plan_hash = "ab" * 32
        self.render = render


class _EpochData:
    """Four micro-batches, grouped by the trainer into two accumulation windows."""

    def epoch(self, epoch: int, *, rank: int, world_size: int, start_update: int = 0):
        del epoch, rank, world_size, start_update

        def render(update_index: int):
            base = update_index * 2
            return [
                {"case_id": [f"c{base}"], "index": base},
                {"case_id": [f"c{base + 1}"], "index": base + 1},
            ]

        return _Stream(render)

    def validation(self, *, rank: int, world_size: int, epoch: int):
        del rank, world_size, epoch
        yield {"case_id": ["valid"], "index": -1}

    def data_state(self, epoch: int, committed_updates: int) -> dict:
        return {"epoch": epoch, "update_in_epoch": committed_updates, "plan_hash": "ab" * 32}

    def identity(self) -> dict:
        return {"name": "epoch-plateau", "role": "train"}


class _PlateauTask:
    validate_on_data_epoch_end = True
    stop_training = False

    def __init__(self, fail_index: int | None) -> None:
        self.fail_index = fail_index
        self.scheduler_values: list[float | None] = []
        self._accu = 0

    def distribute(self, process) -> None:
        del process

    def train_update(self, batches, accumulation: int) -> dict[str, float]:
        applied = False
        for batch in batches:
            if batch["index"] == self.fail_index:
                continue
            self._accu += 1
            if self._accu == accumulation:
                applied = True
                self._accu = 0
        return {"loss": 1.0, "update_applied": 1.0 if applied else 0.0}

    def step_schedulers(self, validation) -> None:
        value = None
        if validation:
            item = (validation.get("losses") or {}).get("generator") or {}
            if item.get("status") == "ok":
                value = float(item["value"])
        self.scheduler_values.append(value)
        self.stop_training = True

    def eval_mode(self) -> None:
        return None

    def train_mode(self) -> None:
        return None

    def learning_rates(self) -> dict[str, float]:
        return {"generator": 0.001}

    def state_dict(self) -> dict:
        return {"loss": "official"}

    def export_for_inference(self):
        return None

    def validation_report(self, batch) -> dict:
        del batch
        return {
            "losses": {"generator": {"status": "ok", "value": 0.5, "reason": None}},
            "pesq": {"status": "na", "value": None, "reason": "not the plateau metric"},
            "stoi": {"status": "na", "value": None, "reason": "not computed"},
        }


class MossFormerEpochPlateauTests(unittest.TestCase):
    def test_task_declares_data_epoch_validation(self) -> None:
        self.assertTrue(MossFormerGANTask.validate_on_data_epoch_end)

    def _fit(self, root: Path, fail_index: int | None) -> _PlateauTask:
        task = _PlateauTask(fail_index)
        trainer = Trainer(
            task,
            plan_batches(2, 1, 1),
            cpu_process(),
            max_steps=4,
            seed=1,
            checkpoint_dir=root,
            checkpoint_interval=0,
            validation_interval=2,
            config={
                "seed": 1,
                "loss": "official",
                "model": {"name": "mossformergan"},
                "best_checkpoint_start_epoch": 40,
                "data": {},
            },
        )
        trainer.fit(_EpochData(), None)
        return task

    def test_clean_epoch_passes_validation_to_plateau(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self._fit(Path(directory), None)
        self.assertEqual(task.scheduler_values, [0.5])

    def test_first_micro_batch_failure_still_passes_epoch_end_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self._fit(Path(directory), 0)
        self.assertEqual(task.scheduler_values, [0.5])

    def test_last_micro_batch_failure_still_passes_epoch_end_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self._fit(Path(directory), 3)
        self.assertEqual(task.scheduler_values, [0.5])


if __name__ == "__main__":
    unittest.main()
