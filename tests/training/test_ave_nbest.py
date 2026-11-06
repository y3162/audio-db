"""ESPnet-style top-5 validation-loss average for chunk checkpoints."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.core.average import average_checkpoint_files, average_state_dicts
from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from tests.training.support import cpu_process


class _Payload:
    def __init__(self) -> None:
        self.weight = 0.0
        self.tracked = 0

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        return {
            "weight": torch.tensor([self.weight], dtype=torch.float32),
            "tracked": torch.tensor([self.tracked], dtype=torch.int64),
        }


def _summary(loss: float) -> dict:
    metric = {"status": "ok", "value": loss, "reason": None}
    return {"losses": {"generator": metric}, "si_snr": {"status": "ok", "value": 1.0, "reason": None}}


class AverageNBestTests(unittest.TestCase):
    def _trainer(self, root: Path, keep: int = 5) -> tuple[Trainer, _Payload]:
        task = _Payload()
        trainer = Trainer(
            task,
            plan_batches(1, 1, 1),
            cpu_process(),
            max_steps=1,
            seed=1,
            checkpoint_dir=root,
            checkpoint_interval=0,
            validation_interval=0,
            config={
                "best_checkpoint_start_epoch": 0,
                "micro_batch_size_per_gpu": 1,
                "chunk_iterator": {"keep_nbest": keep},
            },
        )
        return trainer, task

    def _retain(self, trainer: Trainer, task: _Payload, epoch: int, loss: float) -> None:
        task.weight = float(epoch)
        task.tracked = epoch
        trainer.global_step = epoch * 10
        trainer._retain_chunk_models(epoch, _summary(loss))

    def test_fewer_than_five_links_the_best_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root)
            for epoch in range(1, 5):
                self._retain(trainer, task, epoch, float(epoch))
            self.assertFalse((root / "valid.loss.ave_5best.pt").exists())
            self.assertEqual(len(trainer.chunk_history), 4)
            link = root / "valid.loss.ave_1best.pt"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.readlink().as_posix(), "1epoch.pt")
            self.assertEqual((root / "valid.loss.ave.pt").readlink().as_posix(), "valid.loss.ave_1best.pt")
            loaded = torch.load(link, weights_only=True)
            self.assertTrue(torch.equal(loaded["weight"], torch.tensor([1.0])))

    def test_histories_from_one_through_six(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root)
            for epoch in range(1, 7):
                self._retain(trainer, task, epoch, float(epoch))
                average = root / "valid.loss.ave_5best.pt"
                best = root / "valid.loss.ave_1best.pt"
                if epoch < 5:
                    self.assertFalse(average.exists())
                    self.assertTrue(best.is_symlink())
                    self.assertEqual(best.readlink().as_posix(), "1epoch.pt")
                else:
                    self.assertTrue(average.is_file())
                    self.assertFalse(average.is_symlink())
                    self.assertEqual((root / "valid.loss.ave.pt").readlink().as_posix(), average.name)

    def test_top_five_mean_excludes_a_worse_epoch_and_sums_integers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root)
            # Losses put epoch 6 outside the top five. Ties keep the earlier epoch.
            rows = [(1, 1.0), (2, 1.0), (3, 2.0), (4, 3.0), (5, 4.0), (6, 9.0)]
            for epoch, loss in rows:
                self._retain(trainer, task, epoch, loss)
            path = root / "valid.loss.ave_5best.pt"
            loaded = torch.load(path, weights_only=True)
            self.assertTrue(torch.equal(loaded["weight"], torch.tensor([3.0])))
            self.assertTrue(torch.equal(loaded["tracked"], torch.tensor([15], dtype=torch.int64)))
            meta = json.loads((root / f"{path.name}.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["epochs"], [1, 2, 3, 4, 5])
            self.assertEqual(meta["values"], [1.0, 1.0, 2.0, 3.0, 4.0])
            self.assertEqual(meta["files"], ["1epoch.pt", "2epoch.pt", "3epoch.pt", "4epoch.pt", "5epoch.pt"])
            self.assertFalse((root / "6epoch.pt").exists())

    def test_non_finite_loss_is_not_ranked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root)
            self._retain(trainer, task, 1, 1.0)
            trainer._retain_chunk_models(2, _summary(float("nan")))
            self.assertEqual([row["epoch"] for row in trainer.chunk_history], [1])
            self.assertFalse((root / "2epoch.pt").exists())

    def test_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = {"weight": torch.tensor([1.0])}
            second = {"weight": torch.tensor([3.0])}
            torch.save(first, root / "1epoch.pt")
            torch.save(second, root / "2epoch.pt")
            averaged = average_checkpoint_files([root / "1epoch.pt", root / "2epoch.pt"])
            self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.0])))
            (root / "2epoch.pt").unlink()
            with self.assertRaises(TrainingError):
                average_checkpoint_files([root / "1epoch.pt", root / "2epoch.pt"])

    def test_resume_recomputes_the_average_from_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root)
            for epoch, loss in ((1, 4.0), (2, 3.0), (3, 2.0), (4, 1.0)):
                self._retain(trainer, task, epoch, loss)
            resumed, resumed_task = self._trainer(root)
            resumed.chunk_history = list(trainer.chunk_history)
            self._retain(resumed, resumed_task, 5, 0.5)
            loaded = torch.load(root / "valid.loss.ave_5best.pt", weights_only=True)
            self.assertTrue(torch.equal(loaded["weight"], torch.tensor([3.0])))

    def test_hand_average_matches_the_float_mean(self):
        states = [
            {"weight": torch.tensor([1.0, 2.0]), "tracked": torch.tensor(1)},
            {"weight": torch.tensor([3.0, 6.0]), "tracked": torch.tensor(4)},
        ]
        averaged = average_state_dicts(states)
        self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.equal(averaged["tracked"], torch.tensor(5)))
