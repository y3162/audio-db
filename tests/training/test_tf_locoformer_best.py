"""Research best-checkpoint rule for TF-Locoformer.

``valid.si_snr.best.pt`` maximizes positive SI-SDR in dB. ``valid.loss.best.pt``
minimizes validation loss. The training log key ``si_sdr`` stays the negative
SI-SDR loss. ESPnet's DNS recipe asks ``best_model_criterion`` for ``valid`` /
``si_snr`` / ``max``, but ``SISNRLoss`` records ``si_snr_loss`` as that negative
loss. Keeping the positive SI-SDR ranking is intentional.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.core.batch import plan_batches
from src.training.core.trainer import Trainer
from src.training.se.spectral import si_sdr_loss
from src.training.se.tf_locoformer import TFLocoformerTask, _positive_si_sdr
from tests.se.test_tf_locoformer import TINY
from tests.training.support import cpu_process, paired_dataset, tiny_config


class _Marked:
    def __init__(self) -> None:
        self.mark = 0

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        return {
            "mark": torch.tensor([float(self.mark)]),
            "tracked": torch.tensor([self.mark], dtype=torch.int64),
        }


def _summary(loss: float, si_sdr_db: float) -> dict:
    metric = {"status": "ok", "value": loss, "reason": None}
    return {
        "losses": {"generator": metric},
        "si_snr": {"status": "ok", "value": si_sdr_db, "reason": None},
    }


class PositiveSISDRSelectionTests(unittest.TestCase):
    def _trainer(self, root: Path, keep: int) -> tuple[Trainer, _Marked]:
        task = _Marked()
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

    def _retain(self, trainer: Trainer, task: _Marked, epoch: int, loss: float, si_sdr_db: float) -> None:
        task.mark = epoch
        trainer.global_step = epoch * 10
        trainer._retain_chunk_models(epoch, _summary(loss, si_sdr_db))

    def test_higher_si_sdr_becomes_the_si_snr_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root, keep=5)
            self._retain(trainer, task, 1, 3.0, 5.0)
            self._retain(trainer, task, 2, 2.0, 10.0)
            chosen = torch.load(root / "valid.si_snr.best.pt", weights_only=True)
            meta = json.loads((root / "valid.si_snr.best.pt.json").read_text(encoding="utf-8"))
            self.assertEqual(int(chosen["mark"]), 2)
            self.assertEqual(meta["metric"], "si_snr")
            self.assertEqual(meta["epoch"], 2)
            self.assertEqual(meta["value"], 10.0)
            self.assertEqual(meta["global_step"], 20)

    def test_loss_minimum_and_si_sdr_maximum_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root, keep=5)
            self._retain(trainer, task, 1, 3.0, 5.0)
            self._retain(trainer, task, 2, 2.0, 10.0)
            self._retain(trainer, task, 3, 0.5, 6.0)
            si = torch.load(root / "valid.si_snr.best.pt", weights_only=True)
            loss = torch.load(root / "valid.loss.best.pt", weights_only=True)
            si_meta = json.loads((root / "valid.si_snr.best.pt.json").read_text(encoding="utf-8"))
            loss_meta = json.loads((root / "valid.loss.best.pt.json").read_text(encoding="utf-8"))
            self.assertEqual(int(si["mark"]), 2)
            self.assertEqual(si_meta["epoch"], 2)
            self.assertEqual(si_meta["value"], 10.0)
            self.assertEqual(int(loss["mark"]), 3)
            self.assertEqual(loss_meta["metric"], "loss")
            self.assertEqual(loss_meta["epoch"], 3)
            self.assertEqual(loss_meta["value"], 0.5)

    def test_keep_nbest_five_keeps_the_union_of_the_two_rankings(self):
        # loss rank (low first): 7, 6, 5, 4, 3
        # si-sdr rank (high first, then smaller epoch): 1, 3, 5, 7, 6
        rows = [
            (1, 7.0, 10.0),
            (2, 6.0, 1.0),
            (3, 5.0, 9.0),
            (4, 4.0, 2.0),
            (5, 3.0, 8.0),
            (6, 2.0, 3.0),
            (7, 1.0, 7.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, task = self._trainer(root, keep=5)
            for epoch, loss, si in rows:
                self._retain(trainer, task, epoch, loss, si)
            kept = sorted(
                int(path.name.split("epoch")[0])
                for path in root.glob("*epoch.pt")
                if path.name[0].isdigit()
            )
            self.assertEqual(kept, [1, 3, 4, 5, 6, 7])
            self.assertFalse((root / "2epoch.pt").exists())
            si = torch.load(root / "valid.si_snr.best.pt", weights_only=True)
            loss = torch.load(root / "valid.loss.best.pt", weights_only=True)
            self.assertEqual(int(si["mark"]), 1)
            self.assertEqual(int(loss["mark"]), 7)
            si_meta = json.loads((root / "valid.si_snr.best.pt.json").read_text(encoding="utf-8"))
            loss_meta = json.loads((root / "valid.loss.best.pt.json").read_text(encoding="utf-8"))
            self.assertEqual(si_meta, {"epoch": 1, "global_step": 10, "metric": "si_snr", "value": 10.0})
            self.assertEqual(loss_meta, {"epoch": 7, "global_step": 70, "metric": "loss", "value": 1.0})
            averaged = torch.load(root / "valid.loss.ave_5best.pt", weights_only=True)
            self.assertEqual(float(averaged["mark"]), 5.0)
            self.assertEqual(int(averaged["tracked"]), 7 + 6 + 5 + 4 + 3)
            self.assertEqual(os.readlink(root / "valid.loss.ave.pt"), "valid.loss.ave_5best.pt")
            average_meta = json.loads((root / "valid.loss.ave_5best.pt.json").read_text(encoding="utf-8"))
            self.assertEqual(average_meta["epochs"], [7, 6, 5, 4, 3])
            self.assertEqual(average_meta["average"], "espnet_nbest_float_mean_int_sum")


class OfficialSISNRStatsTests(unittest.TestCase):
    """Confirm the DNS recipe's stats key and sign on the pinned ESPnet tree."""

    def test_dns_si_snr_stats_are_the_negative_loss_under_si_snr_loss(self):
        esp = Path(__file__).resolve().parents[2] / "src/se/official/espnet"
        if not (esp / "espnet2/enh/loss/criterions/time_domain.py").is_file():
            self.skipTest("official ESPnet checkout missing")
        import sys

        inserted = str(esp) not in sys.path
        sys.path.insert(0, str(esp))
        try:
            from espnet2.enh.loss.criterions.time_domain import SISNRLoss
            from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver
        except ModuleNotFoundError as exc:
            self.skipTest(f"official ESPnet loss import unavailable: {exc}")
        finally:
            if inserted and sys.path and sys.path[0] == str(esp):
                sys.path.pop(0)
        criterion = SISNRLoss(eps=1e-7)
        self.assertEqual(criterion.name, "si_snr_loss")
        reference = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
        estimate = reference * 0.5
        loss, stats, _other = FixedOrderSolver(criterion, weight=0.0)([reference], [estimate])
        self.assertEqual(set(stats), {"si_snr_loss"})
        self.assertNotIn("si_snr", stats)
        self.assertLess(float(stats["si_snr_loss"]), 0.0)
        self.assertAlmostEqual(float(stats["si_snr_loss"]), float(loss))


class SISDRSignTests(unittest.TestCase):
    def test_helper_reports_si_sdr_not_the_negative_loss(self):
        clean = torch.tensor([[1.0, -0.5, 0.25, -0.25]])
        estimate = clean * 0.8
        loss = float(si_sdr_loss(estimate, clean))
        metric = _positive_si_sdr(clean, estimate)
        self.assertLess(loss, 0.0)
        self.assertGreater(metric["value"], 0.0)
        self.assertAlmostEqual(metric["value"], -loss)
        self.assertEqual(metric["status"], "ok")

    def test_task_log_and_validation_use_opposite_signs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = paired_dataset(root / "wav", 1, frames=512, segment_size=512, seed=11)
            config = tiny_config(
                root,
                loss="official",
                segment_size=512,
                global_batch_size=1,
                micro_batch_size_per_gpu=1,
                max_steps=1,
            )
            config["optimizer"] = {"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01}
            config["grad_clip"] = 5.0
            config["scheduler"] = {"name": "none"}
            config["model"] = {
                "name": "tf_locoformer",
                "variant": "dns",
                "stage": None,
                "sampling_rate": 16000,
                "architecture": dict(TINY),
                "stft": {},
                "components": {},
            }
            task = TFLocoformerTask(config, torch.device("cpu"))
            batch = data.collate([0])
            logged = []
            original = si_sdr_loss

            def capture(estimate, reference, eps: float = 1e-8):
                value = original(estimate, reference, eps)
                logged.append(float(value.detach()))
                return value

            import src.training.se.tf_locoformer as module

            module.si_sdr_loss = capture
            try:
                metrics = task.train_update([batch], 1)
                report = task.validation_report(batch)
            finally:
                module.si_sdr_loss = original
        self.assertEqual(len(logged), 2)
        self.assertAlmostEqual(metrics["si_sdr"], logged[0])
        self.assertAlmostEqual(report["si_snr"]["value"], -logged[1])
        self.assertNotEqual(metrics["si_sdr"], report["si_snr"]["value"])


if __name__ == "__main__":
    unittest.main()
