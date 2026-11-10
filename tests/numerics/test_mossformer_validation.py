"""MossFormerGAN validation uses the released generator objective."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.experiments.calc.rng import set_determinism
from src.experiments.official.mossformergan.reference import MossFormerGANTask, _GAN, _MAG, _RI, _TIME
from tests.numerics.support import paired_dataset, tiny_config


def tearDownModule():
    set_determinism(False, cuda=False)


def _task(root: Path) -> tuple[MossFormerGANTask, object]:
    config = tiny_config(
        root / "ckpt",
        loss="official",
        segment_size=8000,
        global_batch_size=1,
        micro_batch_size_per_gpu=1,
        max_steps=1,
    )
    config["optimizer"]["lr"] = 5e-4
    config["grad_clip"] = 10.0
    config["scheduler"] = {"name": "none"}
    config["model"] = {
        "name": "mossformergan",
        "variant": "se_16k",
        "sampling_rate": 16000,
        "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
    }
    data = paired_dataset(root / "wav", 1, frames=8000, segment_size=8000, seed=config["seed"])
    return MossFormerGANTask(config, torch.device("cpu")), data


class MossFormerValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_validation_is_the_weighted_generator_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            task, data = _task(Path(directory))
            batch = data.collate([0])
            with patch("src.experiments.official.mossformergan.reference.metric_targets", return_value=torch.ones(1)):
                task.eval_mode()
                self.assertFalse(task.generator.training)
                self.assertTrue(task.discriminator.training)
                before = [parameter.detach().clone() for parameter in task.discriminator.parameters()]
                report = task.validation_report(batch)
                after = [parameter.detach() for parameter in task.discriminator.parameters()]
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, after, strict=True)))
            losses = report["losses"]
            expected = (
                _RI * losses["ri"]["value"]
                + _MAG * losses["magnitude"]["value"]
                + _TIME * losses["time"]["value"]
                + _GAN * losses["gan"]["value"]
            )
            self.assertAlmostEqual(losses["generator"]["value"], expected, places=5)
            self.assertNotAlmostEqual(losses["generator"]["value"], losses["time"]["value"], places=4)
            self.assertEqual(report["pesq_gate"], "included")
            self.assertEqual(losses["discriminator"]["status"], "na")
            self.assertTrue(task.discriminator.training)

    def test_failed_pesq_counts_as_zero_without_an_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            task, data = _task(Path(directory))
            batch = data.collate([0])
            before = [parameter.detach().clone() for parameter in task.generator.parameters()]
            with patch("src.experiments.official.mossformergan.reference.metric_targets", return_value=None):
                report = task.validation_report(batch)
            after = [parameter.detach() for parameter in task.generator.parameters()]
            self.assertEqual(report["pesq_gate"], "denominator_only")
            self.assertIsNone(report["losses"]["generator"]["value"])
            self.assertEqual(report["losses"]["generator"]["status"], "na")
            self.assertFalse(report["generator_loss_included"])
            reduced = task.reduce_validation([report])
            self.assertEqual(reduced["losses"]["generator"]["value"], 0.0)
            self.assertEqual(reduced["validation_coverage"]["pesq_failed_batches"], 1)
            self.assertEqual(reduced["pesq"]["value"], None)
            self.assertEqual(reduced["pesq"]["status"], "na")
            self.assertEqual(report["losses"]["time"]["status"], "na")
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, after, strict=True)))
