"""MossFormerGAN validation uses the released generator objective."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.training.core.rng import set_determinism
from src.training.se.config import load_training_config
from src.training.se.mossformergan import MossFormerGANTask, _GAN, _MAG, _RI, _TIME
from tests.training.support import paired_dataset, tiny_config


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
            with patch("src.training.se.mossformergan.metric_targets", return_value=torch.ones(1)):
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
            with patch("src.training.se.mossformergan.metric_targets", return_value=None):
                report = task.validation_report(batch)
            after = [parameter.detach() for parameter in task.generator.parameters()]
            self.assertEqual(report["pesq_gate"], "counted_as_zero")
            self.assertEqual(report["losses"]["generator"]["value"], 0.0)
            self.assertEqual(report["losses"]["time"]["status"], "na")
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, after, strict=True)))

    def test_smoke_limits_are_optional_config_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = """
seed: 1
deterministic: false
global_batch_size: 4
micro_batch_size_per_gpu: 1
max_steps: 2
segment_size: 32000
loss: official
checkpoint_dir: %s
checkpoint_interval: 2
validation_interval: 2
optimizer:
  name: adamw
  lr: 0.0005
  betas: [0.9, 0.999]
  weight_decay: 0.01
scheduler:
  name: none
model:
  name: mossformergan
  variant: se_16k
  sampling_rate: 16000
  architecture: {}
data:
  pipeline: reproduction
  dataset: mossformergan.vb28.ondemand.official@1
  allow_test_selection: true
  max_updates_per_epoch: 2
  validation_max_batches: 2
runtime:
  device: cpu
  allow_busy_gpu: false
chunk_iterator:
  num_samples_per_epoch: 4
  chunk_length: 48000
  max_epochs: 1
  max_updates: 2
  validation_max_updates: 1
""" % (root / "ckpt")
            path = root / "config.yaml"
            path.write_text(text, encoding="utf-8")
            # chunk_iterator cannot be combined with this reproduction check only when
            # logical epochs are set. The parser accepts both keys; the runner rejects
            # chunk_iterator without the matching dataset later. Parse the two maps
            # by loading a reproduction file and a chunk file separately.
            path.write_text(text.split("chunk_iterator:")[0], encoding="utf-8")
            parsed = load_training_config(path)
            self.assertEqual(parsed["data"]["max_updates_per_epoch"], 2)
            self.assertEqual(parsed["data"]["validation_max_batches"], 2)
            chunk = root / "chunk.yaml"
            chunk.write_text(
                text.replace("pipeline: reproduction", "pipeline: legacy")
                .replace("dataset: mossformergan.vb28.ondemand.official@1\n", "")
                .replace("allow_test_selection: true\n", "")
                .replace("max_updates_per_epoch: 2\n", "")
                .replace("validation_max_batches: 2\n", "")
                .replace("data:\n", "data:\n  database: x\n  train_set: a\n  validation_set: b\n"),
                encoding="utf-8",
            )
            # The replacement above is too brittle. Build the chunk file directly.
            chunk.write_text(
                f"""
seed: 1
deterministic: false
global_batch_size: 4
micro_batch_size_per_gpu: 2
max_steps: 2
segment_size: 48000
loss: official
checkpoint_dir: {root / "ckpt"}
checkpoint_interval: 2
validation_interval: 0
optimizer:
  name: adam
  lr: 0.001
  betas: [0.9, 0.999]
  weight_decay: 0.0
scheduler:
  name: plateau
  factor: 0.7
  patience: 1
model:
  name: tf_gridnet
  variant: ms-snsd
  sampling_rate: 16000
  architecture: {{}}
data:
  database: {root / "db"}
  train_set: train
  validation_set: valid
chunk_iterator:
  num_samples_per_epoch: 4
  chunk_length: 48000
  max_epochs: 1
  max_updates: 2
  validation_max_updates: 1
runtime:
  device: cpu
  allow_busy_gpu: false
""",
                encoding="utf-8",
            )
            chunk_config = load_training_config(chunk)
            self.assertEqual(chunk_config["chunk_iterator"]["max_updates"], 2)
            self.assertEqual(chunk_config["chunk_iterator"]["validation_max_updates"], 1)
            self.assertNotIn("max_updates", load_training_config(path).get("chunk_iterator", {}) or {})
