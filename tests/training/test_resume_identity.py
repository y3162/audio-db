"""Resume refuses every config change except logging."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.core.batch import plan_batches
from src.training.core.checkpoint import save_atomic
from src.training.core.trainer import Trainer
from src.training.errors import TrainingError
from src.training.provenance import build_provenance
from src.training.se.config import load_training_config
from src.training.se.preflight import resume_config_mismatch
from src.training.se.run import execute
from tests.training.support import cpu_process, tiny_config


_YAML = """
seed: 1
deterministic: false
global_batch_size: 1
micro_batch_size_per_gpu: 1
required_world_size: 1
max_steps: 1
segment_size: 16
loss: waveform_l1
checkpoint_dir: {checkpoint}
checkpoint_interval: 0
validation_interval: 0
optimizer: {{name: adamw, lr: 0.001, betas: [0.9, 0.999], weight_decay: 0.0}}
scheduler: {{name: none}}
model:
  name: tf_gridnet
  sampling_rate: 16000
  architecture: {{}}
  stft: {{}}
data:
  database: {database}
  train_set: toy_train
  validation_set: toy_valid
runtime:
  device: cpu
  allow_busy_gpu: false
"""


class _Identity:
    def __init__(self, name: str):
        self.name = name

    def identity(self):
        return {"name": self.name, "role": "train", "snapshot_hash": self.name}


class ResumeIdentityTests(unittest.TestCase):
    def test_only_logging_may_differ(self):
        saved = {
            "logging": {"tensorboard": {"enabled": False}},
            "seed": 1,
            "checkpoint_dir": "runs/a",
            "verification_max_updates": 2,
            "validation_interval": 10,
            "data": {"max_updates_per_epoch": 4, "validation_max_batches": 1},
            "chunk_iterator": {"keep_nbest": 5, "max_updates": 2},
        }
        logging_only = dict(saved)
        logging_only["logging"] = {"tensorboard": {"enabled": True}}
        self.assertEqual(resume_config_mismatch(saved, logging_only), [])
        unlimited = dict(saved)
        unlimited.pop("verification_max_updates")
        self.assertIn("verification_max_updates", resume_config_mismatch(saved, unlimited))
        moved = dict(saved)
        moved["checkpoint_dir"] = "runs/b"
        self.assertEqual(resume_config_mismatch(saved, moved), ["checkpoint_dir"])
        other_valid = dict(saved)
        other_valid["data"] = {"max_updates_per_epoch": 4, "validation_max_batches": 8}
        self.assertEqual(resume_config_mismatch(saved, other_valid), ["data"])

    def test_dataset_identity_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(Path(directory), seed=1, loss="mock", max_steps=1)
            trainer = Trainer(
                object(),
                plan_batches(1, 1, 1),
                cpu_process(),
                max_steps=1,
                seed=1,
                checkpoint_dir=Path(directory),
                checkpoint_interval=0,
                validation_interval=0,
                config=config,
            )
            payload = {
                "batch_plan": trainer.plan.as_dict(),
                "config": config,
                "dataset": _Identity("saved").identity(),
            }
            with self.assertRaisesRegex(TrainingError, "dataset identity"):
                trainer._check_resume(payload, _Identity("other"))

    def test_resume_validation_dataset_mismatch_rejects_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "train.yaml"
            checkpoint_dir = root / "ckpt"
            config_path.write_text(
                _YAML.format(checkpoint=checkpoint_dir, database=root / "unused.duckdb"),
                encoding="utf-8",
            )
            config = load_training_config(config_path)
            saved = dict(config)
            saved["data"] = dict(config["data"])
            saved["data"]["validation_set"] = "other_valid"
            checkpoint = root / "step_00000001.pt"
            save_atomic(
                checkpoint,
                {
                    "format": "audio-db-training-v1",
                    "config": saved,
                    "global_step": 1,
                    "epoch": 0,
                    "micro_batches_consumed": 1,
                },
            )

            def boom(*_args, **_kwargs):
                raise AssertionError("build_task was called")

            with patch("src.training.se.run.build_task", side_effect=boom):
                with self.assertRaisesRegex(TrainingError, "does not match"):
                    execute(config_path, resume=checkpoint, strict_resume=True)

    def test_full_profile_checks_resume_even_when_strict_is_false(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "train.yaml"
            checkpoint_dir = root / "ckpt"
            text = _YAML.format(checkpoint=checkpoint_dir, database=root / "unused.duckdb")
            text = text.replace("loss: waveform_l1\n", "loss: waveform_l1\nexperiment_profile: full\n")
            config_path.write_text(text, encoding="utf-8")
            config = load_training_config(config_path)
            saved = dict(config)
            saved["data"] = dict(config["data"])
            saved["data"]["validation_set"] = "other_valid"
            checkpoint = root / "step_00000001.pt"
            save_atomic(
                checkpoint,
                {
                    "format": "audio-db-training-v1",
                    "config": saved,
                    "global_step": 1,
                    "epoch": 0,
                    "micro_batches_consumed": 1,
                },
            )

            def boom(*_args, **_kwargs):
                raise AssertionError("build_task was called")

            with patch("src.training.se.run.build_task", side_effect=boom):
                with self.assertRaisesRegex(TrainingError, "does not match"):
                    execute(config_path, resume=checkpoint, strict_resume=False)

    def test_provenance_records_git_dirty(self):
        document = build_provenance(
            _Identity("n"),
            {"model": {"name": "mp_senet"}, "seed": 1, "loss": "official"},
            cpu_process(),
        )
        self.assertIn(document["git_dirty"], (True, False, None))
        self.assertTrue(document["git_commit"])


if __name__ == "__main__":
    unittest.main()
