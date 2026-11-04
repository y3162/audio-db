"""A failed checkpoint verification must not publish an incomplete file."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.training.core.checkpoint import load_checkpoint, save_atomic
from src.training.errors import TrainingError


class AtomicCheckpointTests(unittest.TestCase):
    def test_verification_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"previous checkpoint")
            with patch("src.training.core.checkpoint.load_checkpoint", side_effect=TrainingError("corrupt")):
                with self.assertRaisesRegex(TrainingError, "corrupt"):
                    save_atomic(path, {"format": "audio-db-training-v1"}, verify_training=True)
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_verified_checkpoint_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            payload = {"format": "audio-db-training-v1", "global_step": 3, "epoch": 1,
                       "micro_batches_consumed": 2, "task": {}, "rng": {}, "dataset": {},
                       "config": {}, "batch_plan": {}}
            save_atomic(path, payload, verify_training=True)
            self.assertEqual(load_checkpoint(path, torch.device("cpu")), payload)


if __name__ == "__main__":
    unittest.main()
