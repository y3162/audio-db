"""Training configuration and task construction reject unsupported families."""
import unittest

import torch

from src.training.errors import TrainingError
from src.training.se.config import _model
from src.training.se.run import build_task


class RemovedModelsTests(unittest.TestCase):
    def test_removed_models_are_rejected(self):
        for name in ("unipase", "stupase"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(TrainingError, "model.name must be one of"):
                    _model({"name": name}, "official")
                with self.assertRaisesRegex(TrainingError, "no training task"):
                    build_task({"model": {"name": name}, "loss": "official"}, torch.device("cpu"))
