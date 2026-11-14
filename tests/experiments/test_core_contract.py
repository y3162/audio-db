"""Invariants the experiment core keeps while its responsibilities move.

These checks name the public format and the six official trainers. They do
not run a training step.
"""

from __future__ import annotations

import unittest

from src.experiments.execution.checkpoint import FORMAT, SCHEMA_VERSION
from src.experiments.execution.selector import EVENTS
from src.experiments.official.deepfilternet3.trainer import DeepFilterNet3Trainer
from src.experiments.official.mossformergan.trainer import MossFormerGANTrainer
from src.experiments.official.mp_senet.trainer import MPSENetGANTrainer
from src.experiments.official.semambapp.trainer import SEMambaPPTrainer
from src.experiments.official.tf_gridnet.trainer import TFGridNetTrainer
from src.experiments.official.tf_locoformer.trainer import TFLocoformerTrainer


class CoreContractTests(unittest.TestCase):
    def test_checkpoint_format_stays_experiment_v1(self) -> None:
        self.assertEqual(FORMAT, "audio-db-experiment-v1")
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(
            EVENTS,
            ("step", "partial", "boundary_pending", "boundary", "stop", "final"),
        )

    def test_official_trainers_remain_importable(self) -> None:
        trainers = (
            TFGridNetTrainer,
            TFLocoformerTrainer,
            DeepFilterNet3Trainer,
            MPSENetGANTrainer,
            SEMambaPPTrainer,
            MossFormerGANTrainer,
        )
        for trainer in trainers:
            self.assertTrue(callable(getattr(trainer, "run", None)) or callable(getattr(trainer, "fit", None)))
            self.assertTrue(callable(getattr(trainer, "state_dict", None)))
            self.assertTrue(callable(getattr(trainer, "load_state_dict", None)))


if __name__ == "__main__":
    unittest.main()
