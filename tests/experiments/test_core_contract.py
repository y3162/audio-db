"""Invariants the experiment core keeps while its responsibilities move.

These checks name the public format and the six official trainers. They do
not run a training step.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from src.experiments.evaluation.boundary import ResearchSession
from src.experiments.execution.checkpoint import FORMAT, SCHEMA_VERSION
from src.experiments.execution.direct import DirectSession
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
            self.assertTrue(callable(getattr(trainer, "fit", None)))
            self.assertTrue(callable(getattr(trainer, "state_dict", None)))
            self.assertTrue(callable(getattr(trainer, "load_state_dict", None)))

    def test_trainers_do_not_carry_a_checkpoint_manager(self) -> None:
        root = Path(__file__).resolve().parents[2]
        paths = [
            root / "src/experiments/training/trainers.py",
            root / "src/experiments/official/tf_gridnet/trainer.py",
            root / "src/experiments/official/tf_locoformer/trainer.py",
            root / "src/experiments/official/deepfilternet3/trainer.py",
            root / "src/experiments/official/mp_senet/trainer.py",
            root / "src/experiments/official/semambapp/trainer.py",
            root / "src/experiments/official/mossformergan/trainer.py",
        ]
        forbidden = ("CheckpointManager", "lifecycle.checkpoint", "checkpoint.directory", "checkpoint.offer")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path.name} still mentions {token}")
        for method_name in ("metrics", "score"):
            parameters = inspect.signature(getattr(ResearchSession, method_name)).parameters
            self.assertNotIn("checkpoint", parameters)
            self.assertIn("output_directory", parameters)
            self.assertIn("config_identity", parameters)

    def test_direct_session_publishes_directory_and_identity(self) -> None:
        class Stub:
            directory = Path("/tmp/audio-db-direct-surface")

            def offer(self, event: str, **kwargs: object) -> str:
                del kwargs
                return event

        session = DirectSession(Stub())
        session.start()
        self.assertEqual(session.output_directory, Path("/tmp/audio-db-direct-surface"))
        self.assertIsNone(session.config_identity)
        self.assertFalse(session.restored)
        self.assertEqual(session.save_if_needed("boundary", logical_step=1), "boundary")


if __name__ == "__main__":
    unittest.main()
