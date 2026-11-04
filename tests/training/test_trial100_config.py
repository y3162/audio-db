"""Trial configs load and name the VoiceBank splits."""

import unittest
from pathlib import Path

from src.training.se.config import load_training_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "training" / "trial100"


class TrialConfigTests(unittest.TestCase):
    def test_five_models_use_voicebank_and_one_hundred_steps(self):
        names = []
        for path in sorted(CONFIGS.glob("*.yaml")):
            config = load_training_config(path)
            names.append(config["model"]["name"])
            self.assertEqual(config["max_steps"], 100)
            self.assertEqual(config["data"]["train_set"], "voicebank_28_train")
            self.assertEqual(config["data"]["validation_set"], "voicebank_28_valid")
            self.assertNotEqual(config["data"]["database"], "data/db/resource.duckdb")
        self.assertEqual(
            names,
            ["deepfilternet3", "mossformergan", "mp_senet", "semambapp", "tf_gridnet"],
        )


if __name__ == "__main__":
    unittest.main()
