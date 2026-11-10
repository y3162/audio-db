"""Phase 5 resource profile stays off unless the environment variable is set."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.experiments.api import Experiment
from src.experiments.config import load_experiment

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tests/fixtures/configs/experiments/linear_supervised.yaml"


class Phase5ProfileTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("AUDIO_DB_RESOURCE_PROFILE", None)

    def test_profile_is_absent_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "out"
            experiment = Experiment(_config(directory, steps=1))
            experiment.run()
            self.assertFalse((directory / "resource_profile.json").exists())

    def test_profile_records_backward_and_checkpoint_without_changing_the_payload(self) -> None:
        os.environ["AUDIO_DB_RESOURCE_PROFILE"] = "1"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "out"
            experiment = Experiment(_config(directory, steps=1))
            experiment.run()
            payload = json.loads((directory / "resource_profile.json").read_text(encoding="utf-8"))
            names = [item["name"] for item in payload["events"]]
            self.assertIn("backward", names)
            self.assertIn("optimizer_step", names)
            self.assertIn("replicated_state_compare", names)
            self.assertIn("checkpoint_write", names)
            checkpoint = directory / "step_00000001.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((directory / "step_00000001.data_provenance.json").is_file())
            self.assertTrue((directory / "data_provenance.json").is_file())


def _config(directory: Path, steps: int):
    text = EXAMPLE.read_text(encoding="utf-8")
    text = text.replace("directory: runs/linear-supervised", f"directory: {directory}")
    text = text.replace("max_logical_steps: 4", f"max_logical_steps: {steps}")
    path = directory.parent / "linear.yaml"
    path.write_text(text, encoding="utf-8")
    return load_experiment(path)


if __name__ == "__main__":
    unittest.main()
