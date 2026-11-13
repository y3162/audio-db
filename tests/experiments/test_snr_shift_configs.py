"""SNR-shift launch declarations share data and exclude the test split."""

import unittest
from pathlib import Path

from src.experiments.config import load_experiment

ROOT = Path(__file__).resolve().parents[2]


class SNRShiftConfigTests(unittest.TestCase):
    def test_batch_and_validation_configuration(self):
        path = ROOT / "configs/reproduction/experiments/snr_shift"
        expected_batch = {"mp_senet": 4, "semambapp": 8, "tf_gridnet": 4, "tf_locoformer": 4}
        outputs = set()
        for model, global_batch in expected_batch.items():
            with self.subTest(model=model):
                config = load_experiment(path / f"{model}.yaml")
                source = config.components["source"].kwargs
                self.assertEqual(source["snapshot"], f"voicebank.snr_shift.{model}@1")
                self.assertEqual(source["snapshot_root"], "data/db/snapshots-snr-shift-v1-perrank")
                self.assertEqual(source["split"], "train")
                self.assertNotIn("allow_busy_gpu", config.runtime_kwargs)
                batch = config.components["batcher"].kwargs
                accumulation = config.components["trainer"].kwargs["accumulation"]
                self.assertEqual(batch["group_size"], accumulation)
                self.assertEqual(batch["micro_batch"] * accumulation, global_batch)
                outputs.add(config.output_directory)
                if model in ("tf_gridnet", "tf_locoformer"):
                    self.assertEqual(config.components["trainer"].kwargs["early_stopping_patience"], 10)
                else:
                    policy = config.components["early_stop"].kwargs
                    self.assertEqual(policy["metric"], "validation.pesq")
                    self.assertEqual(policy["patience"], 10)
                    self.assertEqual(policy["mode"], "max")
        self.assertEqual(len(outputs), 4)
