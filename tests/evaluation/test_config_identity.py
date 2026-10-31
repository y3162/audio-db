"""Configuration schema and process-stable seeds."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from src.evaluation.config import config_from_dict, config_hash, load_config
from src.evaluation.errors import ConfigError
from src.evaluation.gpu import parse_nvidia_smi
from src.evaluation.identity import derive_seed
from src.evaluation.models.se import MockEnhancer, load_evaluation_enhancer
from src.evaluation.config import ModelConfig
from tests.evaluation.support import experiment_config

ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_smoke_file_loads(self):
        config = load_config(ROOT / "configs/evaluation/smoke.yaml")
        self.assertEqual(config.experiment.name, "se-smoke")
        self.assertEqual(config.models[0].name, "mock")
        self.assertEqual(config.mixing.snr_db, [0.0, 5.0])

    def test_example_accepts_checkpoint_and_manifest_alias(self):
        config = load_config(ROOT / "configs/evaluation/example.yaml")
        self.assertEqual(config.models[0].name, "mp_senet")
        self.assertEqual(config.models[0].variant, "vb")
        self.assertTrue(config.models[0].checkpoint.endswith("g_best_vb"))
        manifest = config_from_dict(
            {
                "experiment": {"name": "m"},
                "dataset": {"source": "manifest", "manifest": "samples.jsonl"},
                "models": [{"name": "mock"}],
            }
        )
        self.assertEqual(manifest.dataset.path, "samples.jsonl")

    def test_rejects_unknown_fields_missing_weights_and_bad_snr(self):
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "experiment": {"name": "m", "extra": 1},
                    "dataset": {"source": "synthetic"},
                    "models": [{"name": "mock"}],
                }
            )
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "experiment": {"name": "m"},
                    "dataset": {"source": "synthetic"},
                    "models": [{"name": "mp_senet", "variant": "vb"}],
                }
            )
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "experiment": {"name": "m"},
                    "dataset": {"source": "synthetic"},
                    "mixing": {"enabled": True, "snr_db": [float("nan")]},
                    "models": [{"name": "mock"}],
                }
            )

    def test_wer_requires_asr_and_duplicates_are_rejected(self):
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "experiment": {"name": "m"},
                    "dataset": {"source": "synthetic"},
                    "models": [{"name": "mock"}],
                    "metrics": {"wer": True},
                }
            )
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "experiment": {"name": "m"},
                    "dataset": {"source": "synthetic"},
                    "models": [{"name": "mock"}, {"name": "mock"}],
                }
            )

    def test_hash_ignores_output_location_and_limit(self):
        left = experiment_config(Path("/tmp/a"), run_id="one")
        right = experiment_config(Path("/tmp/b"), run_id="two")
        right.experiment.limit = 99
        right.output.save_audio = True
        self.assertEqual(config_hash(left), config_hash(right))
        right.experiment.seed = 1
        self.assertNotEqual(config_hash(left), config_hash(right))

    def test_loader_forwards_checkpoints_without_substituting_them(self):
        seen = {}

        def loader(name, checkpoint, device="cpu", config=None, *, variant=None):
            seen.update(name=name, checkpoint=checkpoint, device=device, config=config, variant=variant)
            return MockEnhancer()

        model = ModelConfig(
            name="unipase",
            checkpoints={"dewavlm": "a.pt", "adapter": "b.pt", "vocoder": "c.pt", "postnet": "d.pt"},
            model_config={"width": 1},
        )
        enhancer = load_evaluation_enhancer(model, "cpu", loader=loader)
        self.assertIsInstance(enhancer, MockEnhancer)
        self.assertEqual(seen["name"], "unipase")
        self.assertEqual(seen["checkpoint"]["dewavlm"], "a.pt")
        self.assertEqual(seen["config"], {"width": 1})
        self.assertIsNone(seen["variant"])


class IdentityTests(unittest.TestCase):
    def test_seed_does_not_depend_on_python_hash_seed(self):
        code = "from src.evaluation.identity import derive_seed; print(derive_seed(1, 'utt', 'mp_senet'))"
        values = []
        for hash_seed in ("0", "1"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hash_seed
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            values.append(completed.stdout.strip())
        self.assertEqual(values[0], values[1])
        self.assertEqual(int(values[0]), derive_seed(1, "utt", "mp_senet"))

    def test_nvidia_smi_parser(self):
        rows = parse_nvidia_smi("0, 128, 0\n1, 20000, 90\n")
        self.assertEqual(rows[1]["memory_used_mib"], 20000)
        self.assertEqual(rows[0]["index"], 0)


class MockContractTests(unittest.TestCase):
    def test_mock_rejects_non_finite_audio(self):
        enhancer = MockEnhancer()
        with self.assertRaises(ValueError):
            enhancer.enhance(torch.tensor([float("nan")]), 16000)
        with self.assertRaises(ValueError):
            enhancer.enhance(torch.zeros(2, 8), 16000)
