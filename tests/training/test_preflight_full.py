"""Preflight refuses full-profile launch conditions before any training step."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.errors import TrainingError
from src.training.se.config import load_training_config
from src.training.se.preflight import claim_conflict, run_preflight, smoke_cap_keys
from src.training.se.run import execute

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "configs/training/reproduction/tf_gridnet_ms_snsd_full.yaml"
LOCO = ROOT / "configs/training/reproduction/tf_locoformer_dns_research_valid_full.yaml"

_WORLD_YAML = """
seed: 1
deterministic: false
global_batch_size: 4
micro_batch_size_per_gpu: 2
required_world_size: 2
experiment_profile: full
experiment_claim: official_algorithm_fixed_seed
max_steps: 4
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


class PreflightFullTests(unittest.TestCase):
    def test_full_yamls_have_no_smoke_caps(self):
        for path in (GRID, LOCO):
            config = load_training_config(path)
            self.assertEqual(config["experiment_profile"], "full")
            self.assertEqual(smoke_cap_keys(config), [])
            self.assertIn("required_world_size", config)
            self.assertTrue(config["checkpoint_retention"]["keep_best"])
            self.assertTrue(config["checkpoint_retention"]["protect_nbest"])

    def test_smoke_cap_in_a_full_profile_is_rejected(self):
        text = GRID.read_text(encoding="utf-8")
        self.assertNotIn("verification_max_updates:", text)
        edited = text.replace("max_steps: 553521\n", "max_steps: 553521\nverification_max_updates: 2\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "capped.yaml"
            path.write_text(edited, encoding="utf-8")
            out = root / "out"
            code = run_preflight(path, output_dir=out)
            report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("smoke caps" in item for item in report["failures"]))

    def test_world_size_mismatch_rejects_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "world.yaml"
            path.write_text(
                _WORLD_YAML.format(checkpoint=root / "ckpt", database=root / "unused.duckdb"),
                encoding="utf-8",
            )

            def boom(*_args, **_kwargs):
                raise AssertionError("build_task was called")

            with patch("src.training.se.run.build_task", side_effect=boom):
                with self.assertRaisesRegex(TrainingError, "required_world_size 2"):
                    execute(path)

    def test_official_and_research_claims_cannot_be_mixed(self):
        research = "tf_locoformer.dns.accept_recipe.research_valid_fixed@1"
        official = "tf_gridnet.ms_snsd.espnet_recipe@1"
        self.assertIsNotNone(claim_conflict("official_faithful", research))
        self.assertIsNotNone(claim_conflict("official_algorithm_fixed_seed", research))
        self.assertIsNotNone(claim_conflict("research_checkpoint_selection", official))
        self.assertIsNone(claim_conflict("official_algorithm_fixed_seed", official))
        self.assertIsNone(claim_conflict("research_checkpoint_selection", research))
        self.assertIsNone(claim_conflict("leakage_safe", official))
        text = GRID.read_text(encoding="utf-8").replace(
            "experiment_claim: official_algorithm_fixed_seed",
            "experiment_claim: research_checkpoint_selection",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mixed.yaml"
            path.write_text(text, encoding="utf-8")
            out = root / "out"
            self.assertEqual(run_preflight(path, output_dir=out), 1)
            report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
            self.assertTrue(any("mixed" in item for item in report["failures"]))

    def test_retention_that_drops_best_is_rejected(self):
        text = GRID.read_text(encoding="utf-8").replace("keep_best: true", "keep_best: false")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "drop.yaml"
            path.write_text(text, encoding="utf-8")
            out = root / "out"
            run_preflight(path, output_dir=out)
            report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
            self.assertTrue(any("cannot keep best" in item for item in report["failures"]))

    def test_preflight_writes_json_and_markdown_for_the_full_yamls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, path in (("grid", GRID), ("loco", LOCO)):
                out = root / name
                code = run_preflight(path, output_dir=out)
                report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
                markdown = (out / "preflight.md").read_text(encoding="utf-8")
                self.assertIn(report["status"], ("PASS", "FAIL"))
                self.assertEqual(code, 0 if report["status"] == "PASS" else 1)
                self.assertIn("公式再現の成功ではない", markdown)
                self.assertIn("公式再現の成功ではない", report["note"])
                if report["status"] == "FAIL":
                    self.assertTrue(report["failures"])
                    if report["checks"].get("snapshot_hash") is None:
                        self.assertTrue(any("data gap" in item for item in report["failures"]))
                else:
                    self.assertIsInstance(report["checks"].get("snapshot_hash"), str)
                    self.assertTrue(report["checks"]["snapshot_hash"])


if __name__ == "__main__":
    unittest.main()
