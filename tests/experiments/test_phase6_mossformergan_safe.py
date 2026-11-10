"""Short leakage-safe MossFormerGAN run on the safe VoiceBank snapshot.

World size is 1 and the device is CPU. That is not the 2-GPU production batch.
The production YAML is parsed here and is not launched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiments.api import build_experiment, load_experiment

ROOT = Path(__file__).resolve().parents[2]
READINESS = ROOT / "configs/experiments/readiness/mossformergan_se_16k_leakage_safe.yaml"
PRODUCTION = ROOT / "configs/experiments/production/mossformergan_se_16k_leakage_safe.yaml"
OFFICIAL = ROOT / "configs/experiments/production/mossformergan_se_16k_official_2gpu.yaml"


def _run(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=1200,
    )


class MossFormerLeakageSafeTests(unittest.TestCase):
    def test_production_yaml_keeps_the_official_file_and_a_separate_directory(self) -> None:
        official = yaml.safe_load(OFFICIAL.read_text(encoding="utf-8"))
        safe = yaml.safe_load(PRODUCTION.read_text(encoding="utf-8"))
        official_trainer = official["components"]["trainer"]["kwargs"]
        safe_trainer = safe["components"]["trainer"]["kwargs"]
        self.assertEqual(official_trainer["experiment_claim"], "official_faithful")
        self.assertEqual(safe_trainer["experiment_claim"], "leakage_safe")
        self.assertEqual(official_trainer["accumulation"], safe_trainer["accumulation"])
        self.assertEqual(official_trainer["grad_clip"], safe_trainer["grad_clip"])
        self.assertEqual(official_trainer["checkpoint_interval"], 90)
        self.assertEqual(safe_trainer["checkpoint_interval"], 90)
        self.assertEqual(safe_trainer["max_logical_steps"], 323675)
        self.assertNotEqual(official["output"]["directory"], safe["output"]["directory"])
        self.assertEqual(
            safe["components"]["source"]["kwargs"]["snapshot"],
            "mossformergan.vb28.ondemand.safe@1",
        )
        self.assertEqual(safe["components"]["data"]["kwargs"]["require_profile"], "leakage_safe")
        self.assertEqual(official["components"]["data"]["kwargs"]["require_profile"], "official_faithful")

    def test_safe_snapshot_validates_without_the_test_split_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "run"
            text = READINESS.read_text(encoding="utf-8")
            text = text.replace(
                "data/results/experiments/phase6-readiness/mossformergan-leakage-safe",
                str(output),
            )
            config = root / "experiment.yaml"
            config.write_text(text, encoding="utf-8")
            experiment = build_experiment(load_experiment(config))
            try:
                data = experiment.get("data")
                identity = data.identity()
                source = experiment.get("source")
                valid_ids = set(source.case_ids("valid.cases"))
                test_ids = set(source.case_ids("test.cases"))
                self.assertEqual(identity["profile"], "leakage_safe")
                self.assertEqual(identity["dataset_name"], "mossformergan.vb28.ondemand.safe")
                self.assertEqual(identity["claim_label"], "leakage_safe_comparison")
                self.assertIs(identity["selection_uses_test"], False)
                self.assertTrue(valid_ids)
                self.assertTrue(test_ids.isdisjoint(valid_ids))
                batch = next(iter(data.validation_epoch(0)))
                self.assertIn(batch.sample_id, valid_ids)
                self.assertNotIn(batch.sample_id, test_ids)
                self.assertEqual(experiment.get("trainer").experiment_claim, "leakage_safe")
            finally:
                experiment.close()

            first = _run(config)
            self.assertEqual(first.returncode, 0, first.stderr[-4000:] + first.stdout[-2000:])
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "INTERRUPTED")
            self.assertFalse((output / "complete.json").is_file())
            saved = output / "step_00000001.pt"
            self.assertTrue(saved.is_file())
            provenance = json.loads((output / "data_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["profile"], "leakage_safe")
            self.assertEqual(provenance["claim_label"], "leakage_safe_comparison")
            self.assertIs(provenance["selection_uses_test"], False)
            self.assertEqual(provenance["dataset_name"], "mossformergan.vb28.ondemand.safe")
            payload = torch_load(saved)
            self.assertEqual(payload["states"]["trainer"]["experiment_claim"], "leakage_safe")
            self.assertEqual(payload["logical_step"], 1)
            resume_text = text.replace("      stop_after_logical_steps: 1\n", "")
            resume = root / "resume.yaml"
            resume.write_text(resume_text, encoding="utf-8")
            second = _run(resume, "--resume", str(saved))
            self.assertEqual(second.returncode, 0, second.stderr[-4000:] + second.stdout[-2000:])
            final = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "COMPLETED")
            self.assertTrue((output / "complete.json").is_file())
            rows = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual([row["logical_step"] for row in rows], [1, 2])
            resumed_payload = torch_load(output / "step_00000002.pt")
            self.assertEqual(resumed_payload["logical_step"], 2)
            self.assertEqual(resumed_payload["states"]["trainer"]["optimizer_updates"]["optim_g"], 2)
            self.assertEqual(resumed_payload["states"]["trainer"]["optimizer_updates"]["optim_d"], 2)
            self.assertGreater(int(resumed_payload["states"]["data"]["update_index"]), 0)


def torch_load(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(path.name)
    return payload
