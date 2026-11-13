"""Training-volume helpers keep planned counts and observed runs apart."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.experiments.budget import fixed_pair_kept_samples, logical_steps, observed_budget


class TrainingBudgetTests(unittest.TestCase):
    def test_remainder_micro_batches_are_dropped(self) -> None:
        self.assertEqual(fixed_pair_kept_samples(10700, 2, 2), 10700)
        self.assertEqual(logical_steps(10700, 2, 2), 2675)
        self.assertEqual(fixed_pair_kept_samples(10700, 4, 2), 10696)
        self.assertEqual(logical_steps(10700, 4, 2), 1337)

    def test_missing_run_does_not_invent_actuals(self) -> None:
        with TemporaryDirectory() as tmp:
            report = observed_budget(Path(tmp) / "absent")
        self.assertFalse(report["present"])
        self.assertIsNone(report["logical_steps"])
        self.assertIsNone(report["optimizer_updates"])
        self.assertIsNone(report["samples_seen"])

    def test_metrics_supply_observed_steps(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "metrics.jsonl").write_text(
                "\n".join([
                    json.dumps({"logical_step": 1, "optimizer_updates": {"optim": 1}, "sample_id": "a,b"}),
                    json.dumps({"logical_step": 2, "optimizer_updates": {"optim": 2}, "sample_id": "c"}),
                ]) + "\n",
                encoding="utf-8",
            )
            (directory / "status.json").write_text(json.dumps({"state": "INTERRUPTED"}), encoding="utf-8")
            report = observed_budget(directory)
        self.assertEqual(report["logical_steps"], 2)
        self.assertEqual(report["optimizer_updates"], {"optim": 2})
        self.assertEqual(report["samples_seen"], 3)
        self.assertEqual(report["training_status"], "INTERRUPTED")
