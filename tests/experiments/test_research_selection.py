"""Model selection, namespaced Top-K, and the research export path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.execution.selector import Decision, SavePoint, TopKSelector
from src.experiments.runtime import PreparePolicy, PreparedModel
from src.experiments.selection.export import DIRECTORY, FORMAT, METADATA, WEIGHTS, export_best
from src.experiments.selection.policy import BestMetricSelection


class SelectionPolicyTests(unittest.TestCase):
    def test_min_max_ties_and_incomplete(self) -> None:
        earliest = BestMetricSelection("validation.loss", "min")
        self.assertTrue(_observe(earliest, 2, 1.0))
        self.assertFalse(_observe(earliest, 4, 1.0))
        self.assertEqual(earliest.best_step, 2)
        self.assertTrue(_observe(earliest, 6, 0.5))
        self.assertEqual((earliest.best_step, earliest.best), (6, 0.5))

        later = BestMetricSelection("validation.wer", "min", tie_break="later")
        _observe(later, 2, 0.4)
        self.assertTrue(_observe(later, 5, 0.4))
        self.assertEqual(later.best_step, 5)

        higher = BestMetricSelection("validation.si_sdr", "max")
        _observe(higher, 1, 3.0)
        self.assertFalse(_observe(higher, 2, 2.0))
        self.assertEqual(higher.best_step, 1)

        uncounted = BestMetricSelection("validation.loss", "min")
        self.assertTrue(
            uncounted.observe(
                source="validation",
                name="loss",
                logical_step=None,
                epoch=None,
                status="ok",
                value=1.0,
            )
        )
        self.assertIsNone(uncounted.best_step)
        self.assertEqual(uncounted.best_observation, 1)
        self.assertFalse(
            uncounted.observe(
                source="validation",
                name="loss",
                logical_step=None,
                epoch=None,
                status="ok",
                value=1.0,
            )
        )
        self.assertTrue(
            uncounted.observe(
                source="validation",
                name="loss",
                logical_step=None,
                epoch=None,
                status="ok",
                value=0.5,
            )
        )
        self.assertIsNone(uncounted.best_step)
        self.assertEqual(uncounted.best, 0.5)
        restored = BestMetricSelection("validation.loss", "min")
        restored.load_state_dict(json.loads(json.dumps(uncounted.state_dict())))
        self.assertIsNone(restored.best_step)
        self.assertEqual(restored.best_observation, uncounted.best_observation)

        rejected = BestMetricSelection("validation.loss", "min")
        self.assertFalse(rejected.observe(
            source="validation",
            name="loss",
            logical_step=1,
            epoch=0,
            status="incomplete",
            value=None,
        ))
        self.assertFalse(rejected.observe(
            source="validation",
            name="loss",
            logical_step=2,
            epoch=0,
            status="ok",
            value=float("nan"),
        ))
        self.assertIsNone(rejected.best)
        self.assertEqual(rejected.rejected, 2)
        self.assertFalse(rejected.observe(
            source="train_eval",
            name="loss",
            logical_step=3,
            epoch=0,
            status="ok",
            value=0.1,
        ))
        self.assertEqual(rejected.rejected, 2)

    def test_state_round_trip(self) -> None:
        policy = BestMetricSelection("validation.loss", "min", k=2)
        _observe(policy, 1, 2.0)
        _observe(policy, 2, 1.0)
        restored = BestMetricSelection("validation.loss", "min", k=2)
        restored.load_state_dict(json.loads(json.dumps(policy.state_dict())))
        self.assertEqual(restored.best_step, 2)
        self.assertEqual(restored.ranking[0]["logical_step"], 2)
        self.assertFalse(restored._improved)

    def test_unrelated_observation_keeps_an_improvement(self) -> None:
        policy = BestMetricSelection("validation.loss", "min")
        self.assertTrue(_observe(policy, 1, 1.0))
        self.assertFalse(policy.observe(
            source="validation",
            name="si_snr",
            logical_step=1,
            epoch=0,
            status="ok",
            value=10.0,
        ))
        self.assertTrue(policy._improved)


class TopKNamespaceTests(unittest.TestCase):
    def test_dotted_metric_ranks_steps_and_bare_loss_stays_on_boundaries(self) -> None:
        namespaced = TopKSelector(metric="validation.wer", mode="min", k=1)
        self.assertEqual(
            namespaced.decide(SavePoint("step", 2, {"validation.wer": 0.4})),
            Decision(latest=True, archive=("best_00000002.pt",)),
        )
        bare = TopKSelector(metric="loss", mode="min", k=1)
        self.assertEqual(bare.decide(SavePoint("step", 2, {"loss": 0.4})), Decision(latest=True))
        self.assertEqual(
            bare.decide(SavePoint("boundary", 4, {"loss": 0.4})),
            Decision(latest=True, archive=("best_00000004.pt",)),
        )


class ExportTests(unittest.TestCase):
    def test_export_is_not_an_official_filename(self) -> None:
        module = torch.nn.Linear(2, 2)
        trainer = type("Trainer", (), {})()
        trainer.model = PreparedModel("model", module, tuple(module.parameters()), module, PreparePolicy())
        selection = BestMetricSelection("validation.loss", "min")
        _observe(selection, 3, 0.2)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            export_best(type("Runtime", (), {"rank": 0, "world_size": 1})(), directory, trainer.model, selection, config_sha256="abc")
            weights = directory / DIRECTORY / WEIGHTS
            metadata = json.loads((directory / DIRECTORY / METADATA).read_text(encoding="utf-8"))
            payload = torch.load(weights, map_location="cpu", weights_only=False)
            self.assertEqual(payload["format"], FORMAT)
            self.assertEqual(metadata["logical_step"], 3)
            self.assertFalse((directory / "valid.loss.best.pt").exists())
            self.assertFalse((directory / "best_generator.pt").exists())
            self.assertNotIn("model", metadata)


def _observe(policy: BestMetricSelection, step: int, value: float) -> bool:
    return policy.observe(
        source=policy.source,
        name=policy.name,
        logical_step=step,
        epoch=0,
        status="ok",
        value=value,
    )


if __name__ == "__main__":
    unittest.main()
