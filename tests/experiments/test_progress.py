"""Named progress for schedules, selectors, and checkpoint files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import BuildError, CheckpointError
from src.experiments.evaluation.schedule import EvaluationSchedule
from src.experiments.execution.selector import SavePoint, StandardSelector
from tests.experiments.support import write_config


class NoCounterTrainer:
    """Publishes no counter, so a periodic selector is rejected at build."""

    progress_keys = ()

    def __init__(self, **kwargs: object) -> None:
        del kwargs

    def fit(self) -> None:
        raise AssertionError("this trainer is not run")

    def run(self, runtime: object, checkpoint: object) -> None:
        del runtime, checkpoint
        raise AssertionError("this trainer is not run")

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state


class ProgressTests(unittest.TestCase):
    def test_schedules_read_named_progress_and_reject_a_missing_key(self) -> None:
        steps = EvaluationSchedule({"unit": "logical_step", "interval": 2})
        self.assertFalse(steps.due(kind="step", progress={"logical_step": 1}))
        self.assertTrue(steps.due(kind="step", progress={"logical_step": 2}))
        self.assertTrue(steps.due(kind="step", logical_step=4))
        with self.assertRaisesRegex(BuildError, "logical_step"):
            steps.due(kind="step", progress={"epoch": 1})
        epochs = EvaluationSchedule({"unit": "epoch", "interval": 2})
        self.assertTrue(epochs.due(kind="boundary", progress={"epoch": 0}))
        self.assertFalse(epochs.due(kind="boundary", progress={"epoch": 1}))
        with self.assertRaisesRegex(BuildError, "epoch"):
            epochs.due(kind="boundary", progress={})
        self.assertTrue(EvaluationSchedule(None).due(kind="boundary", progress={}))

    def test_event_saves_do_not_require_a_counter(self) -> None:
        point = SavePoint("final", 0, progress={}, progress_explicit=True)
        self.assertNotIn("logical_step", point.progress)
        decision = StandardSelector().decide(point)
        self.assertTrue(decision.latest)
        step = SavePoint("step", 3, progress={"logical_step": 3}, progress_explicit=True)
        self.assertFalse(StandardSelector(latest_every=2).decide(step).latest)
        with self.assertRaisesRegex(CheckpointError, "updates"):
            StandardSelector(progress_key="updates").decide(step)

    def test_a_trainer_without_the_selector_key_is_refused_at_build(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = write_config(
            directory,
            {
                "model": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 2, "out_features": 2},
                    "prepare": {"trainable": True, "distribute": "none"},
                },
                "trainer": {
                    "role": "trainer",
                    "class": "tests.experiments.test_progress.NoCounterTrainer",
                    "inject": {"model": {"ref": "model"}},
                },
            },
        )
        with self.assertRaisesRegex(BuildError, "progress logical_step"):
            build_experiment(load_experiment(path))

    def test_checkpoint_stores_progress_beside_logical_step(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = write_config(
            directory,
            {
                "model": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 4, "out_features": 4},
                    "prepare": {"trainable": True, "distribute": "none"},
                },
                "loss": {"role": "loss", "class": "torch.nn.MSELoss"},
                "data": {
                    "role": "data",
                    "class": "src.experiments.data.TensorPairData",
                    "kwargs": {"features": 4, "examples": 8, "batch_size": 4},
                },
                "optimizer": {
                    "role": "optimizer",
                    "class": "torch.optim.SGD",
                    "parameters_of": "model",
                    "kwargs": {"lr": 0.1},
                },
                "trainer": {
                    "role": "trainer",
                    "class": "src.experiments.trainers.StandardTrainer",
                    "kwargs": {"max_logical_steps": 1, "accumulation": 1},
                    "inject": {
                        "model": {"ref": "model"},
                        "loss": {"ref": "loss"},
                        "data": {"ref": "data"},
                        "optimizer": {"ref": "optimizer"},
                    },
                },
            },
        )
        result = build_experiment(load_experiment(path)).run()
        payload = torch.load(result.output_dir / "latest.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["logical_step"], 1)
        self.assertEqual(payload["progress"]["logical_step"], 1)


if __name__ == "__main__":
    unittest.main()
