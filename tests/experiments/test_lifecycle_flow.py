"""Lifecycle start, conditional save, and a single finish."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import BuildError, CheckpointError
from src.experiments.execution.lifecycle import Lifecycle
from src.experiments.execution.results import RunResult
from tests.experiments.support import write_config


def _components(*, steps: int = 1) -> dict[str, object]:
    return {
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
            "kwargs": {"max_logical_steps": steps, "accumulation": 1},
            "inject": {
                "model": {"ref": "model"},
                "loss": {"ref": "loss"},
                "data": {"ref": "data"},
                "optimizer": {"ref": "optimizer"},
            },
        },
    }


class LifecycleFlowTests(unittest.TestCase):
    def _built(self) -> object:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = write_config(directory, _components())
        experiment = build_experiment(load_experiment(path))
        self.addCleanup(experiment.close)
        return experiment

    def test_selector_declines_a_step_and_keeps_a_boundary(self) -> None:
        experiment = self._built()
        lifecycle = Lifecycle(experiment._checkpoint, experiment._graph.runtime, experiment.config)
        self.assertIsNone(lifecycle.start(resume=None))
        trainer = experiment.get("trainer")
        trainer.logical_step = 1
        prepared = {"called": False}

        def prepare(runtime: object) -> None:
            del runtime
            prepared["called"] = True

        declined = lifecycle.save_if_needed("step", logical_step=1, prepare=prepare)
        self.assertFalse(declined.saved)
        self.assertFalse(prepared["called"])
        self.assertFalse(experiment._directory.joinpath("latest.pt").is_file())
        kept = lifecycle.save_if_needed(
            "boundary",
            progress={"logical_step": 1},
            metrics={"loss": 0.25},
            prepare=prepare,
        )
        self.assertTrue(kept.saved)
        self.assertTrue(prepared["called"])
        self.assertTrue(experiment._directory.joinpath("latest.pt").is_file())
        with self.assertRaisesRegex(CheckpointError, "already started"):
            lifecycle.start(resume=None)
        result = RunResult("completed", 1, {"optimizer": 0}, experiment._directory)
        lifecycle.finish(result, primary=True)
        self.assertTrue(lifecycle.finished)
        completed = json.loads((experiment._directory / "complete.json").read_text(encoding="utf-8"))
        self.assertEqual(completed["logical_step"], 1)
        with self.assertRaisesRegex(CheckpointError, "already finished"):
            lifecycle.finish(result, primary=True)
        with self.assertRaisesRegex(CheckpointError, "has finished"):
            lifecycle.save_if_needed("final", logical_step=1)

    def test_a_missing_counter_is_refused(self) -> None:
        experiment = self._built()
        lifecycle = Lifecycle(experiment._checkpoint, experiment._graph.runtime, experiment.config)
        lifecycle.start()
        with self.assertRaisesRegex(CheckpointError, "logical_step"):
            lifecycle.save_if_needed("stop")
        with self.assertRaisesRegex(CheckpointError, "does not match"):
            lifecycle.save_if_needed("step", logical_step=1, progress={"logical_step": 2})
        with self.assertRaisesRegex(BuildError, "unknown stop reason"):
            lifecycle.finish(RunResult("nope", 0, {}, experiment._directory), primary=True)

    def test_experiment_run_finishes_once(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        experiment = build_experiment(load_experiment(write_config(directory, _components(steps=1))))
        result = experiment.run()
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(result.logical_step, 1)
        lifecycle = experiment._lifecycle
        self.assertIsInstance(lifecycle, Lifecycle)
        self.assertTrue(lifecycle.started)
        self.assertTrue(lifecycle.finished)
        self.assertEqual(lifecycle.reason, "completed")
        status = json.loads((result.output_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "COMPLETED")
        self.assertEqual(status["logical_step"], 1)


if __name__ == "__main__":
    unittest.main()
