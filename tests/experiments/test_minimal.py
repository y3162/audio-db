"""CPU linear experiment through the Python API and the CLI."""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from unittest import mock

from src.experiments.api import Experiment, build_experiment, preflight_experiment
from src.experiments.config import config_sha256, load_experiment, structural_sha256
from src.experiments.errors import CheckpointError
from src.experiments.trainers import StandardTrainer

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tests/fixtures/configs/experiments/linear_supervised.yaml"


class DropLinear(torch.nn.Module):
    """Linear layer plus dropout, so a preflight that leaks RNG changes step 1."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.drop(self.linear(inputs))


def _weights(experiment: object) -> list[torch.Tensor]:
    module = experiment.get("model").state_module  # type: ignore[attr-defined]
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def _config_text(
    directory: Path,
    steps: int,
    *,
    model: str | None = None,
    preflight: bool | None = None,
) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    text = text.replace("directory: runs/linear-supervised", f"directory: {directory / 'out'}")
    text = text.replace("max_logical_steps: 4", f"max_logical_steps: {steps}")
    if model is not None:
        text = text.replace(
            "class: torch.nn.Linear\n    kwargs: {in_features: 8, out_features: 8}\n",
            f"class: {model}\n    kwargs: {{}}\n",
        )
    if preflight is not None:
        text = text.replace("precision: fp32\n", f"precision: fp32\n    preflight: {str(preflight).lower()}\n")
    path = directory / "linear.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class MinimalExperimentTests(unittest.TestCase):
    def test_standard_trainer_updates_through_runtime(self) -> None:
        source = inspect.getsource(StandardTrainer.fit)
        self.assertNotIn("optimizer.step", source)
        self.assertIn("runtime.step", source)

    def test_python_api_updates_parameters_and_repeats_a_fixed_seed(self) -> None:
        first_dir = Path(self.enterContext(self._directory()))
        second_dir = Path(self.enterContext(self._directory()))
        torch.manual_seed(12345)
        first = self._run(first_dir, steps=2)
        torch.manual_seed(999)
        second = self._run(second_dir, steps=2)
        self.assertEqual(first["result"].logical_step, 2)
        self.assertEqual(first["result"].stop_reason, "completed")
        self.assertEqual(first["result"].optimizer_updates, {"optimizer": 2})
        self.assertTrue(first["losses"])
        self.assertTrue(all(math.isfinite(loss) for loss in first["losses"]))
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(first["before"], first["after"])))
        for left, right in zip(first["after"], second["after"]):
            self.assertTrue(torch.equal(left, right))
        self.assertIs(first["experiment"].get("loss"), first["experiment"].get("loss_for_log"))
        state_ids = first["experiment"]._graph.states.ids()
        self.assertNotIn("loss", state_ids)
        self.assertNotIn("loss_for_log", state_ids)
        self.assertIn("model", state_ids)
        self.assertIn("optimizer", state_ids)
        self.assertIn("data", state_ids)
        self.assertIn("runtime", state_ids)
        metrics = (first["result"].output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(metrics), 2)
        self.assertEqual(json.loads(metrics[-1])["logical_step"], 2)
        status = json.loads((first["result"].output_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "COMPLETED")
        parameter = next(first["experiment"].get("model").state_module.parameters())
        self.assertEqual(int(first["experiment"].get("optimizer").state[parameter]["step"]), 2)
        data = first["experiment"].get("data")
        view = data.train_epoch(0, start_update=0)
        self.assertTrue(torch.equal(view.batch(0)[0].inputs, view.batch(0)[0].inputs.clone()))
        self.assertEqual(data.update_index, 2)

    def test_shipped_example_runs_four_steps(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        outcome = self._run(directory, steps=4)
        self.assertEqual(outcome["result"].logical_step, 4)
        self.assertEqual(outcome["result"].optimizer_updates["optimizer"], 4)

    def test_preflight_does_not_update_optimizer(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        experiment = build_experiment(load_experiment(_config_text(directory, 2)))
        before = _weights(experiment)
        path = preflight_experiment(experiment)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["training_executed"], False)
        self.assertEqual(payload["logical_step"], 0)
        self.assertEqual(payload["batch_probe"]["status"], "ok")
        self.assertTrue(payload["batch_probe"]["finite"])
        self.assertGreater(payload["batch_probe"]["parameters_with_grad"], 0)
        self.assertNotIn("loss", payload["state_ids"])
        after = _weights(experiment)
        for left, right in zip(before, after):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(len(experiment.get("optimizer").state), 0)
        self.assertFalse((experiment._directory / "metrics.jsonl").exists())
        experiment.close()

    def test_cli_runs_and_preflight_skips_updates(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _config_text(directory, 2)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            [sys.executable, "-m", "src.experiments.cli", "--config", str(config)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        status = json.loads((directory / "out" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "COMPLETED")
        self.assertEqual(len((directory / "out" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()), 2)

        preflight_dir = directory / "preflight"
        preflight = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.experiments.cli",
                "--config",
                str(config),
                "--preflight",
                "--preflight-dir",
                str(preflight_dir),
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(preflight.returncode, 0, preflight.stderr)
        report = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
        self.assertFalse(report["training_executed"])
        self.assertEqual(report["batch_probe"]["status"], "ok")
        self.assertFalse((preflight_dir / "metrics.jsonl").exists())

        failed = subprocess.run(
            [sys.executable, "-m", "src.experiments.cli", "--config", str(directory / "missing.yaml")],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(failed.returncode, 1)

    def test_new_run_refuses_a_directory_that_holds_a_run(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        self._run(directory, steps=2)
        out = directory / "out"
        kept = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
        self.assertIn("complete.json", kept)
        self.assertIn("latest.pt", kept)

        def unchanged() -> None:
            self.assertEqual({path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}, kept)

        config = _config_text(directory, 2)
        with self.assertRaisesRegex(CheckpointError, "already holds a run"):
            Experiment(load_experiment(config)).run()
        unchanged()
        # Checkpoints alone are enough. A stopped run has no complete.json.
        (out / "complete.json").unlink()
        del kept["complete.json"]
        with self.assertRaisesRegex(CheckpointError, r"latest\.pt"):
            Experiment(load_experiment(config)).run()
        unchanged()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        again = subprocess.run(
            [sys.executable, "-m", "src.experiments.cli", "--config", str(config)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(again.returncode, 1)
        self.assertIn("--resume", again.stderr)
        unchanged()
        # The same directory still resumes from its own Latest.
        resumed = Experiment(load_experiment(config)).run(resume=out)
        self.assertEqual(resumed.logical_step, 2)
        self.assertEqual(resumed.stop_reason, "completed")
        self.assertEqual(len((out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_preflight_does_not_change_the_first_dropout_step(self) -> None:
        probed = Path(self.enterContext(self._directory()))
        skipped = Path(self.enterContext(self._directory()))
        probed_run = self._run(probed, steps=1, model="tests.experiments.test_minimal.DropLinear")
        skipped_run = self._run(
            skipped,
            steps=1,
            model="tests.experiments.test_minimal.DropLinear",
            preflight=False,
        )
        for left, right in zip(probed_run["after"], skipped_run["after"], strict=True):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(probed_run["losses"], skipped_run["losses"])
        probed_meta = json.loads((probed / "out" / "metadata.json").read_text(encoding="utf-8"))
        skipped_meta = json.loads((skipped / "out" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(probed_meta["launches"][0]["batch_probe"]["status"], "ok")
        self.assertEqual(skipped_meta["launches"][0]["batch_probe"]["status"], "disabled")

    def test_resume_reapplies_the_yaml_learning_rate(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        source = _config_text(directory, 2)
        stopped = source.read_text(encoding="utf-8").replace(
            "max_logical_steps: 2, global_batch_size: 4, micro_batch_size: 4",
            "max_logical_steps: 2, global_batch_size: 4, micro_batch_size: 4, stop_after_logical_steps: 1",
        )
        source.write_text(stopped, encoding="utf-8")
        first = build_experiment(load_experiment(source))
        stopped_result = first.run()
        self.assertEqual(stopped_result.stop_reason, "interrupted")
        self.assertFalse((directory / "out" / "complete.json").exists())
        changed = stopped.replace("{lr: 0.001}", "{lr: 0.05}").replace(
            ", stop_after_logical_steps: 1", ""
        )
        resume_path = directory / "resume.yaml"
        resume_path.write_text(changed, encoding="utf-8")
        original = load_experiment(source)
        updated = load_experiment(resume_path)
        self.assertNotEqual(config_sha256(original), config_sha256(updated))
        self.assertEqual(structural_sha256(original), structural_sha256(updated))
        wider_path = directory / "wider.yaml"
        wider_path.write_text(changed.replace("out_features: 8", "out_features: 4"), encoding="utf-8")
        kept = (directory / "out" / "latest.pt").read_bytes()
        with self.assertRaisesRegex(CheckpointError, "structure"):
            build_experiment(load_experiment(wider_path)).run(resume=directory / "out")
        self.assertEqual((directory / "out" / "latest.pt").read_bytes(), kept)
        resumed = build_experiment(updated)
        result = resumed.run(resume=directory / "out")
        self.assertEqual(result.logical_step, 2)
        self.assertEqual(resumed.get("optimizer").param_groups[0]["lr"], 0.05)
        self.assertTrue(any(not torch.equal(left, right) for left, right in zip(_weights(first), _weights(resumed))))

    def test_ctrl_c_does_not_leave_the_status_running(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        experiment = build_experiment(load_experiment(_config_text(directory, 2)))
        with mock.patch.object(StandardTrainer, "_offer", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                experiment.run()
        status = json.loads((directory / "out" / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")
        self.assertFalse((directory / "out" / "complete.json").exists())

    def _run(
        self,
        directory: Path,
        *,
        steps: int,
        model: str | None = None,
        preflight: bool | None = None,
    ) -> dict[str, object]:
        experiment = build_experiment(load_experiment(_config_text(directory, steps, model=model, preflight=preflight)))
        before = _weights(experiment)
        result = experiment.run()
        return {
            "experiment": experiment,
            "before": before,
            "after": _weights(experiment),
            "losses": list(experiment.get("trainer").losses),
            "result": result,
        }

    def _directory(self):
        import tempfile

        return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
