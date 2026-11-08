"""Phase 2-A checkpoint save, restore, and resume on CPU."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.experiments.api import build_experiment
from src.experiments.checkpoint import FORMAT
from src.experiments.config import config_sha256, load_experiment
from src.experiments.data import TensorPairData
from src.experiments.errors import CheckpointError, ConfigError, StateError

ROOT = Path(__file__).resolve().parents[2]


class CheckpointTests(unittest.TestCase):
    def test_interrupted_resume_matches_a_continuous_run_in_another_process(self) -> None:
        continuous_dir = Path(self.enterContext(self._directory()))
        interrupted_dir = Path(self.enterContext(self._directory()))
        continuous_config = _write_yaml(continuous_dir, "continuous", steps=4, stop_after=None)
        interrupted_config = _write_yaml(interrupted_dir, "interrupted", "run", steps=4, stop_after=2)
        resume_config = _write_yaml(interrupted_dir, "resume", "run", steps=4, stop_after=None)
        self.assertEqual(config_sha256(load_experiment(interrupted_config)), config_sha256(load_experiment(resume_config)))
        self.assertEqual(config_sha256(load_experiment(continuous_config)), config_sha256(load_experiment(resume_config)))

        continuous = _cli(continuous_config)
        self.assertEqual(continuous.returncode, 0, continuous.stderr)
        interrupted = _cli(interrupted_config)
        self.assertEqual(interrupted.returncode, 0, interrupted.stderr)
        interrupted_out = interrupted_dir / "run"
        self.assertEqual(_status(interrupted_out), "INTERRUPTED")
        self.assertFalse((interrupted_out / "complete.json").is_file())
        saved = interrupted_out / "step_00000002.pt"
        self.assertTrue(saved.is_file())
        data = TensorPairData(features=8, examples=32, batch_size=4)
        prefetched = data.train_epoch(0, start_update=0).batch(2)[0].sample_id
        self.assertEqual(data.update_index, 0)
        metrics_path = interrupted_out / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "applied": True,
                        "logical_step": 3,
                        "loss": 123.0,
                        "optimizer_updates": {"optimizer": 3},
                        "sample_id": "not-consumed",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        resumed = _cli(resume_config, "--resume", str(saved))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)

        continuous_out = continuous_dir / "continuous"
        self.assertEqual(_status(continuous_out), "COMPLETED")
        self.assertEqual(_status(interrupted_out), "COMPLETED")
        self.assertTrue((continuous_out / "complete.json").is_file())
        self.assertTrue((interrupted_out / "complete.json").is_file())
        left = torch.load(continuous_out / "step_00000004.pt", map_location="cpu", weights_only=False)
        right = torch.load(interrupted_out / "step_00000004.pt", map_location="cpu", weights_only=False)
        self.assertEqual(left["logical_step"], 4)
        self.assertEqual(right["logical_step"], 4)
        self.assertEqual(left["config_sha256"], right["config_sha256"])
        self.assertEqual(left["optimizer_updates"], right["optimizer_updates"])
        _assert_equal(left["states"], right["states"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        left_metrics = _metrics(continuous_out)
        right_metrics = _metrics(interrupted_out)
        self.assertEqual([row["logical_step"] for row in left_metrics], [1, 2, 3, 4])
        self.assertEqual(left_metrics, right_metrics)
        self.assertNotIn("not-consumed", [row["sample_id"] for row in right_metrics])
        self.assertEqual(right_metrics[2]["sample_id"], prefetched)
        self.assertEqual(right["states"]["data"]["update_index"], 4)
        optimizer_step = int(next(iter(right["states"]["optimizer"]["state"].values()))["step"])
        self.assertEqual(optimizer_step, 4)
        delta = _max_abs(left["states"]["model"], right["states"]["model"])
        self.assertEqual(delta, 0.0)
        self.assertEqual(right["states"]["scheduler"]["last_epoch"], left["states"]["scheduler"]["last_epoch"])

    def test_atomic_replace_keeps_the_previous_checkpoint(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _write_yaml(directory, "out", steps=1, stop_after=None)
        experiment = build_experiment(load_experiment(config))
        result = experiment.run()
        self.assertEqual(result.logical_step, 1)
        finished = directory / "out" / "step_00000001.pt"
        original = finished.read_bytes()
        with patch("src.experiments.checkpoint.os.replace", side_effect=OSError("injected")):
            with self.assertRaises(CheckpointError):
                experiment._checkpoint.save(directory / "out", logical_step=1, label="step")
        self.assertEqual(finished.read_bytes(), original)
        temporary = directory / "out" / ".step_00000001.pt.tmp"
        self.assertTrue(temporary.is_file())
        with self.assertRaises(CheckpointError):
            experiment._checkpoint.restore(temporary, mode="resume")
        truncated = directory / "out" / "truncated.pt"
        truncated.write_bytes(b"not-a-checkpoint")
        with self.assertRaises(CheckpointError):
            experiment._checkpoint.restore(truncated, mode="resume")

    def test_state_identity_and_foreign_formats_are_refused(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _write_yaml(directory, "out", steps=1, stop_after=None)
        trained = build_experiment(load_experiment(config))
        trained.run()
        source = directory / "out" / "step_00000001.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        experiment = build_experiment(load_experiment(config))
        cases = {
            "missing": _without_state(payload, "scheduler"),
            "extra": _with_extra_state(payload),
            "config": _replace(payload, "config_sha256", "0" * 64),
            "plan": _replace_plan(payload),
            "version": _replace(payload, "schema_version", 2),
            "unknown": _replace(payload, "unexpected", 1),
            "world": _replace(payload, "world_size", 2),
        }
        expected = {
            "missing": StateError,
            "extra": StateError,
            "config": CheckpointError,
            "plan": CheckpointError,
            "version": CheckpointError,
            "unknown": CheckpointError,
            "world": CheckpointError,
        }
        for name, mutated in cases.items():
            path = directory / "out" / f"{name}.pt"
            torch.save(mutated, path)
            with self.subTest(name=name):
                with self.assertRaises(expected[name]):
                    experiment._checkpoint.restore(path, mode="resume")
        foreign = directory / "out" / "legacy.pt"
        torch.save({"format": "audio-db-training-v1", "global_step": 1}, foreign)
        with self.assertRaises(CheckpointError) as raised:
            experiment._checkpoint.restore(foreign, mode="resume")
        self.assertIn("audio-db-training-v1", str(raised.exception))

    def test_initialize_copies_weights_only(self) -> None:
        source_dir = Path(self.enterContext(self._directory()))
        config = _write_yaml(source_dir, "out", steps=2, stop_after=None)
        source = build_experiment(load_experiment(config))
        source.run()
        step_path = source_dir / "out" / "step_00000002.pt"
        init_path = source._checkpoint.save(
            source_dir / "out",
            logical_step=2,
            label="init",
            initialize_allow=["model"],
        )
        source_weights = _module_tensors(torch.load(step_path, map_location="cpu", weights_only=False))
        checkpoint_rng = torch.load(step_path, map_location="cpu", weights_only=False)["rng_by_rank"][0]["torch"]

        rejected_dir = Path(self.enterContext(self._directory()))
        rejected = build_experiment(load_experiment(_write_yaml(rejected_dir, "out", steps=1, stop_after=None, seed=99)))
        with self.assertRaises(CheckpointError):
            rejected.run(initialize_from=step_path)

        fresh_dir = Path(self.enterContext(self._directory()))
        fresh = build_experiment(load_experiment(_write_yaml(fresh_dir, "out", steps=1, stop_after=None, seed=99)))
        fresh._checkpoint.restore(init_path, mode="initialize")
        self.assertTrue(_tensors_equal(source_weights, _live_tensors(fresh)))
        self.assertEqual(len(fresh.get("optimizer").state), 0)
        self.assertEqual(fresh.get("trainer").logical_step, 0)
        restored_rng = torch.random.get_rng_state().clone()
        self.assertFalse(torch.equal(restored_rng, checkpoint_rng.cpu()))
        twin_dir = Path(self.enterContext(self._directory()))
        build_experiment(load_experiment(_write_yaml(twin_dir, "out", steps=1, stop_after=None, seed=99)))
        self.assertTrue(torch.equal(restored_rng, torch.random.get_rng_state()))
        result = fresh.run()
        self.assertEqual(result.logical_step, 1)
        parameter = next(fresh.get("model").state_module.parameters())
        self.assertEqual(int(fresh.get("optimizer").state[parameter]["step"]), 1)

    def test_resume_refuses_completion_and_a_copied_directory(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _write_yaml(directory, "out", steps=1, stop_after=None)
        build_experiment(load_experiment(config)).run()
        finished = directory / "out" / "step_00000001.pt"
        again = build_experiment(load_experiment(config))
        with self.assertRaises(CheckpointError):
            again.run(resume=finished)
        copied_dir = Path(self.enterContext(self._directory()))
        copied = copied_dir / "out"
        copied.mkdir()
        target = copied / finished.name
        shutil.copyfile(finished, target)
        copied_config = _write_yaml(copied_dir, "out", steps=1, stop_after=None)
        with self.assertRaises(CheckpointError):
            build_experiment(load_experiment(copied_config)).run(resume=target)
        with self.assertRaises(ConfigError):
            build_experiment(load_experiment(config)).run(resume=finished, initialize_from=finished)

    def test_cli_rejects_resume_and_initialize_together(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _write_yaml(directory, "out", steps=1, stop_after=None)
        completed = _cli(config, "--resume", "checkpoint.pt", "--initialize-from", "init.pt")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("cannot both be set", completed.stderr)

    def _directory(self):
        import tempfile

        return tempfile.TemporaryDirectory()


def _write_yaml(
    directory: Path,
    folder: str,
    output: str | None = None,
    *,
    steps: int,
    stop_after: int | None,
    seed: int = 17,
) -> Path:
    stop = ""
    if stop_after is not None:
        stop = f"      stop_after_logical_steps: {stop_after}\n"
    output_dir = directory / (output or folder)
    text = f"""schema: audio-db-experiment/v1
seed: {seed}
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
output:
  directory: "{output_dir}"
components:
  model:
    role: module
    class: tests.experiments.support.DropoutLinear
    kwargs: {{in_features: 8, out_features: 8, p: 0.5}}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 8
      examples: 32
      batch_size: 4
  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: model
    kwargs: {{lr: 0.001}}
  scheduler:
    role: scheduler
    class: torch.optim.lr_scheduler.StepLR
    kwargs: {{step_size: 1, gamma: 0.5}}
    inject:
      optimizer: {{ref: optimizer}}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: 1
{stop}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
      scheduler: {{ref: scheduler}}
"""
    path = directory / f"{folder}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _status(directory: Path) -> str:
    return json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"]


def _metrics(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _replace(payload: dict[str, object], key: str, value: object) -> dict[str, object]:
    copied = dict(payload)
    copied[key] = value
    return copied


def _without_state(payload: dict[str, object], state_id: str) -> dict[str, object]:
    copied = dict(payload)
    states = dict(copied["states"])
    states.pop(state_id)
    copied["states"] = states
    copied["state_ids"] = [item for item in copied["state_ids"] if item != state_id]
    return copied


def _with_extra_state(payload: dict[str, object]) -> dict[str, object]:
    copied = dict(payload)
    states = dict(copied["states"])
    states["ghost"] = {"value": 1}
    copied["states"] = states
    copied["state_ids"] = [*list(copied["state_ids"]), "ghost"]
    return copied


def _replace_plan(payload: dict[str, object]) -> dict[str, object]:
    copied = dict(payload)
    states = dict(copied["states"])
    data = dict(states["data"])
    data["plan_hash"] = "0" * 64
    states["data"] = data
    copied["states"] = states
    return copied


def _module_tensors(payload: dict[str, object]) -> list[torch.Tensor]:
    tensors = []
    for value in payload["states"]["model"].values():
        if isinstance(value, torch.Tensor):
            tensors.append(value.detach().cpu().clone())
    return tensors


def _live_tensors(experiment: object) -> list[torch.Tensor]:
    module = experiment.get("model").state_module  # type: ignore[attr-defined]
    return [tensor.detach().cpu().clone() for tensor in module.state_dict().values() if isinstance(tensor, torch.Tensor)]


def _tensors_equal(left: list[torch.Tensor], right: list[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))


def _max_abs(left: dict[str, object], right: dict[str, object]) -> float:
    values: list[float] = []

    def walk(lhs: object, rhs: object) -> None:
        if isinstance(lhs, torch.Tensor):
            values.append(float((lhs.detach().cpu() - rhs.detach().cpu()).abs().max()))  # type: ignore[union-attr]
            return
        if isinstance(lhs, dict):
            for key, value in lhs.items():
                walk(value, rhs[key])  # type: ignore[index]

    walk(left, right)
    return max(values)


def _assert_equal(left: object, right: object, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(path)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(path)
        for key, value in left.items():
            _assert_equal(value, right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError(path)
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            _assert_equal(lhs, rhs, f"{path}[{index}]")
        return
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError(path)
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


if __name__ == "__main__":
    unittest.main()
