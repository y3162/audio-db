"""Phase 6 stop boundary: publish the last finished step, then resume it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _yaml(directory: Path, *, steps: int, stop_after: int | None, interval: int) -> Path:
    stop = "null" if stop_after is None else str(stop_after)
    text = textwrap.dedent(
        f"""\
        schema: audio-db-experiment/v1
        checkpoint:
          selector:
            class: tests.experiments.support.StepArchiveSelector
            kwargs:
              interval: {interval}
        seed: 17
        deterministic: true
        entry: trainer
        runtime:
          class: src.experiments.runtime.ProcessRuntime
          kwargs:
            device: cpu
            precision: fp32
        output:
          directory: {directory}
        components:
          model:
            role: module
            class: tests.experiments.support.DropoutLinear
            kwargs: {{in_features: 8, out_features: 8, p: 0.0}}
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
              stop_after_logical_steps: {stop}
            inject:
              model: {{ref: model}}
              loss: {{ref: loss}}
              data: {{ref: data}}
              optimizer: {{ref: optimizer}}
              scheduler: {{ref: scheduler}}
        """
    )
    path = directory.parent / f"{directory.name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _run(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_same(left: object, right: object, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise AssertionError(path)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(path)
        for key in left:
            _assert_same(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError(path)
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            _assert_same(lhs, rhs, f"{path}[{index}]")
        return
    try:
        import numpy as np

        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
                raise AssertionError(path)
            return
    except ModuleNotFoundError:
        pass
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def _metrics(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


class InterruptPublishTests(unittest.TestCase):
    def test_stop_publishes_a_step_the_interval_has_not_reached(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stopped = root / "stopped"
            continuous = root / "continuous"
            stopped.mkdir()
            continuous.mkdir()
            first = _run(_yaml(stopped, steps=3, stop_after=1, interval=3))
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            status = json.loads((stopped / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "INTERRUPTED")
            self.assertFalse((stopped / "complete.json").is_file())
            saved = stopped / "archive/step_00000001.pt"
            self.assertTrue(saved.is_file())
            self.assertFalse((stopped / "archive/step_00000002.pt").is_file())
            resumed = _run(_yaml(stopped, steps=3, stop_after=None, interval=3), "--resume", str(saved))
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            full = _run(_yaml(continuous, steps=3, stop_after=None, interval=3))
            self.assertEqual(full.returncode, 0, full.stderr + full.stdout)
            left = torch.load(continuous / "archive/step_00000003.pt", map_location="cpu", weights_only=False)
            right = torch.load(stopped / "archive/step_00000003.pt", map_location="cpu", weights_only=False)
            self.assertEqual(left["logical_step"], 3)
            self.assertEqual(right["logical_step"], 3)
            self.assertEqual(left["states"]["data"]["update_index"], right["states"]["data"]["update_index"])
            self.assertEqual(left["optimizer_updates"], right["optimizer_updates"])
            _assert_same(left["states"], right["states"])
            _assert_same(left["rng_by_rank"], right["rng_by_rank"])
            rows = _metrics(stopped)
            steps = [row["logical_step"] for row in rows]
            self.assertEqual(steps, [1, 2, 3])
            self.assertEqual(len(steps), len(set(steps)))
            final = json.loads((stopped / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "COMPLETED")
            self.assertTrue((stopped / "complete.json").is_file())
            self.assertFalse((continuous / "complete.json").with_name("missing").exists())
