"""A loss-coefficient change resumes without rewinding a decayed learning rate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.config import config_sha256, load_experiment as load_config, structural_sha256
from src.experiments.errors import CheckpointError


class ScaledMSE(torch.nn.Module):
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        table = {"mse": 1.0} if weights is None else weights
        self.scale = float(table["mse"])

    def forward(self, prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (prediction - targets).square().mean() * self.scale


class RecordingAdamW(torch.optim.AdamW):
    seen: list[float] = []

    def step(self, closure=None):  # type: ignore[no-untyped-def]
        RecordingAdamW.seen.append(float(self.param_groups[0]["lr"]))
        return super().step(closure)


class CoefficientResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingAdamW.seen = []

    def test_numeric_weights_leave_the_structural_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "one.yaml"
            second = root / "two.yaml"
            changed = root / "class.yaml"
            first.write_text(_yaml(root / "out", 1.0, "tests.experiments.test_resume_coefficients.ScaledMSE"), encoding="utf-8")
            second.write_text(_yaml(root / "out", 3.0, "tests.experiments.test_resume_coefficients.ScaledMSE"), encoding="utf-8")
            changed.write_text(_yaml(root / "out", 3.0, "torch.nn.L1Loss"), encoding="utf-8")
            one = load_config(first)
            two = load_config(second)
            other = load_config(changed)
            self.assertEqual(structural_sha256(one), structural_sha256(two))
            self.assertNotEqual(config_sha256(one), config_sha256(two))
            self.assertNotEqual(structural_sha256(one), structural_sha256(other))

    def test_resume_uses_the_new_coefficient_and_the_decayed_rate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            out = root / "out"
            first = root / "one.yaml"
            second = root / "two.yaml"
            first.write_text(
                _yaml(out, 1.0, "tests.experiments.test_resume_coefficients.ScaledMSE", stop_after=1),
                encoding="utf-8",
            )
            second.write_text(
                _yaml(out, 3.0, "tests.experiments.test_resume_coefficients.ScaledMSE"),
                encoding="utf-8",
            )
            started = build_experiment(load_experiment(first))
            try:
                started.run()
            finally:
                started.close()
            continued = build_experiment(load_experiment(second))
            try:
                result = continued.run(resume=out)
            finally:
                continued.close()
            self.assertEqual(result.logical_step, 2)
            self.assertEqual(continued.get("loss").scale, 3.0)
            self.assertEqual(RecordingAdamW.seen, [0.001, 0.0005])
            parameter = next(continued.get("model").state_module.parameters())
            self.assertEqual(int(continued.get("optimizer").state[parameter]["step"]), 2)

    def test_a_shape_change_is_still_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            out = root / "out"
            first = root / "one.yaml"
            wider = root / "wide.yaml"
            first.write_text(
                _yaml(out, 1.0, "tests.experiments.test_resume_coefficients.ScaledMSE", stop_after=1),
                encoding="utf-8",
            )
            wider.write_text(
                _yaml(out, 1.0, "tests.experiments.test_resume_coefficients.ScaledMSE").replace(
                    "out_features: 4", "out_features: 8"
                ),
                encoding="utf-8",
            )
            started = build_experiment(load_experiment(first))
            try:
                started.run()
            finally:
                started.close()
            with self.assertRaises(CheckpointError) as caught:
                build_experiment(load_experiment(wider)).run(resume=out)
            self.assertIn("structure", str(caught.exception).lower())


def _yaml(directory: Path, scale: float, loss_class: str, stop_after: int | None = None) -> str:
    stop = "" if stop_after is None else f"\n      stop_after_logical_steps: {stop_after}"
    weights = ""
    if loss_class.endswith("ScaledMSE"):
        weights = f"\n    kwargs: {{weights: {{mse: {scale}}}}}"
    return f"""schema: audio-db-experiment/v1
seed: 11
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs: {{device: cpu, precision: fp32, preflight: false}}
output:
  directory: {directory}
components:
  model:
    role: module
    class: torch.nn.Linear
    kwargs: {{in_features: 4, out_features: 4}}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: {loss_class}{weights}
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 4
      examples: 8
      micro_batch_size: 4
  optimizer:
    role: optimizer
    class: tests.experiments.test_resume_coefficients.RecordingAdamW
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
      max_logical_steps: 2
      global_batch_size: 4
      micro_batch_size: 4{stop}
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
      scheduler: {{ref: scheduler}}
"""
