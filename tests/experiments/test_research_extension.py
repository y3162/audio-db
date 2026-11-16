"""One research experiment that is not an official reproduction.

The copied model, loss, metrics, frozen parameter group, ``extends``,
scheduler, preflight, and resumed optimizer setting are all exercised
together. Official trainers are not used.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.api import build_experiment, preflight_experiment
from src.experiments.config import config_sha256, load_experiment, structural_sha256
from src.experiments.errors import BuildError


class WaveCopy(torch.nn.Module):
    """Two convolutions. Research code can replace this class without a new trainer."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv1d(1, 1, 1)
        self.decoder = torch.nn.Conv1d(1, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs.unsqueeze(1))).squeeze(1)


class ResearchExtensionDemo(unittest.TestCase):
    def test_copied_model_runs_preflight_metrics_and_resumes_with_a_new_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent.yaml"
            child = root / "child.yaml"
            parent.write_text(_parent(root / "out"), encoding="utf-8")
            child.write_text(_child(), encoding="utf-8")
            loaded = load_experiment(child)
            self.assertEqual(loaded.components["model"].class_path, _MODEL)
            self.assertEqual(loaded.components["loss"].class_path, "torch.nn.L1Loss")
            self.assertEqual(
                loaded.components["scheduler"].class_path,
                "torch.optim.lr_scheduler.StepLR",
            )
            self.assertEqual(
                [spec.class_path for spec in (loaded.components["train_loss"], loaded.components["sisdr"])],
                [
                    "src.experiments.metrics.LossMetric",
                    "src.experiments.metrics.SISDRMetric",
                ],
            )
            experiment = build_experiment(loaded)
            model = experiment.get("model").state_module
            self.assertIsInstance(model, WaveCopy)
            self.assertIsInstance(experiment.get("loss"), torch.nn.L1Loss)
            self.assertFalse(model.decoder.weight.requires_grad)
            self.assertTrue(model.encoder.weight.requires_grad)
            self.assertEqual(experiment.get("optimizer").param_groups[0]["lr"], 0.05)
            self.assertNotIn(id(model.decoder.weight), _owned(experiment))
            decoder_before = model.decoder.weight.detach().clone()
            encoder_before = model.encoder.weight.detach().clone()
            report = json.loads(preflight_experiment(experiment).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["training_executed"])
            self.assertEqual(report["batch_probe"]["status"], "ok")
            self.assertEqual(experiment.get("trainer").logical_step, 0)
            first = experiment.run()
            self.assertEqual(first.stop_reason, "interrupted")
            self.assertEqual(first.logical_step, 1)
            self.assertTrue(torch.equal(model.decoder.weight, decoder_before))
            self.assertFalse(torch.equal(model.encoder.weight, encoder_before))
            resume = root / "resume.yaml"
            resume.write_text(_resume(), encoding="utf-8")
            original = loaded
            updated = load_experiment(resume)
            self.assertNotEqual(config_sha256(original), config_sha256(updated))
            self.assertEqual(structural_sha256(original), structural_sha256(updated))
            kept = (root / "out" / "latest.pt").read_bytes()
            wider = root / "wider.yaml"
            wider.write_text(_resume().replace("modules: [encoder]", "modules: [missing]"), encoding="utf-8")
            with self.assertRaises(BuildError):
                build_experiment(load_experiment(wider)).run(resume=root / "out")
            self.assertEqual((root / "out" / "latest.pt").read_bytes(), kept)
            continued = build_experiment(updated)
            result = continued.run(resume=root / "out")
            self.assertEqual(result.logical_step, 2)
            self.assertEqual(result.stop_reason, "completed")
            scheduler = continued.get("scheduler")
            self.assertEqual(scheduler.base_lrs, [0.08])
            self.assertEqual(continued.get("optimizer").param_groups[0]["lr"], 0.04)
            runtime = continued._graph.runtime
            scores = continued.get("validation").evaluate(runtime, logical_step=2, epoch=0)
            by_name = {item.name: item for item in scores}
            self.assertEqual(set(by_name), {"loss", "si_sdr"})
            self.assertEqual(by_name["loss"].status, "ok")
            self.assertEqual(by_name["si_sdr"].status, "ok")
            self.assertIsNotNone(by_name["loss"].value)
            self.assertIsNotNone(by_name["si_sdr"].value)


_MODEL = "tests.experiments.test_research_extension.WaveCopy"


def _owned(experiment: object) -> set[int]:
    return {
        id(parameter)
        for group in experiment.get("optimizer").param_groups
        for parameter in group["params"]
    }


def _parent(directory: Path) -> str:
    return f"""schema: audio-db-experiment/v1
seed: 3
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs: {{device: cpu, precision: fp32, preflight: true}}
output:
  directory: {directory}
components:
  model:
    role: module
    class: {_MODEL}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: tests.experiments.tf_gridnet_data.ScriptedChunkData
    kwargs:
      updates_per_epoch: 2
      micro_batch_size: 1
      length: 32
      seed: 17
      valid_sizes: [[1]]
  optimizer:
    role: optimizer
    class: torch.optim.SGD
    parameters_of: model
    kwargs: {{lr: 0.2}}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs: {{max_logical_steps: 2, global_batch_size: 1, micro_batch_size: 1}}
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""


def _child() -> str:
    return """extends: parent.yaml
components:
  model:
    prepare: {trainable: true, distribute: none, frozen_modules: [decoder]}
  loss:
    class: torch.nn.L1Loss
  waveform:
    role: loss
    class: src.experiments.losses.WaveformL1
    kwargs: {name: waveform}
  objective:
    role: loss
    class: src.experiments.losses.WeightedObjective
    kwargs:
      weights: {waveform: 1.0}
    inject:
      terms: [{ref: waveform}]
  optimizer:
    parameters_of:
      - component: model
        modules: [encoder]
        lr: 0.05
  scheduler:
    role: scheduler
    class: torch.optim.lr_scheduler.StepLR
    kwargs: {step_size: 1, gamma: 0.5}
    inject:
      optimizer: {ref: optimizer}
  train_loss:
    role: metric
    class: src.experiments.metrics.LossMetric
    kwargs: {name: loss}
  sisdr:
    role: metric
    class: src.experiments.metrics.SISDRMetric
    kwargs: {name: si_sdr}
  validation:
    role: evaluator
    class: src.experiments.evaluation.AudioPairEvaluator
    kwargs:
      source: validation
      every: {unit: epoch, interval: 1}
    inject:
      model: {ref: model}
      data: {ref: data}
      objective: {ref: objective}
      metrics: [{ref: train_loss}, {ref: sisdr}]
  trainer:
    kwargs:
      stop_after_logical_steps: 1
    inject:
      scheduler: {ref: scheduler}
"""


def _resume() -> str:
    return """extends: child.yaml
components:
  optimizer:
    parameters_of:
      - component: model
        modules: [encoder]
        lr: 0.08
  trainer:
    kwargs:
      __replace__: true
      max_logical_steps: 2
      global_batch_size: 1
      micro_batch_size: 1
"""
