"""Named model outputs. A research feature uses the same StandardTrainer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.training.trainers import _named_context
from src.models.features import parse_output_features
from src.models.mp_senet.model import _FORWARD_FIELDS, pack_forward
from src.models.semambapp.model import pack_forward as pack_semamba


class FeatureModel(torch.nn.Module):
    """A copied research model. ``hidden`` is optional and is not a trainer branch."""

    def __init__(self, output_features: list[str] | tuple[str, ...] | None = None) -> None:
        super().__init__()
        self.output_features = parse_output_features(output_features, allowed=frozenset({"hidden"}))
        self.encoder = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = torch.tanh(self.encoder(inputs))
        outputs = {"enhanced": self.head(hidden)}
        if "hidden" in self.output_features:
            outputs["hidden"] = hidden
        return outputs


class FeatureLoss(torch.nn.Module):
    def forward(self, context: dict[str, torch.Tensor]) -> torch.Tensor:
        return (context["hidden"] - context["targets"]).square().mean()


class NamedOutputTests(unittest.TestCase):
    def test_pack_keeps_an_extra_feature_and_rejects_a_tuple(self) -> None:
        values = {name: torch.zeros(1) for name in _FORWARD_FIELDS}
        values["noisy_complex"] = torch.ones(1)
        packed = pack_forward(values)
        self.assertEqual(packed["noisy_complex"].item(), 1.0)
        with self.assertRaises(RuntimeFailure):
            pack_forward(tuple(values[name] for name in _FORWARD_FIELDS))  # type: ignore[arg-type]
        semamba_names = (
            "clean_mag", "clean_phase", "clean_complex", "mag", "phase", "complex", "recomputed", "audio",
        )
        semamba = pack_semamba({name: torch.zeros(1) for name in semamba_names})
        self.assertEqual(set(semamba), set(semamba_names))

    def test_unknown_feature_is_rejected(self) -> None:
        with self.assertRaises(BuildError):
            FeatureModel(output_features=["missing"])
        with self.assertRaises(BuildError):
            parse_output_features("hidden", allowed=frozenset({"hidden"}))

    def test_research_feature_reaches_the_encoder_through_standard_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "run.yaml"
            config.write_text(_yaml(root / "out", '["hidden"]'), encoding="utf-8")
            experiment = build_experiment(load_experiment(config))
            model = experiment.get("model").state_module
            before = model.encoder.weight.detach().clone()
            try:
                result = experiment.run()
            finally:
                experiment.close()
            self.assertEqual(result.logical_step, 1)
            self.assertFalse(torch.equal(model.encoder.weight, before))

    def test_unrequested_feature_is_absent_from_the_loss_context(self) -> None:
        model = FeatureModel()
        produced = model(torch.zeros(2, 4))
        self.assertNotIn("hidden", produced)
        with self.assertRaises(KeyError):
            FeatureLoss()({"targets": torch.zeros(2, 4), **produced})

    def test_a_batch_field_name_is_rejected(self) -> None:
        batch = type("Batch", (), {"inputs": torch.zeros(1), "targets": torch.zeros(1), "sample_id": None})()
        with self.assertRaises(RuntimeFailure):
            _named_context({"targets": torch.zeros(1)}, batch, "trainer")


def _yaml(directory: Path, features: str) -> str:
    return f"""schema: audio-db-experiment/v1
seed: 5
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
    class: tests.experiments.test_named_outputs.FeatureModel
    kwargs:
      output_features: {features}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: tests.experiments.test_named_outputs.FeatureLoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 4
      examples: 4
      micro_batch_size: 2
  optimizer:
    role: optimizer
    class: torch.optim.SGD
    parameters_of: model
    kwargs: {{lr: 0.1}}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs: {{max_logical_steps: 1, global_batch_size: 2, micro_batch_size: 2}}
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""
