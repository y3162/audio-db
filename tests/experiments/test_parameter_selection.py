"""Named freezing and optimizer groups are selected before the optimizer exists."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.errors import BuildError, ConfigError, PrepareError


class TwoPart(torch.nn.Module):
    """Two linear maps used to prove submodule selection."""

    def __init__(self, features: int = 4) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(features, features)
        self.decoder = torch.nn.Linear(features, features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


_DOCUMENT = """\
schema: audio-db-experiment/v1
seed: 1
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs: {{device: cpu, precision: fp32}}
output:
  directory: {directory}
components:
  model:
    role: module
    class: tests.experiments.test_parameter_selection.TwoPart
    kwargs: {{features: 4}}
    prepare: {{trainable: true, distribute: none{prepare}}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs: {{features: 4, examples: 4, batch_size: 2}}
  optimizer:
    role: optimizer
    class: torch.optim.SGD
    parameters_of: {parameters}
    kwargs: {{lr: 0.1}}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs: {{max_logical_steps: 1, accumulation: 1}}
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""


class ParameterSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _path(self, name: str, text: str) -> Path:
        path = Path(self._tmp.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def _document(self, name: str, *, prepare: str = "", parameters: str = "model") -> Path:
        text = _DOCUMENT.format(
            directory=str(Path(self._tmp.name) / name / "out"),
            prepare=prepare,
            parameters=parameters,
        )
        return self._path(f"{name}.yaml", text)

    def test_frozen_module_is_excluded_from_the_optimizer(self) -> None:
        path = self._document("freeze", prepare=", frozen_modules: [decoder]")
        experiment = build_experiment(load_experiment(path))
        try:
            model = experiment.get("model").state_module
            self.assertTrue(model.encoder.weight.requires_grad)
            self.assertFalse(model.decoder.weight.requires_grad)
            self.assertFalse(model.decoder.bias.requires_grad)
            optimizer = experiment.get("optimizer")
            owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            self.assertIn(id(model.encoder.weight), owned)
            self.assertNotIn(id(model.decoder.weight), owned)
            self.assertEqual(len(owned), 2)
        finally:
            experiment.close()

    def test_trainable_modules_freeze_the_rest(self) -> None:
        path = self._document("train", prepare=", trainable_modules: [encoder]")
        experiment = build_experiment(load_experiment(path))
        try:
            model = experiment.get("model").state_module
            self.assertTrue(model.encoder.bias.requires_grad)
            self.assertFalse(model.decoder.weight.requires_grad)
        finally:
            experiment.close()

    def test_missing_duplicate_empty_and_overlap_are_rejected(self) -> None:
        missing = self._document("missing", prepare=", frozen_modules: [missing]")
        with self.assertRaises(PrepareError) as caught:
            build_experiment(load_experiment(missing)).close()
        self.assertIn("missing", str(caught.exception))

        for prepare in (
            ", frozen_modules: []",
            ", frozen_modules: [encoder, encoder]",
            ", trainable_modules: [encoder], frozen_modules: [encoder]",
            ", frozen_modules: null",
        ):
            path = self._document("bad", prepare=prepare)
            with self.assertRaises(ConfigError):
                load_experiment(path)

    def test_named_groups_keep_a_default_learning_rate_for_the_remainder(self) -> None:
        parameters = "\n".join(
            (
                "",
                "      - component: model",
                "        modules: [encoder]",
                "        lr: 0.01",
            )
        )
        path = self._document("groups", parameters=parameters)
        experiment = build_experiment(load_experiment(path))
        try:
            model = experiment.get("model").state_module
            groups = experiment.get("optimizer").param_groups
            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0]["lr"], 0.01)
            self.assertEqual({id(parameter) for parameter in groups[0]["params"]}, {id(model.encoder.weight), id(model.encoder.bias)})
            self.assertEqual(groups[1]["lr"], 0.1)
            self.assertEqual({id(parameter) for parameter in groups[1]["params"]}, {id(model.decoder.weight), id(model.decoder.bias)})
        finally:
            experiment.close()

    def test_one_parameter_cannot_join_two_groups(self) -> None:
        parameters = "\n".join(
            (
                "",
                "      - {component: model, modules: [encoder], lr: 0.01}",
                "      - {component: model, modules: [encoder], lr: 0.02}",
            )
        )
        path = self._document("dup", parameters=parameters)
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(path)).close()
        self.assertIn("more than one group", str(caught.exception))

    def test_extends_can_replace_the_loss_without_a_new_trainer(self) -> None:
        parent = self._document("loss-parent")
        child = self._path(
            "loss-child.yaml",
            "\n".join(
                (
                    f"extends: {parent.name}",
                    "components:",
                    "  loss:",
                    "    class: torch.nn.L1Loss",
                )
            )
            + "\n",
        )
        # The child lives next to the parent, so the relative name resolves.
        experiment = build_experiment(load_experiment(child))
        try:
            self.assertIsInstance(experiment.get("loss"), torch.nn.L1Loss)
            result = experiment.run()
            self.assertEqual(result.logical_step, 1)
            self.assertEqual(result.optimizer_updates["optimizer"], 1)
        finally:
            experiment.close()
