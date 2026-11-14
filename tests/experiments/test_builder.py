"""Builder resolves explicit references and rejects bad graphs."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.experiments.api import Experiment, build_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import BuildError, ConfigError, ExperimentError
from tests.experiments.support import (
    Marker,
    OrderProbe,
    Timeline,
    null_trainer,
    write_config,
)


class BuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        Marker.created = 0
        OrderProbe.reset()
        Timeline.reset()

    def test_single_and_shared_instances(self) -> None:
        path = write_config(
            self._tmp(),
            {
                "loss": {
                    "role": "loss",
                    "class": "tests.experiments.support.Marker",
                    "kwargs": {"value": 3},
                },
                "loss_for_log": {"ref": "loss"},
                "trainer": {
                    "role": "trainer",
                    "class": "tests.experiments.support.CaptureTrainer",
                    "inject": {
                        "loss": {"ref": "loss"},
                        "loss_for_log": {"ref": "loss_for_log"},
                    },
                },
            },
        )
        experiment = build_experiment(load_experiment(path))
        trainer = experiment.get("trainer")
        self.assertIs(trainer.loss, trainer.loss_for_log)
        self.assertIs(experiment.get("loss"), experiment.get("loss_for_log"))
        self.assertEqual(Marker.created, 1)
        self.assertEqual(trainer.loss.value, 3)

    def test_yaml_order_does_not_change_construction_order(self) -> None:
        first = self._order_document(("b_item", "b"), ("a_item", "a"))
        second = self._order_document(("a_item", "a"), ("b_item", "b"))
        OrderProbe.reset()
        build_experiment(load_experiment(self._write_raw("first.yaml", first)))
        self.assertEqual(OrderProbe.labels, ["a", "b"])
        OrderProbe.reset()
        build_experiment(load_experiment(self._write_raw("second.yaml", second)))
        self.assertEqual(OrderProbe.labels, ["a", "b"])

    def test_undefined_reference_builds_nothing(self) -> None:
        path = write_config(
            self._tmp(),
            {
                "a": {
                    "role": "other",
                    "class": "tests.experiments.support.Marker",
                    "inject": {"value": {"ref": "missing"}},
                },
                "trainer": null_trainer(),
            },
        )
        experiment = Experiment(load_experiment(path))
        with self.assertRaises(BuildError) as caught:
            experiment.build()
        self.assertIn("missing", str(caught.exception))
        self.assertIn("a", str(caught.exception))
        self.assertEqual(Marker.created, 0)
        self.assertFalse(experiment._built)
        with self.assertRaises(BuildError):
            experiment.get("a")

    def test_cycle_is_rejected_before_construction(self) -> None:
        path = write_config(
            self._tmp(),
            {
                "a": {
                    "role": "other",
                    "class": "tests.experiments.support.NeedsOther",
                    "inject": {"other": {"ref": "b"}},
                },
                "b": {
                    "role": "other",
                    "class": "tests.experiments.support.NeedsOther",
                    "inject": {"other": {"ref": "a"}},
                },
                "trainer": null_trainer(),
            },
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(path))
        message = str(caught.exception)
        self.assertIn("a", message)
        self.assertIn("b", message)
        self.assertEqual(Timeline.constructed, 0)

    def test_argument_collision(self) -> None:
        path = write_config(
            self._tmp(),
            {
                "other": {"role": "other", "class": "tests.experiments.support.Marker"},
                "holder": {
                    "role": "other",
                    "class": "tests.experiments.support.TakesLr",
                    "kwargs": {"lr": 0.2},
                    "inject": {"lr": {"ref": "other"}},
                },
                "trainer": null_trainer(),
            },
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(path))
        self.assertIn("lr", str(caught.exception))
        self.assertEqual(Timeline.constructed, 0)
        self.assertEqual(Marker.created, 0)

    def test_unknown_key_does_not_start_training(self) -> None:
        directory = self._tmp()
        document = {
            "schema": "audio-db-experiment/v1",
            "seed": 1,
            "deterministic": True,
            "entry": "trainer",
            "train_loop": {"steps": 3},
            "runtime": {
                "class": "src.experiments.runtime.ProcessRuntime",
                "kwargs": {"device": "cpu", "precision": "fp32"},
            },
            "output": {"directory": str(directory / "out")},
            "components": {
                "trainer": {
                    "role": "trainer",
                    "class": "tests.experiments.support.Marker",
                }
            },
        }
        path = directory / "bad.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            load_experiment(path)
        self.assertIn("train_loop", str(caught.exception))
        self.assertEqual(Marker.created, 0)
        self.assertFalse((directory / "out").exists())

    def test_duplicate_component_id(self) -> None:
        directory = self._tmp()
        path = directory / "duplicate.yaml"
        path.write_text(
            "\n".join(
                [
                    "schema: audio-db-experiment/v1",
                    "seed: 1",
                    "deterministic: true",
                    "entry: trainer",
                    "runtime:",
                    "  class: src.experiments.runtime.ProcessRuntime",
                    "  kwargs: {device: cpu, precision: fp32}",
                    "output:",
                    "  directory: out",
                    "components:",
                    "  model:",
                    "    role: other",
                    "    class: tests.experiments.support.Marker",
                    "  model:",
                    "    role: other",
                    "    class: tests.experiments.support.Marker",
                    "  trainer:",
                    "    role: trainer",
                    "    class: tests.experiments.support.NullTrainer",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ConfigError) as caught:
            load_experiment(path)
        self.assertIn("model", str(caught.exception))
        self.assertEqual(Marker.created, 0)

    def test_invalid_role(self) -> None:
        directory = self._tmp()
        path = write_config(
            directory,
            {
                "model": {
                    "role": "not_a_role",
                    "class": "tests.experiments.support.Marker",
                },
                "trainer": null_trainer(),
            },
        )
        with self.assertRaises(ConfigError) as caught:
            load_experiment(path)
        self.assertIn("not_a_role", str(caught.exception))
        self.assertEqual(Marker.created, 0)

    def test_missing_dependency_is_not_inferred(self) -> None:
        path = write_config(
            self._tmp(),
            {
                "loss": {"role": "loss", "class": "torch.nn.MSELoss"},
                "model": {"role": "other", "class": "tests.experiments.support.NeedsExplicit"},
                "trainer": null_trainer(),
            },
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(path))
        self.assertIn("loss", str(caught.exception))
        self.assertEqual(Timeline.constructed, 0)

    def test_later_stage_dependency_is_rejected(self) -> None:
        path = write_config(
            self._tmp(),
            {
                "other": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 1, "out_features": 1},
                },
                "model": {
                    "role": "module",
                    "class": "tests.experiments.support.ModuleWithExtra",
                    "kwargs": {"in_features": 1, "out_features": 1},
                    "inject": {"extra": {"ref": "optim"}},
                },
                "optim": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": "other",
                    "kwargs": {"lr": 0.01},
                },
                "trainer": null_trainer(),
            },
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(path))
        message = str(caught.exception)
        self.assertIn("model", message)
        self.assertIn("optim", message)
        self.assertEqual(Timeline.constructed, 0)

    def test_failed_build_is_not_runnable(self) -> None:
        path = write_config(
            self._tmp(),
            {
                "a": {
                    "role": "other",
                    "class": "tests.experiments.support.NeedsOther",
                    "inject": {"other": {"ref": "b"}},
                },
                "b": {
                    "role": "other",
                    "class": "tests.experiments.support.NeedsOther",
                    "inject": {"other": {"ref": "a"}},
                },
                "trainer": null_trainer(),
            },
        )
        experiment = Experiment(load_experiment(path))
        with self.assertRaises(BuildError):
            experiment.build()
        with self.assertRaises(ExperimentError):
            experiment.run()
        self.assertFalse(experiment._built)

    def test_a_stateless_trainer_is_omitted_unless_save_is_explicit(self) -> None:
        path = write_config(
            self._tmp(),
            {"trainer": {"role": "trainer", "class": "tests.experiments.support.CaptureTrainer"}},
        )
        experiment = build_experiment(load_experiment(path))
        assert experiment._graph is not None
        self.assertNotIn("trainer", experiment._graph.states.ids())
        explicit = write_config(
            self._tmp(),
            {
                "trainer": {
                    "role": "trainer",
                    "class": "tests.experiments.support.CaptureTrainer",
                    "state": {"save": True},
                }
            },
            filename="explicit.yaml",
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(explicit))
        self.assertIn("state.save", str(caught.exception))

    def _tmp(self) -> Path:
        return Path(self.enterContext(self._tempdir()))

    def _tempdir(self):
        import tempfile

        return tempfile.TemporaryDirectory()

    def _write_raw(self, name: str, text: str) -> Path:
        directory = self._tmp()
        path = directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def _order_document(self, first: tuple[str, str], second: tuple[str, str]) -> str:
        directory = self._tmp()
        output = directory / "out"
        return "\n".join(
            [
                "schema: audio-db-experiment/v1",
                "seed: 17",
                "deterministic: true",
                "entry: trainer",
                "runtime:",
                "  class: src.experiments.runtime.ProcessRuntime",
                "  kwargs: {device: cpu, precision: fp32}",
                "output:",
                f"  directory: {output}",
                "components:",
                f"  {first[0]}:",
                "    role: other",
                "    class: tests.experiments.support.OrderProbe",
                f"    kwargs: {{label: {first[1]}}}",
                f"  {second[0]}:",
                "    role: other",
                "    class: tests.experiments.support.OrderProbe",
                f"    kwargs: {{label: {second[1]}}}",
                "  trainer:",
                "    role: trainer",
                "    class: tests.experiments.support.NullTrainer",
                "",
            ]
        )


if __name__ == "__main__":
    unittest.main()
