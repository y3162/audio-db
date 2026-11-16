"""YAML ``extends`` resolves before schema validation and is stored with the run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiments.api import build_experiment, load_experiment
from src.experiments.config import config_sha256, dump_resolved_yaml
from src.experiments.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]

_PARENT = """\
schema: audio-db-experiment/v1
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
    class: torch.nn.Linear
    kwargs: {{in_features: 8, out_features: 8, bias: true}}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: 8
      examples: 8
      micro_batch_size: 4
  optimizer:
    role: optimizer
    class: torch.optim.SGD
    parameters_of: model
    kwargs: {{lr: 0.1}}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs: {{max_logical_steps: 1, global_batch_size: 4, micro_batch_size: 4}}
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


class ExtendsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self) -> None:
        os.chdir(self._cwd)

    def _root(self) -> Path:
        return Path(self._tmp.name)

    def test_retired_batch_names_are_rejected(self) -> None:
        path = self._root() / "old.yaml"
        text = _PARENT.format(directory="runs/old").replace(
            "      micro_batch_size: 4\n",
            "      batch_size: 4\n",
            1,
        )
        _write(path, text)
        with self.assertRaises(ConfigError) as caught:
            load_experiment(path)
        self.assertIn("batch_size", str(caught.exception))
        self.assertIn("global_batch_size", str(caught.exception))

    def test_shipped_document_has_no_parent(self) -> None:
        path = ROOT / "tests/fixtures/configs/experiments/linear_supervised.yaml"
        config = load_experiment(path)
        self.assertEqual(config.extends_chain, (str(path.resolve()),))
        self.assertEqual(config.source_hashes, (_sha(path),))
        self.assertNotIn("extends", config.resolved_document or {})
        self.assertEqual(config_sha256(config), config_sha256(load_experiment(path)))

    def test_merge_overrides_scalars_replaces_lists_and_keeps_null(self) -> None:
        root = self._root()
        parent = root / "official" / "base.yaml"
        child = root / "research" / "child.yaml"
        _write(
            parent,
            _PARENT.format(directory="runs/extend").replace(
                "      micro_batch_size: 4\n",
                "      micro_batch_size: 4\n      tags: [a, b]\n",
            ),
        )
        _write(
            child,
            "\n".join(
                (
                    "extends: ../official/base.yaml",
                    "seed: 3",
                    "components:",
                    "  model:",
                    "    kwargs: {in_features: 4}",
                    "  data:",
                    "    kwargs:",
                    "      tags: [c]",
                    "      note: null",
                    "      features: 4",
                )
            )
            + "\n",
        )
        os.chdir("/")
        config = load_experiment(child)
        self.assertEqual(config.seed, 3)
        self.assertEqual(config.components["model"].kwargs["in_features"], 4)
        self.assertEqual(config.components["model"].kwargs["out_features"], 8)
        self.assertEqual(config.components["model"].kwargs["bias"], True)
        self.assertEqual(config.components["data"].kwargs["tags"], ["c"])
        self.assertIsNone(config.components["data"].kwargs["note"])
        self.assertEqual(config.components["data"].kwargs["examples"], 8)
        self.assertEqual(config.extends_chain, (str(parent.resolve()), str(child.resolve())))
        self.assertEqual(config.source_hashes, (_sha(parent), _sha(child)))
        self.assertNotIn("__replace__", json.dumps(config.resolved_document))

    def test_replace_then_child_override_across_three_files(self) -> None:
        root = self._root()
        grand = root / "grand.yaml"
        parent = root / "parent.yaml"
        child = root / "child.yaml"
        _write(grand, _PARENT.format(directory="runs/extend"))
        _write(
            parent,
            "\n".join(
                (
                    "extends: grand.yaml",
                    "components:",
                    "  model:",
                    "    kwargs:",
                    "      __replace__: true",
                    "      in_features: 3",
                    "      out_features: 3",
                )
            )
            + "\n",
        )
        _write(
            child,
            "\n".join(
                (
                    "extends: parent.yaml",
                    "components:",
                    "  model:",
                    "    kwargs: {bias: false}",
                    "  data:",
                    "    kwargs: {features: 3}",
                )
            )
            + "\n",
        )
        config = load_experiment(child)
        self.assertEqual(
            config.components["model"].kwargs,
            {"in_features": 3, "out_features": 3, "bias": False},
        )
        self.assertEqual(len(config.extends_chain), 3)
        self.assertEqual(config.components["optimizer"].kwargs["lr"], 0.1)

    def test_top_level_replace_drops_parent_keys(self) -> None:
        root = self._root()
        parent = root / "parent.yaml"
        child = root / "child.yaml"
        _write(parent, _PARENT.format(directory="runs/extend"))
        _write(child, "extends: parent.yaml\n__replace__: true\n" + _PARENT.format(directory="runs/other"))
        config = load_experiment(child)
        self.assertEqual(config.output_directory, Path("runs/other"))
        self.assertNotIn("__replace__", config.resolved_document or {})

    def test_false_and_mistyped_replace_are_rejected(self) -> None:
        root = self._root()
        parent = root / "parent.yaml"
        _write(parent, _PARENT.format(directory="runs/extend"))
        for flag in ("false", '"true"', "1", "null"):
            child = root / f"bad-{flag}.yaml"
            _write(
                child,
                f"extends: parent.yaml\ncomponents:\n  model:\n    kwargs:\n      __replace__: {flag}\n",
            )
            with self.assertRaises(ConfigError) as caught:
                load_experiment(child)
            self.assertIn("__replace__", str(caught.exception))

    def test_replace_inside_a_list_is_rejected(self) -> None:
        root = self._root()
        parent = root / "parent.yaml"
        child = root / "child.yaml"
        _write(parent, _PARENT.format(directory="runs/extend"))
        _write(
            child,
            "extends: parent.yaml\ncomponents:\n  data:\n    kwargs:\n      tags:\n        - __replace__: true\n",
        )
        with self.assertRaises(ConfigError) as caught:
            load_experiment(child)
        self.assertIn("__replace__", str(caught.exception))

    def test_cycle_self_missing_parent_and_bad_extends_are_rejected(self) -> None:
        root = self._root()
        left = root / "left.yaml"
        right = root / "right.yaml"
        _write(left, "extends: right.yaml\nschema: audio-db-experiment/v1\n")
        _write(right, "extends: left.yaml\nschema: audio-db-experiment/v1\n")
        with self.assertRaises(ConfigError) as caught:
            load_experiment(left)
        self.assertIn("cycle", str(caught.exception))

        _write(left, "extends: left.yaml\nschema: audio-db-experiment/v1\n")
        with self.assertRaises(ConfigError) as caught:
            load_experiment(left)
        self.assertIn("cycle", str(caught.exception))

        _write(left, "extends: missing.yaml\nschema: audio-db-experiment/v1\n")
        with self.assertRaises(ConfigError) as caught:
            load_experiment(left)
        self.assertIn("does not exist", str(caught.exception))

        _write(right, "[]\n")
        _write(left, "extends: right.yaml\n")
        with self.assertRaises(ConfigError) as caught:
            load_experiment(left)
        self.assertIn("mapping", str(caught.exception))

        for extends in ("[]", "1", "null", '""'):
            _write(left, f"extends: {extends}\n")
            with self.assertRaises(ConfigError):
                load_experiment(left)

    def test_duplicate_key_and_unknown_final_key_are_rejected(self) -> None:
        root = self._root()
        path = root / "dup.yaml"
        _write(path, "schema: audio-db-experiment/v1\nschema: audio-db-experiment/v1\n")
        with self.assertRaises(ConfigError) as caught:
            load_experiment(path)
        self.assertIn("duplicate key", str(caught.exception))

        parent = root / "parent.yaml"
        child = root / "child.yaml"
        _write(parent, _PARENT.format(directory="runs/extend"))
        _write(child, "extends: parent.yaml\nmystery: 1\n")
        with self.assertRaises(ConfigError) as caught:
            load_experiment(child)
        self.assertIn("mystery", str(caught.exception))

    def test_absolute_parent_and_hash_ignores_the_chain(self) -> None:
        root = self._root()
        parent = root / "parent.yaml"
        child = root / "child.yaml"
        flat = root / "flat.yaml"
        body = _PARENT.format(directory="runs/extend")
        _write(parent, body)
        _write(child, f"extends: {parent.resolve()}\nseed: 4\n")
        _write(flat, body.replace("seed: 17", "seed: 4", 1))
        extended = load_experiment(child)
        standalone = load_experiment(flat)
        self.assertEqual(config_sha256(extended), config_sha256(standalone))
        self.assertEqual(extended.extends_chain[0], str(parent.resolve()))
        dumped = dump_resolved_yaml(extended)
        self.assertNotIn("\nextends:", dumped)
        reloaded = yaml.safe_load(dumped)
        self.assertEqual(reloaded["seed"], 4)
        self.assertNotIn("__replace__", dumped)

    def test_builder_sees_overridden_learning_rate(self) -> None:
        root = self._root()
        parent = root / "parent.yaml"
        child = root / "child.yaml"
        out = root / "out"
        _write(parent, _PARENT.format(directory=str(out)))
        _write(
            child,
            "extends: parent.yaml\ncomponents:\n  optimizer:\n    kwargs: {lr: 0.25}\n",
        )
        experiment = build_experiment(load_experiment(child))
        try:
            optimizer = experiment.get("optimizer")
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.25)
            model = experiment.get("model")
            self.assertEqual(tuple(model.module.weight.shape), (8, 8))
        finally:
            experiment.close()

    def test_run_stores_resolved_config_and_resume_keeps_the_run_id(self) -> None:
        root = self._root()
        parent = root / "parent.yaml"
        child = root / "child.yaml"
        out = root / "out"
        text = _PARENT.format(directory=str(out)).replace(
            "max_logical_steps: 1",
            "max_logical_steps: 2, stop_after_logical_steps: 1",
            1,
        )
        _write(parent, text)
        _write(child, "extends: parent.yaml\n")
        config = load_experiment(child)
        experiment = build_experiment(config)
        try:
            result = experiment.run()
        finally:
            experiment.close()
        self.assertEqual(result.stop_reason, "interrupted")
        stored = (out / "config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("extends:", stored)
        reloaded = load_experiment(out / "config.yaml")
        self.assertEqual(config_sha256(reloaded), config_sha256(config))
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["format"], "audio-db-run-record-v1")
        self.assertEqual(len(meta["run_id"]), 32)
        self.assertEqual(len(meta["launches"]), 1)
        self.assertEqual(meta["launches"][0]["resume"], False)
        self.assertEqual(
            meta["launches"][0]["batch"],
            {
                "accumulation_steps": 1,
                "global_batch_size": 4,
                "micro_batch_size": 4,
                "world_size": 1,
            },
        )
        self.assertEqual(meta["source_hashes"], [_sha(parent), _sha(child)])
        self.assertFalse((out / "config_history").exists())

        # stop_after_logical_steps is an execution cap and is outside the identity hash.
        # Resume drops it, so the same experiment can finish the remaining step.
        resume_child = root / "resume.yaml"
        _write(
            resume_child,
            "\n".join(
                (
                    "extends: parent.yaml",
                    "components:",
                    "  trainer:",
                    "    kwargs:",
                    "      __replace__: true",
                    "      max_logical_steps: 2",
                    "      global_batch_size: 4",
                    "      micro_batch_size: 4",
                )
            )
            + "\n",
        )
        resumed_config = load_experiment(resume_child)
        self.assertEqual(config_sha256(resumed_config), config_sha256(config))
        again = build_experiment(resumed_config)
        try:
            finished = again.run(resume=out)
        finally:
            again.close()
        self.assertEqual(finished.stop_reason, "completed")
        self.assertEqual(finished.logical_step, 2)
        meta_again = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta_again["run_id"], meta["run_id"])
        self.assertEqual(len(meta_again["launches"]), 2)
        self.assertEqual(meta_again["launches"][1]["resume"], True)
        self.assertTrue((out / "config_history" / "0001.yaml").is_file())
        self.assertIn("stop_after_logical_steps", (out / "config_history" / "0001.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
