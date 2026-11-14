"""Research evaluation on the real TF-GridNet trainer.

Changing the YAML metric, interval, or selection mode changes the recorded
choice. The official validation loss file stays on its own path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from tests.experiments.test_tf_gridnet_training import _scripted_text


class ResearchTrainingTests(unittest.TestCase):
    def test_selection_mode_changes_the_chosen_step_and_keeps_the_official_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minimum = _run(root / "min", _selection_text(root / "min", mode="min"))
            maximum = _run(root / "max", _selection_text(root / "max", mode="max"))
            self.assertEqual(minimum["mode"], "min")
            self.assertEqual(maximum["mode"], "max")
            self.assertNotEqual(minimum["logical_step"], maximum["logical_step"])
            official = json.loads((root / "min" / "valid.loss.best.pt.json").read_text(encoding="utf-8"))
            self.assertEqual(minimum["logical_step"], official["global_step"])
            self.assertNotEqual(maximum["logical_step"], official["global_step"])
            official_weights = torch.load(root / "min" / "valid.loss.best.pt", map_location="cpu", weights_only=False)
            research_weights = torch.load(
                root / "min" / "research_selection" / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertNotIn("format", official_weights)
            self.assertEqual(research_weights["format"], "audio-db-research-selection-v1")
            self.assertFalse((root / "min" / "research_selection" / "valid.loss.best.pt").exists())

    def test_epoch_interval_changes_how_often_the_metric_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            every = _rows(root / "every", interval=1)
            half = _rows(root / "half", interval=2)
        self.assertEqual([row["logical_step"] for row in every], [1, 2])
        self.assertEqual([row["logical_step"] for row in half], [1])
        self.assertTrue(all(row["source"] == "validation" and row["metric"] == "si_sdr" for row in every))

    def test_wer_is_the_corpus_rate_and_train_eval_is_a_different_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _run(root / "wer", _wer_text(root / "wer"))
            rows = _log(root / "wer")
            wer = [row for row in rows if row["metric"] == "wer"]
            train_eval = [row for row in rows if row["source"] == "train_eval"]
            validation = [row for row in rows if row["record"] == "metric" and row["source"] == "validation"]
            self.assertEqual(len(wer), 1)
            self.assertEqual(wer[0]["status"], "ok")
            self.assertAlmostEqual(wer[0]["value"], 1.0 / 3.0)
            self.assertNotEqual(wer[0]["value"], 0.0)
            self.assertTrue(train_eval)
            self.assertTrue(validation)
            self.assertTrue(all(row["metric"] == "loss" for row in train_eval))
            self.assertTrue((root / "wer" / "valid.loss.best.pt").is_file())

    def test_research_path_does_not_change_the_trained_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain_result = build_experiment(load_experiment(_write(root / "plain", _scripted_text(root / "plain", steps=2, updates=1)))).run()
            self.assertEqual(plain_result.stop_reason, "completed")
            _run(root / "research", _selection_text(root / "research", mode="min", evaluator=True))
            plain = torch.load(root / "plain" / "latest.pt", map_location="cpu", weights_only=False)
            research = torch.load(root / "research" / "latest.pt", map_location="cpu", weights_only=False)
            for name in ("model", "optimizer", "scheduler", "data", "trainer"):
                self.assertTrue(_same(plain["states"][name], research["states"][name]), name)
            self.assertTrue(_same(plain["rng_by_rank"], research["rng_by_rank"]))

    def test_resume_replays_the_pending_boundary_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = _run(root / "full", _selection_text(root / "full", mode="min"))
            partial = root / "partial" / "experiment.yaml"
            _write(partial.parent, _selection_text(partial.parent, mode="min", stop_after=1))
            stopped = build_experiment(load_experiment(partial)).run()
            self.assertEqual(stopped.stop_reason, "interrupted")
            document = yaml.safe_load(partial.read_text(encoding="utf-8"))
            del document["components"]["trainer"]["kwargs"]["stop_after_logical_steps"]
            partial.write_text(yaml.safe_dump(document), encoding="utf-8")
            resumed = build_experiment(load_experiment(partial)).run(resume=partial.parent)
            self.assertEqual(resumed.stop_reason, "completed")
            self.assertEqual(resumed.logical_step, 2)
            self.assertEqual(_choice(partial.parent)["logical_step"], full["logical_step"])
            self.assertEqual(_choice(partial.parent)["value"], full["value"])
            self.assertEqual(len(_log(partial.parent)), len(_log(root / "full")))

    def test_gradient_accumulation_still_completes_one_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = _selection_text(root, mode="min", steps=1, updates=2)
            text = text.replace("      micro_batch: 1\n", "      micro_batch: 1\n      group_size: 2\n", 1)
            text = text.replace("      accumulation: 1\n", "      accumulation: 2\n", 1)
            result = build_experiment(load_experiment(_write(root, text))).run()
            self.assertEqual(result.stop_reason, "completed")
            self.assertEqual(result.logical_step, 1)
            self.assertEqual(result.optimizer_updates, {"optimizer": 1})
            self.assertEqual(len(_log(root)), 1)

    def test_gloo_ranks_agree_on_the_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _write(root, _selection_text(root, mode="min", distribute="ddp"))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.experiments.launcher",
                    "--config",
                    str(path),
                    "--nproc",
                    "2",
                    "--gpu-memory-interval",
                    "0",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
            choice = _choice(root)
            self.assertEqual(choice["metric"], "validation.loss")
            self.assertTrue((root / "latest.pt").is_file())


def _run(directory: Path, text: str) -> dict[str, object]:
    result = build_experiment(load_experiment(_write(directory, text))).run()
    if result.stop_reason != "completed":
        raise AssertionError(result.stop_reason)
    return _choice(directory)


def _rows(directory: Path, *, interval: int) -> list[dict[str, object]]:
    result = build_experiment(load_experiment(_write(directory, _metric_text(directory, interval=interval)))).run()
    if result.stop_reason != "completed":
        raise AssertionError(result.stop_reason)
    return [row for row in _log(directory) if row["record"] == "metric"]


def _write(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(text)
    return path


def _selection_text(
    directory: Path,
    *,
    mode: str,
    evaluator: bool = False,
    steps: int = 2,
    updates: int = 1,
    stop_after: int | None = None,
    distribute: str = "none",
) -> str:
    text = _scripted_text(directory, steps=steps, updates=updates, stop_after=stop_after, distribute=distribute)
    metric = ""
    inject = "      model_selection: {ref: selection}\n"
    if evaluator:
        metric = """  sisdr:
    role: metric
    class: src.experiments.metrics.SISDRMetric
    kwargs:
      name: si_sdr
  evaluator:
    role: evaluator
    class: src.experiments.evaluation.AudioPairEvaluator
    kwargs:
      source: validation
      every: {unit: epoch, interval: 1}
    inject:
      model: {ref: model}
      data: {ref: data}
      metrics: [{ref: sisdr}]
"""
        inject = "      evaluators: {ref: evaluator}\n" + inject
    block = f"""{metric}  selection:
    role: other
    class: src.experiments.selection.BestMetricSelection
    kwargs:
      metric: validation.loss
      mode: {mode}
    state:
      save: true
"""
    return _with(text, block, inject)


def _metric_text(directory: Path, *, interval: int) -> str:
    text = _scripted_text(directory, steps=2, updates=1)
    block = f"""  sisdr:
    role: metric
    class: src.experiments.metrics.SISDRMetric
    kwargs:
      name: si_sdr
  evaluator:
    role: evaluator
    class: src.experiments.evaluation.AudioPairEvaluator
    kwargs:
      source: validation
      every: {{unit: epoch, interval: {interval}}}
    inject:
      model: {{ref: model}}
      data: {{ref: data}}
      metrics: [{{ref: sisdr}}]
  selection:
    role: other
    class: src.experiments.selection.BestMetricSelection
    kwargs:
      metric: validation.si_sdr
      mode: max
    state:
      save: true
"""
    return _with(text, block, "      evaluators: {ref: evaluator}\n      model_selection: {ref: selection}\n")


def _wer_text(directory: Path) -> str:
    text = _scripted_text(directory, steps=1, updates=1)
    text = text.replace(
        "class: tests.experiments.tf_gridnet_data.ScriptedChunkData",
        "class: tests.experiments.research_data.TranscriptChunkData",
        1,
    )
    block = """  recognizer:
    role: other
    class: tests.experiments.research_data.FixedRecognizer
    kwargs:
      hypothesis: alpha bravo
  wer:
    role: metric
    class: src.experiments.metrics.WerMetric
    kwargs:
      name: wer
    inject:
      recognizer: {ref: recognizer}
  train_loss:
    role: metric
    class: src.experiments.metrics.LossMetric
    kwargs:
      name: loss
  train_eval:
    role: evaluator
    class: src.experiments.evaluation.AudioPairEvaluator
    kwargs:
      source: train_eval
      every: {unit: epoch, interval: 1}
    inject:
      model: {ref: model}
      data: {ref: data}
      objective: {ref: objective}
      metrics: [{ref: train_loss}]
  validation_eval:
    role: evaluator
    class: src.experiments.evaluation.AudioPairEvaluator
    kwargs:
      source: validation
      every: {unit: epoch, interval: 1}
    inject:
      model: {ref: model}
      data: {ref: data}
      metrics: [{ref: wer}]
  selection:
    role: other
    class: src.experiments.selection.BestMetricSelection
    kwargs:
      metric: validation.wer
      mode: min
    state:
      save: true
"""
    return _with(
        text,
        block,
        "      evaluators: [{ref: train_eval}, {ref: validation_eval}]\n      model_selection: {ref: selection}\n",
    )


def _with(text: str, block: str, inject: str) -> str:
    text = text.replace("  trainer:\n", block + "  trainer:\n", 1)
    return text.replace("      scheduler: {ref: scheduler}\n", "      scheduler: {ref: scheduler}\n" + inject, 1)


def _same(left: object, right: object) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(item, other) for item, other in zip(left, right))
    return left == right


def _choice(directory: Path) -> dict[str, object]:
    return json.loads((directory / "research_selection" / "best.json").read_text(encoding="utf-8"))


def _log(directory: Path) -> list[dict[str, object]]:
    path = directory / "research_metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
