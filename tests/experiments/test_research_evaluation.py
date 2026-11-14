"""Schedule, side effects, and YAML construction of the research evaluator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.experiments.api import build_experiment
from src.experiments.config import config_sha256, load_experiment
from src.experiments.data.sample import Batch
from src.experiments.errors import BuildError
from src.experiments.evaluation.evaluator import AudioPairEvaluator
from src.experiments.evaluation.schedule import EvaluationSchedule
from src.experiments.metrics.loss import LossMetric
from src.experiments.metrics.signal import SISDRMetric
from src.experiments.metrics.wer import WerMetric
from src.experiments.runtime import PreparePolicy, PreparedModel, ProcessRuntime
from src.experiments.training.objectives import ObjectiveOutput
from tests.experiments.research_data import FixedRecognizer
from tests.experiments.test_tf_gridnet_training import _scripted_text


class ScheduleTests(unittest.TestCase):
    def test_units(self) -> None:
        epoch = EvaluationSchedule({"unit": "epoch", "interval": 2})
        self.assertTrue(epoch.due(logical_step=1, epoch=0, kind="boundary"))
        self.assertFalse(epoch.due(logical_step=2, epoch=1, kind="boundary"))
        self.assertFalse(epoch.due(logical_step=2, epoch=2, kind="step"))
        step = EvaluationSchedule({"unit": "logical_step", "interval": 2})
        self.assertFalse(step.due(logical_step=0, epoch=0, kind="step"))
        self.assertTrue(step.due(logical_step=2, epoch=0, kind="step"))
        self.assertTrue(EvaluationSchedule(None).due(logical_step=3, epoch=1, kind="boundary"))
        self.assertTrue(EvaluationSchedule({"unit": "end"}).due(logical_step=3, epoch=1, kind="end"))
        with self.assertRaises(BuildError):
            EvaluationSchedule({"unit": "boundary", "interval": 2})


class _CursorData:
    def __init__(self) -> None:
        self.epoch = 7
        self.update_index = 3
        self.calls: list[str] = []

    def validation_epoch(self, epoch: int):
        self.calls.append(f"validation:{epoch}")
        torch.randn(4)
        yield _batch()

    def train_eval_epoch(self, epoch: int):
        self.calls.append(f"train_eval:{epoch}")
        yield _batch()

    def reference_transcript(self, sample_id: str) -> str:
        del sample_id
        return "alpha bravo charlie"


class _ConstantObjective(torch.nn.Module):
    def forward(self, context: object) -> ObjectiveOutput:
        del context
        total = torch.tensor(2.5)
        return ObjectiveOutput(total, (("constant", total, 1.0),))


class EvaluatorSideEffectTests(unittest.TestCase):
    def test_eval_restores_mode_rng_cursor_and_does_not_train(self) -> None:
        module = _Echo()
        module.train()
        prepared = PreparedModel("model", module, tuple(module.parameters()), module, PreparePolicy())
        data = _CursorData()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        before = module.scale.detach().clone()
        torch.manual_seed(5)
        torch.randn(1)
        rng = torch.get_rng_state().clone()
        evaluator = AudioPairEvaluator(
            source="validation",
            metrics=(LossMetric(), SISDRMetric()),
            model=prepared,
            data=data,
            every={"unit": "epoch", "interval": 1},
            objective=_ConstantObjective(),
        )
        runtime = ProcessRuntime(device="cpu", precision="fp32", timeout_seconds=30)
        results = evaluator.evaluate(runtime, logical_step=4, epoch=0)
        by_name = {item.name: item for item in results}
        self.assertEqual(by_name["loss"].status, "ok")
        self.assertAlmostEqual(by_name["loss"].value, 2.5)
        self.assertEqual(by_name["si_sdr"].unit, "dB")
        self.assertTrue(module.training)
        self.assertEqual(tuple(module.scale.shape), tuple(before.shape))
        self.assertEqual(float(module.scale.detach()), float(before))
        self.assertIsNone(module.scale.grad)
        self.assertEqual(optimizer.state, {})
        self.assertEqual((data.epoch, data.update_index), (7, 3))
        self.assertEqual(data.calls, ["validation:0"])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(evaluator.due(logical_step=4, epoch=0, kind="boundary"))

        train_eval = AudioPairEvaluator(
            source="train_eval",
            metrics=(SISDRMetric(name="monitor"),),
            model=prepared,
            data=data,
            every={"unit": "boundary"},
        )
        train_eval.evaluate(runtime, logical_step=4, epoch=1)
        self.assertEqual(data.calls[-1], "train_eval:1")
        self.assertEqual((data.epoch, data.update_index), (7, 3))

    def test_wer_uses_the_recognizer_and_is_not_zero_when_it_differs(self) -> None:
        module = _Echo()
        prepared = PreparedModel("model", module, tuple(module.parameters()), module, PreparePolicy())
        data = _CursorData()
        evaluator = AudioPairEvaluator(
            source="validation",
            metrics=(WerMetric(recognizer=FixedRecognizer("alpha bravo")),),
            model=prepared,
            data=data,
        )
        runtime = ProcessRuntime(device="cpu", precision="fp32", timeout_seconds=30)
        result = evaluator.evaluate(runtime, logical_step=1, epoch=0)[0]
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.value, 1.0 / 3.0)

        missing = AudioPairEvaluator(
            source="validation",
            metrics=(WerMetric(recognizer=FixedRecognizer("alpha bravo")),),
            model=prepared,
            data=_NoTranscript(),
        )
        incomplete = missing.evaluate(runtime, logical_step=1, epoch=0)[0]
        self.assertEqual(incomplete.status, "incomplete")
        self.assertIsNone(incomplete.value)


class YamlBuildTests(unittest.TestCase):
    def test_official_faithful_and_metric_state_are_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(BuildError) as caught:
                build_experiment(load_experiment(_yaml(root / "claim", claim=True)))
            self.assertIn("official_faithful", str(caught.exception))
            with self.assertRaises(BuildError) as caught:
                build_experiment(load_experiment(_yaml(root / "save", save_metric=True)))
            self.assertIn("does not save state", str(caught.exception))

    def test_adding_a_metric_changes_the_config_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = load_experiment(_yaml(root / "plain", research=False))
            added = load_experiment(_yaml(root / "added", research=True))
            self.assertNotEqual(config_sha256(plain), config_sha256(added))
            again = load_experiment(_yaml(root / "plain-again", research=False))
            self.assertEqual(config_sha256(plain), config_sha256(again))
        fixture = Path("tests/fixtures/configs/experiments/tf_gridnet_ms_snsd.yaml")
        first = config_sha256(load_experiment(fixture))
        second = config_sha256(load_experiment(fixture))
        self.assertEqual(first, second)
        self.assertNotIn("AudioPairEvaluator", fixture.read_text(encoding="utf-8"))


def _yaml(directory: Path, *, research: bool = True, claim: bool = False, save_metric: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = _scripted_text(directory, steps=1, updates=1)
    if research or claim or save_metric:
        save = "\n    state: {save: true}" if save_metric else ""
        text = text.replace(
            "  trainer:\n",
            f"""  sisdr:
    role: metric
    class: src.experiments.metrics.SISDRMetric
    kwargs:
      name: si_sdr{save}
  trainer:
""",
            1,
        )
    if claim:
        text = text.replace("      max_logical_steps:", "      experiment_claim: official_faithful\n      max_logical_steps:", 1)
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(text)
    return path


class _Echo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale


class _NoTranscript:
    epoch = 0
    update_index = 0

    def validation_epoch(self, epoch: int):
        del epoch
        yield _batch()


def _batch() -> Batch:
    reference = torch.linspace(-0.2, 0.3, 64).unsqueeze(0)
    estimate = torch.linspace(-0.1, 0.2, 64).unsqueeze(0)
    return Batch(
        inputs=estimate,
        targets=reference,
        sample_id="sample",
        realization_id="sample",
        lengths=torch.tensor([64]),
        sample_ids=("sample-0",),
    )


if __name__ == "__main__":
    unittest.main()
