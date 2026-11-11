"""Phase 6 checks the existing evaluation path with a fixed mock ASR.

Corpus WER is the micro aggregate. The mock ASR does not score audio quality.
One real MP-SENet checkpoint is loaded through src.se.load_enhancer. The other
model loaders are checked only as far as the evaluation adapter's call.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.datasets.synthetic import make_synthetic_samples
from src.evaluation.datasets.load import DatasetBundle
from src.evaluation.metrics.wer import corpus_wer
from src.evaluation.models.se import MockEnhancer, load_evaluation_enhancer
from src.evaluation.pipeline import execute
from src.evaluation.provenance import is_confirmed_independent, judge_independence
from src.evaluation.results.reader import read_jsonl
from src.se.api import build_model, registry
from tests.evaluation.support import experiment_config
from tests.suite_resources import apply_intra_threads


ROOT = Path(__file__).resolve().parents[2]


def _scripts() -> dict[str, dict[str, str]]:
    return {
        "synthetic-0000": {
            "clean": "alpha beta gamma",
            "noisy": "alpha beta",
            "enhanced": "alpha beta delta",
        },
        "synthetic-0001": {
            "clean": "one two",
            "noisy": "one two two",
            "enhanced": "two",
        },
    }


def _samples():
    samples = make_synthetic_samples(2, 16000, 0.4, 0)
    samples[0].transcript = "alpha beta gamma"
    samples[1].transcript = "one two"
    return samples


class CorpusWerPathTests(unittest.TestCase):
    def test_mock_asr_records_micro_wer_for_every_condition(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = self._run(root / "first", "first")
            second = self._run(root / "second", "second")
            first_groups = {group["condition"]: group for group in first["summary"]["groups"]}
            second_groups = {group["condition"]: group for group in second["summary"]["groups"]}
            for result, groups in ((first, first_groups), (second, second_groups)):
                self.assertEqual(set(groups), {"clean", "noisy", "enhanced"})
                self.assertEqual(groups["clean"]["edits_normalized"], {"S": 0, "D": 0, "I": 0, "N": 5, "empty_reference": 0})
                self.assertEqual(groups["noisy"]["edits_normalized"], {"S": 0, "D": 1, "I": 1, "N": 5, "empty_reference": 0})
                self.assertEqual(groups["enhanced"]["edits_normalized"], {"S": 1, "D": 1, "I": 0, "N": 5, "empty_reference": 0})
                self.assertEqual(groups["noisy"]["wer_normalized"], corpus_wer(groups["noisy"]["edits_normalized"]))
                self.assertAlmostEqual(groups["noisy"]["wer_normalized"], 2 / 5)
                self.assertNotAlmostEqual(groups["noisy"]["wer_normalized"], (1 / 3 + 1 / 2) / 2)
                record = result["records"][0]
                self.assertEqual(set(record["hypotheses"]), {"clean", "noisy", "enhanced"})
                self.assertEqual(record["wer"]["normalization"], "wer-norm-v1")
            for condition in ("clean", "noisy", "enhanced"):
                self.assertEqual(
                    first_groups[condition]["edits_normalized"],
                    second_groups[condition]["edits_normalized"],
                )
                self.assertEqual(
                    first_groups[condition]["wer_normalized"],
                    second_groups[condition]["wer_normalized"],
                )

    def test_resume_does_not_duplicate_wer_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = self._config(root, "resume")
            bundle = DatasetBundle(samples=_samples())
            first = execute(config, enhancer_factory=lambda model, device: MockEnhancer(), bundle=bundle)
            self.assertEqual(first["summary"]["num_ok"], 2)
            config.output.resume = True
            second = execute(config, enhancer_factory=lambda model, device: MockEnhancer(), bundle=bundle)
            self.assertEqual(second["reused"], 2)
            rows = read_jsonl(Path(second["run_dir"]) / "results.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["utterance_id"] for row in rows}), 2)

    def test_enhancer_failure_is_an_error_and_not_a_zero_wer(self) -> None:
        class Boom:
            def enhance(self, waveform, sample_rate):
                del waveform, sample_rate
                raise RuntimeError("enhancer failed")

        with tempfile.TemporaryDirectory() as raw:
            config = self._config(Path(raw), "boom")
            result = execute(
                config,
                enhancer_factory=lambda model, device: Boom(),
                bundle=DatasetBundle(samples=_samples()[:1]),
            )
            record = result["records"][0]
            self.assertEqual(record["status"], "error")
            self.assertIn("enhancer failed", record["error"]["message"])
            self.assertEqual(record["wer"]["status"], "error")
            self.assertEqual(result["summary"]["num_error"], 1)
            self.assertEqual(result["summary"]["num_ok"], 0)
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["summary"]["official_wer"])
            groups = {group["condition"]: group for group in result["summary"]["groups"]}
            self.assertEqual(set(groups), {"clean", "noisy", "enhanced"})
            self.assertEqual(groups["noisy"]["target_utterance_ids"], groups["enhanced"]["target_utterance_ids"])
            for group in groups.values():
                self.assertFalse(group["wer_official"])
                self.assertIsNone(group["wer_normalized"])
                self.assertNotEqual(group["wer_normalized"], 0)

    def test_one_real_enhancer_reaches_mock_asr(self) -> None:
        apply_intra_threads()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            weights = root / "generator.pt"
            spec = {"dense_channel": 16, "num_tsconformers": 1}
            module = build_model("mp_senet", spec)
            torch.save({"generator": module.state_dict()}, weights)
            config = self._config(root, "real")
            config.metrics.pesq = False
            config.metrics.stoi = False
            config.models[0].name = "mp_senet"
            config.models[0].variant = "vb"
            config.models[0].checkpoint = str(weights)
            config.models[0].model_config = spec
            enhancer = load_evaluation_enhancer(config.models[0], "cpu")
            self.assertFalse(enhancer.model.training)
            result = execute(
                config,
                enhancer_factory=lambda model, device: enhancer,
                bundle=DatasetBundle(samples=_samples()),
            )
            self.assertEqual(result["summary"]["num_ok"], 2)
            self.assertEqual(result["summary"]["num_error"], 0)
            groups = {group["condition"]: group for group in result["summary"]["groups"]}
            self.assertAlmostEqual(groups["enhanced"]["wer_normalized"], 2 / 5)
            self.assertEqual(result["records"][0]["model"]["name"], "mp_senet")
            digest = result["records"][0]["model"]["checkpoint_sha256"]["checkpoint"]
            self.assertEqual(len(digest), 64)

    def test_other_loaders_are_reachable_from_the_evaluation_adapter(self) -> None:
        expected = {
            "tf_gridnet": "ms-snsd",
            "deepfilternet3": "deepfilternet3",
            "mossformergan": "se_16k",
            "semambapp": "semambapp",
            "tf_locoformer": "dns",
        }
        for name, variant in expected.items():
            seen: dict[str, object] = {}

            def loader(called, checkpoint, device="cpu", config=None, variant=None, _name=name):
                del checkpoint, config
                seen["name"] = called
                seen["variant"] = variant
                seen["device"] = device
                if called != _name:
                    raise AssertionError(called)
                return MockEnhancer()

            spec = registry.get(name)
            self.assertTrue(callable(spec.loader))
            from src.evaluation.config import ModelConfig

            model = ModelConfig(name=name, variant=variant, checkpoint="unused.pt")
            self.assertIn(variant, spec.variants)
            load_evaluation_enhancer(model, "cpu", loader=loader)
            self.assertEqual(seen, {"name": name, "variant": variant, "device": "cpu"})

    def test_incomplete_provenance_is_not_promoted(self) -> None:
        judged = judge_independence(
            {
                "schema_version": 1,
                "format": "audio-db-data-provenance-v1",
                "pipeline": "reproduction",
                "materials": [],
                "selection_uses_test": False,
                "profile": "leakage_safe",
            },
            [{"set_name": "eval", "sha256": "a" * 64, "source_sha256": "b" * 64, "lineage": "source"}],
        )
        self.assertNotEqual(judged["independent"], True)
        self.assertFalse(is_confirmed_independent(judged["independent"]))

    def _config(self, root: Path, run_id: str):
        config = experiment_config(root, run_id=run_id)
        config.metrics.pesq = False
        config.metrics.stoi = False
        config.asr.conditions = ["clean", "noisy", "enhanced"]
        config.asr.scripts = _scripts()
        config.asr.default_text = ""
        return config

    def _run(self, root: Path, run_id: str):
        root.mkdir(parents=True, exist_ok=True)
        return execute(
            self._config(root, run_id),
            enhancer_factory=lambda model, device: MockEnhancer(),
            bundle=DatasetBundle(samples=_samples()),
        )
