"""WER completeness. Mock SE and mock ASR only; no real model is loaded.

Official WER is the micro aggregate of every target utterance. A missing
required condition withholds it. Provisional figures stay labeled as such.
"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from src.evaluation.cli import main
from src.evaluation.datasets.load import DatasetBundle
from src.evaluation.datasets.synthetic import make_synthetic_samples
from src.evaluation.errors import ConfigError
from src.evaluation.pipeline import execute
from src.evaluation.results.reader import read_json
from src.evaluation.results.summary import format_summary, summarize_directory
from src.evaluation.sample import AudioSample
from tests.evaluation.support import experiment_config


class _Enhancer:
    def __init__(self):
        self.calls = 0
        self.fail_below = None

    def enhance(self, waveform, sample_rate):
        del sample_rate
        self.calls += 1
        if self.fail_below is not None and float(waveform[0]) < self.fail_below:
            raise RuntimeError("enhancement failed")
        return waveform.clone()


class _Recognizer:
    def __init__(self, texts):
        self.texts = texts
        self.fail = set()
        self.calls = []
        self._utterance = None
        self._condition = None

    def bind(self, utterance_id, condition):
        self._utterance = utterance_id
        self._condition = condition

    def transcribe(self, waveform, sample_rate):
        del waveform, sample_rate
        self.calls.append((self._utterance, self._condition))
        if self._utterance in self.fail or (self._utterance, self._condition) in self.fail:
            raise RuntimeError(f"asr failed for {self._utterance} {self._condition}")
        text = self.texts[self._utterance][self._condition]
        return SimpleNamespace(text=text, model="mock", model_id=None, decode=None)


class WerCompletenessTests(unittest.TestCase):
    def test_asr_exception_withholds_official_wer_and_resume_matches_a_healthy_run(self):
        with self._temporary_dir() as directory:
            root = Path(directory)
            samples = self._samples()
            failed_id = samples[1].utterance_id
            recognizer = _Recognizer(self._texts())
            recognizer.fail.add((failed_id, "enhanced"))
            enhancer = _Enhancer()
            config = self._config(root / "broken", "broken")
            failed = self._run(config, samples, enhancer, recognizer)
            broken = {row["utterance_id"]: row for row in failed["records"]}
            self.assertEqual(broken[samples[0].utterance_id]["status"], "ok")
            self.assertEqual(broken[failed_id]["status"], "error")
            self.assertEqual(broken[failed_id]["evaluation"]["retryable"], True)
            self.assertIn("asr failed", broken[failed_id]["error"]["message"])
            self.assertEqual(broken[failed_id]["evaluation"]["reasons"], ["asr_failed"])
            self.assertIn("enhanced", broken[failed_id]["evaluation"]["missing_conditions"])
            self.assertEqual(failed["status"], "incomplete")
            self.assertEqual(failed["exit_code"], 1)
            self.assertNotEqual(read_json(Path(failed["run_dir"]) / "run.json")["status"], "completed")
            self.assertFalse(failed["summary"]["official_wer"])
            groups = {group["condition"]: group for group in failed["summary"]["groups"]}
            self.assertEqual(groups["noisy"]["target_utterance_ids"], groups["enhanced"]["target_utterance_ids"])
            self.assertEqual(groups["noisy"]["utterances"], 2)
            self.assertIsNone(groups["noisy"]["wer_normalized"])
            self.assertIsNone(groups["enhanced"]["wer_normalized"])
            self.assertFalse(groups["noisy"]["wer_official"])
            self.assertFalse(groups["enhanced"]["wer_official"])
            self.assertEqual(groups["noisy"]["wer_count"], 2)
            self.assertEqual(groups["enhanced"]["wer_count"], 1)
            text = format_summary(failed["summary"])
            self.assertIn("official_wer: false", text)
            self.assertIn("wer=provisional:", text)
            self.assertNotIn("wer=official:", text)

            kept = next(
                path
                for path in (Path(failed["run_dir"]) / "records").glob("*.json")
                if failed_id not in path.read_text(encoding="utf-8")
            )
            before = kept.read_bytes()
            recognizer.fail.clear()
            recognizer.calls.clear()
            enhancer.calls = 0
            config.output.resume = True
            resumed = self._run(config, samples, enhancer, recognizer)
            self.assertEqual(enhancer.calls, 1)
            self.assertEqual({utterance for utterance, _condition in recognizer.calls}, {failed_id})
            self.assertEqual(kept.read_bytes(), before)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["exit_code"], 0)
            self.assertTrue(resumed["summary"]["official_wer"])

            healthy = self._run(
                self._config(root / "healthy", "healthy"),
                samples,
                _Enhancer(),
                _Recognizer(self._texts()),
            )
            self.assertTrue(healthy["summary"]["official_wer"])
            resumed_groups = {group["condition"]: group for group in resumed["summary"]["groups"]}
            healthy_groups = {group["condition"]: group for group in healthy["summary"]["groups"]}
            for condition in ("noisy", "enhanced"):
                self.assertEqual(
                    resumed_groups[condition]["edits_normalized"],
                    healthy_groups[condition]["edits_normalized"],
                )
                self.assertEqual(
                    resumed_groups[condition]["wer_normalized"],
                    healthy_groups[condition]["wer_normalized"],
                )
                self.assertEqual(resumed_groups[condition]["wer_count"], 2)
                self.assertEqual(resumed_groups[condition]["utterances"], 2)
            self.assertEqual(healthy_groups["noisy"]["edits_normalized"]["N"], 5)
            self.assertEqual(healthy_groups["noisy"]["edits_normalized"]["D"], 0)
            self.assertEqual(healthy_groups["enhanced"]["edits_normalized"]["D"], 2)
            self.assertEqual(healthy_groups["enhanced"]["edits_normalized"]["N"], 5)
            self.assertEqual(healthy_groups["enhanced"]["wer_normalized"], 2 / 5)
            self.assertEqual(
                healthy_groups["enhanced"]["wer_normalized"],
                (
                    healthy_groups["enhanced"]["edits_normalized"]["S"]
                    + healthy_groups["enhanced"]["edits_normalized"]["D"]
                    + healthy_groups["enhanced"]["edits_normalized"]["I"]
                )
                / healthy_groups["enhanced"]["edits_normalized"]["N"],
            )

    def test_empty_hypothesis_is_a_scored_recognition(self):
        with self._temporary_dir() as directory:
            samples = self._samples()[:1]
            texts = {
                samples[0].utterance_id: {"noisy": "", "enhanced": ""},
            }
            result = self._run(
                self._config(Path(directory), "empty"),
                samples,
                _Enhancer(),
                _Recognizer(texts),
            )
            record = result["records"][0]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["hypotheses"]["enhanced"]["text"], "")
            self.assertIsNone(record["hypotheses"]["enhanced"]["error"])
            self.assertEqual(record["wer"]["values"]["enhanced"]["normalized"]["D"], 2)
            self.assertEqual(record["wer"]["values"]["enhanced"]["normalized"]["N"], 2)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["exit_code"], 0)
            group = {item["condition"]: item for item in result["summary"]["groups"]}["enhanced"]
            self.assertTrue(group["wer_official"])
            self.assertEqual(group["wer_normalized"], 1.0)
            self.assertEqual(group["edits_normalized"]["D"], 2)
            self.assertEqual(group["wer_count"], 1)

    def test_enhancement_failure_withholds_both_conditions(self):
        with self._temporary_dir() as directory:
            samples = self._samples()
            samples[1].clean_waveform[0] = -12345
            enhancer = _Enhancer()
            enhancer.fail_below = -1000
            result = self._run(
                self._config(Path(directory), "enhance"),
                samples,
                enhancer,
                _Recognizer(self._texts()),
            )
            by_id = {row["utterance_id"]: row for row in result["records"]}
            self.assertEqual(by_id[samples[1].utterance_id]["status"], "error")
            self.assertEqual(by_id[samples[1].utterance_id]["evaluation"]["reasons"], ["enhancement_failed"])
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["summary"]["official_wer"])
            groups = {group["condition"]: group for group in result["summary"]["groups"]}
            self.assertEqual(groups["noisy"]["target_utterance_ids"], groups["enhanced"]["target_utterance_ids"])
            self.assertEqual(groups["noisy"]["utterances"], 2)
            self.assertEqual(groups["enhanced"]["utterances"], 2)
            self.assertEqual(groups["noisy"]["wer_count"], 1)
            self.assertEqual(groups["enhanced"]["wer_count"], 1)
            self.assertIsNone(groups["noisy"]["wer_normalized"])
            self.assertIsNone(groups["enhanced"]["wer_normalized"])
            self.assertFalse(groups["noisy"]["wer_official"])
            self.assertFalse(groups["enhanced"]["wer_official"])
            text = format_summary(result["summary"])
            self.assertNotIn("wer=official:", text)

    def test_reference_and_input_defects_are_validation_failures(self):
        with self._temporary_dir() as directory:
            root = Path(directory)
            missing_text = self._samples()[:1]
            missing_text[0].transcript = "   "
            blank = self._run(
                self._config(root / "blank", "blank"),
                missing_text,
                _Enhancer(),
                _Recognizer({missing_text[0].utterance_id: {"noisy": "alpha beta", "enhanced": "alpha beta"}}),
            )
            self.assertEqual(blank["records"][0]["status"], "validation_failed")
            self.assertEqual(blank["records"][0]["wer"]["reason"], "no_reference_transcript")
            self.assertIn("reference transcript", blank["records"][0]["evaluation"]["message"])
            self.assertNotIn("normalized", blank["records"][0]["wer"]["values"]["enhanced"])
            self.assertEqual(blank["status"], "validation_failed")
            self.assertEqual(blank["exit_code"], 1)
            self.assertFalse(blank["summary"]["official_wer"])

            partial = self._samples()[:1]
            partial_config = self._config(root / "partial", "partial")
            partial_config.dataset.max_seconds = 0.05
            refused = self._run(
                partial_config,
                partial,
                _Enhancer(),
                _Recognizer(self._texts()),
            )
            self.assertTrue(refused["records"][0]["reference_audio"]["truncated"])
            self.assertEqual(refused["records"][0]["status"], "validation_failed")
            self.assertEqual(refused["records"][0]["wer"]["reason"], "partial_audio")
            self.assertIn("prefix", refused["records"][0]["evaluation"]["message"])
            self.assertNotIn("normalized", refused["records"][0]["wer"]["values"]["noisy"])
            self.assertIsNone(refused["summary"]["groups"][0]["wer_normalized"])

            broken = self._samples()[:1]
            broken[0].clean_waveform = torch.tensor([float("nan"), 0.1, -0.1])
            invalid = self._run(
                self._config(root / "nan", "nan"),
                broken,
                _Enhancer(),
                _Recognizer(self._texts()),
            )
            self.assertEqual(invalid["records"][0]["status"], "validation_failed")
            self.assertEqual(invalid["records"][0]["evaluation"]["reasons"], ["preparation_failed"])
            self.assertIn("non-finite", invalid["records"][0]["evaluation"]["message"])
            self.assertEqual(invalid["records"][0]["wer"]["status"], "validation_failed")
            self.assertFalse(invalid["summary"]["official_wer"])

            absent = AudioSample(
                utterance_id="gone",
                dataset="toy",
                split="test",
                sample_rate=16000,
                transcript="alpha beta",
                noisy_path=str(root / "missing.wav"),
            )
            with self.assertRaises(ConfigError) as caught:
                self._run(
                    self._config(root / "missing-file", "missing-file"),
                    [absent],
                    _Enhancer(),
                    _Recognizer({"gone": {"noisy": "alpha beta", "enhanced": "alpha beta"}}),
                )
            self.assertIn("missing", str(caught.exception))

    def test_auxiliary_na_does_not_block_official_wer(self):
        with self._temporary_dir() as directory:
            sample = self._samples()[0]
            sample.clean_waveform = None
            sample.noise_waveform = None
            sample.noisy_waveform = torch.randn(3200) * 0.05
            config = self._config(Path(directory), "noisy-only")
            config.metrics.pesq = True
            result = self._run(
                config,
                [sample],
                _Enhancer(),
                _Recognizer({sample.utterance_id: {"noisy": "alpha beta", "enhanced": "alpha beta"}}),
            )
            record = result["records"][0]
            self.assertEqual(record["pesq"]["status"], "na")
            self.assertEqual(record["pesq"]["reason"], "no_clean_reference")
            self.assertEqual(record["status"], "ok")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["summary"]["official_wer"])
            enhanced = {group["condition"]: group for group in result["summary"]["groups"]}["enhanced"]
            self.assertEqual(enhanced["wer_normalized"], 0.0)
            self.assertIsNone(enhanced["pesq_mean"])

    def test_summarize_incomplete_and_legacy_directories_are_not_official(self):
        with self._temporary_dir() as directory:
            root = Path(directory)
            samples = self._samples()
            recognizer = _Recognizer(self._texts())
            recognizer.fail.add(samples[1].utterance_id)
            failed = self._run(
                self._config(root / "again", "again"),
                samples,
                _Enhancer(),
                recognizer,
            )
            text = io.StringIO()
            with redirect_stdout(text):
                code = main(["--summarize", failed["run_dir"]])
            self.assertEqual(code, 0)
            printed = text.getvalue()
            self.assertIn("official_wer: false", printed)
            self.assertNotIn("wer=official:", printed)
            summarized = summarize_directory(Path(failed["run_dir"]))
            self.assertFalse(summarized["official_wer"])

            legacy = root / "legacy" / "old"
            self._write_legacy(legacy)
            loaded = summarize_directory(legacy)
            self.assertFalse(loaded["official_wer"])
            self.assertEqual(loaded["completeness"]["status"], "unknown")
            self.assertTrue(loaded["groups"])
            for group in loaded["groups"]:
                self.assertIsNone(group["wer_normalized"])
                self.assertFalse(group["wer_official"])
            self.assertEqual(loaded["groups"][0]["wer_provisional"], 0.5)
            legacy_text = format_summary(loaded)
            self.assertIn("official_wer: false", legacy_text)
            self.assertIn("wer=provisional:", legacy_text)
            self.assertNotIn("wer=official:", legacy_text)

            bare = root / "legacy" / "bare"
            (bare / "records").mkdir(parents=True)
            (bare / "records" / "only.json").write_text('{"status": "ok", "record_id": "only"}\n', encoding="utf-8")
            (bare / "run.json").write_text('{"schema_version": 1, "status": "completed"}\n', encoding="utf-8")
            (bare / "results.jsonl").write_text('{"status": "ok", "record_id": "only"}\n', encoding="utf-8")
            bare_summary = summarize_directory(bare)
            self.assertFalse(bare_summary["official_wer"])
            self.assertEqual(bare_summary["num_records"], 1)

    def test_resume_upgrades_a_legacy_success_without_rerunning_models(self):
        with self._temporary_dir() as directory:
            root = Path(directory)
            samples = self._samples()
            enhancer = _Enhancer()
            recognizer = _Recognizer(self._texts())
            config = self._config(root / "migrated", "migrated")
            first = self._run(config, samples, enhancer, recognizer)
            self.assertTrue(first["summary"]["official_wer"])
            first_groups = {group["condition"]: group for group in first["summary"]["groups"]}
            run_dir = Path(first["run_dir"])
            stripped = []
            for path in sorted((run_dir / "records").glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.pop("evaluation", None)
                payload["schema_version"] = 1
                text = json.dumps(payload)
                path.write_text(text, encoding="utf-8")
                stripped.append(text)
            (run_dir / "results.jsonl").write_text("\n".join(stripped) + "\n", encoding="utf-8")
            legacy = summarize_directory(run_dir)
            self.assertFalse(legacy["official_wer"])
            enhancer.calls = 0
            recognizer.calls.clear()
            config.output.resume = True
            resumed = self._run(config, samples, enhancer, recognizer)
            self.assertEqual(enhancer.calls, 0)
            self.assertEqual(recognizer.calls, [])
            self.assertEqual(resumed["status"], "completed")
            self.assertTrue(resumed["summary"]["official_wer"])
            resumed_groups = {group["condition"]: group for group in resumed["summary"]["groups"]}
            for condition in ("noisy", "enhanced"):
                self.assertEqual(
                    resumed_groups[condition]["edits_normalized"],
                    first_groups[condition]["edits_normalized"],
                )
                self.assertEqual(
                    resumed_groups[condition]["wer_normalized"],
                    first_groups[condition]["wer_normalized"],
                )
            stored = json.loads(next((run_dir / "records").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(stored["schema_version"], 2)
            self.assertTrue(stored["evaluation"]["required_wer"])

    def test_cli_returns_nonzero_when_asr_fails(self):
        with self._temporary_dir() as directory:
            root = Path(directory)
            config_path = root / "fail.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment: {name: completeness, seed: 0, limit: 1}",
                        "dataset: {source: synthetic, count: 1, sample_rate: 16000, seconds: 0.2, seed: 0}",
                        "mixing: {enabled: false}",
                        "models: [{name: mock}]",
                        "asr: {enabled: true, name: mock, default_text: hello, conditions: [noisy, enhanced]}",
                        "metrics: {wer: true, pesq: false, stoi: false}",
                        f"output: {{root: {json.dumps(str(root / 'runs'))}, run_id: cli-fail}}",
                        "runtime: {device: cpu}",
                    ]
                ),
                encoding="utf-8",
            )

            class Boom:
                def bind(self, utterance_id, condition):
                    del utterance_id, condition

                def transcribe(self, waveform, sample_rate):
                    del waveform, sample_rate
                    raise RuntimeError("asr failed")

            from unittest.mock import patch

            stdout = io.StringIO()
            with patch("src.evaluation.pipeline.load_recognizer", return_value=Boom()):
                with redirect_stdout(stdout):
                    code = main(
                        ["--config", str(config_path), "--device", "cpu", "--limit", "1"]
                    )
            printed = stdout.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("official_wer: false", printed)
            self.assertIn("status: incomplete", printed)
            self.assertNotIn("wer=official:", printed)

    def _samples(self):
        samples = make_synthetic_samples(2, 16000, 0.25, 0)
        samples[0].transcript = "alpha beta"
        samples[1].transcript = "one two three"
        return samples

    def _texts(self):
        return {
            "synthetic-0000": {"noisy": "alpha beta", "enhanced": "alpha"},
            "synthetic-0001": {"noisy": "one two three", "enhanced": "one two"},
        }

    def _config(self, root, run_id):
        config = experiment_config(root, run_id=run_id)
        config.experiment.limit = None
        config.mixing.enabled = False
        config.metrics.pesq = False
        config.metrics.stoi = False
        config.asr.default_text = ""
        config.asr.conditions = ["noisy", "enhanced"]
        return config

    def _run(self, config, samples, enhancer, recognizer):
        return execute(
            config,
            enhancer_factory=lambda model, device: enhancer,
            recognizer_factory=lambda _asr: recognizer,
            bundle=DatasetBundle(samples=samples),
        )

    def _write_legacy(self, run_dir: Path) -> None:
        counts = {"S": 1, "D": 0, "I": 0, "N": 2, "wer": 0.5}
        record = {
            "schema_version": 1,
            "record_id": "legacy-record",
            "status": "ok",
            "utterance_id": "legacy",
            "identity": {"config_hash": "legacy"},
            "model": {"name": "mock", "variant": None},
            "snr_db": None,
            "wer": {
                "status": "ok",
                "values": {
                    "noisy": {"status": "ok", "raw": counts, "normalized": counts},
                    "enhanced": {"status": "ok", "raw": counts, "normalized": counts},
                },
            },
            "pesq": {"values": {}},
            "stoi": {"values": {}},
            "estoi": {"values": {}},
            "si_sdr": {"values": {}},
            "timing_sec": {"enhance": 0.0},
        }
        (run_dir / "records").mkdir(parents=True)
        (run_dir / "records" / "legacy-record.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )
        (run_dir / "results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        (run_dir / "run.json").write_text(
            json.dumps({"schema_version": 1, "status": "completed", "config_hash": "legacy"}),
            encoding="utf-8",
        )

    def _temporary_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
