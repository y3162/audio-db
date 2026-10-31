"""End-to-end pipeline behavior on synthetic audio and a mock enhancer."""

import unittest
from pathlib import Path

import torch
import torchaudio

from src.evaluation.datasets.load import DatasetBundle
from src.evaluation.datasets.synthetic import make_synthetic_samples
from src.evaluation.errors import ConfigError
from src.evaluation.identity import checkpoint_fingerprint
from src.evaluation.pipeline import execute
from src.evaluation.results.reader import read_jsonl
from tests.evaluation.support import experiment_config


class CountingEnhancer:
    def __init__(self, fail_below=None):
        self.calls = 0
        self.fail_below = fail_below

    def enhance(self, waveform, sample_rate):
        self.calls += 1
        if self.fail_below is not None and float(waveform[0]) < self.fail_below:
            raise RuntimeError("boom")
        return waveform.clone()


class RngEnhancer:
    def enhance(self, waveform, sample_rate):
        return (waveform + torch.randn_like(waveform) * 0.01).contiguous()


class ShortEnhancer:
    def enhance(self, waveform, sample_rate):
        return waveform[: waveform.numel() // 2].clone()


class PipelineTests(unittest.TestCase):
    def test_end_to_end_micro_wer_and_reference_metrics(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="e2e")
            samples = make_synthetic_samples(2, 16000, 1.0, 0)
            samples[0].transcript = "a"
            samples[1].transcript = " ".join(["w"] * 100)
            config.asr.scripts = {
                "synthetic-0000": {"enhanced": "a", "noisy": "b"},
                "synthetic-0001": {
                    "enhanced": samples[1].transcript,
                    "noisy": samples[1].transcript,
                },
            }
            calls = {"n": 0}

            def factory(model, device):
                calls["n"] += 1
                self.assertEqual(device, "cpu")
                return CountingEnhancer()

            result = execute(
                config,
                enhancer_factory=factory,
                bundle=DatasetBundle(samples=samples),
            )
            self.assertEqual(calls["n"], 1)
            self.assertEqual(result["summary"]["num_ok"], 2)
            self.assertEqual(result["summary"]["num_error"], 0)
            groups = {group["condition"]: group for group in result["summary"]["groups"]}
            self.assertAlmostEqual(groups["noisy"]["wer_normalized"], 1 / 101)
            self.assertAlmostEqual(groups["enhanced"]["wer_normalized"], 0.0)
            self.assertNotAlmostEqual(groups["noisy"]["wer_normalized"], 0.5)
            self.assertGreater(groups["enhanced"]["pesq_count"], 0)
            self.assertIsNotNone(groups["enhanced"]["pesq_mean"])
            self.assertIsNotNone(groups["enhanced"]["stoi_mean"])
            record = result["records"][0]
            self.assertEqual(record["mix"]["protocol"], "rms_v1")
            self.assertAlmostEqual(record["mix"]["snr_db_achieved"], 0.0, places=2)
            self.assertIsNotNone(record["git_commit"])
            self.assertEqual(len(record["model"]["checkpoint_sha256"]["mock"]), 64)
            reloaded = read_jsonl(Path(result["run_dir"]) / "results.jsonl")
            self.assertEqual(len(reloaded), 2)

    def test_dry_run_does_not_infer_or_create_a_directory(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="dry")
            calls = {"n": 0}

            def factory(model, device):
                calls["n"] += 1
                return CountingEnhancer()

            result = execute(config, dry_run=True, enhancer_factory=factory)
            self.assertTrue(result["plan"]["ready"])
            self.assertEqual(result["plan"]["planned_records"], 2)
            self.assertEqual(calls["n"], 0)
            self.assertFalse((Path(directory) / "dry").exists())

    def test_resume_skips_completed_work_and_retries_failures(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="resume")
            config.mixing.enabled = False
            config.asr.enabled = False
            config.metrics.wer = False
            config.metrics.pesq = False
            config.metrics.stoi = False
            samples = make_synthetic_samples(2, 16000, 0.5, 0)
            samples[0].clean_waveform[0] = -12345
            bundle = DatasetBundle(samples=samples)
            box = {"enhancer": CountingEnhancer(fail_below=-1000)}

            def factory(model, device):
                return box["enhancer"]

            first = execute(config, enhancer_factory=factory, bundle=bundle)
            self.assertEqual(first["summary"]["num_error"], 1)
            self.assertEqual(first["summary"]["num_ok"], 1)
            self.assertEqual(box["enhancer"].calls, 2)
            ok_path = next(
                path
                for path in (Path(first["run_dir"]) / "records").glob("*.json")
                if '"status": "ok"' in path.read_text(encoding="utf-8")
            )
            before = ok_path.read_bytes()
            config.output.resume = True
            box["enhancer"] = CountingEnhancer()
            second = execute(config, enhancer_factory=factory, bundle=bundle)
            self.assertEqual(box["enhancer"].calls, 1)
            self.assertEqual(ok_path.read_bytes(), before)
            self.assertEqual(second["summary"]["num_ok"], 2)
            self.assertEqual(second["reused"], 1)

    def test_changed_config_is_not_resumed(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="seed")
            config.metrics.pesq = False
            config.metrics.stoi = False
            config.asr.enabled = False
            config.metrics.wer = False
            config.mixing.enabled = False
            execute(config, enhancer_factory=lambda model, device: CountingEnhancer())
            config.output.resume = True
            config.experiment.seed = 1
            with self.assertRaises(ConfigError):
                execute(config, enhancer_factory=lambda model, device: CountingEnhancer())

    def test_existing_run_without_resume_is_rejected(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="once")
            config.metrics.pesq = False
            config.metrics.stoi = False
            config.asr.enabled = False
            config.metrics.wer = False
            config.mixing.enabled = False
            execute(config, enhancer_factory=lambda model, device: CountingEnhancer())
            with self.assertRaises(ConfigError):
                execute(config, enhancer_factory=lambda model, device: CountingEnhancer())

    def test_checkpoint_bytes_invalidate_the_cache(self):
        with self._temporary_dir() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            weights.write_bytes(b"aaa")
            config = experiment_config(directory, run_id="ckpt")
            config.models[0].name = "mp_senet"
            config.models[0].variant = "vb"
            config.models[0].checkpoint = str(weights)
            config.mixing.enabled = False
            config.asr.enabled = False
            config.metrics.wer = False
            config.metrics.pesq = False
            config.metrics.stoi = False
            calls = {"n": 0}

            def factory(model, device):
                calls["n"] += 1
                return CountingEnhancer()

            first = execute(config, enhancer_factory=factory)
            self.assertEqual(calls["n"], 1)
            first_sha = first["records"][0]["model"]["checkpoint_sha256"]["checkpoint"]
            weights.write_bytes(b"bbb")
            config.output.resume = True
            second = execute(config, enhancer_factory=factory)
            self.assertEqual(calls["n"], 2)
            second_sha = second["records"][0]["model"]["checkpoint_sha256"]["checkpoint"]
            self.assertNotEqual(first_sha, second_sha)
            self.assertEqual(second["reused"], 0)
            stored = "\n".join(path.read_text(encoding="utf-8") for path in (root / "ckpt").rglob("*.jsonl"))
            self.assertIn(second_sha, stored)
            self.assertEqual(checkpoint_fingerprint(config.models[0])["checkpoint"], second_sha)

    def test_utterance_order_does_not_change_the_enhanced_waveform(self):
        with self._temporary_dir() as directory:
            samples = make_synthetic_samples(2, 16000, 0.5, 4)
            forward = self._rng_run(directory, "forward", samples)
            backward = self._rng_run(directory, "backward", list(reversed(samples)))
            self.assertEqual(set(forward), set(backward))
            for name, left in forward.items():
                right, rate = torchaudio.load(backward[name])
                left_wave, left_rate = torchaudio.load(left)
                self.assertEqual(left_rate, rate)
                self.assertTrue(torch.allclose(left_wave, right, atol=1e-6))

    def test_edges_skip_or_record_errors_without_zero_scores(self):
        with self._temporary_dir() as directory:
            silent = make_synthetic_samples(1, 16000, 0.5, 0)[0]
            silent.clean_waveform = torch.zeros(8000)
            silent.utterance_id = "silent"
            config = self._bare(directory, "silent")
            result = execute(
                config,
                enhancer_factory=lambda model, device: CountingEnhancer(),
                bundle=DatasetBundle(samples=[silent]),
            )
            self.assertEqual(result["records"][0]["status"], "skipped")
            self.assertEqual(result["records"][0]["skip_reason"], "clean_has_zero_energy")

            noisy_only = make_synthetic_samples(1, 16000, 1.0, 1)[0]
            noisy_only.clean_waveform = None
            noisy_only.noise_waveform = None
            noisy_only.noisy_waveform = torch.randn(16000) * 0.05
            noisy_only.transcript = None
            noisy_only.utterance_id = "noisy-only"
            config = experiment_config(directory, run_id="noclean")
            config.mixing.enabled = False
            result = execute(
                config,
                enhancer_factory=lambda model, device: CountingEnhancer(),
                bundle=DatasetBundle(samples=[noisy_only]),
            )
            record = result["records"][0]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["pesq"]["status"], "na")
            self.assertIsNone(record["pesq"]["values"]["enhanced"]["value"])
            self.assertEqual(record["wer"]["status"], "na")
            self.assertEqual(record["wer"]["reason"], "no_reference_transcript")

            broken = make_synthetic_samples(1, 16000, 0.5, 2)[0]
            broken.clean_waveform = torch.tensor([float("nan")] * 8000)
            broken.noise_waveform = None
            config = self._bare(directory, "nan")
            config.mixing.enabled = False
            result = execute(
                config,
                enhancer_factory=lambda model, device: CountingEnhancer(),
                bundle=DatasetBundle(samples=[broken]),
            )
            self.assertEqual(result["records"][0]["status"], "error")
            self.assertIn("non-finite", result["records"][0]["error"]["message"])

            short = make_synthetic_samples(2, 16000, 0.5, 3)
            config = self._bare(directory, "short")
            config.mixing.enabled = False
            result = execute(
                config,
                enhancer_factory=lambda model, device: ShortEnhancer(),
                bundle=DatasetBundle(samples=short),
            )
            self.assertTrue(all(record["status"] == "error" for record in result["records"]))
            self.assertTrue(all(record["pesq"]["status"] == "na" for record in result["records"]))
            self.assertNotIn(0, [record["pesq"].get("value") for record in result["records"]])

    def test_missing_checkpoint_is_not_replaced(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="missing")
            config.models[0].name = "mp_senet"
            config.models[0].variant = "vb"
            config.models[0].checkpoint = str(Path(directory) / "absent.pt")
            calls = {"n": 0}

            def factory(model, device):
                calls["n"] += 1
                return CountingEnhancer()

            with self.assertRaises(ConfigError):
                execute(config, enhancer_factory=factory)
            self.assertEqual(calls["n"], 0)

    def test_two_models_load_once_each(self):
        with self._temporary_dir() as directory:
            config = experiment_config(directory, run_id="two")
            config.models = []
            from src.evaluation.config import ModelConfig

            config.models = [
                ModelConfig(name="mock", variant="a"),
                ModelConfig(name="mock", variant="b"),
            ]
            config.mixing.enabled = False
            config.asr.enabled = False
            config.metrics.wer = False
            config.metrics.pesq = False
            config.metrics.stoi = False
            calls = {"n": 0}

            def factory(model, device):
                calls["n"] += 1
                return CountingEnhancer()

            result = execute(config, enhancer_factory=factory)
            self.assertEqual(calls["n"], 2)
            self.assertEqual(result["summary"]["num_ok"], 4)

    def _rng_run(self, directory, run_id, samples):
        config = self._bare(directory, run_id)
        config.mixing.enabled = False
        config.output.save_audio = True
        result = execute(
            config,
            enhancer_factory=lambda model, device: RngEnhancer(),
            bundle=DatasetBundle(samples=samples),
        )
        saved = {}
        for record in result["records"]:
            saved[record["utterance_id"]] = str(Path(result["run_dir"]) / record["saved_audio"]["enhanced"])
        return saved

    def _bare(self, directory, run_id):
        config = experiment_config(directory, run_id=run_id)
        config.asr.enabled = False
        config.metrics.wer = False
        config.metrics.pesq = False
        config.metrics.stoi = False
        return config

    def _temporary_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()
