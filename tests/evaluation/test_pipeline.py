"""End-to-end pipeline behavior on synthetic audio and a mock enhancer."""

import unittest
from pathlib import Path

import torch
import torchaudio

from src.evaluation.audio.io import audio_span
from src.evaluation.datasets.load import DatasetBundle
from src.evaluation.datasets.synthetic import make_synthetic_samples
from src.evaluation.sample import AudioSample
from src.evaluation.errors import ConfigError
from src.evaluation.identity import checkpoint_fingerprint
from src.evaluation import pipeline
from src.evaluation.pipeline import execute
from src.evaluation.results.reader import read_json, read_jsonl
from src.evaluation.results.summary import summarize_directory
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


class StoppingEnhancer:
    """Enhances until the given call, then stops the process the way Ctrl-C does."""

    def __init__(self, stop_at):
        self.calls = 0
        self.stop_at = stop_at

    def enhance(self, waveform, sample_rate):
        self.calls += 1
        if self.calls == self.stop_at:
            raise KeyboardInterrupt
        return waveform.clone()


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

    def test_wer_uses_source_length_not_the_presence_of_max_seconds(self):
        with self._temporary_dir() as directory:
            root = Path(directory)
            short = root / "short.wav"
            long = root / "long.wav"
            torchaudio.save(str(short), torch.zeros(1, 1600), 16000)
            torchaudio.save(str(long), torch.zeros(1, 16000), 16000)
            samples = [
                AudioSample(
                    utterance_id="short",
                    dataset="toy",
                    split="test",
                    sample_rate=16000,
                    transcript="sample 0",
                    noisy_path=str(short),
                ),
                AudioSample(
                    utterance_id="long",
                    dataset="toy",
                    split="test",
                    sample_rate=16000,
                    transcript="one two three four",
                    noisy_path=str(long),
                ),
                AudioSample(
                    utterance_id="blank",
                    dataset="toy",
                    split="test",
                    sample_rate=16000,
                    transcript=None,
                    noisy_path=str(short),
                ),
            ]
            config = experiment_config(directory, run_id="span")
            config.experiment.limit = None
            config.mixing.enabled = False
            config.dataset.max_seconds = 0.5
            config.metrics.pesq = False
            config.metrics.stoi = False
            result = execute(
                config,
                enhancer_factory=lambda model, device: CountingEnhancer(),
                bundle=DatasetBundle(samples=samples),
            )
            by_id = {record["utterance_id"]: record for record in result["records"]}
            self.assertEqual(by_id["short"]["status"], "ok")
            self.assertFalse(by_id["short"]["reference_audio"]["truncated"])
            self.assertEqual(by_id["short"]["wer"]["status"], "ok")
            self.assertEqual(by_id["short"]["wer"]["values"]["noisy"]["normalized"]["N"], 2)
            self.assertTrue(by_id["long"]["reference_audio"]["truncated"])
            self.assertEqual(by_id["long"]["wer"]["status"], "na")
            self.assertEqual(by_id["long"]["wer"]["reason"], "partial_audio")
            self.assertIsNotNone(by_id["long"]["hypotheses"]["noisy"]["text"])
            self.assertEqual(by_id["blank"]["wer"]["status"], "na")
            self.assertEqual(by_id["blank"]["wer"]["reason"], "no_reference_transcript")
            groups = {group["condition"]: group for group in result["summary"]["groups"]}
            self.assertEqual(groups["noisy"]["edits_normalized"]["N"], 2)
            self.assertEqual(groups["noisy"]["wer_normalized"], 0.0)
            self.assertEqual(groups["noisy"]["edits_normalized"]["S"], 0)

            full = experiment_config(directory, run_id="full-file")
            full.experiment.limit = 1
            full.mixing.enabled = False
            full.dataset.max_seconds = None
            full.metrics.pesq = False
            full.metrics.stoi = False
            samples[1].transcript = "sample 0"
            scored = execute(
                full,
                enhancer_factory=lambda model, device: CountingEnhancer(),
                bundle=DatasetBundle(samples=[samples[1]]),
            )
            record = scored["records"][0]
            self.assertFalse(record["reference_audio"]["truncated"])
            self.assertEqual(record["wer"]["status"], "ok")
            self.assertEqual(record["wer"]["values"]["enhanced"]["normalized"]["wer"], 0.0)
            self.assertEqual(record["reference_audio"]["source_frames"], 16000)
            self.assertEqual(record["reference_audio"]["used_frames"], 16000)

    def test_audio_span_compares_source_length_with_the_limit(self):
        short = audio_span(1600, 1600, 16000, 0.5)
        long = audio_span(16000, 8000, 16000, 0.5)
        unlimited = audio_span(16000, 16000, 16000, None)
        unknown = audio_span(None, 8000, 16000, 0.5)
        self.assertFalse(short["truncated"])
        self.assertTrue(long["truncated"])
        self.assertFalse(unlimited["truncated"])
        self.assertIsNone(unknown["truncated"])

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

    def test_joined_files_are_not_rewritten_for_every_record(self):
        with self._temporary_dir() as directory:
            calls = {"n": 0}
            publish = pipeline._publish

            def counted(run_dir, records):
                calls["n"] += 1
                return publish(run_dir, records)

            pipeline._publish = counted
            try:
                result = execute(
                    self._plain_config(directory, "throttled"),
                    enhancer_factory=lambda model, device: CountingEnhancer(),
                    bundle=DatasetBundle(samples=make_synthetic_samples(6, 16000, 0.25, 0)),
                )
                self.assertEqual(calls["n"], 1)
                self.assertEqual(len(read_jsonl(Path(result["run_dir"]) / "results.jsonl")), 6)
                self.assertEqual(result["summary"]["num_ok"], 6)
                calls["n"] = 0
                with self._publish_interval(0.0):
                    execute(
                        self._plain_config(directory, "every-record"),
                        enhancer_factory=lambda model, device: CountingEnhancer(),
                        bundle=DatasetBundle(samples=make_synthetic_samples(6, 16000, 0.25, 0)),
                    )
                self.assertEqual(calls["n"], 7)
            finally:
                pipeline._publish = publish

    def test_stopped_run_publishes_the_records_it_finished(self):
        with self._temporary_dir() as directory:
            box = {"enhancer": StoppingEnhancer(stop_at=3)}
            with self.assertRaises(KeyboardInterrupt):
                execute(
                    self._plain_config(directory, "stopped"),
                    enhancer_factory=lambda model, device: box["enhancer"],
                    bundle=DatasetBundle(samples=make_synthetic_samples(4, 16000, 0.25, 0)),
                )
            run_dir = Path(directory) / "stopped"
            self.assertEqual(len(list((run_dir / "records").glob("*.json"))), 2)
            self.assertEqual(len(read_jsonl(run_dir / "results.jsonl")), 2)
            self.assertEqual(read_json(run_dir / "summary.json")["num_records"], 2)
            self.assertEqual(read_json(run_dir / "run.json")["status"], "running")

    def test_stopped_resume_keeps_the_records_already_on_disk(self):
        with self._temporary_dir() as directory:
            samples = make_synthetic_samples(4, 16000, 0.25, 0)
            config = self._plain_config(directory, "resumed")
            first = execute(
                config,
                enhancer_factory=lambda model, device: CountingEnhancer(),
                bundle=DatasetBundle(samples=samples),
            )
            run_dir = Path(first["run_dir"])
            self.assertEqual(len(read_jsonl(run_dir / "results.jsonl")), 4)
            by_utterance = {record["utterance_id"]: record["record_id"] for record in first["records"]}
            for utterance in ("synthetic-0000", "synthetic-0002"):
                (run_dir / "records" / f"{by_utterance[utterance]}.json").unlink()
            config.output.resume = True
            box = {"enhancer": StoppingEnhancer(stop_at=2)}
            with self._publish_interval(0.0), self.assertRaises(KeyboardInterrupt):
                execute(config, enhancer_factory=lambda model, device: box["enhancer"], bundle=DatasetBundle(samples=samples))
            on_disk = {path.stem for path in (run_dir / "records").glob("*.json")}
            self.assertEqual(len(on_disk), 3)
            published = {record["record_id"] for record in read_jsonl(run_dir / "results.jsonl")}
            self.assertEqual(published, on_disk)
            self.assertEqual(read_json(run_dir / "summary.json")["num_records"], 3)
            # The earlier run finished, but this one did not.
            self.assertEqual(read_json(run_dir / "run.json")["status"], "running")
            # A hard kill leaves the joined file behind records/. The summary is rebuilt from records/.
            (run_dir / "results.jsonl").write_text("", encoding="utf-8")
            self.assertEqual(summarize_directory(run_dir)["num_records"], 3)
            box["enhancer"] = CountingEnhancer()
            final = execute(config, enhancer_factory=lambda model, device: box["enhancer"], bundle=DatasetBundle(samples=samples))
            self.assertEqual(box["enhancer"].calls, 1)
            self.assertEqual(final["reused"], 3)
            self.assertEqual(len(read_jsonl(run_dir / "results.jsonl")), 4)
            self.assertEqual(read_json(run_dir / "run.json")["status"], "completed")
            self.assertEqual(summarize_directory(run_dir)["num_records"], 4)

    def _plain_config(self, directory, run_id):
        config = experiment_config(directory, run_id=run_id)
        config.experiment.limit = None
        config.mixing.enabled = False
        config.asr.enabled = False
        config.metrics.wer = False
        config.metrics.pesq = False
        config.metrics.stoi = False
        return config

    def _publish_interval(self, seconds):
        from unittest import mock

        return mock.patch.object(pipeline, "_PUBLISH_INTERVAL_SECONDS", seconds)

    def _temporary_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()
