"""The evaluation adapter calls src.asr and still accepts the mock recognizer."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.asr import ASRResult
from src.evaluation.config import config_from_dict
from src.evaluation.errors import ConfigError
from src.evaluation.models.asr import load_recognizer
from src.evaluation.pipeline import execute
from tests.evaluation.support import experiment_config


def _enabled(name, model_id, **asr):
    raw = {
        "experiment": {"name": "m"},
        "dataset": {"source": "synthetic", "count": 1, "seconds": 0.1},
        "models": [{"name": "mock"}],
        "asr": {"enabled": True, "name": name, "model_id": model_id, **asr},
        "metrics": {"wer": True},
    }
    return config_from_dict(raw)


class AsrAdapterTests(unittest.TestCase):
    def test_mock_still_transcribes_without_a_model_id(self):
        with tempfile.TemporaryDirectory() as directory:
            config = experiment_config(Path(directory))
            recognizer = load_recognizer(config.asr)
            result = recognizer.transcribe(torch.zeros(160), 16000)
            self.assertIsInstance(result, ASRResult)
            self.assertEqual(result.text, "sample 0")
            with self.assertRaises(ConfigError):
                config_from_dict(
                    {
                        "experiment": {"name": "m"},
                        "dataset": {"source": "synthetic"},
                        "models": [{"name": "mock"}],
                        "asr": {"enabled": True, "name": "mock", "model_id": "tiny"},
                    }
                )

    def test_real_model_requires_a_catalog_id_and_uses_src_asr(self):
        config = _enabled("parakeet", "nvidia/parakeet-tdt-0.6b-v2", language="en")
        self.assertEqual(config.asr.decode, None)
        self.assertEqual(config.asr.model_id, "nvidia/parakeet-tdt-0.6b-v2")
        with self.assertRaises(ConfigError):
            _enabled("parakeet", "nvidia/parakeet-tdt-0.6b-v2", decode="ctc")
        with self.assertRaises(ConfigError):
            _enabled("hubert", "facebook/hubert-large-ls960")
        with self.assertRaises(ConfigError):
            _enabled("wavlm", "microsoft/wavlm-base")
        with self.assertRaises(ConfigError):
            _enabled("no-such-model", "tiny")
        captured = {}

        class Fake:
            def transcribe(self, waveform, sample_rate=16000):
                captured["samples"] = waveform.numel()
                return ASRResult(text="Hello there", model="parakeet", model_id="nvidia/parakeet-tdt-0.6b-v2", decode="tdt")

        def factory(name, *, model_id, device, language, decode):
            captured.update(name=name, model_id=model_id, device=device, language=language, decode=decode)
            return Fake()

        with patch("src.evaluation.models.asr.dependency_problem", return_value=None):
            with patch("src.evaluation.models.asr.load_asr", factory):
                recognizer = load_recognizer(config.asr)
                text = recognizer.transcribe(torch.zeros(1600), 16000).text
        self.assertEqual(text, "Hello there")
        self.assertEqual(captured["name"], "parakeet")
        self.assertEqual(captured["model_id"], "nvidia/parakeet-tdt-0.6b-v2")
        self.assertEqual(captured["language"], "en")

    def test_pipeline_stores_raw_text_from_the_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            config = experiment_config(Path(directory), run_id="asr-adapter")
            config.experiment.limit = 1

            class Fake:
                def transcribe(self, waveform, sample_rate=16000):
                    return ASRResult(
                        text="Raw, TEXT!",
                        model="parakeet",
                        model_id="nvidia/parakeet-tdt-0.6b-v2",
                        decode="tdt",
                    )

            result = execute(config, recognizer_factory=lambda _config: Fake())
            hypotheses = result["records"][0]["hypotheses"]["enhanced"]
            self.assertEqual(hypotheses["text"], "Raw, TEXT!")
            self.assertEqual(hypotheses["model_id"], "nvidia/parakeet-tdt-0.6b-v2")
            self.assertEqual(hypotheses["decode"], "tdt")
            self.assertEqual(result["records"][0]["wer"]["normalization"], "wer-norm-v1")
