"""ASR catalog and waveform checks. These tests do not download weights."""

import unittest

import torch

from src.asr import ASRResult, SpeechRecognizer, available_families, load_recognizer, model_catalog
from src.asr.catalog import resolve_model
from src.asr.core.audio import prepare_waveform


class CatalogTests(unittest.TestCase):
    def test_families_and_decode_follow_the_checkpoint(self):
        self.assertEqual(available_families(), ("parakeet", "whisper", "wav2vec2", "hubert", "wavlm"))
        tdt = resolve_model("parakeet", "nvidia/parakeet-tdt-0.6b-v2")
        ctc = resolve_model("parakeet", "nvidia/parakeet-ctc-0.6b")
        rnnt = resolve_model("parakeet", "nvidia/parakeet-rnnt-1.1b")
        self.assertEqual((tdt.decode, tdt.backend), ("tdt", "nemo"))
        self.assertEqual((ctc.decode, ctc.backend), ("ctc", "nemo"))
        self.assertEqual((rnnt.decode, rnnt.backend), ("rnnt", "nemo"))
        self.assertEqual(resolve_model("whisper", "openai/whisper-tiny").model_id, "tiny")
        self.assertEqual(resolve_model("wav2vec2", "facebook/wav2vec2-base-960h").decode, "ctc")
        self.assertEqual(resolve_model("hubert", "facebook/hubert-large-ls960-ft").backend, "transformers")
        self.assertTrue(resolve_model("wavlm", "patrickvonplaten/wavlm-libri-clean-100h-base-plus").asr_finetuned)
        ids = {card.model_id for card in model_catalog()}
        self.assertIn("nvidia/parakeet-tdt-0.6b-v2", ids)
        self.assertIn("large-v3", ids)

    def test_rejects_unknown_ssl_and_mixed_decoders(self):
        with self.assertRaises(ValueError):
            resolve_model("quartznet", "nvidia/quartznet")
        with self.assertRaises(ValueError):
            resolve_model("hubert", "facebook/hubert-large-ls960")
        with self.assertRaises(ValueError):
            resolve_model("wavlm", "microsoft/wavlm-base-plus")
        with self.assertRaises(ValueError):
            resolve_model("wav2vec2", "facebook/wav2vec2-base")
        with self.assertRaises(ValueError):
            resolve_model("parakeet", "nvidia/parakeet-tdt_ctc-110m")
        with self.assertRaises(ValueError):
            resolve_model("parakeet", "nvidia/parakeet-tdt-0.6b-v2", decode="ctc")
        with self.assertRaises(ValueError):
            resolve_model("whisper", "tiny.en", language="ja")
        with self.assertRaises(ValueError):
            resolve_model("hubert", "facebook/wav2vec2-base-960h")
        with self.assertRaises(ValueError):
            load_recognizer("parakeet", model_id="nvidia/parakeet-tdt-0.6b-v2", device="")

    def test_dynamic_parakeet_id_keeps_one_decoder(self):
        card = resolve_model("parakeet", "nvidia/parakeet-ctc-0.6b-custom")
        self.assertEqual(card.decode, "ctc")
        self.assertEqual(card.backend, "nemo")


class RecognizerTests(unittest.TestCase):
    def test_transcribe_checks_shape_and_returns_raw_text(self):
        card = resolve_model("whisper", "tiny")
        seen = {}

        def run(audio: torch.Tensor) -> str:
            seen["shape"] = tuple(audio.shape)
            seen["dtype"] = audio.dtype
            seen["rate_samples"] = audio.numel()
            return "  Hello, World  "

        recognizer = SpeechRecognizer(card, device="cpu", language="en", runner=run, sample_rate=16000)
        result = recognizer.transcribe(torch.zeros(8000), sample_rate=8000)
        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.text, "  Hello, World  ")
        self.assertEqual(result.model, "whisper")
        self.assertEqual(result.model_id, "tiny")
        self.assertEqual(result.decode, "whisper")
        self.assertEqual(seen["dtype"], torch.float32)
        self.assertEqual(seen["shape"][0], 16000)
        self.assertEqual(recognizer.settings()["backend"], "whisper")
        wave = torch.zeros(1600)
        with self.assertRaises(ValueError):
            recognizer.transcribe(wave.reshape(1, -1), 16000)
        with self.assertRaises(ValueError):
            recognizer.transcribe(torch.zeros(16, dtype=torch.int16), 16000)
        with self.assertRaises(ValueError):
            recognizer.transcribe(torch.tensor([float("nan")]), 16000)
        with self.assertRaises(ValueError):
            recognizer.transcribe(torch.zeros(16), 0)
        with self.assertRaises(ValueError):
            prepare_waveform(torch.zeros(0), 16000)


if __name__ == "__main__":
    unittest.main()
