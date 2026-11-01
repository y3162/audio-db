"""Reference metrics treat a missing clean signal as unavailable, not zero."""

import unittest

import torch

from src.evaluation.metrics.pesq import score_pesq
from src.evaluation.metrics.sisdr import score_si_sdr
from src.evaluation.metrics.stoi import score_stoi
from src.evaluation.metrics.compute import score_utterance
from src.evaluation.config import MetricsConfig


class MetricTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(1)
        self.reference = torch.randn(16000, generator=generator) * 0.1
        self.estimate = self.reference + torch.randn(16000, generator=generator) * 0.01

    def test_pesq_and_stoi_score_a_pair(self):
        pesq_score = score_pesq(self.reference, self.estimate, 16000)
        stoi_score = score_stoi(self.reference, self.estimate, 16000)
        self.assertEqual(pesq_score["status"], "ok")
        self.assertGreater(pesq_score["value"], 1.0)
        self.assertEqual(pesq_score["mode"], "wb")
        self.assertEqual(stoi_score["status"], "ok")
        self.assertGreater(stoi_score["value"], 0.5)
        identical = score_stoi(self.reference, self.reference, 16000)
        self.assertAlmostEqual(identical["value"], 1.0, places=5)

    def test_identity_and_scaled_noise_metric_sanity(self):
        # A shared gain must not affect SI-SDR; independent noise must lower it.
        identical = score_pesq(self.reference, self.reference, 16000)
        self.assertEqual(identical["status"], "ok")
        self.assertGreater(identical["value"], 4.4)
        self.assertAlmostEqual(score_stoi(self.reference, self.reference, 16000)["value"], 1.0, places=5)
        base = score_si_sdr(self.reference, self.estimate)["value"]
        scaled = score_si_sdr(self.reference * 0.1, self.estimate * 0.1)["value"]
        self.assertGreater(base, 15.0)
        self.assertAlmostEqual(base, scaled, places=4)
        self.assertEqual(score_si_sdr(self.reference, self.reference)["reason"], "numerically_infinite")

    def test_missing_reference_is_not_zero(self):
        metrics = MetricsConfig(wer=True, pesq=True, stoi=True, si_sdr=True)
        scored = score_utterance(
            metrics,
            clean=None,
            estimates={"noisy": self.estimate, "enhanced": self.estimate},
            sample_rate=16000,
            reference=None,
            hypotheses={},
        )
        for name in ("pesq", "stoi", "si_sdr"):
            self.assertEqual(scored[name]["status"], "na")
            self.assertEqual(scored[name]["reason"], "no_clean_reference")
            self.assertIsNone(scored[name]["values"]["enhanced"]["value"])
            self.assertNotEqual(scored[name]["values"]["enhanced"]["value"], 0)
        self.assertEqual(scored["wer"]["status"], "na")
        self.assertEqual(scored["wer"]["reason"], "no_reference_transcript")

    def test_wer_requires_the_full_utterance_and_a_transcript(self):
        metrics = MetricsConfig(wer=True)
        hypotheses = {"enhanced": {"text": "hello"}, "noisy": {"text": "hello"}}
        full = score_utterance(
            metrics,
            clean=None,
            estimates={"enhanced": self.estimate},
            sample_rate=16000,
            reference="hello",
            hypotheses=hypotheses,
            reference_truncated=False,
        )
        self.assertEqual(full["wer"]["status"], "ok")
        self.assertEqual(full["wer"]["values"]["enhanced"]["normalized"]["N"], 1)
        self.assertEqual(full["wer"]["values"]["enhanced"]["normalized"]["wer"], 0.0)

        partial = score_utterance(
            metrics,
            clean=None,
            estimates={"enhanced": self.estimate},
            sample_rate=16000,
            reference="hello world",
            hypotheses=hypotheses,
            reference_truncated=True,
        )
        self.assertEqual(partial["wer"]["status"], "na")
        self.assertEqual(partial["wer"]["reason"], "partial_audio")
        self.assertNotIn("normalized", partial["wer"]["values"]["enhanced"])
        self.assertIsNone(partial["wer"]["values"]["enhanced"].get("wer"))

        missing = score_utterance(
            metrics,
            clean=None,
            estimates={"enhanced": self.estimate},
            sample_rate=16000,
            reference="   ",
            hypotheses=hypotheses,
            reference_truncated=False,
        )
        self.assertEqual(missing["wer"]["status"], "na")
        self.assertEqual(missing["wer"]["reason"], "no_reference_transcript")
        self.assertEqual(missing["wer"]["values"], {})

        unknown = score_utterance(
            metrics,
            clean=None,
            estimates={"enhanced": self.estimate},
            sample_rate=16000,
            reference="hello",
            hypotheses=hypotheses,
            reference_truncated=None,
        )
        self.assertEqual(unknown["wer"]["status"], "na")
        self.assertEqual(unknown["wer"]["reason"], "unknown_audio_span")

    def test_si_sdr_matches_the_definition(self):
        reference = torch.tensor([1.0, 0.0, -1.0, 0.0])
        estimate = torch.tensor([1.0, 0.5, -1.0, 0.0])
        scored = score_si_sdr(reference, estimate)
        self.assertEqual(scored["status"], "ok")
        self.assertAlmostEqual(scored["value"], 10.280287236002435, places=5)
        perfect = score_si_sdr(reference, reference * 2)
        self.assertIsNone(perfect["value"])
        self.assertEqual(perfect["reason"], "numerically_infinite")

    def test_invalid_and_short_audio_are_not_scores(self):
        silence = torch.zeros_like(self.reference)
        for score in (lambda a, b: score_pesq(a, b, 16000),
                      lambda a, b: score_stoi(a, b, 16000), score_si_sdr):
            self.assertIsNone(score(self.reference, silence)["value"])
            bad = self.reference.clone()
            bad[0] = float("nan")
            self.assertNotEqual(score(self.reference, bad)["status"], "ok")
        self.assertIsNone(score_stoi(self.reference[:2000], self.estimate[:2000], 16000)["value"])
        self.assertIsNone(score_pesq(self.reference[:100], self.estimate[:100], 16000)["value"])
        low_energy = self.reference * 1e-9
        self.assertEqual(score_si_sdr(low_energy, low_energy)["reason"], "reference_has_zero_energy")
        for result in (score_pesq(low_energy, low_energy, 16000), score_stoi(low_energy, low_energy, 16000)):
            self.assertIn(result['status'], ('ok', 'na', 'error'))
            if result['status'] != 'ok':
                self.assertIsNone(result['value'])
        orthogonal = score_si_sdr(torch.tensor([1., -1., 1., -1.]), torch.tensor([1., 1., -1., -1.]))
        self.assertEqual(orthogonal["reason"], "negative_infinite")

    def test_noise_delay_and_length(self):
        generator = torch.Generator().manual_seed(9)
        noise = torch.randn(self.reference.shape, generator=generator)
        weak = self.reference + noise * .01
        strong = self.reference + noise * .5
        for score in (lambda x: score_pesq(self.reference, x, 16000),
                      lambda x: score_stoi(self.reference, x, 16000),
                      lambda x: score_si_sdr(self.reference, x)):
            self.assertGreater(score(weak)["value"], score(strong)["value"])
            self.assertLess(score(torch.roll(self.reference, 1000))["value"], score(weak)["value"])
            self.assertEqual(score(weak[:-100])["trimmed_samples"], 100)
