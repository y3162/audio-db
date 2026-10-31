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

    def test_si_sdr_matches_the_definition(self):
        reference = torch.tensor([1.0, 0.0, -1.0, 0.0])
        estimate = torch.tensor([1.0, 0.5, -1.0, 0.0])
        scored = score_si_sdr(reference, estimate)
        self.assertEqual(scored["status"], "ok")
        self.assertAlmostEqual(scored["value"], 10.280287236002435, places=5)
        perfect = score_si_sdr(reference, reference * 2)
        self.assertIsNone(perfect["value"])
        self.assertEqual(perfect["reason"], "numerically_infinite")
