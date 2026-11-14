"""Units of the research metrics. They do not train and do not fill gaps with zero."""

from __future__ import annotations

import math
import unittest

import torch

from src.experiments.calc.spectral import si_sdr_loss
from src.experiments.errors import BuildError
from src.experiments.metrics.loss import LossMetric
from src.experiments.metrics.signal import ESTOIMetric, PESQMetric, SISDRMetric, STOIMetric
from src.experiments.metrics.wer import WerMetric


class LossMetricTests(unittest.TestCase):
    def test_weighted_mean_and_non_finite_is_not_zero(self) -> None:
        metric = LossMetric()
        metric.update(1.0, 1)
        metric.update(3.0, 3)
        result = metric.result_from(metric.local_stats())
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.unit, "loss")
        self.assertEqual(result.direction, "min")
        self.assertAlmostEqual(result.value, 2.5)

        metric.update(float("nan"), 2)
        failed = metric.result_from(metric.local_stats())
        self.assertEqual(failed.status, "incomplete")
        self.assertIsNone(failed.value)
        self.assertNotEqual(failed.value, 0.0)
        self.assertEqual(failed.failures, 2)
        self.assertAlmostEqual(failed.partial_value, 2.5)

    def test_blank_name_is_rejected(self) -> None:
        with self.assertRaises(BuildError):
            LossMetric(name="validation.loss")


class SignalMetricTests(unittest.TestCase):
    def test_si_sdr_is_positive_db_and_not_the_training_loss(self) -> None:
        reference = torch.linspace(-0.4, 0.4, 1600)
        estimate = reference * 0.5 + 0.02 * torch.sin(torch.linspace(0, 30, 1600))
        metric = SISDRMetric()
        metric.score(reference, estimate)
        result = metric.result_from(metric.local_stats())
        loss = float(si_sdr_loss(estimate.unsqueeze(0), reference.unsqueeze(0)))
        self.assertLess(loss, 0.0)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.unit, "dB")
        self.assertGreater(result.value, 0.0)
        self.assertAlmostEqual(result.value, -loss, places=3)

        perfect = SISDRMetric()
        perfect.score(reference, reference.clone())
        perfect_result = perfect.result_from(perfect.local_stats())
        self.assertEqual(perfect_result.status, "incomplete")
        self.assertIsNone(perfect_result.value)

    def test_pesq_stoi_and_estoi_do_not_fill_failures_with_zero(self) -> None:
        reference = torch.zeros(160)
        estimate = torch.zeros(160)
        for metric in (PESQMetric(), STOIMetric(), ESTOIMetric()):
            metric.score(reference, estimate, 16000)
            result = metric.result_from(metric.local_stats())
            self.assertIn(result.status, {"incomplete", "ok"})
            if result.status != "ok":
                self.assertIsNone(result.value)
                self.assertNotEqual(result.partial_value, 0.0)
                self.assertTrue(result.failures or result.excluded)
            else:
                self.assertTrue(math.isfinite(result.value))
                self.assertNotEqual(result.value, 0.0)


class WerMetricTests(unittest.TestCase):
    def test_corpus_micro_average(self) -> None:
        metric = WerMetric()
        metric.update("alpha bravo", "alpha extra")
        metric.update("charlie delta", "charlie delta")
        result = metric.result_from(metric.local_stats())
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.value, 0.25)
        self.assertEqual(result.weight, 4.0)

    def test_missing_reference_is_incomplete_and_not_zero(self) -> None:
        metric = WerMetric()
        metric.update("alpha bravo", "alpha bravo")
        metric.update("", "alpha bravo")
        result = metric.result_from(metric.local_stats())
        self.assertEqual(result.status, "incomplete")
        self.assertIsNone(result.value)
        self.assertNotEqual(result.value, 0.0)
        self.assertEqual(result.failures, 1)
        self.assertAlmostEqual(result.partial_value, 0.0)

    def test_recognizer_must_transcribe(self) -> None:
        with self.assertRaises(BuildError):
            WerMetric(recognizer=object())


if __name__ == "__main__":
    unittest.main()
