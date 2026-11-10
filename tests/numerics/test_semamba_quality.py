"""Quality-comparison helpers. These do not import the reference tree or a dummy PESQ."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.metrics.pesq import score_pesq
from src.evaluation.metrics.sisdr import score_si_sdr
from src.experiments.calc.dataset import _normalize
from tests.numerics.reference.semamba_quality import (
    TOLERANCES,
    assert_tensor_roundtrip,
    choose_max_seconds,
    energy_normalize,
    score_conditions,
    summarize_step,
    tolerance_result,
    waveform_gap,
)


class QualityComparisonTest(unittest.TestCase):
    def test_length_cap_rejects_short_sources(self) -> None:
        with self.assertRaises(ValueError):
            choose_max_seconds(2.0)
        self.assertEqual(choose_max_seconds(5.855), 5.855)
        self.assertEqual(choose_max_seconds(14.55), 10.0)

    def test_energy_normalize_matches_training(self) -> None:
        noisy = torch.randn(1600)
        clean = torch.randn(1600)
        scaled_noisy, scaled_clean, scale = energy_normalize(noisy, clean)
        expected_noisy, expected_clean = _normalize(noisy, clean)
        self.assertTrue(torch.equal(scaled_noisy, expected_noisy))
        self.assertTrue(torch.equal(scaled_clean, expected_clean))
        self.assertAlmostEqual(float(torch.sqrt(torch.mean(scaled_noisy * scaled_noisy))), 1.0, places=5)
        self.assertGreater(scale, 0.0)

    def test_waveform_roundtrip_is_exact(self) -> None:
        payload = {"clean": torch.randn(32), "nested": {"official": torch.tensor([1.0, -2.0])}}
        with tempfile.TemporaryDirectory() as directory:
            digest = assert_tensor_roundtrip(Path(directory) / "waves.pt", payload)
        self.assertEqual(len(digest), 64)

    def test_non_finite_wave_is_not_zero(self) -> None:
        gap = waveform_gap(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 2.0]))
        self.assertEqual(gap["status"], "na")
        self.assertIsNone(gap["max_abs"])
        self.assertIsNone(gap["cosine"])

    def test_missing_metric_is_not_zero(self) -> None:
        clean = torch.zeros(8)
        noisy = torch.ones(8)
        official = torch.tensor([1.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        independent = torch.ones(8)
        scored = score_conditions(clean, noisy, official, independent, 16000)
        for metric in ("pesq", "stoi", "si_sdr"):
            self.assertIsNone(scored["official_minus_independent"][metric]["value"])
            self.assertNotEqual(scored["official_minus_independent"][metric]["value"], 0)
            self.assertIsNone(scored["improvement_over_noisy"]["official"][metric]["value"])
        present = _summary_row("present", 0.05, 0.01, 0.4)
        summary = summarize_step([{**scored, "case_id": "missing"}, present])
        self.assertGreater(summary["metrics"]["pesq"]["abs_official_minus_independent"]["n_missing"], 0)
        self.assertNotEqual(summary["metrics"]["pesq"]["abs_official_minus_independent"]["value"], 0.0)

    def test_tolerance_uses_every_utterance(self) -> None:
        self.assertFalse(tolerance_result([])["passed"])
        inside = [_finite_utterance("a", 0.05, 0.01, 0.4), _finite_utterance("b", -0.02, 0.0, -0.2)]
        self.assertTrue(tolerance_result(inside)["passed"])
        self.assertEqual(TOLERANCES["pesq_abs"], 0.10)
        self.assertEqual(TOLERANCES["stoi_abs"], 0.02)
        self.assertEqual(TOLERANCES["si_sdr_abs_db"], 1.0)
        outside = [_finite_utterance("a", 0.05, 0.01, 0.4), _finite_utterance("b", 0.25, 0.0, 0.0)]
        result = tolerance_result(outside)
        self.assertFalse(result["passed"])
        self.assertTrue(any(item.startswith("b:pesq:") for item in result["failures"]))
        missing = [_finite_utterance("a", None, 0.0, 0.0)]
        self.assertFalse(tolerance_result(missing)["passed"])
        unequal = _finite_utterance("short", 0.0, 0.0, 0.0)
        unequal["alignment"] = {"same_length": False}
        self.assertIn("short:alignment:length_mismatch", tolerance_result([unequal])["failures"])

    def test_identical_signal_is_not_scored_as_zero_si_sdr(self) -> None:
        signal = torch.randn(1600)
        result = score_si_sdr(signal, signal.clone())
        self.assertIsNone(result["value"])
        self.assertNotEqual(result["value"], 0)

    def test_real_pesq_is_not_the_dummy_zero(self) -> None:
        signal = torch.sin(2 * torch.pi * 440 * torch.arange(16000) / 16000)
        result = score_pesq(signal, signal.clone(), 16000)
        if result["status"] == "ok":
            self.assertGreater(result["value"], 1.0)
        else:
            self.assertIsNone(result["value"])
            self.assertNotEqual(result["status"], "ok")


def _summary_row(case_id: str, pesq: float, stoi: float, si_sdr: float) -> dict:
    def metric(value: float) -> dict:
        return {"status": "ok", "reason": None, "value": value}

    deltas = {"pesq": metric(pesq), "stoi": metric(stoi), "si_sdr": metric(si_sdr)}
    zeros = {"pesq": metric(0.0), "stoi": metric(0.0), "si_sdr": metric(0.0)}
    return {
        "case_id": case_id,
        "conditions": {"noisy": zeros, "official": deltas, "independent": zeros},
        "official_minus_independent": deltas,
        "improvement_over_noisy": {"official": deltas, "independent": zeros},
        "wave": {"status": "ok", "reason": None, "cosine": 0.9, "max_abs": 0.1},
    }


def _finite_utterance(case_id: str, pesq: float | None, stoi: float | None, si_sdr: float | None) -> dict:
    def metric(value: float | None) -> dict:
        if value is None:
            return {"status": "na", "reason": "missing", "value": None}
        return {"status": "ok", "reason": None, "value": value}

    return {
        "case_id": case_id,
        "official_minus_independent": {"pesq": metric(pesq), "stoi": metric(stoi), "si_sdr": metric(si_sdr)},
    }


if __name__ == "__main__":
    unittest.main()
