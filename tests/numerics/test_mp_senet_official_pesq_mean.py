"""Official validation PESQ keeps failures as -1. The research path still drops na."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.experiments.calc.metrics import _combine_metric
from src.experiments.official.mp_senet.reference import (
    MPSENetTask,
    _annotate_research_pesq,
    _combine_scores,
    combine_official_pesq_values,
    official_eval_pesq,
)
from tests.numerics.support import tiny_config


def _task(loss, **model):
    config = tiny_config(Path("unused"), loss=loss)
    config["model"].update(model)
    task = MPSENetTask(config, torch.device("cpu"))
    task.eval_mode()
    return task


def _batch(length, rows=1, *, fill="randn", seed=0):
    if fill == "zeros":
        clean = torch.zeros(rows, length)
        noisy = torch.zeros(rows, length)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        clean = torch.randn(rows, length, generator=generator)
        noisy = torch.randn(rows, length, generator=generator)
    return {"clean": clean, "noisy": noisy}


class OfficialPesqMeanTests(unittest.TestCase):
    def test_official_mean_keeps_negative_one_and_research_drops_na(self):
        official = combine_official_pesq_values([4.0, -1.0])
        self.assertEqual(official["value"], 1.5)
        self.assertEqual(official["status"], "ok")
        self.assertEqual(official["count"], 2)
        self.assertEqual(official["pesq_failures"], 1)
        self.assertEqual(official["pesq_scored"], 1)
        self.assertEqual(official["pesq_policy"], "official_include_negative_one")
        aggregated = _combine_metric(
            [combine_official_pesq_values([4.0]), combine_official_pesq_values([-1.0])]
        )
        self.assertEqual(aggregated["status"], "ok")
        self.assertEqual(aggregated["value"], 1.5)

        research = _annotate_research_pesq(
            [
                {"status": "ok", "value": 4.0, "reason": None},
                {"status": "na", "value": None, "reason": "zero_energy_audio"},
            ]
        )
        self.assertEqual(research["value"], 4.0)
        self.assertEqual(research["pesq_policy"], "research_exclude_na")
        self.assertEqual(research["pesq_scored"], 1)
        self.assertEqual(research["pesq_failures"], 1)
        excluded = _combine_scores(
            [
                {"status": "ok", "value": 4.0, "reason": None},
                {"status": "na", "value": None, "reason": "zero_energy_audio"},
            ]
        )
        self.assertEqual(excluded["value"], 4.0)
        self.assertNotEqual(research["pesq_policy"], official["pesq_policy"])

    def test_validation_report_returns_the_official_mean_for_the_trainer(self):
        task = _task("official", n_fft=400, hop_size=100, win_size=400)
        with patch("src.experiments.official.mp_senet.reference.official_eval_pesq", side_effect=[4.0, -1.0]):
            with patch("src.evaluation.metrics.pesq.score_pesq", side_effect=AssertionError("research")):
                report = task.validation_report(_batch(1600, rows=2, seed=5))
        self.assertEqual(report["pesq"]["status"], "ok")
        self.assertEqual(report["pesq"]["value"], 1.5)
        self.assertEqual(report["pesq_failures"], 1)
        self.assertEqual(report["pesq_scored"], 1)
        self.assertEqual(report["pesq_policy"], "official_include_negative_one")
        summary = _combine_metric([report["pesq"]])
        self.assertEqual(summary["value"], 1.5)

    def test_pesq_receives_the_istft_length_not_the_old_truncation_as_input(self):
        import pesq as pesq_module

        task = _task("official", n_fft=400, hop_size=100, win_size=400)
        seen = []
        real = pesq_module.pesq

        def wrapped(sample_rate, reference, estimate, mode="wb", on_error=pesq_module.PesqError.RAISE_EXCEPTION):
            seen.append(
                (
                    int(np.asarray(reference).shape[-1]),
                    int(np.asarray(estimate).shape[-1]),
                    on_error,
                )
            )
            return real(sample_rate, reference, estimate, mode, on_error)

        with patch("pesq.pesq", wrapped):
            report = task.validation_report(_batch(1650, seed=6))
        self.assertTrue(seen)
        self.assertTrue(all(reference == 1650 and estimate == 1600 for reference, estimate, _mode in seen))
        self.assertTrue(any(mode == pesq_module.PesqError.RETURN_VALUES for _reference, _estimate, mode in seen))
        row = report["audio_lengths"][0]
        self.assertEqual(row["model_input"], 1650)
        self.assertEqual(row["pesq_reference"], 1650)
        self.assertEqual(row["pesq_estimate"], 1600)
        self.assertEqual(report["pesq"]["status"], "ok")
        self.assertEqual(report["pesq"]["value"], -1.0)
        self.assertEqual(report["pesq_failures"], 1)
        self.assertEqual(report["pesq_scored"], 0)
        self.assertEqual(report["pesq_policy"], "official_include_negative_one")

    def test_silence_and_library_errors_become_negative_one_on_the_official_path(self):
        silence = np.zeros(4096, dtype=np.float32)
        short = np.zeros(1600, dtype=np.float32)
        self.assertEqual(official_eval_pesq(silence, silence, 16000), -1.0)
        self.assertEqual(official_eval_pesq(short, short, 16000), -1.0)
        tone = (0.1 * np.sin(2 * np.pi * 220.0 * np.arange(16000) / 16000.0)).astype(np.float32)
        degraded = tone * np.float32(0.8)
        from pesq import pesq as raw_pesq

        raw = float(raw_pesq(16000, tone, degraded, "wb"))
        self.assertGreater(raw, 1.0)
        self.assertEqual(official_eval_pesq(tone, degraded, 16000), raw)
        ones = np.ones(32, dtype=np.float32)
        with patch("pesq.pesq", return_value=-6.0):
            self.assertEqual(official_eval_pesq(ones, ones, 16000), -1.0)
        with patch("pesq.pesq", side_effect=RuntimeError("No utterances detected")):
            self.assertEqual(official_eval_pesq(ones, ones, 16000), -1.0)

        task = _task("official", n_fft=32, hop_size=8, win_size=32)
        report = task.validation_report(_batch(4096, fill="zeros"))
        self.assertEqual(report["audio_lengths"][0]["model_input"], 4096)
        self.assertEqual(report["audio_lengths"][0]["audio"], 4096)
        self.assertEqual(report["pesq"]["status"], "ok")
        self.assertEqual(report["pesq"]["value"], -1.0)
        self.assertEqual(report["pesq_failures"], 1)
        self.assertEqual(report["pesq_policy"], "official_include_negative_one")

        research = _task("reconstruction", n_fft=32, hop_size=8, win_size=32)
        research_report = research.validation_report(_batch(4096, fill="zeros"))
        self.assertEqual(research_report["pesq"]["status"], "na")
        self.assertIsNone(research_report["pesq"]["value"])
        self.assertEqual(research_report["pesq_policy"], "research_exclude_na")
        self.assertNotEqual(research_report["pesq"]["value"], -1.0)
        self.assertNotEqual(research_report["pesq"]["value"], 0)

    def test_missing_library_stays_unavailable(self):
        batch = _batch(1600, seed=7)
        with patch("src.evaluation.metrics.pesq.library_status", return_value="pesq is not installed"):
            official = _task("official", n_fft=400, hop_size=100, win_size=400).validation_report(batch)
            research = _task("reconstruction", n_fft=400, hop_size=100, win_size=400).validation_report(batch)
        for report in (official, research):
            self.assertEqual(report["pesq"]["status"], "unavailable")
            self.assertIsNone(report["pesq"]["value"])
            self.assertIsNone(report["pesq_failures"])
            self.assertNotEqual(report["pesq"]["value"], -1)
            self.assertNotEqual(report["pesq"]["value"], 0)
        self.assertEqual(official["pesq_policy"], "official_include_negative_one")
        self.assertEqual(research["pesq_policy"], "research_exclude_na")

    def test_research_report_does_not_use_the_official_mean(self):
        task = _task("reconstruction")
        calls = {"n": 0}

        def fake_score(_reference, _estimate, _sample_rate):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"status": "ok", "value": 4.0, "reason": None}
            return {"status": "na", "value": None, "reason": "zero_energy_audio"}

        with patch("src.evaluation.metrics.pesq.score_pesq", fake_score):
            report = task.validation_report(_batch(160, rows=2, seed=8))
        self.assertEqual(report["pesq"]["value"], 4.0)
        self.assertNotEqual(report["pesq"]["value"], 1.5)
        self.assertEqual(report["pesq_policy"], "research_exclude_na")
        self.assertEqual(report["pesq_scored"], 1)
        self.assertEqual(report["pesq_failures"], 1)


if __name__ == "__main__":
    unittest.main()
