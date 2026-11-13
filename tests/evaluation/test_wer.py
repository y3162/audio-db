"""Word error counts and micro aggregation."""

import unittest

from src.evaluation.metrics.wer import (
    corpus_wer,
    edit_counts,
    empty_counts,
    add_counts,
    normalize_text,
    score_pair,
)
from src.evaluation.results.summary import build_summary


class WordErrorTests(unittest.TestCase):
    def test_substitution_deletion_insertion(self):
        substitution = edit_counts(["a", "b", "c"], ["a", "x", "c"])
        self.assertEqual((substitution["S"], substitution["D"], substitution["I"], substitution["N"]), (1, 0, 0, 3))
        deletion = edit_counts(["a", "b"], ["a"])
        self.assertEqual((deletion["S"], deletion["D"], deletion["I"]), (0, 1, 0))
        insertion = edit_counts(["a"], ["a", "b"])
        self.assertEqual((insertion["S"], insertion["D"], insertion["I"]), (0, 0, 1))

    def test_empty_reference_is_not_zero_wer(self):
        counts = edit_counts([], ["a"])
        self.assertEqual(counts["N"], 0)
        self.assertIsNone(counts["wer"])
        self.assertEqual(counts["I"], 1)

    def test_normalization_v1(self):
        scored = score_pair("Hello, World!", "hello world", "wer-norm-v1")
        self.assertEqual(scored["normalized"]["wer"], 0.0)
        self.assertGreater(scored["raw"]["wer"], 0.0)
        self.assertEqual(normalize_text("hello-world"), "hello world")

    def test_summary_uses_micro_average(self):
        records = [
            _record("u1", substitutions=1, length=1),
            _record("u2", substitutions=0, length=100),
        ]
        summary = build_summary(records)
        group = summary["groups"][0]
        self.assertEqual(group["condition"], "enhanced")
        self.assertEqual(group["edits_normalized"]["S"], 1)
        self.assertEqual(group["edits_normalized"]["N"], 101)
        self.assertAlmostEqual(group["wer_normalized"], 1 / 101)
        self.assertTrue(group["wer_official"])
        self.assertTrue(summary["official_wer"])
        macro = (1.0 + 0.0) / 2.0
        self.assertNotAlmostEqual(group["wer_normalized"], macro)
        self.assertEqual(summary["wer_aggregation"], "micro")

    def test_missing_wer_is_not_an_official_subset(self):
        scored = _record("full", substitutions=0, length=2)
        refused = _record("partial", substitutions=9, length=40)
        # A status of ok with a missing condition is the old shape. It must
        # not become the official WER of the remaining utterance.
        refused["wer"] = {
            "status": "na",
            "reason": "partial_audio",
            "values": {"enhanced": {"status": "na", "reason": "partial_audio"}},
        }
        summary = build_summary([scored, refused])
        group = summary["groups"][0]
        self.assertEqual(group["utterances"], 2)
        self.assertEqual(group["wer_count"], 1)
        self.assertFalse(summary["official_wer"])
        self.assertFalse(group["wer_official"])
        self.assertIsNone(group["wer_normalized"])
        self.assertIsNone(group["edits_normalized"])
        self.assertEqual(group["edits_provisional"]["N"], 2)
        self.assertEqual(group["edits_provisional"]["S"], 0)
        self.assertEqual(group["wer_provisional"], 0.0)
        only_missing = build_summary([refused])
        self.assertEqual(only_missing["groups"][0]["wer_count"], 0)
        self.assertIsNone(only_missing["groups"][0]["wer_normalized"])
        self.assertIsNone(only_missing["groups"][0]["wer_provisional"])
        self.assertNotEqual(only_missing["groups"][0]["wer_normalized"], 0)

    def test_corpus_helper_skips_empty_references(self):
        total = empty_counts()
        add_counts(total, {"S": 0, "D": 0, "I": 1, "N": 0})
        add_counts(total, {"S": 1, "D": 0, "I": 0, "N": 1})
        self.assertEqual(total["empty_reference"], 1)
        self.assertEqual(corpus_wer(total), 1.0)


def _record(utterance_id, substitutions, length):
    counts = {"S": substitutions, "D": 0, "I": 0, "N": length, "wer": None if length == 0 else substitutions / length}
    return {
        "schema_version": 1,
        "status": "ok",
        "identity": {"config_hash": "abc"},
        "model": {"name": "mock", "variant": None},
        "snr_db": 0.0,
        "utterance_id": utterance_id,
        "evaluation": {
            "required_wer": True,
            "required_conditions": ["enhanced"],
            "status": "ok",
            "retryable": False,
            "missing_conditions": [],
            "reasons": [],
            "message": None,
            "conditions": {"enhanced": {"metric": "wer", "status": "ok", "reason": None}},
        },
        "wer": {"status": "ok", "values": {"enhanced": {"status": "ok", "raw": counts, "normalized": counts}}},
        "pesq": {"values": {}},
        "stoi": {"values": {}},
        "estoi": {"values": {}},
        "si_sdr": {"values": {}},
        "timing_sec": {"enhance": 0.0},
    }
