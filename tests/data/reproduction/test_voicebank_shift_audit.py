import unittest

from src.experiments.data.errors import ReproductionError
from src.experiments.data.prepare.voicebank_shift_audit import audit_recipes


def row(utterance, split, noise, start, end):
    return {"utterance_id": utterance, "split": split, "noise_sha256": noise * 64,
            "noise_start_48k": start, "noise_end_exclusive_48k": end, "status": "ready"}


class VoiceBankShiftAuditTests(unittest.TestCase):
    def test_merges_train_spans_without_double_counting_validation(self):
        rows = [row("p226_001", "train", "a", 0, 10),
                row("p227_001", "train", "a", 5, 15),
                row("p286_001", "train", "a", 3, 20),
                row("p232_001", "test", "b", 0, 9)]
        report = audit_recipes(rows)
        valid = report["comparisons"]["train_valid"]
        self.assertEqual(valid["overlap_samples"], 12)
        self.assertEqual(valid["target_noise_samples"], 17)
        self.assertEqual(valid["target_utterances_with_noise_overlap"], 1)
        self.assertEqual(valid["shared_noise_sha256_count"], 1)
        self.assertAlmostEqual(valid["target_noise_overlap_fraction"], 12 / 17)
        self.assertFalse(report["validation_noise_independent"])
        self.assertEqual(report["comparisons"]["train_test"]["overlap_samples"], 0)
        self.assertEqual(valid["shared_speakers"], [])

    def test_duplicate_and_invalid_spans_are_rejected(self):
        one = row("p226_001", "train", "a", 0, 10)
        with self.assertRaises(ReproductionError):
            audit_recipes([one, one])
        with self.assertRaises(ReproductionError):
            audit_recipes([row("p226_001", "train", "a", 4, 4)])
