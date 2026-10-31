"""Disjointness rules for the LibriSpeech+DEMAND smoke set."""

import unittest

from src.data.training_sets import assert_disjoint, plan_pairs


def _speech(prefix: str, subset: str, count: int) -> list[dict]:
    return [
        {
            "path": f"{prefix}/speaker{index}/utt.flac",
            "speaker_id": f"{prefix}-s{index}",
            "subset": subset,
            "transcript": "a",
        }
        for index in range(count)
    ]


def _noise(scenes: list[str]) -> list[dict]:
    return [{"path": f"DEMAND/{scene}/ch01.wav", "subset": scene} for scene in scenes]


class PlanTests(unittest.TestCase):
    def test_train_and_validation_do_not_share_speech_noise_or_speakers(self):
        train = plan_pairs(_speech("train-clean-100", "train-clean-100", 10), _noise(["OOFFICE", "OMEETING"]), role="train", count=4)
        valid = plan_pairs(_speech("dev-clean", "dev-clean", 6), _noise(["NPARK"]), role="validation", count=2)
        assert_disjoint(train, valid)
        self.assertEqual([item["snr_db"] for item in train], [0.0, 5.0, 10.0, 15.0])

    def test_test_clean_is_rejected(self):
        train = plan_pairs(_speech("test-clean", "test-clean", 2), _noise(["OOFFICE"]), role="train", count=1)
        valid = plan_pairs(_speech("dev-clean", "dev-clean", 2), _noise(["NPARK"]), role="validation", count=1)
        with self.assertRaises(ValueError):
            assert_disjoint(train, valid)


if __name__ == "__main__":
    unittest.main()
