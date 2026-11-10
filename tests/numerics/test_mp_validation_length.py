import unittest

from src.experiments.calc.errors import TrainingError
from src.experiments.official.mp_senet.reference import align_validation_length


class ValidationLengthTests(unittest.TestCase):
    def test_exact_segment_is_unchanged(self):
        self.assertEqual(align_validation_length(32000, 100, 400), 32000)

    def test_remainder_is_trimmed_to_the_hop_grid(self):
        self.assertEqual(align_validation_length(29083, 100, 400), 29000)

    def test_short_remainder_is_rejected(self):
        with self.assertRaises(TrainingError):
            align_validation_length(150, 100, 400)


if __name__ == "__main__":
    unittest.main()
