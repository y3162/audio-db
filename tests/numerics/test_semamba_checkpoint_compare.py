import unittest

import numpy as np
import torch

from tests.numerics.reference.compare_semamba_checkpoints import compare


class CheckpointComparisonTest(unittest.TestCase):
    def test_nested_optimizer_and_rng_differences_are_detected(self):
        original = {"optimizer": {"state": [torch.tensor([1.0, 2.0])]}, "rng": np.array([1, 2])}
        same = {"optimizer": {"state": [torch.tensor([1.0, 2.0])]}, "rng": np.array([1, 2])}
        self.assertTrue(compare(original, same)["equal"])
        same["optimizer"]["state"][0][1] = 3.0
        result = compare(original, same)
        self.assertFalse(result["equal"])
        self.assertEqual(result["max_abs"], 1.0)
        self.assertIn("optimizer", result["worst_path"])
        same["rng"][0] = 3
        self.assertEqual(compare(original, same)["different_tensors"], 2)

    def test_missing_state_is_not_equal(self):
        self.assertFalse(compare({"scheduler": {"epoch": 1}}, {})["equal"])
