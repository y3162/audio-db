"""Global batch planning."""

import unittest

from src.training.core.batch import plan_batches
from src.training.errors import TrainingError


class BatchPlanTests(unittest.TestCase):
    def test_examples_from_the_spec(self):
        planned = plan_batches(64, 8, 4)
        self.assertEqual((planned.micro_batch_size, planned.accumulation_steps), (8, 2))
        planned = plan_batches(48, 6, 3)
        self.assertEqual((planned.micro_batch_size, planned.accumulation_steps), (4, 4))
        with self.assertRaises(TrainingError):
            plan_batches(50, 8, 4)

    def test_largest_divisor_under_the_cap(self):
        planned = plan_batches(16, 7, 1)
        self.assertEqual(planned.micro_batch_size, 4)
        self.assertEqual(planned.accumulation_steps, 4)
        self.assertEqual(
            planned.micro_batch_size * planned.world_size * planned.accumulation_steps,
            planned.global_batch_size,
        )

    def test_single_process_cpu_keeps_the_global_batch(self):
        planned = plan_batches(8, 3, 1)
        self.assertEqual((planned.micro_batch_size, planned.accumulation_steps, planned.world_size), (2, 4, 1))

    def test_non_positive_sizes_are_rejected(self):
        with self.assertRaises(TrainingError):
            plan_batches(0, 1, 1)
        with self.assertRaises(TrainingError):
            plan_batches(4, 0, 1)
        with self.assertRaises(TrainingError):
            plan_batches(True, 1, 1)
