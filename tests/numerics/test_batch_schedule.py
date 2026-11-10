import unittest
from src.experiments.calc.batch_schedule import scheduled_batch_size


class BatchScheduleTest(unittest.TestCase):
    def test_deepfilter_cap_and_boundaries(self):
        schedule = [(0, 16), (2, 24), (5, 32), (10, 64), (20, 128), (40, 9999)]
        self.assertEqual([scheduled_batch_size(e, 64, schedule) for e in (0, 1, 2, 4, 5, 9, 10, 20, 40)],
                         [16, 16, 24, 24, 32, 32, 64, 64, 64])
        self.assertEqual(scheduled_batch_size(40, 24, schedule), 24)

    def test_rejects_invalid_schedule(self):
        for schedule in ([], [(1, 16)], [(0, 1), (0, 2)], [(0, 0)]):
            with self.assertRaises(ValueError):
                scheduled_batch_size(0, 64, schedule)
