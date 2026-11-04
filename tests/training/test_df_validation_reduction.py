"""DeepFilterNet validation reduction: utterance mean versus padded batch mean.

The research trainer scores each validation utterance alone and averages those
losses. Official ``df/train.py`` averages the per-batch ``mse_loss``. Those
agree for equal lengths and no padding. They do not agree once lengths differ.
"""

import unittest

import torch
import torch.nn.functional as F


def _mse(wave: torch.Tensor) -> float:
    return float(wave.square().mean())


class ValidationReductionTest(unittest.TestCase):
    def test_equal_lengths_match(self):
        items = [torch.randn(128) for _ in range(4)]
        utterance_mean = sum(_mse(item) for item in items) / len(items)
        batch_mean = float(torch.stack(items).square().mean())
        self.assertAlmostEqual(utterance_mean, batch_mean, places=6)

    def test_padded_batch_mean_differs(self):
        short = torch.ones(8)
        long = torch.full((32,), 2.0)
        utterance_mean = (_mse(short) + _mse(long)) / 2
        padded = torch.stack([F.pad(short, (0, 24)), long])
        batch_mean = float(padded.square().mean())
        self.assertNotAlmostEqual(utterance_mean, batch_mean, places=3)

    def test_schedule_step_count_stops_inside_the_cosine_table(self):
        utterances = 10700
        spans = ((2, 16), (3, 24), (5, 32), (110, 64))
        steps = sum(epochs * (utterances // batch) for epochs, batch in spans)
        self.assertEqual(steps, 22711)
        last_batch = 64
        steps_per_epoch = utterances // last_batch
        last_index = 119 * steps_per_epoch + (steps_per_epoch - 1)
        self.assertEqual(last_index, 120 * steps_per_epoch - 1)
