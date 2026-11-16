"""Tail batches stay in the epoch and the loss uses the real sample count."""

from __future__ import annotations

import unittest

import torch

from src.experiments.data.sample import SampleRef
from src.experiments.data.fixed_pair import MicroBatcher
from src.experiments.data.tensor import TensorPairData
from src.experiments.training.sample_weight import sample_scales, window_is_uniform


def _ref(index: int) -> SampleRef:
    return SampleRef(sample_id=str(index), realization_id=str(index), index=index)


class ScaleTests(unittest.TestCase):
    def test_uniform_window_matches_divide_by_accumulation(self) -> None:
        counts = [[4, 4], [4, 4]]
        self.assertTrue(window_is_uniform(counts, 2))
        self.assertFalse(window_is_uniform([[4, 4], [4, 1]], 2))
        self.assertFalse(window_is_uniform([[4]], 2))

    def test_uneven_scales_reconstruct_the_global_mean(self) -> None:
        local = [4.0, 1.0]
        counts = [4, 1]
        scales = sample_scales(counts, world_size=2, n_global=7)
        other = sample_scales([2], world_size=2, n_global=7)
        weighted = sum(loss * scale for loss, scale in zip(local, scales))
        other_weighted = 9.0 * other[0]
        self.assertAlmostEqual(weighted + other_weighted, (4 * 4 + 1 * 1 + 2 * 9) / 7 * 2)
        self.assertEqual(sample_scales([0], world_size=2, n_global=5), [0.0])
        self.assertEqual(sample_scales([3], world_size=1, n_global=0), [0.0])


class TailBatchTests(unittest.TestCase):
    def test_keep_partial_keeps_the_short_group_and_records_drops(self) -> None:
        rows = [_ref(index) for index in range(5)]
        kept = MicroBatcher(2, 2)
        self.assertTrue(kept.keep_partial)
        self.assertEqual([len(group) for group in kept.groups(rows)], [2, 2, 1])
        self.assertEqual(kept.dropped_samples, 0)
        dropped = MicroBatcher(2, 2, keep_partial=False)
        self.assertEqual([len(group) for group in dropped.groups(rows)], [2, 2])
        self.assertEqual(dropped.dropped_samples, 1)

    def test_tensor_pairs_keep_the_tail_without_copying(self) -> None:
        data = TensorPairData(features=3, examples=5, batch_size=2, seed=1)
        data.setup(type("Runtime", (), {"device": torch.device("cpu"), "world_size": 1, "rank": 0})())
        view = data.train_epoch(0, start_update=0)
        self.assertEqual(view.updates, 3)
        self.assertEqual(view.batch(0)[0].inputs.shape[0], 2)
        self.assertEqual(view.batch(2)[0].inputs.shape[0], 1)
        self.assertEqual(view.batch(2)[0].sample_id, "ex-4")
        explicit = TensorPairData(features=3, examples=5, batch_size=2, seed=1, drop_last=True)
        explicit.setup(type("Runtime", (), {"device": torch.device("cpu"), "world_size": 1, "rank": 0})())
        self.assertEqual(explicit.train_epoch(0, start_update=0).updates, 2)

    def test_short_rank_gets_no_copied_sample(self) -> None:
        data = TensorPairData(features=3, examples=5, batch_size=2, seed=1)
        runtime = type("Runtime", (), {"device": torch.device("cpu"), "world_size": 2, "rank": 1})()
        data.setup(runtime)
        view = data.train_epoch(0, start_update=0)
        self.assertEqual(view.updates, 2)
        self.assertEqual(view.batch(0)[0].inputs.shape[0], 2)
        self.assertEqual(view.batch(1), [])


if __name__ == "__main__":
    unittest.main()
