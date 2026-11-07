"""MossFormer validation split and gradient-clip order."""

from __future__ import annotations

import math
import unittest

import torch

from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.order import (
    distributed_sampler_coverage,
    rank_order,
    recipe_validation_positions,
    sampler_batch_count,
)
from src.training.se.mossformergan import average_new_grad, official_validation_mean


def _official_rank_slices(size: int, world: int, epoch: int = 0) -> list[list[int]]:
    num_samples = int(math.ceil(size / world))
    total = num_samples * world
    generator = torch.Generator()
    generator.manual_seed(epoch)
    ind = torch.randperm(int(size / world), generator=generator) * world
    indices: list[int] = []
    for offset in range(world):
        indices.extend((ind + offset).tolist())
    indices += indices[: total - len(indices)]
    return [indices[rank * num_samples : (rank + 1) * num_samples] for rank in range(world)]


class MossFormerSamplerTests(unittest.TestCase):
    def test_rank_slices_match_the_official_sampler(self):
        for size, world in ((8, 2), (10, 3), (11, 4), (5, 2)):
            official = _official_rank_slices(size, world, epoch=0)
            ours = [
                rank_order("rank_mod_fixed_v1", size, seed=99, epoch=0, world_size=world, rank=rank)
                for rank in range(world)
            ]
            self.assertEqual(ours, official)
            self.assertEqual(len({len(row) for row in ours}), 1)
            coverage = distributed_sampler_coverage(size, world)
            self.assertEqual(coverage["num_samples_per_rank"], len(ours[0]))
            self.assertEqual(coverage["dropped_tail"], size % world)
            self.assertEqual(coverage["scored_slots"], coverage["num_samples_per_rank"] * world)
            flat = [item for row in ours for item in row]
            self.assertEqual(len(flat) - len(set(flat)), coverage["padded_duplicates"])
            self.assertTrue(set(range((size // world) * world)).issuperset(set(flat)))
            self.assertTrue(set(range((size // world) * world, size)).isdisjoint(flat))

    def test_fewer_items_than_ranks_is_not_the_official_split(self):
        left, left_label = recipe_validation_positions(1, 2, 0)
        right, right_label = recipe_validation_positions(1, 2, 1)
        self.assertEqual(left, [0])
        self.assertEqual(right, [0])
        self.assertEqual(left_label, "undersized_same_list")
        self.assertEqual(right_label, "undersized_same_list")
        with self.assertRaises(ReproductionError):
            rank_order("rank_mod_fixed_v1", 1, seed=0, epoch=0, world_size=2, rank=0)
        official = rank_order("rank_mod_fixed_v1", 824, seed=0, epoch=0, world_size=2, rank=0)
        got, label = recipe_validation_positions(824, 2, 0)
        self.assertEqual(label, "rank_mod_fixed_v1")
        self.assertEqual(got, official)
        self.assertEqual(len(got), 412)

    def test_short_last_batch_counts_as_one_batch_on_every_rank(self):
        coverage = distributed_sampler_coverage(10, 3)
        self.assertEqual(coverage["num_samples_per_rank"], 4)
        self.assertEqual(coverage["dropped_tail"], 1)
        self.assertEqual(coverage["padded_duplicates"], 3)
        self.assertEqual(sampler_batch_count(4, 3), 2)
        self.assertEqual(sampler_batch_count(4, 4), 1)

    def test_validation_mean_keeps_failures_in_the_denominator(self):
        equal = official_validation_mean([3.0, 3.0])
        self.assertEqual(equal, 3.0)
        unequal = official_validation_mean([3.0, 6.0])
        self.assertEqual(unequal, 4.5)
        self.assertNotEqual(unequal, (3.0 + 6.0 * 1) / 3)

    def test_new_gradient_is_averaged_before_clip(self):
        merged = average_new_grad(None, torch.tensor([30.0]), [torch.tensor([0.0])])
        parameter = torch.nn.Parameter(torch.zeros(1))
        parameter.grad = merged
        torch.nn.utils.clip_grad_norm_(parameter, 10.0)
        self.assertAlmostEqual(float(parameter.grad), 10.0, places=5)
        self.assertNotAlmostEqual(float(parameter.grad), 5.0)
        carried = average_new_grad(torch.tensor([4.0]), torch.tensor([2.0]), [torch.tensor([2.0])])
        self.assertAlmostEqual(float(carried), 6.0)


if __name__ == "__main__":
    unittest.main()
