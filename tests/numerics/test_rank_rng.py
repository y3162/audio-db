"""Per-rank checkpoint RNG selection."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng, resume_rng_plan
from src.experiments.calc.errors import TrainingError


class RankRngPlanTest(unittest.TestCase):
    def test_one_process_keeps_the_legacy_snapshot(self) -> None:
        kind, state = resume_rng_plan({"marker": "legacy"}, [{"marker": "rank"}], rank=0, world_size=1)
        self.assertEqual(kind, "legacy")
        self.assertEqual(state["marker"], "legacy")

    def test_several_processes_refuse_a_rank0_only_snapshot(self) -> None:
        with self.assertRaises(TrainingError) as caught:
            resume_rng_plan({"cuda": []}, None, rank=1, world_size=2)
        self.assertIn("per-rank", str(caught.exception))

    def test_several_processes_restore_only_their_rank(self) -> None:
        ranks = [
            {"python": "a", "numpy": "a", "torch": "a", "id": 0},
            {"python": "b", "numpy": "b", "torch": "b", "id": 1},
        ]
        kind, state = resume_rng_plan({"ignored": True}, ranks, rank=1, world_size=2)
        self.assertEqual(kind, "rank")
        self.assertEqual(state["id"], 1)

    def test_incomplete_rank_state_is_refused(self) -> None:
        with self.assertRaises(TrainingError):
            resume_rng_plan(None, None, rank=0, world_size=1)
        ranks = [
            {"python": 1, "numpy": 1},
            {"python": 1, "numpy": 1, "torch": 1},
        ]
        with self.assertRaises(TrainingError):
            resume_rng_plan({}, ranks, rank=0, world_size=2)

    def test_cpu_rank_state_round_trips(self) -> None:
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        random.random()
        np.random.rand()
        torch.rand(4)
        saved = capture_rank_rng(cuda=False)
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        restore_rank_rng(saved, cuda=False)
        restored = capture_rank_rng(cuda=False)
        self.assertEqual(saved["python"], restored["python"])
        self.assertTrue(np.array_equal(saved["numpy"][1], restored["numpy"][1]))
        self.assertEqual(saved["numpy"][2], restored["numpy"][2])
        self.assertTrue(torch.equal(saved["torch"], restored["torch"]))
        self.assertNotEqual(id(saved["numpy"][1]), id(np.random.get_state()[1]))


if __name__ == "__main__":
    unittest.main()
