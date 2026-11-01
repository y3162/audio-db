"""CPU seeding must not reset or initialize the CUDA random stream."""

import unittest
from unittest.mock import patch

import torch

from src.training.core.rng import seed_model_init, seed_training_stream


class RandomStreamTest(unittest.TestCase):
    def test_model_initialization_restores_cpu_and_does_not_seed_cuda(self):
        before = torch.random.get_rng_state().clone()
        with patch.object(torch.cuda, "manual_seed_all") as seed_cuda:
            with seed_model_init(123):
                first = torch.rand(5)
            with seed_model_init(123):
                second = torch.rand(5)
        seed_cuda.assert_not_called()
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_cpu_training_does_not_seed_cuda(self):
        with patch.object(torch.cuda, "manual_seed_all") as seed_cuda:
            seed_training_stream(123, cuda=False)
        seed_cuda.assert_not_called()

    def test_cuda_training_seeds_cuda_once(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "manual_seed_all"
        ) as seed_cuda:
            seed_training_stream(123, cuda=True)
        seed_cuda.assert_called_once()
