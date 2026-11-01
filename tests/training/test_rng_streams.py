"""CPU seeding must not reset or initialize the CUDA random stream."""

import unittest
from unittest.mock import patch

import torch

from src.training.core.rng import capture_rng, restore_rng, seed_model_init, seed_training_stream


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

    def test_restore_moves_checkpoint_rng_tensors_to_cpu(self):
        class MappedState:
            def __init__(self, value):
                self.value = value
                self.called = False

            def cpu(self):
                self.called = True
                return self.value

        state = capture_rng(cuda=False)
        state['torch'] = MappedState(state['torch'])
        state['cuda'] = [MappedState(torch.tensor([1], dtype=torch.uint8))]
        with patch.object(torch.cuda, 'set_rng_state_all') as restore_cuda:
            restore_rng(state, cuda=True)
        self.assertTrue(state['torch'].called)
        self.assertTrue(state['cuda'][0].called)
        restore_cuda.assert_called_once()
        self.assertEqual(restore_cuda.call_args.args[0][0].device.type, 'cpu')
