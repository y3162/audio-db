"""Regression for the ESPnet multi-resolution loss backward graph."""
import unittest
import warnings

import torch

from src.experiments.calc.spectral import multi_resolution_l1


class TFGridNetLossOrderTest(unittest.TestCase):
    def test_separate_alignment_branches_match_reference_gradient(self):
        torch.manual_seed(3)
        target = torch.randn(1, 256)
        estimate = torch.randn(1, 256)
        actual = estimate.clone().requires_grad_()
        expected = estimate.clone().requires_grad_()
        scale = (expected * target).sum(-1, keepdim=True) / (expected.square().sum(-1, keepdim=True) + 1e-8)
        time = (expected * scale - target).abs().sum(-1)
        spectral = torch.zeros_like(time)
        for size in (32, 64):
            def magnitude(x):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="A window was not provided")
                    return torch.stft(x, size, hop_length=size // 2, window=None, return_complex=True).abs()
            spectral = spectral + (magnitude(expected * scale) - magnitude(target)).abs().sum((-2, -1))
        reference = (time * 0.5 + 0.5 * spectral / 2).mean()
        loss = multi_resolution_l1(target, actual, (32, 64))
        self.assertTrue(torch.equal(loss, reference))
        loss.backward(); reference.backward()
        self.assertTrue(torch.equal(actual.grad, expected.grad))


if __name__ == '__main__':
    unittest.main()
