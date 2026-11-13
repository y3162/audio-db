"""Gradient accumulation matches a single micro-batch when the reduction is linear.

BatchNorm running statistics are updated per micro-batch, so the same sum is
not an exact match. That case is recorded separately and is not given a wide
tolerance.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn


def _step(module: nn.Module, batches: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
    module.zero_grad(set_to_none=True)
    total = None
    scale = len(batches)
    for batch, target in zip(batches, targets, strict=True):
        loss = torch.mean((module(batch) - target) ** 2) / scale
        loss.backward()
        total = loss.detach() if total is None else total + loss.detach()
    return total


class AccumulationLinearTests(unittest.TestCase):
    def test_linear_loss_and_gradient_match(self) -> None:
        torch.manual_seed(0)
        full = torch.randn(4, 6)
        target = torch.randn(4, 3)
        left = nn.Linear(6, 3)
        right = nn.Linear(6, 3)
        right.load_state_dict(left.state_dict())
        whole = _step(left, [full], [target])
        parts = _step(right, [full[:2], full[2:]], [target[:2], target[2:]])
        self.assertLess(abs(float(whole) - float(parts)), 1e-6)
        for a, b in zip(left.parameters(), right.parameters(), strict=True):
            self.assertTrue(torch.allclose(a.grad, b.grad, rtol=0, atol=1e-6))
        opt_a = torch.optim.SGD(left.parameters(), lr=0.05)
        opt_b = torch.optim.SGD(right.parameters(), lr=0.05)
        opt_a.step()
        opt_b.step()
        for a, b in zip(left.parameters(), right.parameters(), strict=True):
            self.assertTrue(torch.allclose(a, b, rtol=0, atol=1e-6))

    def test_batchnorm_micro_batches_do_not_match(self) -> None:
        torch.manual_seed(1)
        full = torch.randn(4, 2, 8)
        target = torch.randn(4, 2, 8)
        left = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(8, 8))
        right = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(8, 8))
        right.load_state_dict(left.state_dict())
        left.train()
        right.train()
        _step(left, [full], [target])
        _step(right, [full[:2], full[2:]], [target[:2], target[2:]])
        left_weight = left[0].weight.grad
        right_weight = right[0].weight.grad
        assert left_weight is not None and right_weight is not None
        self.assertFalse(torch.allclose(left_weight, right_weight, rtol=0, atol=1e-5))
        self.assertFalse(torch.equal(left[0].running_mean, right[0].running_mean))
