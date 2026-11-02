"""Compare the checked-out MP-SENet train.py loss statements on fixed tensors."""
import unittest
from pathlib import Path

import torch


@unittest.skipUnless((Path(__file__).resolve().parents[2] / "src/se/official/mp-senet/train.py").exists(),
                     "MP-SENet official checkout unavailable")
class OfficialLossPathTest(unittest.TestCase):
    def test_loss_values_and_gradients(self):
        from tests.training.reference.compare_mp_senet import (
            _components, _official_components, _official_discriminator_loss,
        )
        from src.training.se.losses import discriminator_loss

        torch.manual_seed(4)
        parts = {key: torch.randn(2, 8, 9, requires_grad=True) for key in
                 ("clean_mag", "magnitude", "clean_phase", "phase")}
        parts.update({key: torch.randn(2, 8, 9, 2, requires_grad=True) for key in
                      ("clean_complex", "complex", "recomputed")})
        parts["audio"] = torch.randn(2, 128, requires_grad=True)
        clean = torch.randn(2, 128)
        score = torch.randn(2, 1, requires_grad=True)
        ones = torch.ones(2)
        official = _official_components(parts, score, ones, clean)
        independent = _components(parts, score, ones, clean)
        for key in official:
            torch.testing.assert_close(official[key], independent[key], rtol=0, atol=0)
        inputs = (parts["magnitude"], parts["phase"], parts["complex"],
                  parts["recomputed"], parts["audio"], score)
        a = torch.autograd.grad(official["total"], inputs, retain_graph=True)
        b = torch.autograd.grad(independent["total"], inputs)
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for target in (None, torch.rand(2)):
            real, fake = torch.randn(2, 1), torch.randn(2, 1)
            torch.testing.assert_close(_official_discriminator_loss(real, fake, ones, target),
                                       discriminator_loss(real, fake, ones, target), rtol=0, atol=0)
