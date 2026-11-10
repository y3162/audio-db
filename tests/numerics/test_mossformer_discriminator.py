"""Regression checks for the released-width MossFormerGAN scorer."""
import unittest
import torch
from src.experiments.official.mossformergan.discriminator import Discriminator
from src.experiments.official.mossformergan.reference import _uncompress, _istft

class MossFormerDiscriminatorTests(unittest.TestCase):
    def test_reconstruction_gradient_uses_shared_phase(self):
        torch.manual_seed(1)
        real = torch.randn(1, 1, 201, 21)
        imag = torch.randn_like(real, requires_grad=True)
        ref_imag = imag.detach().clone().requires_grad_()
        reference = torch.complex(real, ref_imag)
        magnitude, phase = reference.abs() ** (1 / .3), reference.angle()
        spectrum = torch.stack([magnitude * phase.cos(), magnitude * phase.sin()], -1)
        audio = _istft(_uncompress(real, imag).squeeze(1), 400, 100, 400, 2000)
        expected = _istft(spectrum.squeeze(1), 400, 100, 400, 2000)
        target = torch.randn(1, 2000)
        (audio[..., :2000] - target).abs().mean().backward()
        (expected[..., :2000] - target).abs().mean().backward()
        self.assertTrue(torch.equal(imag.grad, ref_imag.grad))

    def test_shapes_forward_backward_and_update(self):
        torch.manual_seed(24)
        model = Discriminator(ndf=16)
        self.assertEqual(tuple(model.cnn.conv1.weight.shape), (16, 2, 3, 3))
        self.assertEqual(tuple(model.att[0].attn.fn.to_q.weight.shape), (384, 384))
        self.assertEqual(tuple(model.att[0].conv.net[4].conv.weight.shape), (768, 1, 31))
        self.assertEqual(tuple(model.sigmoid.slope.shape), (1,))
        x = torch.rand(1, 1, 201, 19, requires_grad=True)
        y = torch.rand_like(x, requires_grad=True)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.sigmoid.slope.detach().clone()
        output = model(x, y)
        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertTrue(((output > 0) & (output < 1)).all())
        loss = (output - 1).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        opt.step()
        self.assertFalse(torch.equal(before, model.sigmoid.slope))
