"""SEMamba inference returns the input length, including non-hop multiples."""
import unittest
import torch
from src.se.models.semambapp.inference import _enhance_16k


class IdentitySpectrum(torch.nn.Module):
    def forward(self, magnitude, phase):
        return magnitude, phase


class SemambaLengthTest(unittest.TestCase):
    def test_identity_across_hop_boundaries(self):
        window = torch.hann_window(400)
        for length in (401, 499, 500, 501, 599, 600, 601, 680):
            with self.subTest(length=length):
                x = torch.randn(length, generator=torch.Generator().manual_seed(length))
                spec = torch.stft(x[None], 400, 100, 400, window, center=True, pad_mode='reflect', return_complex=True)
                self.assertEqual(spec.shape[-1], 1 + length // 100)
                official_length = torch.istft(spec, 400, 100, 400, window, center=True).shape[-1]
                self.assertEqual(official_length, 100 * (length // 100))
                actual = _enhance_16k(IdentitySpectrum(), x)
                self.assertEqual(actual.numel(), length)
                torch.testing.assert_close(actual, x, atol=2e-6, rtol=2e-5)
