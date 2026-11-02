"""The SEMamba discriminator's analysis must follow the published STFT."""
import importlib.util
from pathlib import Path
import unittest

import torch

REFERENCE = Path(__file__).resolve().parents[2] / 'src/se/official/semambapp/model/discriminator.py'


@unittest.skipUnless(REFERENCE.is_file(), 'official SEMamba source required')
class SEMambaMRDReferenceTest(unittest.TestCase):
    def test_spectrogram_and_waveform_gradient(self):
        from src.training.se.semamba_disc import ResolutionDiscriminator

        spec = importlib.util.spec_from_file_location('semamba_reference_disc', REFERENCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resolution = (1024, 120, 600)
        official = module.DiscriminatorR(resolution)
        independent = ResolutionDiscriminator(resolution)
        torch.manual_seed(43)
        wave = torch.randn(1, 1, 8000)
        a, b = wave.clone().requires_grad_(), wave.clone().requires_grad_()
        x, y = official.spectrogram(a), independent._magnitude(b)
        weight = torch.randn_like(x)
        (x * weight).sum().backward()
        (y * weight).sum().backward()
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
