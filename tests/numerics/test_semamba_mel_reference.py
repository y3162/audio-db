"""Compare the SEMamba mel loss with the optional reference source."""
import importlib.util
from pathlib import Path
import unittest

import torch

REFERENCE = Path(__file__).resolve().parents[2] / 'src/se/official/semambapp/model/loss.py'


@unittest.skipUnless(REFERENCE.is_file(), 'official SEMamba source required')
class SEMambaMelReferenceTest(unittest.TestCase):
    def test_loss_and_waveform_gradient(self):
        from src.experiments.calc.spectral import summed_slaney_log_mel

        spec = importlib.util.spec_from_file_location('semamba_reference_loss', REFERENCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(41)
        reference = torch.randn(1, 8000)
        signal = torch.randn(1, 8000)
        a = signal.clone().requires_grad_()
        b = signal.clone().requires_grad_()
        left = module.MultiScaleMelSpectrogramLoss(sampling_rate=16000)(reference[:, None], a[:, None])
        right = summed_slaney_log_mel(reference, b, 16000)
        left.backward()
        right.backward()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
