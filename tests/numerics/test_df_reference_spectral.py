"""Compare loss gradients with the optional published DeepFilterNet code."""
import os
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "src/se/official/deepfilternet/DeepFilterNet/df/loss.py"
CONFIG = Path(os.environ.get("DF_REFERENCE_CONFIG", ROOT / "data/checkpoints/deepfilternet3/DeepFilterNet3/config.ini"))


@unittest.skipUnless(REFERENCE.is_file() and CONFIG.is_file(), "official DeepFilterNet source and config required")
class DeepFilterNetReferenceSpectralTest(unittest.TestCase):
    def test_loss_and_gradient_on_distinct_waveforms(self):
        from tests.numerics.four_process.real_models import DeepFilterNet3
        from src.experiments.official.deepfilternet3.reference import _multi_resolution_spectral

        adapter = DeepFilterNet3()
        adapter.configure({})
        adapter.build("official", torch.device("cpu"))
        torch.manual_seed(29)
        signal = torch.randn(1, 4800)
        target = torch.randn(1, 4800)
        official = signal.clone().requires_grad_()
        independent = signal.clone().requires_grad_()
        left = adapter._official_loss["spectral"](official, target)
        right = _multi_resolution_spectral(independent, target)
        left.backward()
        right.backward()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(official.grad, independent.grad, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA required")
    def test_cuda_deterministic_fallback_remains_finite(self):
        from src.experiments.official.deepfilternet3.reference import _multi_resolution_spectral

        torch.manual_seed(29)
        signal = torch.randn(1, 4800, device="cuda", requires_grad=True)
        target = torch.randn(1, 4800, device="cuda")
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            _multi_resolution_spectral(signal, target).backward()
            self.assertTrue(torch.isfinite(signal.grad).all())
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
