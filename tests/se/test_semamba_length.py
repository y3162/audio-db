"""SEMamba inference returns the input length, including non-hop multiples."""
import unittest
import os
import torch
from src.se.models.semambapp.inference import _enhance_16k


class IdentitySpectrum(torch.nn.Module):
    def forward(self, magnitude, phase):
        return magnitude, phase


class SemambaLengthTest(unittest.TestCase):
    def test_fused_backend_cpu_fallback_and_checkpoint_keys(self):
        from src.se.models.semambapp.network import SelectiveSSM

        torch.manual_seed(4)
        reference = SelectiveSSM(8, 3, 4, 2)
        fused = SelectiveSSM(8, 3, 4, 2, scan_backend='fused')
        fused.load_state_dict(reference.state_dict(), strict=True)
        x = torch.randn(2, 9, 8)
        torch.testing.assert_close(reference(x), fused(x))
        with self.assertRaises(ValueError):
            SelectiveSSM(8, scan_backend='invalid')
        from src.se.models.semambapp.network import _fused_xz
        projected = _fused_xz(fused, x)
        self.assertEqual(tuple(projected.stride()), (x.shape[1], x.shape[0] * x.shape[1], 1))

    @unittest.skipUnless(os.environ.get('SEMAMBA_TEST_FUSED_CUDA') == '1', 'explicit CUDA extension test')
    def test_fused_scan_forward_backward_cuda(self):
        from src.se.models.semambapp.network import SelectiveSSM

        torch.manual_seed(5)
        reference = SelectiveSSM(8, 3, 4, 2).cuda()
        fused = SelectiveSSM(8, 3, 4, 2, scan_backend='fused').cuda()
        fused.load_state_dict(reference.state_dict(), strict=True)
        x = torch.randn(2, 31, 8, device='cuda')
        a = x.clone().requires_grad_()
        b = x.clone().requires_grad_()
        y1, y2 = reference(a), fused(b)
        torch.testing.assert_close(y1, y2, rtol=2e-4, atol=2e-5)
        y1.square().mean().backward()
        y2.square().mean().backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-5)
        for p, q in zip(reference.parameters(), fused.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-5)

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

    def test_checkpointed_scan_matches_the_eager_scan(self):
        from torch.utils.checkpoint import checkpoint

        from src.se.models.semambapp.network import _selective_scan

        torch.manual_seed(0)
        arguments = tuple(
            value.detach().requires_grad_(True)
            for value in (
                torch.randn(2, 4, 7),
                torch.rand(2, 4, 7),
                -torch.rand(4, 3),
                torch.randn(2, 3, 7),
                torch.randn(2, 3, 7),
                torch.randn(4),
            )
        )
        eager = _selective_scan(*arguments)
        eager.sum().backward()
        eager_grads = [value.grad.detach().clone() for value in arguments]
        for value in arguments:
            value.grad = None
        recomputed = checkpoint(_selective_scan, *arguments, use_reentrant=False)
        recomputed.sum().backward()
        torch.testing.assert_close(recomputed, eager)
        for left, value in zip(eager_grads, arguments, strict=True):
            torch.testing.assert_close(value.grad, left)
