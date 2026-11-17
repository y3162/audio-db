"""SEMamba analysis STFT joint backward matches centered ``torch.stft``.

The reflect pad is an index reflection so strict deterministic CUDA does not
call ``reflection_pad1d_backward``. Its backward still has to add the reflected
edges before they join other waveform losses. A split ``slice`` / ``flip`` /
``cat`` graph does not.
"""

from __future__ import annotations

import unittest

import torch

from src.experiments.calc.stft import semamba_mag_phase_stft


def _centered(waveform: torch.Tensor, addeps: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.hann_window(400, dtype=waveform.dtype).to(waveform.device)
    spectrum = torch.stft(
        waveform,
        400,
        hop_length=100,
        win_length=400,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    if addeps:
        epsilon = 1e-10
        magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + epsilon)
        phase = torch.atan2(spectrum.imag + epsilon, spectrum.real + epsilon)
    else:
        magnitude = spectrum.abs()
        phase = spectrum.angle()
    magnitude = magnitude.pow(0.3)
    compressed = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    return magnitude, phase, compressed


def _joint_grad(function, waveform: torch.Tensor, addeps: bool) -> torch.Tensor:
    value = waveform.detach().requires_grad_(True)
    magnitude, phase, compressed = function(value, addeps)
    (
        magnitude.square().mean()
        + phase.square().mean()
        + compressed.square().mean()
        + value.square().mean()
    ).backward()
    return value.grad.detach()


class SemambaStftJointGradTest(unittest.TestCase):
    def test_forward_and_joint_gradient_match_centered_stft(self) -> None:
        for seed, length, addeps in (
            (23, 8000, False),
            (23, 8000, True),
            (99, 4000, False),
            (7, 48000, True),
        ):
            with self.subTest(seed=seed, length=length, addeps=addeps):
                torch.manual_seed(seed)
                waveform = torch.randn(1, length)
                left = semamba_mag_phase_stft(waveform, 400, 100, 400, 0.3, addeps=addeps)
                right = _centered(waveform, addeps)
                for item, other in zip(left, right, strict=True):
                    self.assertTrue(torch.equal(item, other))
                audio = _joint_grad(lambda value, flag: semamba_mag_phase_stft(value, 400, 100, 400, 0.3, addeps=flag), waveform, addeps)
                official = _joint_grad(_centered, waveform, addeps)
                self.assertTrue(torch.equal(audio, official))

    def test_strict_deterministic_joint_gradient_matches(self) -> None:
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            torch.manual_seed(23)
            waveform = torch.randn(1, 8000)
            audio = _joint_grad(
                lambda value, flag: semamba_mag_phase_stft(value, 400, 100, 400, 0.3, addeps=flag),
                waveform,
                False,
            )
            official = _joint_grad(_centered, waveform, False)
            self.assertTrue(torch.equal(audio, official))
        finally:
            torch.use_deterministic_algorithms(previous)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SemambaStftJointGradCudaTest(unittest.TestCase):
    def test_strict_deterministic_cuda_matches_centered_stft(self) -> None:
        device = torch.device("cuda:0")
        torch.manual_seed(23)
        waveform = torch.randn(1, 8000, device=device)
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            audio = _joint_grad(
                lambda value, flag: semamba_mag_phase_stft(value, 400, 100, 400, 0.3, addeps=flag),
                waveform,
                True,
            )
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            official = _joint_grad(_centered, waveform, True)
        finally:
            torch.use_deterministic_algorithms(previous)
        self.assertTrue(torch.equal(audio, official))


if __name__ == "__main__":
    unittest.main()
