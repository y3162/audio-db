"""One MP-SENet training step against the official modules.

The official tree is imported only by this test. Training code does not import it.
Errors are the maximum absolute difference. A relative error is recorded when
the reference magnitude is nonzero.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.se import build_model
from src.training.core.rng import seed_model_init, set_determinism
from src.training.se.discriminator import MetricDiscriminator
from src.training.se.losses import discriminator_loss, generator_losses, metric_targets
from src.training.se.stft import mag_pha_istft, mag_pha_stft

OFFICIAL = Path(__file__).resolve().parents[2] / "src" / "se" / "official" / "mp-senet"


def _gap(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    delta = (reference.detach().float() - other.detach().float()).abs()
    maximum = float(delta.max())
    denom = reference.detach().float().abs().clamp_min(1e-8)
    return {"max_abs": maximum, "max_rel": float((delta / denom).max())}


class OfficialStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OFFICIAL / "models" / "model.py").is_file():
            raise unittest.SkipTest("official MP-SENet tree is not present")
        joblib = types.ModuleType("joblib")
        joblib.Parallel = lambda *args, **kwargs: None
        joblib.delayed = lambda fn: fn
        sys.modules.setdefault("joblib", joblib)
        sys.modules.setdefault("librosa", types.ModuleType("librosa"))
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.use = lambda *args, **kwargs: None
        sys.modules.setdefault("matplotlib", matplotlib)
        sys.modules.setdefault("matplotlib.pylab", types.ModuleType("matplotlib.pylab"))
        sys.path.insert(0, str(OFFICIAL))
        set_determinism(True, cuda=False)

    @classmethod
    def tearDownClass(cls):
        set_determinism(False, cuda=False)
        if sys.path and sys.path[0] == str(OFFICIAL):
            sys.path.pop(0)

    def test_one_step_matches_the_official_modules(self):
        from models.discriminator import MetricDiscriminator as OfficialDiscriminator
        from models.model import MPNet

        h = SimpleNamespace(
            dense_channel=4,
            num_tsconformers=1,
            n_fft=64,
            hop_size=16,
            win_size=64,
            compress_factor=0.3,
            beta=2.0,
            sampling_rate=16000,
        )
        torch.manual_seed(7)
        with seed_model_init(7):
            generator = build_model("mp_senet", vars(h)).cpu()
            discriminator = MetricDiscriminator().cpu()
        official_generator = MPNet(h, num_tsblocks=1).cpu()
        official_discriminator = OfficialDiscriminator().cpu()
        official_generator.load_state_dict(generator.state_dict(), strict=True)
        official_discriminator.load_state_dict(discriminator.state_dict(), strict=True)

        torch.manual_seed(11)
        clean = torch.randn(2, 2048)
        noisy = clean * 0.5 + 0.05
        for module in (generator, official_generator, discriminator, official_discriminator):
            module.eval()
        from dataset import mag_pha_istft as official_istft
        from dataset import mag_pha_stft as official_stft
        from models.model import phase_losses

        ours = _step(generator, discriminator, clean, noisy, h, mag_pha_stft, mag_pha_istft, generator_losses)
        official = _official_step(official_generator, official_discriminator, clean, noisy, h, official_stft, official_istft, phase_losses)
        errors = {
            "stft_magnitude": _gap(official["noisy_mag"], ours["noisy_mag"]),
            "generator_magnitude": _gap(official["magnitude"], ours["magnitude"]),
            "generator_phase": _gap(official["phase"], ours["phase"]),
            "discriminator": _gap(official["score"], ours["score"]),
            "loss_abs": abs(float(official["total"].detach()) - float(ours["total"].detach())),
        }
        for name in ("stft_magnitude", "generator_magnitude", "generator_phase", "discriminator"):
            self.assertLess(errors[name]["max_abs"], 1e-5, errors)
        self.assertLess(errors["loss_abs"], 1e-5, errors)
        ours["total"].backward()
        official["total"].backward()
        grad_gap = 0.0
        disc_grad = 0.0
        for left, right in zip(generator.parameters(), official_generator.parameters(), strict=True):
            grad_gap = max(grad_gap, float((left.grad - right.grad).abs().max()))
        for left, right in zip(discriminator.parameters(), official_discriminator.parameters(), strict=True):
            if left.grad is None or right.grad is None:
                continue
            disc_grad = max(disc_grad, float((left.grad - right.grad).abs().max()))
        optim_a = torch.optim.AdamW(generator.parameters(), lr=5e-4, betas=(0.8, 0.99), weight_decay=0.01)
        optim_b = torch.optim.AdamW(official_generator.parameters(), lr=5e-4, betas=(0.8, 0.99), weight_decay=0.01)
        optim_a.step()
        optim_b.step()
        weight_gap = max(float((left - right).abs().max()) for left, right in zip(generator.parameters(), official_generator.parameters(), strict=True))
        errors["generator_grad_max_abs"] = grad_gap
        errors["discriminator_grad_max_abs"] = disc_grad
        errors["weight_max_abs"] = weight_gap
        print("MP-SENet L3", errors)
        self.assertLess(grad_gap, 1e-6, errors)
        # AdamW amplifies a sub-1e-7 gradient difference through the 1e-8 epsilon.
        self.assertLess(weight_gap, 5e-4, errors)
        self.assertTrue(torch.isfinite(ours["disc"]).all())


def _step(generator, discriminator, clean, noisy, h, stft, istft, loss_fn):
    noisy_mag, noisy_phase, _ = stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
    audio = istft(magnitude, phase, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    length = min(clean.shape[-1], audio.shape[-1])
    audio = audio[..., :length]
    clean = clean[..., :length]
    clean_mag, clean_phase, clean_complex = stft(clean, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    hat_mag, _, recomputed = stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    score = discriminator(clean_mag, hat_mag)
    ones = torch.ones(clean.shape[0])
    losses = loss_fn(
        clean_mag, magnitude, clean_phase, phase, clean_complex, complex_spectrum, recomputed, clean, audio, score, ones
    )
    targets = metric_targets(clean.detach(), audio.detach(), h.sampling_rate)
    real = discriminator(clean_mag.detach(), clean_mag.detach())
    fake = discriminator(clean_mag.detach(), hat_mag.detach())
    return {
        "noisy_mag": noisy_mag,
        "magnitude": magnitude,
        "phase": phase,
        "score": score,
        "total": losses["total"],
        "disc": discriminator_loss(real, fake, ones, targets),
    }


def _official_step(generator, discriminator, clean, noisy, h, stft, istft, phase_losses):
    noisy_mag, noisy_phase, _ = stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    magnitude, phase, complex_spectrum = generator(noisy_mag, noisy_phase)
    audio = istft(magnitude, phase, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    length = min(clean.shape[-1], audio.shape[-1])
    audio = audio[..., :length]
    clean = clean[..., :length]
    clean_mag, clean_phase, clean_complex = stft(clean, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    hat_mag, _, recomputed = stft(audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    score = discriminator(clean_mag, hat_mag)
    ones = torch.ones(clean.shape[0])
    phase_loss = sum(phase_losses(clean_phase, phase))
    total = (
        torch.nn.functional.mse_loss(clean_mag, magnitude) * 0.9
        + phase_loss * 0.3
        + torch.nn.functional.mse_loss(clean_complex, complex_spectrum) * 2 * 0.1
        + torch.nn.functional.mse_loss(complex_spectrum, recomputed) * 2 * 0.1
        + torch.nn.functional.mse_loss(score.flatten(), ones) * 0.05
        + torch.nn.functional.l1_loss(clean, audio) * 0.2
    )
    return {"noisy_mag": noisy_mag, "magnitude": magnitude, "phase": phase, "score": score, "total": total}


if __name__ == "__main__":
    unittest.main()
