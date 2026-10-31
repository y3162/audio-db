"""SNR mixing protocol rms_v1."""

import math
import unittest

import torch

from src.evaluation.audio.mixing import align_noise, mix_snr, rms_snr_db


class MixingTests(unittest.TestCase):
    def test_requested_snrs_are_reproduced(self):
        generator = torch.Generator().manual_seed(3)
        clean = torch.randn(16000, generator=generator)
        noise = torch.randn(16000, generator=generator)
        for snr in (-10, -5, 0, 5):
            mixed = mix_snr(clean, noise, 16000, 16000, snr, seed=7)
            self.assertAlmostEqual(mixed.snr_db_achieved, float(snr), places=3)
            self.assertEqual(mixed.noisy.numel(), clean.numel())
            self.assertFalse(mixed.clean_silent)
            self.assertEqual(mixed.protocol, "rms_v1")

    def test_silence_inside_the_utterance_stays_in_the_mean(self):
        clean = torch.cat((torch.zeros(800), torch.ones(800)))
        noise = torch.ones(1600)
        mixed = mix_snr(clean, noise, 16000, 16000, 5, seed=1)
        self.assertAlmostEqual(mixed.snr_db_achieved, 5.0, places=3)
        self.assertAlmostEqual(rms_snr_db(mixed.clean, mixed.noise), 5.0, places=3)

    def test_short_noise_is_tiled(self):
        clean = torch.ones(100)
        noise = torch.tensor([0.2, -0.4, 0.6, -0.8, 0.3])
        mixed = mix_snr(clean, noise, 16000, 16000, 0, seed=0)
        self.assertEqual(mixed.noisy.numel(), 100)
        self.assertAlmostEqual(mixed.snr_db_achieved, 0.0, places=3)

    def test_seed_changes_the_crop_and_repeats(self):
        clean = torch.randn(100)
        noise = torch.randn(1000)
        left = mix_snr(clean, noise, 16000, 16000, 0, seed=1)
        right = mix_snr(clean, noise, 16000, 16000, 0, seed=2)
        again = mix_snr(clean, noise, 16000, 16000, 0, seed=1)
        self.assertNotEqual(left.noise_start, right.noise_start)
        self.assertTrue(torch.equal(left.noisy, again.noisy))

    def test_resample_before_snr(self):
        clean = torch.randn(16000)
        noise = torch.randn(8000)
        mixed = mix_snr(clean, noise, 16000, 8000, -5, seed=4)
        self.assertEqual(mixed.sample_rate, 16000)
        self.assertEqual(mixed.noisy.numel(), 16000)
        self.assertAlmostEqual(mixed.snr_db_achieved, -5.0, places=2)

    def test_silent_clean_does_not_divide_by_zero(self):
        mixed = mix_snr(torch.zeros(128), torch.randn(128), 16000, 16000, 0, seed=0)
        self.assertTrue(mixed.clean_silent)
        self.assertIsNone(mixed.snr_db_achieved)
        self.assertTrue(torch.equal(mixed.noisy, torch.zeros(128)))

    def test_silent_noise_is_rejected(self):
        with self.assertRaises(ValueError):
            mix_snr(torch.randn(128), torch.zeros(128), 16000, 16000, 0, seed=0)

    def test_non_finite_and_invalid_snr_are_rejected(self):
        clean = torch.randn(32)
        noise = torch.randn(32)
        with self.assertRaises(ValueError):
            mix_snr(torch.tensor([float("nan")]), noise, 16000, 16000, 0, seed=0)
        with self.assertRaises(ValueError):
            mix_snr(clean, noise, 16000, 16000, float("inf"), seed=0)
        with self.assertRaises(ValueError):
            mix_snr(clean, noise, 16000, 16000, float("nan"), seed=0)

    def test_prevent_clip_keeps_snr(self):
        clean = torch.ones(1000)
        noise = torch.ones(1000)
        mixed = mix_snr(clean, noise, 16000, 16000, 0, seed=0, prevent_clip=True)
        self.assertTrue(mixed.peak_scaled)
        self.assertLessEqual(float(mixed.noisy.abs().max()), 0.99 + 1e-5)
        self.assertAlmostEqual(mixed.snr_db_achieved, 0.0, places=3)
        self.assertFalse(mixed.hard_clipped)

    def test_hard_clip_is_flagged_and_snr_is_pre_clip(self):
        clean = torch.ones(200)
        noise = torch.ones(200)
        mixed = mix_snr(clean, noise, 16000, 16000, 0, seed=0, hard_clip=True)
        self.assertTrue(mixed.hard_clipped)
        self.assertLessEqual(float(mixed.noisy.abs().max()), 1.0)
        self.assertAlmostEqual(mixed.snr_db_achieved, 0.0, places=3)

    def test_align_noise_start_is_inside_the_file(self):
        noise, start = align_noise(torch.arange(50), 10, seed=9)
        self.assertEqual(noise.numel(), 10)
        self.assertGreaterEqual(start, 0)
        self.assertLessEqual(start, 40)
        self.assertTrue(math.isfinite(float(noise.sum())))
