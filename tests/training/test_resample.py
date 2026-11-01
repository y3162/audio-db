import unittest
from pathlib import Path

import torch

from src.training.data.dataset import _match_rate, _segment
from src.training.data.sampler import epoch_indices
from src.training.errors import TrainingError


class ResampleTests(unittest.TestCase):
    def test_same_rate_is_unchanged_object_values(self) -> None:
        wave = torch.arange(160, dtype=torch.float32)
        out = _match_rate(wave, 16000, 16000, "case")
        self.assertTrue(torch.equal(wave, out))

    def test_48k_to_16k_keeps_duration_and_is_deterministic(self) -> None:
        source = torch.sin(torch.linspace(0, 40, 48000))
        first = _match_rate(source, 48000, 16000, "case")
        second = _match_rate(source, 48000, 16000, "case")
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.numel(), 16000)
        self.assertTrue(torch.isfinite(first).all())
        clean = _match_rate(source * 0.5, 48000, 16000, "case")
        self.assertEqual(first.numel(), clean.numel())

    def test_crop_after_resample_repeats_for_the_same_epoch(self) -> None:
        wave = _match_rate(torch.sin(torch.linspace(0, 20, 48000)), 48000, 16000, "id")
        teacher = wave * 0.25
        a, b = _segment(wave, teacher, "id", seed=7, epoch=3, segment_size=1600)
        c, d = _segment(wave, teacher, "id", seed=7, epoch=3, segment_size=1600)
        self.assertTrue(torch.equal(a, c))
        self.assertTrue(torch.equal(b, d))
        self.assertEqual(a.numel(), 1600)
        resumed = epoch_indices(8, seed=7, epoch=3, world_size=1, rank=0)
        self.assertEqual(resumed, epoch_indices(8, seed=7, epoch=3, world_size=1, rank=0))

    def test_bad_rate(self) -> None:
        with self.assertRaises(TrainingError):
            _match_rate(torch.ones(8), 0, 16000, "case")

    def test_real_voicebank_pair(self) -> None:
        root = Path("data/corpora/VoiceBank-DEMAND")
        clean = root / "clean_trainset_28spk_wav" / "p226_001.wav"
        noisy = root / "noisy_trainset_28spk_wav" / "p226_001.wav"
        if not clean.is_file() or not noisy.is_file():
            self.skipTest("VoiceBank-DEMAND pair is not linked")
        import torchaudio

        clean_wave, clean_rate = torchaudio.load(clean)
        noisy_wave, noisy_rate = torchaudio.load(noisy)
        self.assertEqual(clean_rate, 48000)
        self.assertEqual(noisy_rate, 48000)
        self.assertEqual(clean_wave.shape[-1], noisy_wave.shape[-1])
        clean_16 = _match_rate(clean_wave[0], clean_rate, 16000, "p226_001")
        noisy_16 = _match_rate(noisy_wave[0], noisy_rate, 16000, "p226_001")
        self.assertEqual(clean_16.numel(), noisy_16.numel())
        expected = int(round(clean_wave.shape[-1] * 16000 / 48000))
        self.assertLessEqual(abs(clean_16.numel() - expected), 16)
        cropped, teacher = _segment(noisy_16, clean_16, "p226_001", seed=1, epoch=0, segment_size=16000)
        again, teacher_again = _segment(noisy_16, clean_16, "p226_001", seed=1, epoch=0, segment_size=16000)
        self.assertTrue(torch.equal(cropped, again))
        self.assertTrue(torch.equal(teacher, teacher_again))
        self.assertEqual(cropped.numel(), 16000)
