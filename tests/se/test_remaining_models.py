"""Load and short-audio checks for DeepFilterNet3, MossFormerGAN, and SEMamba++."""
import unittest
from pathlib import Path

import torch

from src.se import available_enhancers, load_enhancer

ROOT = Path(__file__).resolve().parents[2]
DFN = ROOT / 'data/checkpoints/deepfilternet3/DeepFilterNet3'
MOSS = ROOT / 'data/checkpoints/mossformergan/last_best_checkpoint.pt'
SEMAMBA = ROOT / 'data/checkpoints/semambapp/semambapp.pth'


def _ok(wave):
    return wave.ndim == 1 and torch.isfinite(wave).all()


class RemainingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registry_lists_implemented_families(self):
        names = set(available_enhancers())
        self.assertEqual({'mp_senet', 'tf_gridnet', 'tf_locoformer', 'deepfilternet3', 'mossformergan', 'semambapp'}, names)

    @unittest.skipUnless((DFN / 'checkpoints/model_120.ckpt.best').exists(), 'checkpoint unavailable')
    def test_deepfilternet3(self):
        enhancer = load_enhancer('deepfilternet3', DFN / 'checkpoints/model_120.ckpt.best', config=DFN / 'config.ini')
        torch.manual_seed(3)
        noisy = torch.randn(4800) * 0.05
        first = enhancer.enhance(noisy, 48000)
        second = enhancer.enhance(noisy, 48000)
        self.assertTrue(_ok(first) and first.shape == noisy.shape)
        self.assertEqual(float((first - second).abs().max()), 0.0)
        batch, lengths = enhancer.enhance_batch(torch.stack((noisy, noisy[:2400].new_zeros(4800))), 48000, [4800, 2400])
        self.assertEqual(tuple(batch.shape), (2, 4800))
        self.assertTrue(torch.isfinite(batch).all())
        self.assertEqual(lengths.tolist(), [4800, 2400])
        with self.assertRaises(ValueError):
            load_enhancer('deepfilternet3', {'model': DFN / 'checkpoints/model_120.ckpt.best', 'extra': DFN / 'config.ini'})

    @unittest.skipUnless(MOSS.exists(), 'checkpoint unavailable')
    def test_mossformergan(self):
        enhancer = load_enhancer('mossformergan', MOSS)
        torch.manual_seed(3)
        noisy = torch.randn(1600) * 0.05
        enhanced = enhancer.enhance(noisy, 16000)
        self.assertTrue(_ok(enhanced) and enhanced.shape == noisy.shape)
        self.assertEqual(float((enhanced - enhancer.enhance(noisy, 16000)).abs().max()), 0.0)
        self.assertTrue(torch.equal(enhancer.enhance(torch.zeros(800), 16000), torch.zeros(800)))

    @unittest.skipUnless(SEMAMBA.exists(), 'checkpoint unavailable')
    def test_semambapp(self):
        enhancer = load_enhancer('semambapp', SEMAMBA)
        torch.manual_seed(3)
        noisy = torch.randn(1200) * 0.05
        enhanced = enhancer.enhance(noisy, 16000)
        self.assertTrue(_ok(enhanced) and enhanced.shape == noisy.shape)
        self.assertEqual(float((enhanced - enhancer.enhance(noisy, 16000)).abs().max()), 0.0)
