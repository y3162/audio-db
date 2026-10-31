"""Common-API checks that do not need the official packages."""
import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio.functional as AF

from src.se import load_enhancer

ROOT = Path(__file__).resolve().parents[2]
MP = ROOT / 'data/checkpoints/mp-senet/g_best_vb'
DFN = ROOT / 'data/checkpoints/deepfilternet3/DeepFilterNet3'
MOSS = ROOT / 'data/checkpoints/mossformergan/last_best_checkpoint.pt'
GRID = ROOT / 'data/checkpoints/tf-gridnet/urgent24/exp/enh_train_enh_tfgridnet_raw/21epoch.pth'
SEMAMBA = ROOT / 'data/checkpoints/semambapp/semambapp.pth'


def _finite_same_shape(testcase, enhanced, source):
    testcase.assertEqual(tuple(enhanced.shape), tuple(source.shape))
    testcase.assertEqual(enhanced.dtype, source.dtype)
    testcase.assertTrue(torch.isfinite(enhanced).all())


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rejects_bad_input(self):
        if not MP.exists():
            self.skipTest('checkpoint unavailable')
        enhancer = load_enhancer('mp_senet', {'generator': MP}, variant='vb')
        with self.assertRaises(ValueError):
            enhancer.enhance(torch.tensor([float('nan')]), 16000)
        with self.assertRaises(ValueError):
            enhancer.enhance(torch.randn(2, 800), 16000)

    def test_mp_senet_shapes_rates_and_symlink(self):
        if not MP.exists():
            self.skipTest('checkpoint unavailable')
        torch.manual_seed(4)
        source = torch.randn(1600)
        direct = load_enhancer('mp_senet', {'generator': MP}, variant='vb')
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / 'weights.pt'
            link.symlink_to(MP)
            linked = load_enhancer('mp_senet', {'generator': link}, variant='vb')
        left = direct.enhance(source, 16000)
        right = linked.enhance(source, 16000)
        self.assertEqual(float((left - right).abs().max()), 0.0)
        _finite_same_shape(self, left, source)
        self.assertEqual(float((left - direct.enhance(source, 16000)).abs().max()), 0.0)
        for rate in (8000, 48000):
            wave = torch.randn(rate)
            _finite_same_shape(self, direct.enhance(wave, rate), wave)
        batch, lengths = direct.enhance_batch(torch.stack((source, source)), 16000, [1600, 800])
        self.assertEqual(tuple(batch.shape), (2, 1600))
        self.assertEqual(lengths.tolist(), [1600, 800])
        self.assertTrue(torch.equal(direct.enhance(torch.zeros(800), 16000), torch.zeros(800)))

    @unittest.skipUnless(GRID.exists(), 'checkpoint unavailable')
    def test_tf_gridnet_urgent_keeps_length_at_8_and_16k(self):
        enhancer = load_enhancer('tf_gridnet', GRID, variant='urgent24')
        for rate in (8000, 16000):
            wave = torch.randn(rate // 2)
            _finite_same_shape(self, enhancer.enhance(wave, rate), wave)

    @unittest.skipUnless((DFN / 'checkpoints/model_120.ckpt.best').exists(), 'checkpoint unavailable')
    def test_deepfilternet_16k_matches_explicit_48k_sandwich(self):
        enhancer = load_enhancer(
            'deepfilternet3', DFN / 'checkpoints/model_120.ckpt.best', config=DFN / 'config.ini')
        torch.manual_seed(6)
        wave = torch.randn(16000) * 0.05
        via_api = enhancer.enhance(wave, 16000)
        _finite_same_shape(self, via_api, wave)
        up = AF.resample(wave, 16000, 48000)
        native = enhancer.enhance(up, 48000)
        down = AF.resample(native, 48000, 16000)
        if down.numel() < wave.numel():
            down = torch.nn.functional.pad(down, (0, wave.numel() - down.numel()))
        down = down[:wave.numel()]
        # The API applies the same torchaudio resample around the 48 kHz model.
        self.assertLess(float((via_api - down).abs().max()), 1e-6)
        self.assertEqual(float((via_api - enhancer.enhance(wave, 16000)).abs().max()), 0.0)

    @unittest.skipUnless(MOSS.exists(), 'checkpoint unavailable')
    def test_mossformer_8k_and_repeat(self):
        enhancer = load_enhancer('mossformergan', MOSS)
        wave = torch.randn(8000)
        enhanced = enhancer.enhance(wave, 8000)
        _finite_same_shape(self, enhanced, wave)
        self.assertEqual(float((enhanced - enhancer.enhance(wave, 8000)).abs().max()), 0.0)

    def test_unipase_long_windows_tile_the_input(self):
        from src.se.models.unipase.inference import _infer_long

        class Echo(torch.nn.Module):
            def forward(self, x, sr_in, sr_out=None, enable_plc=True):
                return x

        for seconds in (21, 24, 30):
            audio = torch.arange(seconds * 16000, dtype=torch.float32)
            out = _infer_long(Echo(), audio, 16000)
            self.assertTrue(torch.equal(out, audio))

    @unittest.skipUnless(SEMAMBA.exists(), 'checkpoint unavailable')
    def test_semamba_48k_input_returns_48k_length(self):
        enhancer = load_enhancer('semambapp', SEMAMBA)
        wave = torch.randn(4800)
        _finite_same_shape(self, enhancer.enhance(wave, 48000), wave)
