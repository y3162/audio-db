import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch

from src.se import available_enhancers, build_model, load_enhancer

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / 'data/checkpoints/mp-senet'
OFFICIAL = ROOT / 'src/se/official/mp-senet'


def reference_model():
    """Load the reference only in this optional comparison test."""
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    transformer = load('reference_transformer', OFFICIAL / 'models/transformer.py')
    from src.se.models.mp_senet.sigmoid import LearnableSigmoid2d
    names = ['models', 'models.transformer', 'utils', 'pesq', 'joblib']
    previous = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules['models'] = types.ModuleType('models')
        sys.modules['models.transformer'] = transformer
        utils = types.ModuleType('utils')
        utils.LearnableSigmoid2d = LearnableSigmoid2d
        sys.modules['utils'] = utils
        pesq = types.ModuleType('pesq')
        pesq.pesq = None
        sys.modules['pesq'] = pesq
        joblib = types.ModuleType('joblib')
        joblib.Parallel = joblib.delayed = None
        sys.modules['joblib'] = joblib
        return load('reference_mp', OFFICIAL / 'models/model.py').MPNet
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class MPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_build_and_gradient(self):
        self.assertIn('mp_senet', available_enhancers())
        model = build_model('mp_senet', {'dense_channel': 8, 'n_fft': 16, 'num_tsconformers': 1})
        x = torch.rand(1, 9, 4, requires_grad=True)
        model(x, x)[0].mean().backward()
        self.assertIsNotNone(x.grad)

    @unittest.skipUnless((CHECKPOINTS / 'g_best_vb').exists(), 'checkpoint unavailable')
    def test_checkpoint_and_inference(self):
        enhancer = load_enhancer('mp_senet', CHECKPOINTS / 'g_best_vb')
        torch.manual_seed(7)
        x = torch.randn(1001) * 0.05
        y = enhancer.enhance(x, 16000)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.equal(enhancer.enhance(torch.zeros(500), 16000), torch.zeros(500)))
        self.assertEqual(enhancer.enhance(torch.randn(800), 8000).shape, (800,))

    @unittest.skipUnless((OFFICIAL / 'models/model.py').exists() and (CHECKPOINTS / 'g_best_vb').exists(), 'reference unavailable')
    def test_official_generator_parity(self):
        config = json.loads((CHECKPOINTS / 'config.json').read_text())
        ref = reference_model()(types.SimpleNamespace(**config)).eval()
        model = build_model('mp_senet', config).eval()
        state = torch.load(CHECKPOINTS / 'g_best_vb', map_location='cpu', weights_only=True)['generator']
        ref.load_state_dict(state, strict=True)
        model.load_state_dict(state, strict=True)
        torch.manual_seed(7)
        mag, phase = torch.rand(1, 201, 8), torch.randn(1, 201, 8)
        with torch.inference_mode():
            expected, actual = ref(mag, phase), model(mag, phase)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    @unittest.skipUnless((OFFICIAL / 'models/model.py').exists() and (CHECKPOINTS / 'g_best_dns').exists(), 'reference unavailable')
    def test_official_waveform_parity_both_checkpoints(self):
        config = json.loads((CHECKPOINTS / 'config.json').read_text())
        torch.manual_seed(7)
        x = torch.randn(1000) * 0.05
        for variant in ('vb', 'dns'):
            with self.subTest(variant=variant):
                checkpoint = CHECKPOINTS / ('g_best_' + variant)
                ref = reference_model()(types.SimpleNamespace(**config)).eval()
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)['generator']
                ref.load_state_dict(state, strict=True)
                enhancer = load_enhancer('mp_senet', checkpoint)
                with torch.inference_mode():
                    norm = torch.sqrt(len(x) / torch.sum(x ** 2.0))
                    wav = (x * norm).unsqueeze(0)
                    window = torch.hann_window(config['win_size'])
                    spec = torch.view_as_real(torch.stft(wav, config['n_fft'], config['hop_size'], config['win_size'], window, return_complex=True))
                    mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9).pow(config['compress_factor'])
                    phase = torch.atan2(spec[..., 1] + 1e-10, spec[..., 0] + 1e-5)
                    mag, phase, _ = ref(mag, phase)
                    mag = mag.pow(1.0 / config['compress_factor'])
                    spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
                    expected = torch.istft(spec, config['n_fft'], config['hop_size'], config['win_size'], window).squeeze(0) / norm
                    actual = enhancer.enhance(x, 16000)
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
