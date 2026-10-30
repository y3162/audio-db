import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
from src.se import build_model, load_enhancer

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'src/se/official/espnet/espnet2/enh/separator/tfgridnetv3_separator.py'
CKPT = ROOT / 'data/checkpoints/tf-gridnet/urgent24/exp/enh_train_enh_tfgridnet_raw/21epoch.pth'


def reference():
    """Load the official separator for a comparison, with minimal import stubs."""
    names = ['espnet2', 'espnet2.enh', 'espnet2.enh.layers',
             'espnet2.enh.layers.complex_utils', 'espnet2.enh.separator',
             'espnet2.enh.separator.abs_separator', 'espnet2.torch_utils',
             'espnet2.torch_utils.get_layer_from_string']
    old = {n: sys.modules.get(n) for n in names}
    try:
        for n in names:
            sys.modules[n] = types.ModuleType(n)
        complex_utils = sys.modules[names[3]]
        complex_utils.is_complex = torch.is_complex
        complex_utils.new_complex_like = lambda unused, pair: torch.complex(*pair)
        sys.modules[names[5]].AbsSeparator = torch.nn.Module
        sys.modules[names[7]].get_layer = lambda n: next(getattr(torch.nn, x) for x in dir(torch.nn) if x.lower() == n.lower())
        spec = importlib.util.spec_from_file_location('reference_grid', REF)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TFGridNetV3
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def reference_v1():
    names = ['espnet2', 'espnet2.enh', 'espnet2.enh.decoder', 'espnet2.enh.decoder.stft_decoder',
             'espnet2.enh.encoder', 'espnet2.enh.encoder.stft_encoder', 'espnet2.enh.layers',
             'espnet2.enh.layers.complex_utils', 'espnet2.enh.separator',
             'espnet2.enh.separator.abs_separator', 'espnet2.torch_utils',
             'espnet2.torch_utils.get_layer_from_string']
    old = {n: sys.modules.get(n) for n in names}
    class Encoder(torch.nn.Module):
        def __init__(self, n_fft, win_length, hop, **kwargs):
            super().__init__(); self.n_fft, self.hop = n_fft, hop
        def forward(self, x, lengths):
            w = torch.hann_window(self.n_fft, device=x.device)
            spec = torch.stft(x.transpose(1, 2).reshape(-1, x.shape[1]), self.n_fft, self.hop, self.n_fft, w, return_complex=True)
            return spec.transpose(1, 2).unsqueeze(2), lengths
    class Decoder(torch.nn.Module):
        def __init__(self, n_fft, win_length, hop, **kwargs):
            super().__init__(); self.n_fft, self.hop = n_fft, hop
        def forward(self, x, lengths):
            w = torch.hann_window(self.n_fft, device=x.device)
            return torch.istft(x.transpose(1, 2), self.n_fft, self.hop, self.n_fft, w, length=lengths.max()), lengths
    try:
        for n in names: sys.modules[n] = types.ModuleType(n)
        sys.modules[names[3]].STFTDecoder = Decoder
        sys.modules[names[5]].STFTEncoder = Encoder
        sys.modules[names[7]].new_complex_like = lambda unused, pair: torch.complex(*pair)
        sys.modules[names[9]].AbsSeparator = torch.nn.Module
        sys.modules[names[11]].get_layer = lambda n: next(getattr(torch.nn, x) for x in dir(torch.nn) if x.lower() == n.lower())
        path = REF.with_name('tfgridnet_separator.py')
        spec = importlib.util.spec_from_file_location('reference_grid_v1', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.TFGridNet
    finally:
        for name, value in old.items():
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value


class GridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gradient(self):
        m = build_model('tf_gridnet', {'n_layers': 1, 'emb_dim': 4, 'lstm_hidden_units': 4, 'attn_n_head': 1})
        x = torch.randn(1, 4, 5, dtype=torch.complex64, requires_grad=True)
        m(x).abs().mean().backward()
        self.assertIsNotNone(x.grad)

    @unittest.skipUnless(REF.exists() and CKPT.exists(), 'reference or checkpoint unavailable')
    def test_reference(self):
        ref = reference()(input_dim=0, n_srcs=1, n_layers=6, lstm_hidden_units=200,
                          attn_n_head=4, attn_qk_output_channel=2, emb_dim=48, emb_ks=4, emb_hs=1).eval()
        model = build_model('tf_gridnet').eval()
        state = torch.load(CKPT, map_location='cpu', weights_only=True)
        state = {k.removeprefix('separator.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        ref.load_state_dict(state, strict=True)
        torch.manual_seed(11)
        x = torch.randn(1, 8, 129, dtype=torch.complex64)
        with torch.inference_mode():
            a = ref(x, torch.tensor([8]))[0][0]
            b = model(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    @unittest.skipUnless(CKPT.exists(), 'checkpoint unavailable')
    def test_inference_and_batch(self):
        e = load_enhancer('tf_gridnet', CKPT, variant='urgent24')
        torch.manual_seed(2)
        x = torch.randn(2, 600)
        y, lengths = e.enhance_batch(x, 8000, [600, 512])
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(lengths.tolist(), [600, 512])
        torch.testing.assert_close(y[1, :512], e.enhance(x[1, :512], 8000), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(y).all())

    @unittest.skipUnless(REF.exists() and (ROOT / 'data/checkpoints/tf-gridnet/ms-snsd/valid.loss.best.pth').exists(), 'reference unavailable')
    def test_ms_snsd_reference(self):
        path = ROOT / 'data/checkpoints/tf-gridnet/ms-snsd/valid.loss.best.pth'
        ref = reference_v1()(input_dim=0, n_srcs=1, n_fft=512, stride=256, n_layers=4,
                              lstm_hidden_units=128, attn_n_head=4, attn_approx_qk_dim=512,
                              emb_dim=32, emb_ks=4, emb_hs=4, use_builtin_complex=True).eval()
        model = build_model('tf_gridnet', variant='ms-snsd').eval()
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = {k.removeprefix('separator.'): v for k,v in state.items()}
        ref.load_state_dict(state, strict=True); model.load_state_dict(state, strict=True)
        torch.manual_seed(11)
        x = torch.randn(1, 1001)
        with torch.inference_mode():
            a = ref(x, torch.tensor([x.shape[-1]]))[0][0]
            b = model(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
