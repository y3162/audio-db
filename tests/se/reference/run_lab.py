"""CPU comparisons that the existing lab environment can run without extra packages.

Official file readers (librosa) are not required: the functions below call the
official network and the official STFT helpers on the same tensor the independent
enhancer receives.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.se import load_enhancer
from tests.se.reference.metrics import compare, format_row

OFFICIAL_MP = ROOT / 'src/se/official/mp-senet'
CKPT_MP = ROOT / 'data/checkpoints/mp-senet'
CKPT_GRID = ROOT / 'data/checkpoints/tf-gridnet'


def _load_py(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mp_reference():
    transformer = _load_py('reference_transformer', OFFICIAL_MP / 'models/transformer.py')
    from src.se.models.mp_senet.sigmoid import LearnableSigmoid2d
    names = ['models', 'models.transformer', 'utils', 'pesq', 'joblib', 'librosa']
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
        sys.modules['librosa'] = types.ModuleType('librosa')
        model_cls = _load_py('reference_mp', OFFICIAL_MP / 'models/model.py').MPNet
        dataset = _load_py('reference_dataset', OFFICIAL_MP / 'dataset.py')
        return model_cls, dataset.mag_pha_stft, dataset.mag_pha_istft
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def official_mp_wave(model_cls, stft, istft, config, state, wave):
    model = model_cls(types.SimpleNamespace(**config)).eval()
    model.load_state_dict(state, strict=True)
    norm = torch.sqrt(wave.numel() / wave.square().sum())
    with torch.inference_mode():
        mag, phase, _ = stft((wave * norm).unsqueeze(0), config['n_fft'], config['hop_size'],
                             config['win_size'], config['compress_factor'])
        mag, phase, _ = model(mag, phase)
        return (istft(mag, phase, config['n_fft'], config['hop_size'], config['win_size'],
                      config['compress_factor']).squeeze(0) / norm)


def grid_v3():
    path = ROOT / 'src/se/official/espnet/espnet2/enh/separator/tfgridnetv3_separator.py'
    names = ['espnet2', 'espnet2.enh', 'espnet2.enh.layers', 'espnet2.enh.layers.complex_utils',
             'espnet2.enh.separator', 'espnet2.enh.separator.abs_separator',
             'espnet2.torch_utils', 'espnet2.torch_utils.get_layer_from_string']
    old = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[names[3]].is_complex = torch.is_complex
        sys.modules[names[3]].new_complex_like = lambda unused, pair: torch.complex(*pair)
        sys.modules[names[5]].AbsSeparator = torch.nn.Module
        sys.modules[names[7]].get_layer = lambda n: next(
            getattr(torch.nn, x) for x in dir(torch.nn) if x.lower() == n.lower())
        return _load_py('reference_grid_v3', path).TFGridNetV3
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def grid_v1():
    path = ROOT / 'src/se/official/espnet/espnet2/enh/separator/tfgridnet_separator.py'
    names = ['espnet2', 'espnet2.enh', 'espnet2.enh.decoder', 'espnet2.enh.decoder.stft_decoder',
             'espnet2.enh.encoder', 'espnet2.enh.encoder.stft_encoder', 'espnet2.enh.layers',
             'espnet2.enh.layers.complex_utils', 'espnet2.enh.separator',
             'espnet2.enh.separator.abs_separator', 'espnet2.torch_utils',
             'espnet2.torch_utils.get_layer_from_string']
    old = {name: sys.modules.get(name) for name in names}

    class Encoder(torch.nn.Module):
        def __init__(self, n_fft, win_length, hop, **kwargs):
            super().__init__()
            self.n_fft, self.hop = n_fft, hop

        def forward(self, x, lengths):
            window = torch.hann_window(self.n_fft, device=x.device)
            spec = torch.stft(x.transpose(1, 2).reshape(-1, x.shape[1]), self.n_fft, self.hop,
                              self.n_fft, window, return_complex=True)
            return spec.transpose(1, 2).unsqueeze(2), lengths

    class Decoder(torch.nn.Module):
        def __init__(self, n_fft, win_length, hop, **kwargs):
            super().__init__()
            self.n_fft, self.hop = n_fft, hop

        def forward(self, x, lengths):
            window = torch.hann_window(self.n_fft, device=x.device)
            return torch.istft(x.transpose(1, 2), self.n_fft, self.hop, self.n_fft, window,
                               length=int(lengths.max())), lengths

    try:
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[names[3]].STFTDecoder = Decoder
        sys.modules[names[5]].STFTEncoder = Encoder
        sys.modules[names[7]].new_complex_like = lambda unused, pair: torch.complex(*pair)
        sys.modules[names[9]].AbsSeparator = torch.nn.Module
        sys.modules[names[11]].get_layer = lambda n: next(
            getattr(torch.nn, x) for x in dir(torch.nn) if x.lower() == n.lower())
        return _load_py('reference_grid_v1', path).TFGridNet
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def official_urgent(separator_cls, state, wave, sample_rate):
    model = separator_cls(
        input_dim=0, n_srcs=1, n_imics=1, n_layers=6, lstm_hidden_units=200, attn_n_head=4,
        attn_qk_output_channel=2, emb_dim=48, emb_ks=4, emb_hs=1, activation='prelu', eps=1e-5,
    ).eval()
    model.load_state_dict({key.removeprefix('separator.'): value for key, value in state.items()}, strict=True)
    n_fft, hop = 256 * sample_rate // 8000, 128 * sample_rate // 8000
    scale = wave.std()
    window = torch.hann_window(n_fft)
    with torch.inference_mode():
        spec = torch.stft((wave / scale).unsqueeze(0), n_fft, hop, n_fft, window, return_complex=True)
        spec = spec.transpose(1, 2)
        enhanced, _, _ = model(spec, torch.tensor([spec.shape[1]]))
        out = torch.istft(enhanced[0].transpose(1, 2), n_fft, hop, n_fft, window, length=wave.numel())
    return (out.squeeze(0) * scale)


def make_waves():
    torch.manual_seed(11)
    return {
        'short_1s': torch.randn(16000) * 0.05,
        'mid_5s': torch.randn(16000 * 5) * 0.05,
        'silence': torch.zeros(16000),
        'small': torch.randn(16000) * 1e-4,
        'noisy': torch.sin(2 * torch.pi * 220 * torch.arange(16000) / 16000) * 0.2
                 + torch.randn(16000) * 0.05,
    }


def main():
    torch.set_num_threads(4)
    config = json.loads((CKPT_MP / 'config.json').read_text())
    model_cls, stft, istft = mp_reference()
    waves = make_waves()
    print('== MP-SENet official mag_pha vs enhancer ==')
    for variant, filename in (('vb', 'g_best_vb'), ('dns', 'g_best_dns')):
        state = torch.load(CKPT_MP / filename, map_location='cpu', weights_only=True)['generator']
        enhancer = load_enhancer('mp_senet', CKPT_MP / filename, device='cpu')
        for name, wave in waves.items():
            expected = official_mp_wave(model_cls, stft, istft, config, state, wave)
            actual = enhancer.enhance(wave, 16000)
            print(variant, format_row(name, compare(expected, actual)))
        for rate, count in ((8000, 8000), (48000, 48000)):
            wave = torch.randn(count) * 0.05
            # Official file CLI would resample with librosa. Compare the 16 kHz
            # network on the same resampled tensor the enhancer uses internally.
            resampled = torch.nn.functional.interpolate(wave.view(1, 1, -1), size=int(count * 16000 / rate),
                                                        mode='linear', align_corners=False).view(-1)
            del resampled
            actual = enhancer.enhance(wave, rate)
            print(variant, format_row(f'api_only_{rate}', {'max_abs': float(actual.abs().max()) if actual.numel() else None,
                                                            'mae': float('nan'), 'rmse': float('nan'),
                                                            'relative_mae': float('nan'), 'si_sdr': float('nan'),
                                                            'correlation': float('nan'), 'lag_samples': 0,
                                                            'length_ref': count, 'length_act': actual.numel(),
                                                            'nan': not bool(torch.isfinite(actual).all())}))

    print('== TF-GridNet ==')
    urgent_state = torch.load(
        CKPT_GRID / 'urgent24/exp/enh_train_enh_tfgridnet_raw/21epoch.pth',
        map_location='cpu', weights_only=True)
    separator = grid_v3()
    enhancer = load_enhancer('tf_gridnet', CKPT_GRID / 'urgent24/exp/enh_train_enh_tfgridnet_raw/21epoch.pth',
                             device='cpu', variant='urgent24')
    for rate in (8000, 16000):
        torch.manual_seed(11)
        wave = torch.randn(rate) * 0.05
        expected = official_urgent(separator, urgent_state, wave, rate)
        actual = enhancer.enhance(wave, rate)
        print(format_row(f'urgent24_{rate}', compare(expected, actual)))

    v1_state = torch.load(CKPT_GRID / 'ms-snsd/valid.loss.best.pth', map_location='cpu', weights_only=True)
    v1 = grid_v1()
    model = v1(input_dim=0, n_srcs=1, n_fft=512, stride=256, window='hann', n_imics=1, n_layers=4,
               lstm_hidden_units=128, attn_n_head=4, attn_approx_qk_dim=512, emb_dim=32,
               emb_ks=4, emb_hs=4, activation='prelu', eps=1e-5).eval()
    model.load_state_dict({key.removeprefix('separator.'): value for key, value in v1_state.items()}, strict=True)
    enhancer = load_enhancer('tf_gridnet', CKPT_GRID / 'ms-snsd/valid.loss.best.pth',
                             device='cpu', variant='ms-snsd')
    torch.manual_seed(11)
    wave = torch.randn(16000) * 0.05
    with torch.inference_mode():
        expected, _, _ = model(wave.unsqueeze(0), torch.tensor([wave.numel()]))
    actual = enhancer.enhance(wave, 16000)
    print(format_row('ms-snsd_16k', compare(expected[0].squeeze(0), actual)))


if __name__ == '__main__':
    main()
