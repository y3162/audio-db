"""MossFormerGAN_SE_16K inference."""
import math

import torch
import torchaudio.functional as AF

from ...core.audio import Enhancer
from ...core.checkpoint import checkpoint_paths
from ...core.config import load_config
from .generator import SyncANet

SAMPLE_RATE = 16000
# Released MossFormerGAN_SE_16K weights use 201 one-sided bins (n_fft 400), not the CLI default 512.
N_FFT = 400
WIN_LEN = 400
HOP = 100
WINDOW = 'hamming'


def build(config, variant=None):
    if variant not in (None, 'se_16k'):
        raise ValueError(f'unsupported MossFormerGAN variant: {variant!r}')
    values = load_config(config)
    unknown = set(values) - {'n_fft', 'win_len', 'hop', 'num_channel'}
    if unknown:
        raise ValueError(f'unknown MossFormerGAN config keys: {sorted(unknown)}')
    n_fft = int(values.get('n_fft', N_FFT))
    channels = int(values.get('num_channel', 64))
    model = SyncANet(num_channel=channels, num_features=n_fft // 2 + 1)
    model.n_fft = n_fft
    return model


def load(checkpoint, device, config, variant):
    model = build(config, variant)
    path = checkpoint_paths(checkpoint, ('model',))['model']
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    model.load_state_dict(state, strict=True)
    device = torch.device(device)
    return MossFormerEnhancer(model.to(device).eval(), device, model)


class MossFormerEnhancer(Enhancer):
    def __init__(self, model, device, graph):
        self.model = model
        self.device = device
        self.n_fft = getattr(graph, 'n_fft', N_FFT)
        self.win_len = WIN_LEN
        self.hop = HOP

    def _enhance_one(self, waveform, sample_rate):
        original_device, original_dtype, length = waveform.device, waveform.dtype, waveform.numel()
        audio = waveform.to(device=self.device, dtype=torch.float32)
        if sample_rate != SAMPLE_RATE:
            audio = AF.resample(audio, sample_rate, SAMPLE_RATE)
        if not torch.any(audio):
            return waveform.clone()
        if audio.numel() <= self.n_fft // 2:
            raise ValueError(f'input is too short for centered STFT (need > {self.n_fft // 2} samples at {SAMPLE_RATE} Hz)')
        with torch.inference_mode():
            enhanced = _enhance_16k(self.model, audio, self.n_fft, self.win_len, self.hop)
        if sample_rate != SAMPLE_RATE:
            enhanced = AF.resample(enhanced, SAMPLE_RATE, sample_rate)
        if enhanced.numel() < length:
            enhanced = torch.nn.functional.pad(enhanced, (0, length - enhanced.numel()))
        return enhanced[:length].to(device=original_device, dtype=original_dtype)


def _enhance_16k(model, audio, n_fft, win_len, hop):
    """Match ClearerVoice's non-segment MossFormerGAN_SE_16K decode."""
    original = audio.shape[-1]
    norm = torch.sqrt(audio.new_full((), original / audio.square().sum().clamp_min(1e-12)))
    frames = math.ceil(original / hop)
    padded = frames * hop
    if padded > original:
        audio = torch.cat((audio, audio[:padded - original]))
    window = torch.hamming_window(win_len, periodic=True, device=audio.device)
    spec = torch.view_as_real(torch.stft((audio * norm).view(1, -1), n_fft, hop, win_len, window=window, center=True, return_complex=True))
    compressed = _compress(spec).permute(0, 1, 3, 2)
    real, imag = model(compressed)
    restored = _uncompress(real.permute(0, 1, 3, 2), imag.permute(0, 1, 3, 2)).squeeze(1)
    waveform = torch.istft(torch.view_as_complex(restored), n_fft, hop, win_len, window=window, center=True)
    return (waveform / norm).squeeze(0)[:original]


def _compress(spec):
    value = torch.complex(spec[..., 0], spec[..., 1])
    magnitude = value.abs().pow(0.3)
    return torch.stack((magnitude * torch.cos(value.angle()), magnitude * torch.sin(value.angle())), 1)


def _uncompress(real, imag):
    value = torch.complex(real, imag)
    magnitude = value.abs().pow(1.0 / 0.3)
    return torch.stack((magnitude * torch.cos(value.angle()), magnitude * torch.sin(value.angle())), -1)
