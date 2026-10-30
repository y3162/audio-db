"""SEMamba++ inference at 16 kHz."""
import torch
import torchaudio.functional as AF

from ...core.audio import Enhancer
from ...core.checkpoint import checkpoint_paths
from ...core.config import load_config
from .network import SEMambaPP

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 100
WIN = 400
COMPRESS = 0.3

_DEFAULTS = dict(hid_feature=48, num_tfmamba=4, d_state=16, d_conv=4, expand=2,
                 norm_epsilon=1e-5, input_channel=2, output_channel=1, n_fft=N_FFT)


def build(config, variant=None):
    if variant not in (None, 'semambapp'):
        raise ValueError(f'unsupported SEMamba++ variant: {variant!r}')
    values = dict(_DEFAULTS)
    raw = load_config(config)
    unknown = set(raw) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f'unknown SEMamba++ config keys: {sorted(unknown)}')
    values.update(raw)
    return SEMambaPP(values)


def load(checkpoint, device, config, variant):
    model = build(config, variant)
    path = checkpoint_paths(checkpoint, ('generator',))['generator']
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'generator' in state and isinstance(state['generator'], dict):
        state = state['generator']
    model.load_state_dict(state, strict=True)
    device = torch.device(device)
    return SEMambaEnhancer(model.to(device).eval(), device)


class SEMambaEnhancer(Enhancer):
    def __init__(self, model, device):
        self.model, self.device = model, device

    def _enhance_one(self, waveform, sample_rate):
        original_device, original_dtype, length = waveform.device, waveform.dtype, waveform.numel()
        audio = waveform.to(device=self.device, dtype=torch.float32)
        if sample_rate != SAMPLE_RATE:
            audio = AF.resample(audio, sample_rate, SAMPLE_RATE)
        if not torch.any(audio):
            return waveform.clone()
        if audio.numel() <= N_FFT // 2:
            raise ValueError(f'input is too short for reflect-padded STFT (need > {N_FFT // 2} samples at {SAMPLE_RATE} Hz)')
        peak = audio.abs().max().clamp_min(1e-9)
        with torch.inference_mode():
            enhanced = _enhance_16k(self.model, audio / peak) * peak
        if sample_rate != SAMPLE_RATE:
            enhanced = AF.resample(enhanced, SAMPLE_RATE, sample_rate)
        if enhanced.numel() < length:
            enhanced = torch.nn.functional.pad(enhanced, (0, length - enhanced.numel()))
        return enhanced[:length].to(device=original_device, dtype=original_dtype)


def _enhance_16k(model, audio):
    window = torch.hann_window(WIN, device=audio.device)
    spec = torch.stft(audio.view(1, -1), N_FFT, HOP, WIN, window, center=True, pad_mode='reflect', return_complex=True)
    magnitude = spec.abs().pow(COMPRESS)
    phase = spec.angle()
    magnitude, phase = model(magnitude, phase)
    restored = torch.complex(magnitude.pow(1 / COMPRESS) * torch.cos(phase), magnitude.pow(1 / COMPRESS) * torch.sin(phase))
    return torch.istft(restored, N_FFT, HOP, WIN, window, center=True, length=audio.numel()).squeeze(0)
