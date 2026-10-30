from collections.abc import Mapping
import torch
from ...core.audio import Enhancer
from ...core.checkpoint import checkpoint_paths
from ...core.config import load_config
from .model import TFGridNet, TFGridNetV3
import torchaudio.functional as AF


def build(config, variant=None):
    if variant in (None, "urgent24"):
        return TFGridNetV3(**load_config(config))
    if variant == "ms-snsd":
        return TFGridNet(**load_config(config))
    raise ValueError(f"unsupported TF-GridNet variant: {variant!r}")


def load(checkpoint, device, config, variant):
    model = build(config, variant)
    path = checkpoint_paths(checkpoint, ('model',))['model']
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, Mapping) or any(not k.startswith('separator.') for k in state):
        raise ValueError('expected an ESPnet state dictionary containing only separator keys')
    model.load_state_dict({k.removeprefix('separator.'): v for k, v in state.items()}, strict=True)
    return TFEnhancer(model.to(device).eval(), torch.device(device), variant)


class TFEnhancer(Enhancer):
    def __init__(self, model, device, variant):
        self.model, self.device, self.variant = model, device, variant

    def _enhance_one(self, waveform, sample_rate):
        if self.variant == 'ms-snsd':
            x = waveform.to(device=self.device, dtype=torch.float32)
            if sample_rate != 16000:
                x = AF.resample(x, sample_rate, 16000)
            if x.numel() <= self.model.n_fft // 2:
                raise ValueError('input too short for the configured STFT')
            if x.std() == 0:
                return torch.zeros_like(waveform)
            with torch.inference_mode():
                out = self.model(x.unsqueeze(0)).squeeze(0)
                if sample_rate != 16000:
                    out = AF.resample(out, 16000, sample_rate)
                out = torch.nn.functional.pad(out[:waveform.numel()], (0, max(0, waveform.numel() - out.numel())))
            return out.to(device=waveform.device, dtype=waveform.dtype)
        # The URGENT configuration scales the 8 kHz STFT to the input rate.
        n_fft, hop = 256 * sample_rate // 8000, 128 * sample_rate // 8000
        if n_fft < 2 or hop < 1 or waveform.numel() <= n_fft // 2:
            raise ValueError('input too short for the configured STFT')
        x = waveform.to(device=self.device, dtype=torch.float32)
        scale = x.std()
        if scale == 0:
            return waveform.clone() if not torch.any(x) else torch.zeros_like(waveform)
        with torch.inference_mode():
            window = torch.hann_window(n_fft, device=self.device)
            spec = torch.stft((x / scale).unsqueeze(0), n_fft, hop, n_fft, window, return_complex=True).transpose(1, 2)
            enhanced = self.model(spec).transpose(1, 2)
            out = torch.istft(enhanced, n_fft, hop, n_fft, window, length=x.numel()).squeeze(0) * scale
        return out.to(device=waveform.device, dtype=waveform.dtype)
