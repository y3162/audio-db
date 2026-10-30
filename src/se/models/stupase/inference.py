"""StuPASE inference. Sampling defaults match the official CLI: 8 steps, CFG 0.5, sway -1."""
import torch
import torchaudio.functional as AF

from ...core.audio import Enhancer
from ...core.checkpoint import checkpoint_paths
from .stupase import StuPASE

_PARTS = ('dewavlm', 'cfm', 'vocoder')


def build(config, variant=None):
    if variant not in (None, 'stupase'):
        raise ValueError(f'unsupported StuPASE variant: {variant!r}')
    paths = checkpoint_paths(config, _PARTS)
    return StuPASE(str(paths['dewavlm']), str(paths['cfm']), str(paths['vocoder']))


def load(checkpoint, device, config, variant):
    model = build(checkpoint if config is None else config, variant)
    return StuPASEEnhancer(model.to(torch.device(device)).eval(), torch.device(device))


class StuPASEEnhancer(Enhancer):
    def __init__(self, model, device):
        self.model, self.device = model, device

    def _enhance_one(self, waveform, sample_rate):
        original_device, original_dtype, length = waveform.device, waveform.dtype, waveform.numel()
        audio = waveform.to(device=self.device, dtype=torch.float32)
        if sample_rate != 16000:
            audio = AF.resample(audio, sample_rate, 16000)
        if not torch.any(audio):
            return waveform.clone()
        scale = audio.abs().max()
        with torch.inference_mode():
            enhanced = self.model(audio.unsqueeze(0), sr=16000, steps=8, cfg_strength=0.5, sway_sampling_coef=-1.0)
        enhanced = enhanced.squeeze(0)
        enhanced = enhanced / enhanced.abs().max().clamp_min(1e-8) * scale
        if sample_rate != 16000:
            enhanced = AF.resample(enhanced, 16000, sample_rate)
        if enhanced.numel() < length:
            enhanced = torch.nn.functional.pad(enhanced, (0, length - enhanced.numel()))
        return enhanced[:length].to(device=original_device, dtype=original_dtype)
