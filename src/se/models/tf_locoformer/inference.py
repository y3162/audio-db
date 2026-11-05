"""TF-Locoformer construction and inference."""

from collections.abc import Mapping

import torch
import torchaudio.functional as AF

from ...core.audio import Enhancer
from ...core.checkpoint import checkpoint_paths
from ...core.config import load_config
from .model import TFLocoformer


def build(config, variant=None):
    if variant not in (None, "dns"):
        raise ValueError(f"unsupported TF-Locoformer variant: {variant!r}")
    return TFLocoformer(**load_config(config))


def load(checkpoint, device, config, variant):
    model = build(config, variant)
    path = checkpoint_paths(checkpoint, ("model",))["model"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("expected a state dictionary")
    if state and all(isinstance(key, str) and key.startswith("separator.") for key in state):
        state = {key.removeprefix("separator."): value for key, value in state.items()}
    model.separator.load_state_dict(state, strict=True)
    return TFLocoformerEnhancer(model.to(device).eval(), torch.device(device))


class TFLocoformerEnhancer(Enhancer):
    def __init__(self, model, device):
        self.model, self.device = model, device

    def _enhance_one(self, waveform, sample_rate):
        x = waveform.to(device=self.device, dtype=torch.float32)
        if sample_rate != 16000:
            x = AF.resample(x, sample_rate, 16000)
        if x.numel() <= self.model.n_fft // 2:
            raise ValueError("input too short for the configured STFT")
        if not torch.any(x) or torch.std(x) == 0:
            return torch.zeros_like(waveform)
        with torch.inference_mode():
            out = self.model(x.unsqueeze(0)).squeeze(0)
            if sample_rate != 16000:
                out = AF.resample(out, 16000, sample_rate)
            out = torch.nn.functional.pad(out[: waveform.numel()], (0, max(0, waveform.numel() - out.numel())))
        return out.to(device=waveform.device, dtype=waveform.dtype)
