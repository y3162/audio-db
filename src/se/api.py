"""Speech enhancement model construction and inference."""
from types import SimpleNamespace

import torch
import torchaudio.functional as AF

from .models.mp_senet.model import MPNet
from .core.audio import Enhancer
from .core.checkpoint import checkpoint_paths, load_state
from .core.config import load_config
from .core.registry import ModelSpec, registry

_DEFAULTS = dict(dense_channel=64, n_fft=400, beta=2.0, num_tsconformers=4,
                 sampling_rate=16000, hop_size=100, win_size=400, compress_factor=0.3)


def available_enhancers():
    """Return names of implemented model families."""
    return registry.names()


def model_metadata(name):
    """Return immutable model metadata."""
    return registry.get(name)


def _config(config):
    values = _DEFAULTS.copy()
    values.update(load_config(config))
    return SimpleNamespace(**values)


def _build_mp(config, variant=None):
    h = _config(config)
    return MPNet(h, num_tsblocks=h.num_tsconformers)


def _load_mp(checkpoint, device, config, variant):
    if variant not in (None, 'vb', 'dns'):
        raise ValueError(f'unknown MP-SENet variant: {variant!r}')
    h = _config(config)
    model = _build_mp(vars(h))
    paths = checkpoint_paths(checkpoint, ('generator',))
    load_state(model, paths['generator'], key='generator')
    return _MPEnhancer(model.to(device).eval(), h, torch.device(device))


def build_model(name, config=None, *, variant=None):
    spec = registry.get(name)
    if variant is not None and variant not in spec.variants:
        raise ValueError(f'unknown variant {variant!r} for {name!r}')
    return spec.builder(config, variant)


def load_enhancer(name, checkpoint, device='cpu', config=None, *, variant=None):
    return registry.get(name).loader(checkpoint, device, config, variant)


class _MPEnhancer(Enhancer):
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device

    def _enhance_one(self, waveform, sample_rate):
        h = self.config
        original_device, original_dtype, length = waveform.device, waveform.dtype, waveform.numel()
        x = waveform.to(device=self.device, dtype=torch.float32)
        if sample_rate != h.sampling_rate:
            x = AF.resample(x, sample_rate, h.sampling_rate)
        if not torch.any(x):
            return waveform.clone()
        # torch.stft's reflect padding requires more than n_fft/2 samples.
        if x.numel() <= h.n_fft // 2:
            raise ValueError(f"input is too short for reflect-padded STFT (need > {h.n_fft // 2} samples at {h.sampling_rate} Hz)")
        with torch.inference_mode():
            norm = torch.sqrt(x.numel() / x.square().sum())
            x = (x * norm).unsqueeze(0)
            window = torch.hann_window(h.win_size, device=self.device)
            spec = torch.stft(x, h.n_fft, h.hop_size, h.win_size, window, return_complex=True)
            mag = (torch.view_as_real(spec).pow(2).sum(-1) + 1e-9).sqrt().pow(h.compress_factor)
            phase = torch.atan2(spec.imag + 1e-10, spec.real + 1e-5)
            mag, phase, _ = self.model(mag, phase)
            mag = mag.pow(1.0 / h.compress_factor)
            spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
            output = torch.istft(spec, h.n_fft, h.hop_size, h.win_size, window, length=x.shape[-1]).squeeze(0) / norm
            if sample_rate != h.sampling_rate:
                output = AF.resample(output, h.sampling_rate, sample_rate)
            output = torch.nn.functional.pad(output[:length], (0, max(0, length - output.numel())))
        return output.to(device=original_device, dtype=original_dtype)


registry.register(ModelSpec('mp_senet', _build_mp, _load_mp, 16000,
                            aliases=('mp-senet',), variants=('vb', 'dns')))

from .models.tf_gridnet.inference import build as _build_grid, load as _load_grid
registry.register(ModelSpec('tf_gridnet', _build_grid, _load_grid, 8000,
                            aliases=('tf-gridnet',), variants=('urgent24', 'ms-snsd')))

from .models.deepfilternet3.inference import build as _build_dfn, load as _load_dfn
registry.register(ModelSpec('deepfilternet3', _build_dfn, _load_dfn, 48000,
                            aliases=('deepfilternet', 'dfn3'), variants=('deepfilternet3',)))

from .models.mossformergan.inference import build as _build_moss, load as _load_moss
registry.register(ModelSpec('mossformergan', _build_moss, _load_moss, 16000,
                            aliases=('mossformergan_se_16k',), variants=('se_16k',)))

from .models.semambapp.inference import build as _build_semamba, load as _load_semamba
registry.register(ModelSpec('semambapp', _build_semamba, _load_semamba, 16000,
                            aliases=('semamba++', 'se_mamba_pp'), variants=('semambapp',)))

from .models.unipase.inference import build as _build_unipase, load as _load_unipase
registry.register(ModelSpec('unipase', _build_unipase, _load_unipase, 16000,
                            aliases=('uni-pase',), variants=('unipase',)))

from .models.stupase.inference import build as _build_stupase, load as _load_stupase
registry.register(ModelSpec('stupase', _build_stupase, _load_stupase, 16000,
                            aliases=('stu-pase',), variants=('stupase',)))
