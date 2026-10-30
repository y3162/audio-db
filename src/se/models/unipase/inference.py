"""UniPASE inference. The network configuration is stored inside the checkpoints."""
import torch

from ...core.audio import Enhancer
from ...core.checkpoint import checkpoint_paths
from .unipase import UniPASE

_PARTS = ('dewavlm', 'adapter', 'vocoder', 'postnet')


def build(config, variant=None):
    if variant not in (None, 'unipase'):
        raise ValueError(f'unsupported UniPASE variant: {variant!r}')
    paths = _paths(config)
    return UniPASE(
        dewavlm_ckpt_path=str(paths['dewavlm']),
        adapter_ckpt_path=str(paths['adapter']),
        vocoder_ckpt_path=str(paths['vocoder']),
        postnet_ckpt_path=str(paths['postnet']),
    )


def load(checkpoint, device, config, variant):
    paths = _paths(checkpoint if config is None else config)
    # Named checkpoints are the architecture source; an explicit config must match them.
    if config is not None:
        configured = _paths(config)
        if {key: str(value) for key, value in configured.items()} != {key: str(value) for key, value in paths.items()}:
            paths = configured
    model = UniPASE(
        dewavlm_ckpt_path=str(paths['dewavlm']),
        adapter_ckpt_path=str(paths['adapter']),
        vocoder_ckpt_path=str(paths['vocoder']),
        postnet_ckpt_path=str(paths['postnet']),
    )
    return UniPASEEnhancer(model.to(torch.device(device)).eval(), torch.device(device))


def _paths(checkpoint):
    if checkpoint is None:
        raise ValueError('UniPASE requires named checkpoints: dewavlm, adapter, vocoder, postnet')
    return checkpoint_paths(checkpoint, _PARTS)


class UniPASEEnhancer(Enhancer):
    def __init__(self, model, device):
        self.model, self.device = model, device

    def _enhance_one(self, waveform, sample_rate):
        original_device, original_dtype = waveform.device, waveform.dtype
        audio = waveform.to(device=self.device, dtype=torch.float32)
        if not torch.any(audio):
            return waveform.clone()
        scale = audio.abs().max()
        with torch.inference_mode():
            enhanced = self.model(audio.unsqueeze(0), sr_in=sample_rate, sr_out=sample_rate, enable_plc=True)
        enhanced = enhanced.squeeze(0)
        enhanced = enhanced / enhanced.abs().max().clamp_min(1e-8) * scale
        if enhanced.numel() < audio.numel():
            enhanced = torch.nn.functional.pad(enhanced, (0, audio.numel() - enhanced.numel()))
        return enhanced[:audio.numel()].to(device=original_device, dtype=original_dtype)
