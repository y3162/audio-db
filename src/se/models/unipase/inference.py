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


# Official README sends inputs longer than 20 s through inference_long.py.
_LONG_SECONDS = 20


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
            if audio.numel() > _LONG_SECONDS * sample_rate:
                enhanced = _infer_long(self.model, audio, sample_rate)
            else:
                enhanced = self.model(audio.unsqueeze(0), sr_in=sample_rate, sr_out=sample_rate, enable_plc=True)
                enhanced = enhanced.squeeze(0)
        # inference.py / inference_long.py: y / (max(|y|) + 1e-8) * input_peak
        enhanced = enhanced / (enhanced.abs().max() + 1e-8) * scale
        if enhanced.numel() < audio.numel():
            enhanced = torch.nn.functional.pad(enhanced, (0, audio.numel() - enhanced.numel()))
        return enhanced[:audio.numel()].to(device=original_device, dtype=original_dtype)


def _infer_long(model, audio, sample_rate):
    """8 s windows, 4 s hop, then the hard overlap trim from inference_long.py."""
    segment = sample_rate * 8
    hop = sample_rate * 4
    patches = []
    start = 0
    end_tmp = 0
    total = audio.numel()
    while start < total:
        end = start + segment
        if end > total:
            if end - total < 2 * sample_rate:
                patches.append(audio[start:])
            else:
                patches[-1] = torch.cat((patches[-1], audio[end_tmp:total]))
            break
        patches.append(audio[start:end])
        end_tmp = end
        start += hop
    enhanced = []
    for patch in patches:
        enhanced.append(model(patch.unsqueeze(0), sr_in=sample_rate, sr_out=sample_rate, enable_plc=True).squeeze(0))
    if len(enhanced) == 1:
        return enhanced[0]
    keep = sample_rate * 6
    edge = sample_rate * 2
    pieces = [enhanced[0][:keep]]
    for patch in enhanced[1:-1]:
        pieces.append(patch[edge:keep])
    pieces.append(enhanced[-1][edge:])
    return torch.cat(pieces, dim=0)
