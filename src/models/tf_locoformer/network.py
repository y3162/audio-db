"""Waveform TF-Locoformer for DNS-Interspeech 2020 enhancement.

The separator is the standalone network. STFT settings and variance
normalization follow the ESPnet DNS recipe: Hann window, n_fft 256, hop 128,
and ``model_conf.normalize_variance``. ``torch.std`` keeps PyTorch's default
unbiased estimate, matching ``ESPnetEnhModel``.
"""

from __future__ import annotations

import torch
from torch import nn

from .separator import TFLocoformerSeparator

# Float32 variance of a constant such as 0.2 is a few ulps, not a true signal.
# Ordinary speech is far above this floor. Below it, division is refused.
_VARIANCE_EPS = 1e-6


_DNS_SEPARATOR = dict(
    num_spk=1,
    n_layers=6,
    emb_dim=128,
    norm_type="rmsgroupnorm",
    num_groups=4,
    tf_order="ft",
    n_heads=4,
    flash_attention=False,
    attention_dim=128,
    pos_enc="rope",
    ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
    ffn_hidden_dim=[384, 384],
    conv1d_kernel=4,
    conv1d_shift=1,
    dropout=0.0,
    eps=1.0e-5,
)


class TFLocoformer(nn.Module):
    def __init__(self, n_fft: int = 256, hop_length: int = 128, normalize_variance: bool = True, **separator):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.normalize_variance = bool(normalize_variance)
        options = dict(_DNS_SEPARATOR)
        options.update(separator)
        if int(options["num_spk"]) != 1:
            raise ValueError("speech enhancement requires num_spk=1")
        self.separator = TFLocoformerSeparator(**options)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape (batch, samples)")
        if waveform.shape[-1] <= self.n_fft // 2:
            raise ValueError("input is too short for the configured STFT")
        if self.normalize_variance:
            scale = torch.std(waveform, dim=1, keepdim=True)
            # A constant or all-zero chunk has unbiased std 0. Dividing would
            # put NaN into the optimizer. Inference rejects the same inputs
            # before forward and returns silence.
            if not torch.isfinite(scale).all() or bool((scale < _VARIANCE_EPS).any().item()):
                raise ValueError("mixture standard deviation is zero; refusing variance normalization")
        else:
            scale = waveform.new_ones((waveform.shape[0], 1))
        window = torch.hann_window(self.n_fft, device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform / scale,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window,
            center=True,
            return_complex=True,
        ).transpose(1, 2)
        enhanced = self.separator(spectrum)
        if enhanced.shape[1] != 1:
            raise ValueError("speech enhancement returns one speaker")
        restored = torch.istft(
            enhanced[:, 0].transpose(1, 2),
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window,
            center=True,
            length=waveform.shape[-1],
        )
        return restored * scale
