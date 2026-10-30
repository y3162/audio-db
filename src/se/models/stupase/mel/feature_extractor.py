# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0


import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class Mel_feat(nn.Module):
    def __init__(self, n_fft=1024, hop_length=320, win_length=1024,
                 n_mels=100, sample_rate=16000, f_min=0, f_max=None, power=1.0):
        super().__init__()
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (B, L) or (B, 1, L)
        Returns:
            feat (Tensor): Shape (B, n_mels, T)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        x = x / (x.abs().max() + 1e-8) * 0.9

        feat = self.mel_spec(x)
        feat = torch.log(torch.clamp(feat, min=1e-7))
        
        return feat
