"""TF-GridNet v3 spectral network; no ESPnet runtime dependency."""
import torch
from torch import nn
from .blocks_v3 import GridNetV3Block


class TFGridNetV3(nn.Module):
    def __init__(self, n_srcs=1, n_layers=6, lstm_hidden_units=200,
                 attn_n_head=4, attn_qk_output_channel=2, emb_dim=48,
                 emb_ks=4, emb_hs=1, activation='prelu', eps=1e-5):
        super().__init__()
        if n_srcs != 1:
            raise ValueError('speech enhancement requires n_srcs=1')
        self.conv = nn.Sequential(nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
                                  nn.GroupNorm(1, emb_dim, eps=eps))
        self.blocks = nn.ModuleList(GridNetV3Block(
            emb_dim, emb_ks, emb_hs, lstm_hidden_units, n_head=attn_n_head,
            qk_output_channel=attn_qk_output_channel, activation=activation, eps=eps)
            for _ in range(n_layers))
        self.deconv = nn.ConvTranspose2d(emb_dim, 2, (3, 3), padding=(1, 1))

    def forward(self, spectrum):
        """Map complex spectra with shape (batch, frames, frequencies)."""
        if not torch.is_complex(spectrum) or spectrum.ndim != 3:
            raise ValueError('spectrum must be complex (batch, frames, frequencies)')
        x = torch.stack((spectrum.real, spectrum.imag), dim=1)
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.deconv(x)
        return torch.complex(x[:, 0], x[:, 1])


class TFGridNet(nn.Module):
    """The waveform-domain, single-source TF-GridNet used by MS-SNSD."""
    def __init__(self, n_fft=512, stride=256, n_layers=4, lstm_hidden_units=128,
                 attn_n_head=4, attn_approx_qk_dim=512, emb_dim=32,
                 emb_ks=4, emb_hs=4, activation='prelu', eps=1e-5):
        from .blocks_v1 import GridNetBlock
        super().__init__()
        self.n_fft, self.stride = n_fft, stride
        self.conv = nn.Sequential(nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
                                  nn.GroupNorm(1, emb_dim, eps=eps))
        self.blocks = nn.ModuleList(GridNetBlock(
            emb_dim, emb_ks, emb_hs, n_fft // 2 + 1, lstm_hidden_units,
            n_head=attn_n_head, approx_qk_dim=attn_approx_qk_dim,
            activation=activation, eps=eps) for _ in range(n_layers))
        self.deconv = nn.ConvTranspose2d(emb_dim, 2, (3, 3), padding=(1, 1))

    def forward(self, waveform):
        if waveform.ndim != 2:
            raise ValueError('waveform must have shape (batch, samples)')
        scale = waveform.std(dim=1, keepdim=True)
        window = torch.hann_window(self.n_fft, device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(waveform / scale, self.n_fft, self.stride, self.n_fft,
                              window, return_complex=True).transpose(1, 2)
        x = self.conv(torch.stack((spectrum.real, spectrum.imag), dim=1))
        for block in self.blocks:
            x = block(x)
        x = self.deconv(x)
        spectrum = torch.complex(x[:, 0], x[:, 1]).transpose(1, 2)
        x = torch.istft(spectrum, self.n_fft, self.stride, self.n_fft, window, length=waveform.shape[-1])
        return x * scale
