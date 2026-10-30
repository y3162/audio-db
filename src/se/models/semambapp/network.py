"""SEMamba++ generator reconstructed from the published configuration and checkpoint layout."""
import math

import torch
from torch import nn
import torch.nn.functional as F


def _padding(kernel, dilation=1):
    return int((kernel * dilation - dilation) / 2)


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SelectiveSSM(nn.Module):
    """Mamba mixer using the reference selective scan, not the CUDA kernel."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1, groups=inner)
        self.x_proj = nn.Linear(inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        self.A_log = nn.Parameter(torch.empty(inner, d_state))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        x, z = self.in_proj(hidden).chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :length]
        x = F.silu(x).transpose(1, 2)
        delta, b_mat, c_mat = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.linear(delta, self.dt_proj.weight)
        delta = F.softplus(delta + self.dt_proj.bias).transpose(1, 2)
        state_a = -torch.exp(self.A_log.float())
        scan = _selective_scan(x.transpose(1, 2).float(), delta.float(), state_a, b_mat.transpose(1, 2).float(), c_mat.transpose(1, 2).float(), self.D.float())
        mixed = scan * F.silu(z.transpose(1, 2).float())
        return self.out_proj(mixed.transpose(1, 2).type_as(hidden))


def _selective_scan(u, delta, a_matrix, b_mat, c_mat, skip):
    batch, dim, length = u.shape
    nstate = a_matrix.shape[1]
    delta_a = torch.exp(delta.unsqueeze(-1) * a_matrix.view(1, dim, 1, nstate))
    delta_b_u = delta.unsqueeze(-1) * b_mat.permute(0, 2, 1).unsqueeze(1) * u.unsqueeze(-1)
    state = u.new_zeros(batch, dim, nstate)
    outputs = []
    for index in range(length):
        state = delta_a[:, :, index] * state + delta_b_u[:, :, index]
        outputs.append(torch.einsum('bdn,bn->bd', state, c_mat[:, :, index]))
    return torch.stack(outputs, dim=-1) + u * skip.view(1, -1, 1)


class MambaResidual(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, eps):
        super().__init__()
        self.mixer = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.norm = RMSNorm(d_model, eps)

    def forward(self, hidden, residual):
        residual = hidden if residual is None else hidden + residual
        return self.mixer(self.norm(residual)), residual


class BidirectionalMamba(nn.Module):
    def __init__(self, channels, d_state, d_conv, expand, eps):
        super().__init__()
        self.forward_blocks = nn.ModuleList([MambaResidual(channels, d_state, d_conv, expand, eps)])
        self.backward_blocks = nn.ModuleList([MambaResidual(channels, d_state, d_conv, expand, eps)])

    def forward(self, x):
        forward, residual = x, None
        for block in self.forward_blocks:
            forward, residual = block(forward, residual)
        forward = forward + residual
        backward, residual = torch.flip(x, [1]), None
        for block in self.backward_blocks:
            backward, residual = block(backward, residual)
        backward = torch.flip(backward + residual, [1])
        return torch.cat((forward, backward), dim=-1)


class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.25):
        super().__init__()
        periodic = int(output_dim * p_ratio)
        gated = output_dim - 2 * periodic
        self.input_linear_p = nn.Linear(input_dim, periodic)
        self.input_linear_g = nn.Linear(input_dim, gated)

    def forward(self, src):
        periodic = self.input_linear_p(src)
        gated = F.gelu(self.input_linear_g(src))
        return torch.cat((torch.cos(periodic), torch.sin(periodic), gated), dim=-1)


class FrequencyGate(nn.Module):
    def __init__(self, bins, expansion):
        super().__init__()
        hidden = int(bins * expansion)
        self.FAN1 = FANLayer(bins, hidden)
        self.FAN2 = FANLayer(hidden, hidden)
        self.Linear = nn.Linear(hidden, bins * 2)

    def forward(self, src):
        x = self.FAN2(self.FAN1(F.gelu(src)))
        first, second = self.Linear(x).chunk(2, dim=-1)
        return first * torch.sigmoid(second)


class ChannelGate(nn.Module):
    def __init__(self, channels, expansion):
        super().__init__()
        hidden = channels * expansion
        self.layernorm = nn.LayerNorm(channels)
        self.FAN1 = FANLayer(channels, hidden)
        self.FAN2 = FANLayer(hidden, hidden)
        self.Linear = nn.Linear(hidden, channels * 2)

    def forward(self, src):
        x = self.layernorm(F.gelu(src).permute(0, 2, 3, 1))
        x = self.FAN2(self.FAN1(x))
        first, second = self.Linear(x).chunk(2, dim=-1)
        return (first * torch.sigmoid(second)).permute(0, 3, 1, 2)


class LocalFrequencyMix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act3 = nn.GELU()

    def forward(self, x):
        batch, channels, time, freq = x.shape
        value = x.permute(0, 2, 1, 3).reshape(batch * time, channels, freq)
        mixed = self.act3(self.conv3(self.act2(self.conv2(self.act1(self.conv1(value))))))
        return (value + mixed).view(batch, time, channels, freq).permute(0, 2, 1, 3)


class FrequencyGLP(nn.Module):
    def __init__(self, bins, channels, expansion):
        super().__init__()
        self.global_branch = FrequencyGate(bins, expansion)
        self.local_branch = LocalFrequencyMix(channels)
        self.linear = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, src):
        return self.linear(torch.cat((self.global_branch(src), self.local_branch(src)), dim=1))


class DenseBlock(nn.Module):
    def __init__(self, channels, depth=4):
        super().__init__()
        layers = []
        for index in range(depth):
            dilation = 2 ** index
            layers.append(nn.Sequential(
                nn.Conv2d(channels * (index + 1), channels, (3, 3), dilation=(dilation, 1),
                          padding=(_padding(3, dilation), _padding(3, 1))),
                nn.InstanceNorm2d(channels, affine=True),
                nn.PReLU(channels)))
        self.dense_block = nn.ModuleList(layers)

    def forward(self, x):
        skip = x
        for layer in self.dense_block:
            x = layer(skip)
            skip = torch.cat((x, skip), dim=1)
        return x


class LearnableSoftplus(nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(bins))

    def forward(self, x):
        beta = torch.exp(self.beta.clamp(max=40)).view(1, -1, 1)
        return (1 / beta + 1e-6) * F.softplus(beta * x)


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg['hid_feature']
        features = [width, width, width]
        pad = ((3 - 1) // 2, (4 - 2) // 2)
        self.downsamples = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(nn.Conv2d(features[0], features[1], (3, 4), stride=(1, 2), padding=pad),
                          nn.InstanceNorm2d(features[1], affine=True), nn.PReLU(features[1])),
            nn.Sequential(nn.Conv2d(features[1], features[2], (3, 4), stride=(1, 2), padding=pad),
                          nn.InstanceNorm2d(features[2], affine=True), nn.PReLU(features[2])),
        ])
        spec = (cfg['d_state'], cfg['d_conv'], cfg['expand'], cfg['norm_epsilon'])
        self.time_mambas = nn.ModuleList([BidirectionalMamba(features[i], *spec) for i in range(3)])
        self.freq_ffns = nn.ModuleList([FrequencyGLP(bins, features[i], 2) for i, bins in enumerate((100, 50, 25))])
        self.freq_layernorm = nn.ModuleList([nn.LayerNorm(features[i]) for i in range(3)])
        self.channel_ffns = nn.ModuleList([ChannelGate(features[i], 2) for i in range(3)])
        self.tlinears = nn.ModuleList([nn.ConvTranspose1d(features[i] * 2, features[i], 1) for i in range(3)])
        self.upsamples = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(features[2], features[1], (3, 4), stride=(1, 2), padding=(1, 1)),
                          nn.InstanceNorm2d(features[1], affine=True), nn.PReLU(features[1])),
            nn.Sequential(nn.ConvTranspose2d(features[1], features[0], (3, 4), stride=(1, 2), padding=(1, 1)),
                          nn.InstanceNorm2d(features[0], affine=True), nn.PReLU(features[0])),
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(features[1] * 2, features[1], 1), nn.GELU(), ChannelLayerNorm(features[1])),
            nn.Sequential(nn.Conv2d(features[0] * 2, features[0], 1), nn.GELU(), ChannelLayerNorm(features[0])),
        ])

    def _time(self, value, index):
        batch, channels, time, freq = value.shape
        sequence = value.permute(0, 3, 2, 1).reshape(batch * freq, time, channels)
        mixed = self.tlinears[index](self.time_mambas[index](sequence).permute(0, 2, 1)).permute(0, 2, 1)
        sequence = mixed + sequence
        sequence = sequence.view(batch, freq, time, channels).permute(0, 2, 1, 3).reshape(batch * time, freq, channels)
        sequence = self.freq_layernorm[index](sequence)
        spectrum = sequence.view(batch, time, freq, channels).permute(0, 3, 1, 2)
        spectrum = self.freq_ffns[index](spectrum) + spectrum
        return self.channel_ffns[index](spectrum) + spectrum

    def forward(self, x):
        base = self.downsamples[0](x)
        mid = self.downsamples[1](base)
        low = self.downsamples[2](mid)
        level1 = self._time(base, 0)
        level2 = self._time(mid, 1)
        level3 = self._time(low, 2)
        merged = self.gates[0](torch.cat((level2, self.upsamples[0](level3)), dim=1))
        merged = self.gates[1](torch.cat((level1, self.upsamples[1](merged)), dim=1))
        return level1 + merged


class SEMambaPP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg['hid_feature']
        bins = cfg['n_fft'] // 2 + 1
        self.dense_encoder = nn.Module()
        self.dense_encoder.dense_conv_1 = nn.Sequential(
            nn.Conv2d(cfg['input_channel'], channels, (1, 1)),
            nn.InstanceNorm2d(channels, affine=True), nn.PReLU(channels))
        self.dense_encoder.dense_block = DenseBlock(channels)
        self.dense_encoder.dense_conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(channels, affine=True), nn.PReLU(channels))
        self.TSMamba = nn.ModuleList([Bottleneck(cfg) for _ in range(cfg['num_tfmamba'])])
        self.mask_decoder = nn.Module()
        self.mask_decoder.dense_block = DenseBlock(channels)
        self.mask_decoder.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (1, 3), stride=(1, 2)),
            nn.Conv2d(channels, cfg['output_channel'], (1, 1)),
            nn.InstanceNorm2d(cfg['output_channel'], affine=True),
            nn.PReLU(cfg['output_channel']),
            nn.Conv2d(cfg['output_channel'], cfg['output_channel'], (1, 1)))
        self.mask_decoder.softplus = LearnableSoftplus(bins)
        self.phase_decoder = nn.Module()
        self.phase_decoder.dense_block = DenseBlock(channels)
        self.phase_decoder.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(channels, affine=True), nn.PReLU(channels))
        self.phase_decoder.phase_conv_r = nn.Conv2d(channels, cfg['output_channel'], (1, 1))
        self.phase_decoder.phase_conv_i = nn.Conv2d(channels, cfg['output_channel'], (1, 1))

    def _encode(self, magnitude, phase):
        x = torch.cat((magnitude.transpose(1, 2).unsqueeze(1), phase.transpose(1, 2).unsqueeze(1)), dim=1)
        x = self.dense_encoder.dense_conv_2(self.dense_encoder.dense_block(self.dense_encoder.dense_conv_1(x)))
        for block in self.TSMamba:
            x = block(x)
        return x

    def forward(self, magnitude, phase):
        x = self._encode(magnitude, phase)
        mask = self.mask_decoder.mask_conv(self.mask_decoder.dense_block(x))
        mask = self.mask_decoder.softplus(mask.squeeze(1).transpose(1, 2))
        phase_h = self.phase_decoder.phase_conv(self.phase_decoder.dense_block(x))
        phase_out = torch.atan2(self.phase_decoder.phase_conv_i(phase_h), self.phase_decoder.phase_conv_r(phase_h))
        phase_out = phase_out.squeeze(1).transpose(1, 2)
        return mask, phase_out
