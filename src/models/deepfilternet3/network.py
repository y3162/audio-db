"""DeepFilterNet3 generator.

Module names and parameter shapes follow the released DeepFilterNet3 checkpoint
(conv encoder, two GRUs, ERB mask, order-5 deep filter). Adapted from the
MIT/Apache-2.0 DeepFilterNet sources; see LICENSE.
"""
import math
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn


@dataclass
class DFN3Params:
    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96
    norm_tau: float = 1.0
    lsnr_max: float = 35.0
    lsnr_min: float = -15.0
    min_nb_freqs: int = 2
    df_order: int = 5
    df_lookahead: int = 2
    conv_lookahead: int = 2
    conv_ch: int = 64
    emb_hidden_dim: int = 256
    emb_num_layers: int = 3
    enc_lin_groups: int = 32
    lin_groups: int = 16
    df_hidden_dim: int = 256
    df_num_layers: int = 2
    df_gru_skip: str = 'groupedlinear'
    df_pathway_kernel_size_t: int = 5
    enc_concat: bool = False
    conv_kernel: tuple = (1, 3)
    convt_kernel: tuple = (1, 3)
    conv_kernel_inp: tuple = (3, 3)
    emb_gru_skip: str = 'none'
    emb_gru_skip_enc: str = 'none'
    df_n_iter: int = 1
    mask_pf: bool = False
    pf_beta: float = 0.02
    lsnr_dropout: bool = False


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    pair = tuple(value)
    return (int(pair[0]), int(pair[1]))


class Conv2dNormAct(nn.Sequential):
    """Causal convolution over time. Input layout is [B, C, T, F]."""

    def __init__(self, in_ch, out_ch, kernel_size, fstride=1, separable=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias=False):
        kernel = _pair(kernel_size)
        fpad = kernel[1] // 2
        layers = []
        time_pad = (0, 0, kernel[0] - 1, 0)
        if any(time_pad):
            layers.append(nn.ConstantPad2d(time_pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        # A 1x1 depthwise kernel keeps grouped convolution and skips the extra pointwise layer.
        pointwise = separable and groups != 1 and max(kernel) != 1
        layers.append(nn.Conv2d(in_ch, out_ch, kernel, padding=(0, fpad), stride=(1, fstride), groups=groups, bias=bias))
        if pointwise:
            layers.append(nn.Conv2d(out_ch, out_ch, 1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, fstride=1, separable=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias=False):
        kernel = _pair(kernel_size)
        fpad = kernel[1] // 2
        layers = []
        time_pad = (0, 0, kernel[0] - 1, 0)
        if any(time_pad):
            layers.append(nn.ConstantPad2d(time_pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        separable = separable and groups != 1
        if not separable:
            groups = 1
        layers.append(nn.ConvTranspose2d(
            in_ch, out_ch, kernel, padding=(kernel[0] - 1, fpad), output_padding=(0, fpad),
            stride=(1, fstride), groups=groups, bias=bias))
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, 1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class GroupedLinearEinsum(nn.Module):
    def __init__(self, input_size, hidden_size, groups):
        super().__init__()
        if input_size % groups or hidden_size % groups:
            raise ValueError(f'grouped linear sizes {input_size}, {hidden_size} not divisible by {groups}')
        self.groups = groups
        self.ws = input_size // groups
        self.weight = nn.Parameter(torch.empty(groups, input_size // groups, hidden_size // groups))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        batch, time, _ = x.shape
        x = x.view(batch, time, self.groups, self.ws)
        return torch.einsum('btgi,gih->btgh', x, self.weight).flatten(2, 3)


class SqueezedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, num_layers=1,
                 linear_groups=8, gru_skip=None, linear_act=None):
        super().__init__()
        act = linear_act or (lambda: nn.Identity())
        self.linear_in = nn.Sequential(GroupedLinearEinsum(input_size, hidden_size, linear_groups), act())
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.gru_skip = gru_skip() if gru_skip is not None else None
        if output_size is None:
            self.linear_out = nn.Identity()
        else:
            self.linear_out = nn.Sequential(GroupedLinearEinsum(hidden_size, output_size, linear_groups), act())

    def forward(self, inputs, state=None):
        projected = self.linear_in(inputs)
        hidden, state = self.gru(projected, state)
        hidden = self.linear_out(hidden)
        if self.gru_skip is not None:
            hidden = hidden + self.gru_skip(inputs)
        return hidden, state


class Mask(nn.Module):
    def __init__(self, erb_inverse):
        super().__init__()
        self.register_buffer('erb_inv_fb', erb_inverse)

    def forward(self, spec, mask):
        mask = mask.matmul(self.erb_inv_fb).unsqueeze(-1)
        return spec * mask


class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        conv = partial(Conv2dNormAct, in_ch=p.conv_ch, out_ch=p.conv_ch, kernel_size=p.conv_kernel, separable=True)
        self.erb_conv0 = Conv2dNormAct(1, p.conv_ch, p.conv_kernel_inp, separable=True)
        self.erb_conv1 = conv(fstride=2)
        self.erb_conv2 = conv(fstride=2)
        self.erb_conv3 = conv(fstride=1)
        self.df_conv0 = Conv2dNormAct(2, p.conv_ch, p.conv_kernel_inp, separable=True)
        self.df_conv1 = conv(fstride=2)
        emb_in = p.conv_ch * p.nb_erb // 4
        self.df_fc_emb = nn.Sequential(
            GroupedLinearEinsum(p.conv_ch * p.nb_df // 2, emb_in, p.enc_lin_groups), nn.ReLU(inplace=True))
        self.combine = torch.cat if p.enc_concat else torch.add
        self.emb_gru = SqueezedGRU(
            emb_in * (2 if p.enc_concat else 1), p.emb_hidden_dim, output_size=emb_in, num_layers=1,
            linear_groups=p.lin_groups, linear_act=partial(nn.ReLU, inplace=True))
        self.lsnr_fc = nn.Sequential(nn.Linear(emb_in, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(self, feat_erb, feat_spec):
        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        c0 = self.df_conv0(feat_spec)
        c1 = self.df_conv1(c0)
        combined = self.combine(
            e3.permute(0, 2, 3, 1).flatten(2), self.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2)))
        embedding, _ = self.emb_gru(combined)
        lsnr = self.lsnr_fc(embedding) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, embedding, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        width = p.conv_ch * p.nb_erb // 4
        self.emb_gru = SqueezedGRU(
            width, p.emb_hidden_dim, output_size=width, num_layers=p.emb_num_layers - 1,
            linear_groups=p.lin_groups, linear_act=partial(nn.ReLU, inplace=True))
        conv = partial(Conv2dNormAct, bias=False, separable=True)
        self.conv3p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = ConvTranspose2dNormAct(p.conv_ch, p.conv_ch, p.convt_kernel, fstride=2, separable=True)
        self.conv1p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = ConvTranspose2dNormAct(p.conv_ch, p.conv_ch, p.convt_kernel, fstride=2, separable=True)
        self.conv0p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv(p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid)

    def forward(self, embedding, e3, e2, e1, e0):
        batch, _, time, freq = e3.shape
        hidden, _ = self.emb_gru(embedding)
        hidden = hidden.view(batch, time, freq, -1).permute(0, 3, 1, 2)
        e3 = self.convt3(self.conv3p(e3) + hidden)
        e2 = self.convt2(self.conv2p(e2) + e3)
        e1 = self.convt1(self.conv1p(e1) + e2)
        return self.conv0_out(self.conv0p(e0) + e1)


class DfDecoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2
        self.df_convp = Conv2dNormAct(
            p.conv_ch, self.df_out_ch, kernel_size=(p.df_pathway_kernel_size_t, 1), separable=True)
        emb_in = p.conv_ch * p.nb_erb // 4
        self.df_gru = SqueezedGRU(
            emb_in, p.df_hidden_dim, num_layers=p.df_num_layers, linear_act=partial(nn.ReLU, inplace=True))
        skip = p.df_gru_skip.lower()
        if skip == 'none':
            self.df_skip = None
        elif skip == 'groupedlinear':
            self.df_skip = GroupedLinearEinsum(emb_in, p.df_hidden_dim, p.lin_groups)
        elif skip == 'identity':
            self.df_skip = nn.Identity()
        else:
            raise ValueError(f'unsupported df_gru_skip {p.df_gru_skip!r}')
        self.df_out = nn.Sequential(
            GroupedLinearEinsum(p.df_hidden_dim, p.nb_df * self.df_out_ch, p.lin_groups), nn.Tanh())
        self.df_fc_a = nn.Sequential(nn.Linear(p.df_hidden_dim, 1), nn.Sigmoid())

    def forward(self, embedding, c0):
        batch, time, _ = embedding.shape
        hidden, _ = self.df_gru(embedding)
        if self.df_skip is not None:
            hidden = hidden + self.df_skip(embedding)
        local = self.df_convp(c0).permute(0, 2, 3, 1)
        coefs = self.df_out(hidden).view(batch, time, self.df_bins, self.df_out_ch)
        return coefs + local


class DeepFilter(nn.Module):
    def __init__(self, n_freqs, order, lookahead):
        super().__init__()
        self.n_freqs = n_freqs
        self.order = order
        self.pad = nn.ConstantPad2d((0, 0, order - 1 - lookahead, lookahead), 0.0)

    def forward(self, spec, coefs):
        complex_spec = torch.view_as_complex(spec)
        unfolded = self.pad(complex_spec).unfold(2, self.order, 1)
        coefs = torch.view_as_complex(coefs)
        coefs = coefs.view(coefs.shape[0], -1, self.order, coefs.shape[2], coefs.shape[3])
        filtered = torch.einsum('...tfn,...ntf->...tf', unfolded.narrow(-2, 0, self.n_freqs), coefs)
        spec = spec.clone()
        spec[..., :self.n_freqs, :] = torch.view_as_real(filtered)
        return spec


class DfNet(nn.Module):
    def __init__(self, erb_fb, erb_inverse, params):
        super().__init__()
        if params.df_n_iter != 1:
            raise ValueError('DeepFilterNet3 inference supports df_n_iter=1')
        if params.nb_erb % 8:
            raise ValueError('nb_erb must be divisible by 8')
        self.params = params
        self.nb_df = params.nb_df
        self.df_order = params.df_order
        self.freq_bins = params.fft_size // 2 + 1
        self.erb_bins = params.nb_erb
        self.pad_feat = nn.ConstantPad2d((0, 0, -params.conv_lookahead, params.conv_lookahead), 0.0)
        self.register_buffer('erb_fb', erb_fb)
        self.enc = Encoder(params)
        self.erb_dec = ErbDecoder(params)
        self.mask = Mask(erb_inverse)
        self.df_dec = DfDecoder(params)
        self.df_op = DeepFilter(params.nb_df, params.df_order, params.df_lookahead)
        self.run_erb = params.nb_df + 1 < self.freq_bins
        self.post_filter = params.mask_pf
        self.post_filter_beta = params.pf_beta

    def forward(self, spec, feat_erb, feat_spec):
        feat_spec = self.pad_feat(feat_spec.squeeze(1).permute(0, 3, 1, 2))
        feat_erb = self.pad_feat(feat_erb)
        e0, e1, e2, e3, embedding, c0, lsnr = self.enc(feat_erb, feat_spec)
        if self.run_erb:
            erb_mask = self.erb_dec(embedding, e3, e2, e1, e0)
            spec_m = self.mask(spec, erb_mask)
        else:
            erb_mask = spec.new_zeros(())
            spec_m = spec
        coefs = self.df_dec(embedding, c0)
        coefs = coefs.view(*coefs.shape[:-1], -1, 2).permute(0, 3, 1, 2, 4)
        enhanced = self.df_op(spec, coefs)
        enhanced[..., self.nb_df:, :] = spec_m[..., self.nb_df:, :]
        if self.post_filter:
            beta, eps = self.post_filter_beta, 1e-12
            clean = torch.view_as_complex(enhanced).abs()
            noisy = torch.view_as_complex(spec).abs().add(eps)
            gain = (clean / noisy).clamp(eps, 1)
            sine = (gain * torch.sin(math.pi * gain / 2)).clamp_min(eps)
            enhanced = enhanced * ((1 + beta) / (1 + beta * gain.div(sine).square())).unsqueeze(-1)
        return enhanced, erb_mask, lsnr, coefs
