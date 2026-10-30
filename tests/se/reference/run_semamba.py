"""Compare SEMamba++ to the official generator using Mamba's reference scan.

The fused CUDA kernel is not required for this check. ``mamba_ssm`` is replaced,
before the official modules import it, by the slow path in mamba-ssm 2.2.4:
causal conv, softplus(dt + dt_bias), and the published selective-scan recurrence.
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.se.reference.metrics import compare, format_row


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, **_ignored):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, layer_idx=0, **_ignored):
        super().__init__()
        inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = (d_model + 15) // 16
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1, groups=inner, bias=True)
        self.x_proj = nn.Linear(inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(inner, d_state))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape
        flat = hidden_states.reshape(batch * seqlen, -1).t()
        projected = self.in_proj.weight @ flat
        xz = projected.view(-1, batch, seqlen).transpose(0, 1)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])
        mixed = self.x_proj(x.transpose(1, 2).reshape(batch * seqlen, -1))
        dt, b_mat, c_mat = torch.split(mixed, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (self.dt_proj.weight @ dt.t()).view(-1, batch, seqlen).transpose(0, 1)
        b_mat = b_mat.view(batch, seqlen, self.d_state).transpose(1, 2)
        c_mat = c_mat.view(batch, seqlen, self.d_state).transpose(1, 2)
        delta = F.softplus(dt + self.dt_proj.bias.view(1, -1, 1))
        state_a = -torch.exp(self.A_log.float())
        delta_a = torch.exp(torch.einsum('bdl,dn->bdln', delta.float(), state_a))
        delta_b_u = torch.einsum('bdl,bnl,bdl->bdln', delta.float(), b_mat.float(), x.float())
        state = x.new_zeros(batch, x.shape[1], self.d_state)
        outputs = []
        for index in range(seqlen):
            state = delta_a[:, :, index] * state + delta_b_u[:, :, index]
            outputs.append(torch.einsum('bdn,bn->bd', state, c_mat[:, :, index].float()))
        scanned = torch.stack(outputs, dim=2) + x.float() * self.D.float().view(-1, 1)
        scanned = scanned * F.silu(z.float())
        return self.out_proj(scanned.transpose(1, 2).type_as(hidden_states))


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.mlp = None if mlp_cls is nn.Identity else mlp_cls(dim)

    def forward(self, hidden_states, residual=None, inference_params=None, **_ignored):
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.mixer(self.norm(residual.to(dtype=self.norm.weight.dtype)))
        return hidden_states, residual


def _install_stub():
    modules = {
        'mamba_ssm': types.ModuleType('mamba_ssm'),
        'mamba_ssm.modules': types.ModuleType('mamba_ssm.modules'),
        'mamba_ssm.modules.mamba_simple': types.ModuleType('mamba_ssm.modules.mamba_simple'),
        'mamba_ssm.modules.block': types.ModuleType('mamba_ssm.modules.block'),
        'mamba_ssm.models': types.ModuleType('mamba_ssm.models'),
        'mamba_ssm.models.mixer_seq_simple': types.ModuleType('mamba_ssm.models.mixer_seq_simple'),
        'mamba_ssm.ops': types.ModuleType('mamba_ssm.ops'),
        'mamba_ssm.ops.triton': types.ModuleType('mamba_ssm.ops.triton'),
        'mamba_ssm.ops.triton.layer_norm': types.ModuleType('mamba_ssm.ops.triton.layer_norm'),
    }
    for name, module in modules.items():
        sys.modules[name] = module
    modules['mamba_ssm.modules.mamba_simple'].Mamba = Mamba
    modules['mamba_ssm.modules.block'].Block = Block
    modules['mamba_ssm.models.mixer_seq_simple']._init_weights = lambda *args, **kwargs: None
    modules['mamba_ssm.ops.triton.layer_norm'].RMSNorm = RMSNorm


def main():
    torch.set_num_threads(4)
    _install_stub()
    sys.path.insert(0, str(ROOT / 'src/se/official/semambapp'))
    from model.semambapp import SEMambapp
    from src.se.models.semambapp.network import SEMambaPP
    from src.se import load_enhancer

    config = {
        'model_cfg': {'hid_feature': 48, 'num_tfmamba': 4, 'd_state': 16, 'd_conv': 4,
                      'expand': 2, 'norm_epsilon': 1e-5, 'input_channel': 2, 'output_channel': 1,
                      'beta': 2.0, 'dropout': 0.1, 'unet_expansion': 1},
        'stft_cfg': {'n_fft': 400, 'hop_size': 100, 'win_size': 400, 'compress_factor': 0.3, 'sampling_rate': 16000},
    }
    official = SEMambapp(config).eval()
    state = torch.load(ROOT / 'data/checkpoints/semambapp/semambapp.pth', map_location='cpu', weights_only=False)['generator']
    missing, unexpected = official.load_state_dict(state, strict=False)
    print('official missing', len(missing), 'unexpected', len(unexpected), flush=True)
    if missing[:8] or unexpected[:8]:
        print('missing sample', missing[:8], flush=True)
        print('unexpected sample', unexpected[:8], flush=True)
    enhancer = load_enhancer('semambapp', ROOT / 'data/checkpoints/semambapp/semambapp.pth', device='cpu')
    independent = enhancer.model
    torch.manual_seed(9)
    magnitude = torch.rand(1, 201, 16)
    phase = torch.randn(1, 201, 16)
    with torch.inference_mode():
        mag_o, pha_o = official(magnitude, phase)[:2]
        mag_i, pha_i = independent(magnitude, phase)
    print(format_row('semamba_mag', compare(mag_o, mag_i)), flush=True)
    print(format_row('semamba_phase', compare(pha_o, pha_i)), flush=True)
    wave = torch.randn(16000) * 0.05
    # Official infer.py peak-normalizes, and iSTFT is called without an explicit length.
    peak = wave.abs().max() + 1e-9
    window = torch.hann_window(400)
    spec = torch.stft((wave / peak).view(1, -1), 400, 100, 400, window, return_complex=True)
    with torch.inference_mode():
        mag, pha, _ = official(spec.abs().pow(0.3), spec.angle())
    restored = torch.complex(mag.pow(1 / 0.3) * torch.cos(pha), mag.pow(1 / 0.3) * torch.sin(pha))
    official_wave = torch.istft(restored, 400, 100, 400, window, center=True).squeeze(0) * peak
    print(format_row('semamba_wave', compare(official_wave, enhancer.enhance(wave, 16000))), flush=True)


if __name__ == '__main__':
    main()
