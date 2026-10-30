"""Stand-ins for the x-transformers 2.14.2 rotary path and Euler ODE integration.

The rotation matches ``x_transformers.x_transformers`` at tag 2.14.2: adjacent-pair
``rotate_half``, frequencies interleaved as ``(f0, f0, f1, f1, ...)``, and a leading
batch axis of size 1. ``use_xpos`` stays off, which is how StuPASE constructs it.
The integrator is the explicit Euler step used when ``odeint_kwargs['method']`` is
``euler`` (the released CFM checkpoint).
"""
import torch
from torch import nn


def _rotate_half(values):
    paired = values.reshape(*values.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).reshape(*values.shape)


def apply_rotary_pos_emb(values, freqs, scale=1.0):
    rot_dim, seq_len, dtype = freqs.shape[-1], values.shape[-2], values.dtype
    # x-transformers indexes the sequence axis, which is dim 1 of (1, seq, dim).
    freqs = freqs[:, -seq_len:]
    if torch.is_tensor(scale):
        scale = scale[:, -seq_len:]
    if values.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
        if torch.is_tensor(scale):
            scale = scale.unsqueeze(1)
    rotated = values[..., :rot_dim]
    rest = values[..., rot_dim:]
    rotated = rotated * freqs.cos() * scale + _rotate_half(rotated) * freqs.sin() * scale
    return torch.cat((rotated, rest), dim=-1).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, use_xpos=False, scale_base=512, interpolation_factor=1.0,
                 base=10000, base_rescale_factor=1.0):
        super().__init__()
        if use_xpos:
            raise NotImplementedError('StuPASE rotary embeddings do not use xpos')
        if interpolation_factor != 1.0:
            raise NotImplementedError('StuPASE rotary embeddings do not interpolate positions')
        base = base * (base_rescale_factor ** (dim / (dim - 2)))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward_from_seq_len(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device).unsqueeze(0)
        freqs = torch.einsum('bi,j->bij', positions.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
        # The library returns the float 1 when xpos is disabled. Attention then
        # uses that scale and its reciprocal, both of which are 1.
        return freqs, 1.0


def odeint(fn, y0, t, method='euler', **_ignored):
    if method != 'euler':
        raise NotImplementedError(f'StuPASE sampling implements Euler integration, got {method!r}')
    steps = [y0]
    state = y0
    for index in range(t.shape[0] - 1):
        dt = t[index + 1] - t[index]
        state = state + dt * fn(t[index], state)
        steps.append(state)
    return torch.stack(steps, dim=0)
