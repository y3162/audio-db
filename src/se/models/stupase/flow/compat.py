"""Minimal stand-ins for x-transformers rotary embeddings and an Euler ODE solver."""
import torch
from torch import nn


def _rotate_half(values):
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(values, freqs, scale=1.0):
    length = values.shape[-2]
    freqs = freqs[-length:]
    rotated = values[..., :freqs.shape[-1]]
    rest = values[..., freqs.shape[-1]:]
    rotated = rotated * freqs.cos() * scale + _rotate_half(rotated) * freqs.sin() * scale
    return torch.cat((rotated, rest), dim=-1).type_as(values)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward_from_seq_len(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, None


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
