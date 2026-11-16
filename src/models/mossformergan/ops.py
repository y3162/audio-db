"""Small stand-ins for the einops and rotary-embedding-torch calls this model uses.

Rotary embedding follows lucidrains/rotary-embedding-torch 0.8.3 (MIT): language
frequencies ``1/theta**(arange(0, dim, 2)/dim)`` and partial rotation of the leading
dimensions. The checkpoint stores those frequencies in ``rotary_pos_emb.freqs``.
"""
import torch
from torch import nn


def rearrange(tensor, pattern, **axes):
    pattern = ' '.join(pattern.split())
    if pattern == 'j -> 1 j':
        return tensor.unsqueeze(0)
    if pattern == 'i -> i 1':
        return tensor.unsqueeze(1)
    if pattern == 'i j 1 -> i j':
        return tensor.squeeze(-1)
    if pattern == '... -> ... 1':
        return tensor.unsqueeze(-1)
    if pattern == 'b (g n) d -> b g n d':
        batch, grouped, dim = tensor.shape
        width = axes['n']
        return tensor.reshape(batch, grouped // width, width, dim)
    if pattern == 'b g n d -> b (g n) d':
        batch, groups, width, dim = tensor.shape
        return tensor.reshape(batch, groups * width, dim)
    if pattern == 'b (g j) -> b g 1 j':
        batch, grouped = tensor.shape
        width = axes['j']
        return tensor.reshape(batch, grouped // width, 1, width)
    raise ValueError(f'unsupported rearrange pattern: {pattern}')


def _rotate_half(values):
    paired = values.reshape(*values.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).reshape(*values.shape)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        exponent = torch.arange(0, dim, 2)[:dim // 2].float() / dim
        self.freqs = nn.Parameter(1.0 / (theta ** exponent), requires_grad=False)

    def rotate_queries_or_keys(self, values, seq_dim=-2):
        length = values.shape[seq_dim]
        positions = torch.arange(length, device=values.device, dtype=self.freqs.dtype)
        freqs = torch.einsum('i,f->if', positions, self.freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        if values.ndim == 3:
            freqs = freqs[-length:]
        rotated = freqs.shape[-1]
        left = values[..., :0]
        middle = values[..., :rotated]
        right = values[..., rotated:]
        middle = middle * freqs.cos() + _rotate_half(middle) * freqs.sin()
        return torch.cat((left, middle, right), dim=-1).type(values.dtype)
