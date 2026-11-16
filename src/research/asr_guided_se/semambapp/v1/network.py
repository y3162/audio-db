"""SEMamba++ with the fixed ASR fusion after the dense encoder.

The official ``SEMambaPP.forward`` is unchanged. Features off is that method.
Features on repeats its dense line, fuses, then runs the same bottleneck and
decoders. Alignment is nearest sample centers, ``100 j`` against ``1280 i``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.semambapp.network import SEMambaPP
from src.research.asr_guided_se.semambapp.v1.config import ASR_HOP, GUIDANCE_DIM, STFT_HOP


def nearest_align(
    hidden: torch.Tensor,
    n_stft: int,
    asr_lengths: torch.Tensor | None = None,
    stft_hop: int = STFT_HOP,
    asr_hop: int = ASR_HOP,
) -> torch.Tensor:
    """Align ``[B, T_asr, D]`` onto STFT frames."""

    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, T_asr, D], got {tuple(hidden.shape)}")
    if n_stft < 1 or hidden.shape[1] < 1:
        raise ValueError(f"alignment needs at least one frame, got STFT {n_stft} and ASR {hidden.shape[1]}")
    batch_size, n_asr, dim = hidden.shape
    device = hidden.device
    stft_centers = torch.arange(n_stft, device=device, dtype=hidden.dtype) * stft_hop
    asr_centers = torch.arange(n_asr, device=device, dtype=hidden.dtype) * asr_hop
    distance = (stft_centers[:, None] - asr_centers[None, :]).abs()
    if asr_lengths is None:
        indices = distance.argmin(dim=1)
        return hidden.index_select(1, indices)
    valid = torch.arange(n_asr, device=device)[None, :] < asr_lengths.to(device)[:, None]
    batched = distance[None, :, :].expand(batch_size, -1, -1)
    batched = batched.masked_fill(~valid[:, None, :], torch.inf)
    indices = batched.argmin(dim=-1)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, dim)
    return hidden.gather(1, gather_index)


def broadcast_frequency(projected: torch.Tensor, n_freq: int) -> torch.Tensor:
    """``[B, T, C]`` to ``[B, C, T, F]``."""

    channel_first = projected.transpose(1, 2).unsqueeze(-1)
    return channel_first.expand(-1, -1, -1, n_freq).contiguous()


def project_asr_features(
    proj: nn.Linear,
    hidden: torch.Tensor,
    n_stft: int,
    asr_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project ``[B, D, T_asr]`` and align it to ``[B, T_stft, guidance]``."""

    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, D, T_asr], got {tuple(hidden.shape)}")
    if hidden.shape[1] != proj.in_features:
        raise ValueError(
            f"ASR channel {hidden.shape[1]} does not match the projection {proj.in_features}"
        )
    projected = proj(hidden.transpose(1, 2))
    return nearest_align(projected, n_stft, asr_lengths)


def fuse_asr_features(fuse: nn.Conv2d, se_feat: torch.Tensor, aligned: torch.Tensor) -> torch.Tensor:
    """Concat the dense map with the broadcast ASR map and project back."""

    asr_feat = broadcast_frequency(aligned, se_feat.size(-1))
    return fuse(torch.cat((se_feat, asr_feat), dim=1))


class GuidedSEMambaPP(SEMambaPP):
    """Official SEMamba++ plus the v1 projection and 1x1 fusion."""

    def __init__(self, cfg, encoder_dim: int, encoder_layer: int) -> None:
        super().__init__(cfg)
        if isinstance(encoder_dim, bool) or int(encoder_dim) < 1:
            raise ValueError(f"encoder_dim must be positive, got {encoder_dim}")
        if isinstance(encoder_layer, bool) or int(encoder_layer) < 0:
            raise ValueError(f"encoder_layer must be >= 0, got {encoder_layer}")
        channels = int(cfg["hid_feature"])
        self.features_enabled = True
        self.asr_proj = nn.Linear(int(encoder_dim), GUIDANCE_DIM)
        self.asr_fuse = nn.Conv2d(channels + GUIDANCE_DIM, channels, (1, 1))
        self.register_buffer(
            "encoder_layer_index",
            torch.tensor(int(encoder_layer), dtype=torch.long),
            persistent=True,
        )
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_pending(self, hidden: torch.Tensor, asr_lengths: torch.Tensor) -> None:
        self._pending = (hidden, asr_lengths)

    def clear_pending(self) -> None:
        self._pending = None

    def forward(self, magnitude, phase, hidden=None):
        if hidden is not None:
            raise RuntimeError("pass ASR features with set_pending; forward does not take a hidden argument")
        if self._pending is None:
            raise RuntimeError("ASR features are enabled but no encoder hidden was provided")
        hidden_tensor, asr_lengths = self._pending
        x = torch.cat(
            (magnitude.transpose(1, 2).unsqueeze(1), phase.transpose(1, 2).unsqueeze(1)),
            dim=1,
        )
        x = self.dense_encoder.dense_conv_2(
            self.dense_encoder.dense_block(self.dense_encoder.dense_conv_1(x))
        )
        aligned = project_asr_features(self.asr_proj, hidden_tensor, int(x.shape[2]), asr_lengths)
        x = fuse_asr_features(self.asr_fuse, x, aligned)
        for block in self.TSMamba:
            x = block(x)
        mask = self.mask_decoder.mask_conv(self.mask_decoder.dense_block(x))
        mask = self.mask_decoder.softplus(mask.squeeze(1).transpose(1, 2))
        phase_h = self.phase_decoder.phase_conv(self.phase_decoder.dense_block(x))
        phase_out = torch.atan2(
            self.phase_decoder.phase_conv_i(phase_h),
            self.phase_decoder.phase_conv_r(phase_h),
        )
        phase_out = phase_out.squeeze(1).transpose(1, 2)
        return mask, phase_out
