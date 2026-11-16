"""CPU stand-in for the frozen Parakeet. Training tests use this instead of NeMo."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.experiments.errors import BuildError


class _Token:
    def __init__(self, token_id: int, start: int, end: int) -> None:
        self.token_id = token_id
        self.start_offset = start
        self.end_offset = end


class _Recognition:
    def __init__(self) -> None:
        self.text = "alpha"
        self.tokens = (_Token(7, 0, 2),)


class FakeParakeet(nn.Module):
    """Frozen encoder with the v1 hop and a waveform-connected stand-in loss."""

    def __init__(self, model_id: str = "nvidia/parakeet-tdt-0.6b-v2", tdt_loss: bool = False) -> None:
        super().__init__()
        if not isinstance(tdt_loss, bool):
            raise BuildError("tdt_loss must be a boolean")
        if model_id != "nvidia/parakeet-tdt-0.6b-v2":
            raise BuildError("ASR-guided SE v1 requires nvidia/parakeet-tdt-0.6b-v2")
        self.model_id = model_id
        self.encoder_dim = 8
        self.encoder_layers = 16
        self.samples_per_frame = 1280
        self.source = nn.Parameter(torch.ones(self.encoder_dim))
        self.loss_calls = 0
        self.recognize_calls = 0
        self.seen_layers: list[int] = []
        self._tdt = False
        if tdt_loss:
            self.ensure_tdt()
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> nn.Module:
        del mode
        return super().train(False)

    def to_device(self, device: torch.device | str) -> FakeParakeet:
        self.to(device)
        self.requires_grad_(False)
        return self

    def ensure_tdt(self) -> None:
        self._tdt = True

    def encode_layer(self, waveforms: torch.Tensor, lengths: torch.Tensor, layer: int):
        del lengths
        self.seen_layers.append(int(layer))
        batch = int(waveforms.shape[0])
        hidden = (self.source.view(1, -1, 1) + float(layer)).expand(batch, self.encoder_dim, 2).contiguous()
        encoded = torch.full((batch,), 2, device=waveforms.device, dtype=torch.long)
        return hidden, encoded

    def recognize(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        del waveforms, lengths
        self.recognize_calls += 1
        return (_Recognition(),)

    def loss_from_ids(self, waveforms, lengths, token_ids, *, deterministic_tdt: bool = False):
        del lengths, token_ids, deterministic_tdt
        self.loss_calls += 1
        return waveforms.square().mean()

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "decode": "fake",
            "tdt_loss": self._tdt,
            "samples_per_encoder_frame": self.samples_per_frame,
        }

    def parameters(self, recurse: bool = True):
        del recurse
        yield self.source

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del waveforms, lengths
        raise RuntimeError("FakeParakeet is not a generator layer")
