"""Frozen Parakeet used by the experiment. This module does not import NeMo."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from src.asr.models.parakeet import TDT_V2_MODEL_ID
from src.experiments.errors import BuildError


class FrozenParakeet(nn.Module):
    """Checkpoint ASR that stays in eval and out of the generator optimizer.

    The NeMo module is not registered as a child, so ``parameters()`` of a
    generator that holds this object in a list does not include it.
    """

    def __init__(self, model_id: str = TDT_V2_MODEL_ID, tdt_loss: bool = False) -> None:
        super().__init__()
        if not isinstance(tdt_loss, bool):
            raise BuildError("tdt_loss must be a boolean")
        if model_id != TDT_V2_MODEL_ID:
            raise BuildError(f"ASR-guided SE v1 requires {TDT_V2_MODEL_ID}")
        from src.asr.models.parakeet_tdt import ParakeetTDT

        self.model_id = model_id
        self._holder = [ParakeetTDT(model_id)]
        self._tdt_requested = tdt_loss
        if tdt_loss:
            self.ensure_tdt()
        self.eval()

    @property
    def inner(self):
        return self._holder[0]

    @property
    def samples_per_frame(self) -> int:
        return int(self.inner.samples_per_encoder_frame)

    @property
    def encoder_dim(self) -> int:
        return int(self.inner.encoder_dim)

    @property
    def encoder_layers(self) -> int:
        return int(self.inner.encoder_layers)

    def train(self, mode: bool = True) -> nn.Module:
        del mode
        super().train(False)
        self.inner.train(False)
        return self

    def to_device(self, device: torch.device | str) -> FrozenParakeet:
        self.inner.to(device)
        self.inner.train(False)
        return self

    def ensure_tdt(self) -> None:
        self.inner.enable_tdt_loss()
        self._tdt_requested = True

    def encode_layer(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.inner.encode(waveforms, lengths, layers=(int(layer),))
        hidden = encoded.layer_outputs[int(layer)]
        return hidden, encoded.encoded_length

    def recognize(self, waveforms: torch.Tensor, lengths: torch.Tensor):
        return self.inner.recognize(waveforms, lengths)

    def transcribe(self, audio: torch.Tensor, sample_rate: int) -> str:
        """One utterance hypothesis. The weights stay frozen."""

        if isinstance(sample_rate, bool) or int(sample_rate) != 16000:
            raise RuntimeError(f"Parakeet transcription expects 16000 Hz, got {sample_rate}")
        wave = audio.detach()
        if wave.ndim == 2:
            if int(wave.shape[0]) != 1:
                raise RuntimeError("transcribe accepts one waveform")
            wave = wave[0]
        if wave.ndim != 1 or int(wave.shape[0]) < 1:
            raise RuntimeError(f"transcribe waveform must be [time], got {tuple(wave.shape)}")
        self.to_device(wave.device)
        length = torch.tensor([int(wave.shape[0])], device=wave.device, dtype=torch.long)
        recognition = self.recognize(wave.to(dtype=torch.float32).view(1, -1), length)[0]
        text = getattr(recognition, "text", None)
        if not isinstance(text, str):
            raise RuntimeError("Parakeet recognition has no text")
        return text

    def loss_from_ids(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        token_ids: Sequence[Sequence[int]],
        *,
        deterministic_tdt: bool = False,
    ) -> torch.Tensor:
        return self.inner.loss_from_ids(
            waveforms,
            lengths,
            token_ids,
            deterministic_tdt=deterministic_tdt,
        )

    def metadata(self) -> dict[str, Any]:
        fingerprint = self.inner.software_fingerprint()
        return {
            "model_id": self.model_id,
            "decode": "tdt",
            "decoding": dict(self.inner.effective_decoding),
            "fingerprint": fingerprint,
        }

    def parameters(self, recurse: bool = True):
        del recurse
        return self.inner.parameters()

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("FrozenParakeet is not a generator layer; call encode_layer or loss_from_ids")


class TeacherCache:
    """Full-utterance recognitions keyed by audio bytes and the ASR fingerprint."""

    def __init__(self, fingerprint: str, writer: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.fingerprint = fingerprint
        self.writer = writer
        self._items: dict[tuple[str, str, str], Any] = {}

    def recognize(self, asr: Any, audio: torch.Tensor, kind: str):
        if audio.ndim != 1:
            raise ValueError(f"teacher audio must be one utterance, got {tuple(audio.shape)}")
        blob = audio.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
        digest = hashlib.sha256(blob).hexdigest()
        key = (digest, kind, self.fingerprint)
        found = self._items.get(key)
        if found is not None:
            return found
        length = torch.tensor([audio.shape[0]], device=audio.device, dtype=torch.long)
        recognition = asr.recognize(audio.view(1, -1), length)[0]
        self._items[key] = recognition
        if self.writer is not None:
            self.writer(
                {
                    "audio_sha256": digest,
                    "kind": kind,
                    "text": recognition.text,
                    "token_ids": [token.token_id for token in recognition.tokens],
                    "start_offsets": [token.start_offset for token in recognition.tokens],
                    "end_offsets": [token.end_offset for token in recognition.tokens],
                    "fingerprint": self.fingerprint,
                }
            )
        return recognition


def fingerprint_text(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True, default=str)
