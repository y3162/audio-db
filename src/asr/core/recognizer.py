"""Loaded recognizer. Backends supply a function that maps a 16 kHz waveform to text."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from src.asr.catalog import ModelCard
from src.asr.core.audio import prepare_waveform
from src.asr.core.result import ASRResult


class SpeechRecognizer:
    """One loaded checkpoint. ``transcribe`` accepts a mono waveform."""

    def __init__(
        self,
        card: ModelCard,
        *,
        device: str,
        language: str | None,
        runner: Callable[[torch.Tensor], str],
        sample_rate: int = 16000,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.card = card
        self.name = card.family
        self.model_id = card.model_id
        self.backend = card.backend
        self.decode = card.decode
        self.device = device
        self.language = language
        self.sample_rate = sample_rate
        self.extra = dict(extra or {})
        self._runner = runner

    def transcribe(self, waveform: torch.Tensor, sample_rate: int = 16000) -> ASRResult:
        audio = prepare_waveform(waveform, sample_rate, self.sample_rate)
        text = self._runner(audio)
        if not isinstance(text, str):
            raise TypeError(f"recognizer returned {type(text).__name__}, expected str")
        extra: dict[str, Any] = {
            "backend": self.backend,
            "device": self.device,
            "sample_rate": self.sample_rate,
        }
        extra.update(self.extra)
        return ASRResult(
            text=text,
            model=self.name,
            language=self.language,
            model_id=self.model_id,
            decode=self.decode,
            extra=extra,
        )

    def settings(self) -> dict[str, str | None]:
        values: dict[str, str | None] = {
            "family": self.name,
            "model_id": self.model_id,
            "backend": self.backend,
            "decode": self.decode,
            "device": self.device,
            "language": self.language,
        }
        for key, value in self.extra.items():
            if isinstance(value, dict):
                for inner, item in value.items():
                    values[f"{key}.{inner}"] = None if item is None else str(item)
            else:
                values[str(key)] = None if value is None else str(value)
        return values
