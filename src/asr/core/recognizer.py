"""Loaded recognizer. Backends supply a function that maps a 16 kHz waveform to text."""

from __future__ import annotations

from collections.abc import Callable

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
    ) -> None:
        self.card = card
        self.name = card.family
        self.model_id = card.model_id
        self.backend = card.backend
        self.decode = card.decode
        self.device = device
        self.language = language
        self.sample_rate = sample_rate
        self._runner = runner

    def transcribe(self, waveform: torch.Tensor, sample_rate: int = 16000) -> ASRResult:
        audio = prepare_waveform(waveform, sample_rate, self.sample_rate)
        text = self._runner(audio)
        if not isinstance(text, str):
            raise TypeError(f"recognizer returned {type(text).__name__}, expected str")
        return ASRResult(
            text=text,
            model=self.name,
            language=self.language,
            model_id=self.model_id,
            decode=self.decode,
            extra={
                "backend": self.backend,
                "device": self.device,
                "sample_rate": self.sample_rate,
            },
        )

    def settings(self) -> dict[str, str | None]:
        return {
            "family": self.name,
            "model_id": self.model_id,
            "backend": self.backend,
            "decode": self.decode,
            "device": self.device,
            "language": self.language,
        }
