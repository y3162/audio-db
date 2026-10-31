"""ASR interface.

Real backends are named so a later implementation can be dropped in. Only the
mock recognizer is executable in this environment. ASR training is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.evaluation.config import ASRConfig, PLANNED_ASR
from src.evaluation.errors import ConfigError


@dataclass
class ASRResult:
    text: str
    model: str
    language: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class MockRecognizer:
    """Return scripted text. The waveform is validated and otherwise ignored."""

    name = "mock"

    def __init__(self, default_text: str = "", scripts: dict[str, Any] | None = None):
        self.default_text = default_text
        self.scripts = scripts or {}
        self._utterance_id: str | None = None
        self._condition: str | None = None

    def bind(self, utterance_id: str, condition: str) -> None:
        self._utterance_id = utterance_id
        self._condition = condition

    def transcribe(self, waveform: torch.Tensor, sample_rate: int = 16000) -> ASRResult:
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
            raise ValueError("waveform must be a one-dimensional torch.Tensor")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        entry = self.scripts.get(self._utterance_id, self.default_text)
        if isinstance(entry, dict):
            text = entry.get(self._condition, self.default_text)
        else:
            text = entry
        if text is None:
            text = self.default_text
        return ASRResult(text=str(text), model=self.name)


def load_recognizer(config: ASRConfig):
    if not config.enabled:
        return None
    if config.name == "mock":
        return MockRecognizer(default_text=config.default_text, scripts=config.scripts)
    if config.name in PLANNED_ASR:
        label = PLANNED_ASR[config.name]
        raise ConfigError(
            f"ASR backend {config.name!r} ({label}) is not installed. "
            "Use asr.name=mock, or add a class with transcribe(waveform, sample_rate) "
            "in src/evaluation/models/asr.py. ASR training is out of scope."
        )
    raise ConfigError(f"unknown ASR model: {config.name}")


def recognizer_settings(config: ASRConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "name": config.name,
        "conditions": list(config.conditions),
        "sample_rate": config.sample_rate,
        "device": config.device,
        "language": config.language,
        "decode": config.decode,
        "checkpoint": config.checkpoint,
        "params": config.params,
    }
