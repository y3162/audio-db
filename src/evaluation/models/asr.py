"""Evaluation adapter for ASR.

Mock stays here. Real checkpoints are loaded from ``src.asr``.
"""

from __future__ import annotations

from typing import Any

import torch

from src.asr import ASRResult, load_recognizer as load_asr
from src.asr.catalog import backend_for, dependency_problem
from src.evaluation.config import ASRConfig
from src.evaluation.errors import ConfigError


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
    if not config.model_id:
        raise ConfigError("asr.model_id is required when a real ASR model is enabled")
    problem = dependency_problem(backend_for(config.name))
    if problem:
        raise ConfigError(problem)
    try:
        return load_asr(
            config.name,
            model_id=config.model_id,
            device=config.device or "cpu",
            language=config.language,
            decode=config.decode,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def recognizer_settings(config: ASRConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "name": config.name,
        "model_id": config.model_id,
        "conditions": list(config.conditions),
        "sample_rate": config.sample_rate,
        "device": config.device,
        "language": config.language,
        "decode": config.decode,
        "checkpoint": config.checkpoint,
        "params": config.params,
    }
