"""Shared ASR types and waveform checks."""

from src.asr.core.audio import prepare_waveform
from src.asr.core.result import ASRResult

__all__ = ["ASRResult", "prepare_waveform"]
