"""Corpus-independent utterance description."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AudioSample:
    """One evaluation item.

    Clean, noisy, and noise are all optional. A sample may carry filesystem
    paths, in-memory waveforms, or both. In-memory waveforms are not written
    into result files.
    """

    utterance_id: str
    dataset: str
    split: str | None = None
    audio_path: str | None = None
    sample_rate: int | None = None
    transcript: str | None = None
    clean_path: str | None = None
    noisy_path: str | None = None
    noise_path: str | None = None
    snr_db: float | None = None
    speaker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    clean_waveform: torch.Tensor | None = None
    noisy_waveform: torch.Tensor | None = None
    noise_waveform: torch.Tensor | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "dataset": self.dataset,
            "split": self.split,
            "speaker_id": self.speaker_id,
            "sample_rate": self.sample_rate,
            "transcript": self.transcript,
            "audio_path": self.audio_path,
            "clean_path": self.clean_path,
            "noisy_path": self.noisy_path,
            "noise_path": self.noise_path,
            "snr_db": self.snr_db,
            "in_memory": {
                "clean": self.clean_waveform is not None,
                "noisy": self.noisy_waveform is not None,
                "noise": self.noise_waveform is not None,
            },
            "metadata": self.metadata,
        }
