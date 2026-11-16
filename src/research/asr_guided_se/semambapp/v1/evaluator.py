"""Waveform prediction for the existing research evaluator. This is not a new evaluator type."""

from __future__ import annotations

import torch


def predict_audio(module: torch.nn.Module, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Return enhanced audio. ``noisy`` is the model input and ``clean`` is the pair target."""

    outputs = module(clean, noisy)
    audio = outputs["audio"]
    if audio.ndim != 2:
        raise RuntimeError(f"enhanced audio must be [batch, time], got {tuple(audio.shape)}")
    return audio
