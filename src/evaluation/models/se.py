"""Run speech enhancement through the existing src.se API.

Model-specific inference stays in src.se. This module loads an enhancer once
and sets a per-utterance torch seed immediately before enhance().
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch

from src.evaluation.config import MOCK_MODEL, ModelConfig
from src.evaluation.identity import derive_seed


class MockEnhancer:
    """Identity copy used to exercise the pipeline without a checkpoint.

    This is not a speech enhancement model and its scores are not model quality.
    Input checks match src.se.core.audio.validate_audio. That module is not
    imported here, because importing src.se loads every enhancement model.
    """

    def enhance(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        _validate_waveform(waveform, sample_rate)
        return waveform.clone()


def _validate_waveform(waveform: torch.Tensor, sample_rate: int) -> None:
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
        raise ValueError("waveform must be a one-dimensional torch.Tensor")
    if not waveform.is_floating_point() or not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must contain finite floating-point samples")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0 or waveform.numel() == 0:
        raise ValueError("sample_rate and waveform length must be positive")


@dataclass
class Enhancement:
    waveform: torch.Tensor
    seconds: float
    seed: int


def load_evaluation_enhancer(model: ModelConfig, device: str, loader=None):
    if model.name == MOCK_MODEL:
        return MockEnhancer()
    if loader is None:
        from src.se import load_enhancer as loader

    checkpoint = model.checkpoints if model.checkpoints is not None else model.checkpoint
    return loader(
        model.name,
        checkpoint,
        device=device,
        config=model.model_config,
        variant=model.variant,
    )


def enhancement_seed(experiment_seed: int, utterance_id: str, model_name: str, variant: str | None, snr_db) -> int:
    return derive_seed(experiment_seed, "enhance", utterance_id, model_name, variant, snr_db)


def enhance_utterance(enhancer, waveform: torch.Tensor, sample_rate: int, seed: int) -> Enhancement:
    """Seed torch and random, then enhance. The same seed reproduces one utterance."""
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    enhanced = enhancer.enhance(waveform, sample_rate)
    elapsed = time.perf_counter() - started
    if not isinstance(enhanced, torch.Tensor) or enhanced.ndim != 1:
        raise ValueError("enhancer must return a one-dimensional waveform")
    if enhanced.numel() != waveform.numel():
        raise ValueError(
            f"enhancer changed the waveform length: input {waveform.numel()}, output {enhanced.numel()}"
        )
    output = enhanced.detach().cpu().to(torch.float32).contiguous()
    if not torch.isfinite(output).all():
        raise ValueError("enhancer returned non-finite audio")
    return Enhancement(waveform=output, seconds=elapsed, seed=seed)
