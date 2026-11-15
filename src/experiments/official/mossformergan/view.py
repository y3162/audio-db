"""Bound view for MossFormerGAN spectra and validation calls."""

from __future__ import annotations

import torch

from src.experiments.official.mossformergan.reference import (
    forward_spectra,
    generator_parts,
    generator_terms,
    reduce_validation,
    validation_report,
)


class OfficialCall:
    """Binds the current task methods so ``self`` is this view, not the task.

    ``_generator_terms`` and ``validation_report`` call sibling methods. A
    ``SimpleNamespace`` does not bind those functions.
    """

    _forward = forward_spectra
    _generator_parts = generator_parts
    _generator_terms = generator_terms
    validation_report = validation_report
    reduce_validation = reduce_validation

    def __init__(
        self,
        *,
        n_fft: int,
        hop: int,
        win: int,
        sample_rate: int,
        generator: torch.nn.Module | None,
        discriminator: torch.nn.Module | None,
        device: torch.device,
        validation_requires_all_ranks: bool,
    ) -> None:
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.sample_rate = sample_rate
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.validation_requires_all_ranks = validation_requires_all_ranks
