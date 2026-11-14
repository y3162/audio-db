"""Waveform adapter around the MP-SENet generator.

The generator is ``src.se.build_model("mp_senet")``. STFT and iSTFT are the
training functions ``mag_pha_stft`` and ``mag_pha_istft``. This module does
not call ``enhance()`` or ``load_enhancer()``, and it does not own a loss or
an optimizer.

Checkpoint keys for the generator are the adapter ``state_dict`` entries
prefixed with ``generator.``. Those parameters are the only trainable
parameters of this module.
"""

from __future__ import annotations

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.se import build_model
from src.se.models.mp_senet.model import MPNet
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft
from src.experiments.official.mp_senet.models import architecture_config, network_config


class MPSENetWaveformAdapter(torch.nn.Module):
    """Map a ``[batch, time]`` waveform through MPNet and back to a waveform."""

    def __init__(
        self,
        *,
        dense_channel: int,
        num_tsconformers: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        compress_factor: float,
        beta: float,
        sampling_rate: int,
        n_heads: int = 4,
        dense_depth: int = 4,
        ts_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config = architecture_config(
            dense_channel=dense_channel,
            num_tsconformers=num_tsconformers,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            beta=beta,
            sampling_rate=sampling_rate,
            n_heads=n_heads,
            dense_depth=dense_depth,
            ts_dropout=ts_dropout,
        )
        try:
            generator = build_model("mp_senet", network_config(config))
        except (TypeError, ValueError) as exc:
            raise BuildError(f"failed to build mp_senet: {exc}") from exc
        if not isinstance(generator, MPNet):
            raise BuildError(f"mp_senet builder returned {type(generator).__name__}, expected MPNet")
        self.generator = generator
        self.dense_channel = int(config["dense_channel"])
        self.num_tsconformers = int(config["num_tsconformers"])
        self.n_fft = int(config["n_fft"])
        self.hop_size = int(config["hop_size"])
        self.win_size = int(config["win_size"])
        self.compress_factor = float(config["compress_factor"])
        self.beta = float(config["beta"])
        self.sampling_rate = int(config["sampling_rate"])
        self.n_heads = int(config["n_heads"])
        self.dense_depth = int(config["dense_depth"])
        self.ts_dropout = float(config["ts_dropout"])

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return an enhanced waveform with the same shape as ``waveform``."""

        if not isinstance(waveform, torch.Tensor):
            raise RuntimeFailure("MP-SENet input must be a tensor")
        if waveform.ndim != 2:
            raise RuntimeFailure(f"MP-SENet input must be [batch, time], got rank {waveform.ndim}")
        length = int(waveform.shape[-1])
        if length <= self.n_fft // 2:
            raise RuntimeFailure(
                f"waveform length {length} is too short for reflect-padded STFT "
                f"(need more than {self.n_fft // 2} samples)"
            )
        try:
            magnitude, phase, _complex = mag_pha_stft(
                waveform,
                self.n_fft,
                self.hop_size,
                self.win_size,
                self.compress_factor,
            )
            enhanced_magnitude, enhanced_phase, _enhanced_complex = self.generator(magnitude, phase)
            audio = mag_pha_istft(
                enhanced_magnitude,
                enhanced_phase,
                self.n_fft,
                self.hop_size,
                self.win_size,
                self.compress_factor,
            )
        except TrainingError as exc:
            raise RuntimeFailure(str(exc)) from exc
        if audio.shape != waveform.shape:
            raise RuntimeFailure(
                f"iSTFT length {int(audio.shape[-1])} does not match the waveform length {length}"
            )
        if not bool(torch.isfinite(audio).all()):
            raise RuntimeFailure("MP-SENet waveform is not finite")
        return audio
