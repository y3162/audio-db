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

_SAMPLE_RATE = 16000


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
    ) -> None:
        super().__init__()
        dense_channel = _positive_int("dense_channel", dense_channel)
        num_tsconformers = _positive_int("num_tsconformers", num_tsconformers)
        n_fft = _positive_int("n_fft", n_fft)
        hop_size = _positive_int("hop_size", hop_size)
        win_size = _positive_int("win_size", win_size)
        sampling_rate = _positive_int("sampling_rate", sampling_rate)
        compress_factor = _positive_float("compress_factor", compress_factor)
        beta = _positive_float("beta", beta)
        if dense_channel % 4 != 0:
            raise BuildError("dense_channel must be divisible by the 4 attention heads")
        if win_size > n_fft:
            raise BuildError(f"win_size {win_size} is larger than n_fft {n_fft}")
        if sampling_rate != _SAMPLE_RATE:
            raise BuildError(f"mp_senet sampling_rate must be {_SAMPLE_RATE}, got {sampling_rate}")
        config = {
            "beta": beta,
            "compress_factor": compress_factor,
            "dense_channel": dense_channel,
            "hop_size": hop_size,
            "n_fft": n_fft,
            "num_tsconformers": num_tsconformers,
            "sampling_rate": sampling_rate,
            "win_size": win_size,
        }
        try:
            generator = build_model("mp_senet", config)
        except (TypeError, ValueError) as exc:
            raise BuildError(f"failed to build mp_senet: {exc}") from exc
        if not isinstance(generator, MPNet):
            raise BuildError(f"mp_senet builder returned {type(generator).__name__}, expected MPNet")
        self.generator = generator
        self.dense_channel = dense_channel
        self.num_tsconformers = num_tsconformers
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.beta = beta
        self.sampling_rate = sampling_rate

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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)
