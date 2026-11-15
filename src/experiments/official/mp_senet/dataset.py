"""MP-SENet waveform lengths and the waveform adapter.

Lengths are shared by the generator, the GAN update, and validation.
``align_validation_length`` is the old truncation. The validation report does
not call it. The waveform adapter is the YAML class
``src.experiments.official.mp_senet.waveform.MPSENetWaveformAdapter``;
``waveform.py`` re-exports it.
"""

from __future__ import annotations

import torch

from src.experiments.calc.errors import TrainingError
from src.experiments.calc.stft import mag_pha_istft, mag_pha_stft
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mp_senet.model import architecture_config, network_config
from src.se import build_model
from src.se.models.mp_senet.model import MPNet

# --- STFT length and discriminator resolution ---

def official_center_istft_length(length: int, hop: int) -> int:
    """Samples restored by official ``center=True`` iSTFT for this input length.

    With ``center=True``, the frame count is ``1 + length // hop`` and iSTFT
    without an explicit length returns ``(frames - 1) * hop``. That is the
    largest multiple of ``hop`` that does not exceed ``length``.
    """

    if hop < 1:
        raise TrainingError("hop must be positive")
    if length < 0:
        raise TrainingError("length must be non-negative")
    return (int(length) // int(hop)) * int(hop)


def align_validation_length(length: int, hop: int, n_fft: int) -> int:
    """Old validation truncation: largest hop multiple at most ``length``.

    The validation report does not call this to cut the generator input.
    The update path still requires the waveform and iSTFT lengths to match.
    Lengths whose hop multiple is at most ``n_fft // 2`` are rejected here
    because the old path refused to build that shorter STFT.
    """

    if hop < 1:
        raise TrainingError("hop must be positive")
    if length % hop == 0:
        return length
    aligned = (length // hop) * hop
    if aligned <= n_fft // 2:
        raise TrainingError(f"utterance length {length} is too short to align to hop {hop}")
    return aligned


def _conv_out(size: int) -> int:
    return (size + 2 - 4) // 2 + 1


def require_discriminator_resolution(n_freq: int, n_time: int) -> None:
    """InstanceNorm rejects a 1x1 map while the discriminator is training.

    Four stride-2 convolutions shrink the spectrogram before that norm. The
    official 400-point STFT and 2-second segments stay large enough. A smaller
    setup is rejected instead of switching the norm to evaluation statistics.
    """

    freq, time = int(n_freq), int(n_time)
    for _ in range(4):
        freq, time = _conv_out(freq), _conv_out(time)
        if freq < 1 or time < 1 or freq * time <= 1:
            raise TrainingError(
                "official loss needs a larger spectrogram so the metric discriminator "
                "InstanceNorm sees more than one spatial location. Increase segment_size or n_fft."
            )

# --- Waveform adapter. YAML class path stays ``waveform.MPSENetWaveformAdapter``. ---

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
