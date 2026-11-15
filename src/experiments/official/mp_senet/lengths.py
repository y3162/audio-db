"""STFT lengths and the discriminator's minimum spectrogram size.

Training rejects an iSTFT that is shorter than the waveform. Validation keeps
the full waveform and may receive the official center length instead.
``align_validation_length`` is the old truncation. The validation report does
not call it.
"""

from __future__ import annotations

from src.experiments.calc.errors import TrainingError


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
