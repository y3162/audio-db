"""MP-SENet generator and metric discriminator for official GAN training.

The generator is one ``src.se.build_model("mp_senet")`` MPNet. The
discriminator is the existing ``MetricDiscriminator``. Neither module is
reimplemented here, and the waveform L1 adapter is a different class.

``MPSENetTask`` builds the generator and then the discriminator inside one
``seed_model_init`` scope. Builder stage 0 does not keep that order, because
other components share the stage. Each class therefore enters
``seed_model_init`` on its own. The discriminator constructor builds a
generator that it does not store, so the discriminator's random draws start
where they start in ``MPSENetTask``. That discarded generator is not a second
registered owner of the training parameters.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.se import build_model
from src.se.models.mp_senet.model import MPNet
from src.training.core.rng import seed_model_init
from src.training.errors import TrainingError
from src.training.se.discriminator import MetricDiscriminator
from src.training.se.mp_senet import official_center_istft_length
from src.training.se.stft import mag_pha_istft, mag_pha_stft

_SAMPLE_RATE = 16000
_ARCHITECTURE_KEYS = (
    "beta",
    "compress_factor",
    "dense_channel",
    "hop_size",
    "n_fft",
    "num_tsconformers",
    "sampling_rate",
    "win_size",
)
# Tuple order is the trainer's dict order. DDP forwards this tuple.
_FORWARD_FIELDS = (
    "clean_mag",
    "clean_phase",
    "clean_complex",
    "noisy_mag",
    "noisy_phase",
    "mag",
    "phase",
    "complex",
    "audio",
    "mag_hat",
    "phase_hat",
    "recomputed_complex",
)


def architecture_config(
    *,
    dense_channel: int,
    num_tsconformers: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float,
    beta: float,
    sampling_rate: int,
) -> dict[str, Any]:
    """Return the MPNet config after the same checks as the waveform adapter."""

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
    return {
        "beta": beta,
        "compress_factor": compress_factor,
        "dense_channel": dense_channel,
        "hop_size": hop_size,
        "n_fft": n_fft,
        "num_tsconformers": num_tsconformers,
        "sampling_rate": sampling_rate,
        "win_size": win_size,
    }


def pack_forward(values: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the generator tuple to the names used by the official update."""

    if len(values) != len(_FORWARD_FIELDS):
        raise RuntimeFailure(
            f"MP-SENet generator returned {len(values)} tensors, expected {len(_FORWARD_FIELDS)}"
        )
    packed = dict(zip(_FORWARD_FIELDS, values, strict=True))
    for name, value in packed.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeFailure(f"MP-SENet field {name} is not a tensor")
    return packed


class MPSENetGANGenerator(torch.nn.Module):
    """One MPNet plus the official STFT pair used by GAN training.

    ``forward`` runs one generator evaluation. The caller keeps the graph and
    does not call this again for the discriminator step. The returned complex
    spectra use the training stack layout from ``mag_pha_stft``.
    """

    def __init__(
        self,
        *,
        seed: int,
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
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            dense_channel=dense_channel,
            num_tsconformers=num_tsconformers,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            beta=beta,
            sampling_rate=sampling_rate,
        )
        with seed_model_init(self.seed):
            generator = build_model("mp_senet", self.architecture)
        if not isinstance(generator, MPNet):
            raise BuildError(f"mp_senet builder returned {type(generator).__name__}, expected MPNet")
        self.generator = generator
        self.n_fft = int(self.architecture["n_fft"])
        self.hop_size = int(self.architecture["hop_size"])
        self.win_size = int(self.architecture["win_size"])
        self.compress_factor = float(self.architecture["compress_factor"])

    def forward(self, noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the official generation tuple for one micro-batch.

        Training rejects an iSTFT length that differs from the waveform.
        Validation uses the task report, which allows the official center
        length. This method does not.
        """

        if not isinstance(noisy, torch.Tensor) or not isinstance(clean, torch.Tensor):
            raise RuntimeFailure("MP-SENet GAN inputs must be tensors")
        if noisy.ndim != 2 or clean.shape != noisy.shape:
            raise RuntimeFailure(
                f"expected matching [batch, time] waveforms, got {tuple(noisy.shape)} and {tuple(clean.shape)}"
            )
        length = int(clean.shape[-1])
        if length <= self.n_fft // 2:
            raise RuntimeFailure(
                f"waveform length {length} is too short for reflect-padded STFT "
                f"(need more than {self.n_fft // 2} samples)"
            )
        spec = self.architecture
        try:
            clean_mag, clean_phase, clean_complex = mag_pha_stft(
                clean, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
            )
            noisy_mag, noisy_phase, _noisy_complex = mag_pha_stft(
                noisy, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
            )
            magnitude, phase, complex_spectrum = self.generator(noisy_mag, noisy_phase)
            audio = mag_pha_istft(
                magnitude, phase, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
            )
        except TrainingError as exc:
            raise RuntimeFailure(str(exc)) from exc
        if int(audio.shape[-1]) != length:
            official = official_center_istft_length(length, self.hop_size)
            raise RuntimeFailure(
                f"iSTFT length {int(audio.shape[-1])} does not match the waveform length {length} "
                f"(official center length would be {official})"
            )
        try:
            magnitude_hat, phase_hat, recomputed = mag_pha_stft(
                audio, spec["n_fft"], spec["hop_size"], spec["win_size"], spec["compress_factor"]
            )
        except (RuntimeError, TrainingError) as exc:
            raise RuntimeFailure(str(exc)) from exc
        return (
            clean_mag,
            clean_phase,
            clean_complex,
            noisy_mag,
            noisy_phase,
            magnitude,
            phase,
            complex_spectrum,
            audio,
            magnitude_hat,
            phase_hat,
            recomputed,
        )

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        """Generator weights only. The discriminator is not part of enhancement."""

        return {"generator": self.generator.state_dict()}


class MPSENetMetricDiscriminator(torch.nn.Module):
    """Existing metric discriminator, initialized after a discarded generator.

    The discarded ``build_model`` call is only there to consume the same
    ``seed_model_init`` draws as ``MPSENetTask``. It is not registered and it
    is not the training generator.
    """

    def __init__(
        self,
        *,
        seed: int,
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
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            dense_channel=dense_channel,
            num_tsconformers=num_tsconformers,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            compress_factor=compress_factor,
            beta=beta,
            sampling_rate=sampling_rate,
        )
        with seed_model_init(self.seed):
            build_model("mp_senet", self.architecture)
            self.discriminator = MetricDiscriminator()

    def forward(self, reference_magnitude: torch.Tensor, estimate_magnitude: torch.Tensor) -> torch.Tensor:
        return self.discriminator(reference_magnitude, estimate_magnitude)


def _seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise BuildError("seed must be an integer >= 0")
    return seed


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise BuildError(f"{name} must be a positive number")
    return float(value)
