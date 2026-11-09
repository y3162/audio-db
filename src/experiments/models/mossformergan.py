"""MossFormerGAN generator and discriminator for official GAN training.

ClearerVoice ``train.py`` builds two network wrappers. Each wrapper builds
SyncANet and then the NISQA discriminator. The kept weights are the first
generator (G1) and the fourth discriminator (D4). Builder stage 0 does not
preserve that order, so each class enters ``seed_model_init`` alone and
replays the draws it has to discard. Discarded modules are not registered
and are not an optimizer target.

The classes wrap ``src.se.build_model("mossformergan")`` and the existing
``Discriminator``. They do not reimplement either network.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.errors import BuildError
from src.se import build_model
from src.se.models.mossformergan.generator import SyncANet
from src.training.core.rng import seed_model_init
from src.training.se.mossformer_discriminator import Discriminator

_BUILD_KEYS = ("hop", "n_fft", "num_channel", "win_len")


def architecture_config(
    *,
    variant: str,
    n_fft: int,
    win_len: int,
    hop: int,
    num_channel: int,
    sampling_rate: int,
) -> dict[str, Any]:
    """Return the architecture both modules must share."""

    if variant != "se_16k":
        raise BuildError(f"MossFormerGAN variant must be se_16k, got {variant!r}")
    n_fft = _positive_int("n_fft", n_fft)
    win_len = _positive_int("win_len", win_len)
    hop = _positive_int("hop", hop)
    num_channel = _positive_int("num_channel", num_channel)
    sampling_rate = _positive_int("sampling_rate", sampling_rate)
    if sampling_rate != 16000:
        raise BuildError(f"MossFormerGAN se_16k requires 16000 Hz, got {sampling_rate}")
    if win_len != n_fft:
        raise BuildError(f"MossFormerGAN win_len {win_len} must equal n_fft {n_fft}")
    return {
        "hop": hop,
        "n_fft": n_fft,
        "num_channel": num_channel,
        "sampling_rate": sampling_rate,
        "variant": variant,
        "win_len": win_len,
    }


class MossFormerGANGenerator(torch.nn.Module):
    """G1. The constructor draws only the first SyncANet."""

    def __init__(
        self,
        *,
        seed: int,
        variant: str = "se_16k",
        n_fft: int = 400,
        win_len: int = 400,
        hop: int = 100,
        num_channel: int = 64,
        sampling_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            variant=variant,
            n_fft=n_fft,
            win_len=win_len,
            hop=hop,
            num_channel=num_channel,
            sampling_rate=sampling_rate,
        )
        with seed_model_init(self.seed):
            generator = _build_generator(self.architecture)
        self.generator = generator
        self.n_fft = int(self.architecture["n_fft"])
        self.win = int(self.architecture["win_len"])
        self.hop = int(self.architecture["hop"])
        self.sampling_rate = int(self.architecture["sampling_rate"])

    def forward(self, noisy_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real and imaginary spectra. ``_generator_terms`` calls this."""

        if not isinstance(noisy_spec, torch.Tensor):
            raise BuildError("MossFormerGAN generator input must be a tensor")
        return self.generator(noisy_spec)

    def export_for_inference(self) -> dict[str, Any]:
        """Inference checkpoint. The value is the SyncANet state."""

        return {"model": self.generator.state_dict()}


class MossFormerGANDiscriminator(torch.nn.Module):
    """D4. The constructor replays G1, D1, G2, D2, G3, D3, and G4, then keeps D4."""

    def __init__(
        self,
        *,
        seed: int,
        variant: str = "se_16k",
        n_fft: int = 400,
        win_len: int = 400,
        hop: int = 100,
        num_channel: int = 64,
        sampling_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            variant=variant,
            n_fft=n_fft,
            win_len=win_len,
            hop=hop,
            num_channel=num_channel,
            sampling_rate=sampling_rate,
        )
        # ``MossFormerGANTask`` moves G1 onto the device between D2 and G3.
        # That move does not draw random numbers. It is not repeated here.
        with seed_model_init(self.seed):
            _discard(_build_generator(self.architecture))
            _discard(Discriminator(ndf=16))
            _discard(_build_generator(self.architecture))
            _discard(Discriminator(ndf=16))
            _discard(_build_generator(self.architecture))
            _discard(Discriminator(ndf=16))
            _discard(_build_generator(self.architecture))
            discriminator = Discriminator(ndf=16)
        if not isinstance(discriminator, Discriminator):
            raise BuildError(f"discriminator builder returned {type(discriminator).__name__}")
        self.discriminator = discriminator
        self.n_fft = int(self.architecture["n_fft"])
        self.win = int(self.architecture["win_len"])
        self.hop = int(self.architecture["hop"])
        self.sampling_rate = int(self.architecture["sampling_rate"])

    def forward(self, reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        """Score one real or fake magnitude pair."""

        return self.discriminator(reference, estimate)


def _build_generator(architecture: dict[str, Any]) -> SyncANet:
    config = {key: architecture[key] for key in _BUILD_KEYS}
    generator = build_model("mossformergan", config, variant=str(architecture["variant"]))
    if not isinstance(generator, SyncANet):
        raise BuildError(f"mossformergan builder returned {type(generator).__name__}, expected SyncANet")
    return generator


def _discard(module: torch.nn.Module) -> None:
    """Drop a module after its constructor has consumed the init stream."""

    del module


def _seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BuildError("seed must be an integer")
    return seed


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{name} must be a positive integer")
    return value
