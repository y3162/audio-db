"""MossFormerGAN generator and discriminator wrappers.

Each wrapper enters ``seed_model_init`` alone. The discriminator network is
the third-party package ``discriminator/`` and is not reimplemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.experiments.calc.rng import seed_model_init
from src.experiments.errors import BuildError
from src.models.features import parse_output_features
from src.models.mossformergan.discriminator import Discriminator
from src.models import build_model
from src.models.mossformergan.generator import SyncANet

_BUILD_KEYS = (
    "approx_qk_dim",
    "dense_depth",
    "dropout",
    "expansion_factor",
    "hop",
    "n_fft",
    "n_head",
    "n_layers",
    "num_channel",
    "query_key_dim",
    "win_len",
)

OFFICIAL_ARCHITECTURE = {
    "approx_qk_dim": 512,
    "dense_depth": 4,
    "dropout": 0.1,
    "expansion_factor": 4.0,
    "hop": 100,
    "n_fft": 400,
    "n_head": 4,
    "n_layers": 6,
    "num_channel": 64,
    "query_key_dim": 128,
    "sampling_rate": 16000,
    "variant": "se_16k",
    "win_len": 400,
}

def architecture_config(
    *,
    variant: str,
    n_fft: int,
    win_len: int,
    hop: int,
    num_channel: int,
    sampling_rate: int,
    n_layers: int = 6,
    n_head: int = 4,
    approx_qk_dim: int = 512,
    dense_depth: int = 4,
    dropout: float = 0.1,
    query_key_dim: int = 128,
    expansion_factor: float = 4.0,
) -> dict[str, Any]:
    """Return the architecture both modules must share."""

    if variant != "se_16k":
        raise BuildError(f"MossFormerGAN variant must be se_16k, got {variant!r}")
    n_fft = _positive_int("n_fft", n_fft)
    win_len = _positive_int("win_len", win_len)
    hop = _positive_int("hop", hop)
    num_channel = _positive_int("num_channel", num_channel)
    sampling_rate = _positive_int("sampling_rate", sampling_rate)
    n_layers = _positive_int("n_layers", n_layers)
    n_head = _positive_int("n_head", n_head)
    approx_qk_dim = _positive_int("approx_qk_dim", approx_qk_dim)
    dense_depth = _positive_int("dense_depth", dense_depth)
    query_key_dim = _positive_int("query_key_dim", query_key_dim)
    if sampling_rate != 16000:
        raise BuildError(f"MossFormerGAN se_16k requires 16000 Hz, got {sampling_rate}")
    if win_len != n_fft:
        raise BuildError(f"MossFormerGAN win_len {win_len} must equal n_fft {n_fft}")
    if num_channel % n_head != 0:
        raise BuildError(f"num_channel must be divisible by n_head ({n_head})")
    dropout = _unit_interval("dropout", dropout)
    expansion_factor = _even_expansion(expansion_factor)
    return {
        "approx_qk_dim": approx_qk_dim,
        "dense_depth": dense_depth,
        "dropout": dropout,
        "expansion_factor": expansion_factor,
        "hop": hop,
        "n_fft": n_fft,
        "n_head": n_head,
        "n_layers": n_layers,
        "num_channel": num_channel,
        "query_key_dim": query_key_dim,
        "sampling_rate": sampling_rate,
        "variant": variant,
        "win_len": win_len,
    }

def require_official_architecture(architecture: Mapping[str, Any], claim: str) -> tuple[str, ...]:
    """Record published-structure differences. A different width still runs.

    Channel divisibility and the ``se_16k`` variant stay in
    ``architecture_config``. This function does not refuse a run.
    """

    if claim != "official_faithful":
        return ()
    if dict(architecture) == OFFICIAL_ARCHITECTURE:
        return ()
    differences: list[str] = []
    for key in sorted(set(architecture) | set(OFFICIAL_ARCHITECTURE)):
        actual = architecture.get(key)
        published = OFFICIAL_ARCHITECTURE.get(key)
        if actual != published:
            differences.append(f"mossformergan.{key} is {actual!r}, published value is {published!r}")
    return tuple(differences)

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
        n_layers: int = 6,
        n_head: int = 4,
        approx_qk_dim: int = 512,
        dense_depth: int = 4,
        dropout: float = 0.1,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
        output_features: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.output_features = parse_output_features(output_features, allowed=frozenset())
        self.seed = _seed(seed)
        self.architecture = architecture_config(
            variant=variant,
            n_fft=n_fft,
            win_len=win_len,
            hop=hop,
            num_channel=num_channel,
            sampling_rate=sampling_rate,
            n_layers=n_layers,
            n_head=n_head,
            approx_qk_dim=approx_qk_dim,
            dense_depth=dense_depth,
            dropout=dropout,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
        )
        with seed_model_init(self.seed):
            generator = _build_generator(self.architecture)
        self.generator = generator
        self.n_fft = int(self.architecture["n_fft"])
        self.win = int(self.architecture["win_len"])
        self.hop = int(self.architecture["hop"])
        self.sampling_rate = int(self.architecture["sampling_rate"])

    def forward(self, noisy_spec: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the named SyncANet spectra. The SyncANet tuple itself is unchanged."""

        if not isinstance(noisy_spec, torch.Tensor):
            raise BuildError("MossFormerGAN generator input must be a tensor")
        real, imag = self.generator(noisy_spec)
        return {"predicted_real": real, "predicted_imag": imag}

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
        n_layers: int = 6,
        n_head: int = 4,
        approx_qk_dim: int = 512,
        dense_depth: int = 4,
        dropout: float = 0.1,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
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
            n_layers=n_layers,
            n_head=n_head,
            approx_qk_dim=approx_qk_dim,
            dense_depth=dense_depth,
            dropout=dropout,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
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

def _unit_interval(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{name} must be a number from 0 to 1")
    number = float(value)
    if number < 0.0 or number > 1.0 or number != number:
        raise BuildError(f"{name} must be a number from 0 to 1")
    return number

def _even_expansion(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError("expansion_factor must be a positive even integer")
    number = float(value)
    if number != int(number) or int(number) < 2 or int(number) % 2 != 0:
        raise BuildError("expansion_factor must be a positive even integer")
    return float(int(number))
