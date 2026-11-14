"""MS-SNSD TF-GridNet for the experiment framework.

The module is ``src.se.build_model("tf_gridnet", architecture, variant="ms-snsd")``.
ESPnet ``xavier_uniform`` initialization runs inside ``seed_model_init`` before
this constructor returns, which is before the builder creates an optimizer.
The generator is the only submodule, so its parameters are this module's
parameters. There is no second copy for inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.se import build_model
from src.se.models.tf_gridnet.model import TFGridNet
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.initialization import apply_espnet_initialization


class TFGridNetModel(torch.nn.Module):
    """Waveform-to-waveform MS-SNSD TF-GridNet with one initialized generator."""

    def __init__(
        self,
        *,
        seed: int,
        architecture: Mapping[str, Any] | None = None,
        variant: str = "ms-snsd",
        init: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if variant != "ms-snsd":
            raise BuildError(f"TFGridNetModel implements variant ms-snsd, got {variant!r}")
        if not isinstance(init, str) or not init:
            raise BuildError("init must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BuildError("seed must be an integer >= 0")
        if architecture is None:
            provided: dict[str, Any] = {}
        elif isinstance(architecture, Mapping):
            provided = _numeric_architecture(dict(architecture))
        else:
            raise BuildError("architecture must be a mapping")
        config = _effective_architecture(provided)
        try:
            with seed_model_init(seed):
                generator = build_model("tf_gridnet", config, variant="ms-snsd")
                apply_espnet_initialization(generator, init)
        except TrainingError as exc:
            raise BuildError(str(exc)) from exc
        except (TypeError, ValueError, AssertionError, AttributeError) as exc:
            raise BuildError(f"failed to build tf_gridnet: {exc}") from exc
        if not isinstance(generator, TFGridNet):
            raise BuildError(f"tf_gridnet builder returned {type(generator).__name__}, expected TFGridNet")
        self.generator = generator
        self.variant = "ms-snsd"
        self.init = init
        self.seed = seed
        self.architecture = config
        self.n_fft = int(generator.n_fft)
        self.stride = int(generator.stride)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return an enhanced waveform with the same shape as ``waveform``."""

        if not isinstance(waveform, torch.Tensor):
            raise RuntimeFailure("TF-GridNet input must be a tensor")
        if waveform.ndim != 2:
            raise RuntimeFailure(f"TF-GridNet input must be [batch, time], got rank {waveform.ndim}")
        if waveform.shape[-1] <= self.n_fft // 2:
            raise RuntimeFailure(
                f"waveform length {int(waveform.shape[-1])} is too short for n_fft {self.n_fft}"
            )
        enhanced = self.generator(waveform)
        if enhanced.shape != waveform.shape:
            raise RuntimeFailure(
                f"TF-GridNet length {tuple(enhanced.shape)} does not match the input {tuple(waveform.shape)}"
            )
        return enhanced

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        """ESPnet separator state. Integer tensors keep their dtype."""

        exported: dict[str, torch.Tensor] = {}
        for name, value in self.generator.state_dict().items():
            if not torch.is_tensor(value):
                raise RuntimeFailure(f"parameter {name} is not a tensor")
            exported["separator." + name] = value.detach().cpu().clone()
        return exported


_GRID_DEFAULTS = {
    "activation": "prelu",
    "attn_approx_qk_dim": 512,
    "attn_n_head": 4,
    "emb_dim": 32,
    "emb_hs": 4,
    "emb_ks": 4,
    "eps": 1e-5,
    "lstm_hidden_units": 128,
    "n_fft": 512,
    "n_layers": 4,
    "stride": 256,
}
_GRID_INTS = frozenset(
    {
        "attn_approx_qk_dim",
        "attn_n_head",
        "emb_dim",
        "emb_hs",
        "emb_ks",
        "lstm_hidden_units",
        "n_fft",
        "n_layers",
        "stride",
    }
)


def _effective_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill omitted MS-SNSD keys from the published constructor defaults."""

    unknown = set(config) - set(_GRID_DEFAULTS)
    if unknown:
        raise BuildError(f"unknown tf_gridnet architecture keys: {sorted(unknown)}")
    merged = dict(_GRID_DEFAULTS)
    merged.update(config)
    for key in _GRID_INTS:
        value = merged[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BuildError(f"{key} must be a positive integer")
    if not isinstance(merged["activation"], str) or not merged["activation"]:
        raise BuildError("activation must be a non-empty string")
    eps = merged["eps"]
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not (float(eps) > 0):
        raise BuildError("eps must be a positive number")
    merged["eps"] = float(eps)
    if merged["emb_dim"] % merged["attn_n_head"] != 0:
        raise BuildError("emb_dim must be divisible by attn_n_head")
    return merged


_INTEGER_ARCHITECTURE = frozenset(
    {
        "attn_approx_qk_dim",
        "attn_n_head",
        "emb_dim",
        "emb_hs",
        "emb_ks",
        "lstm_hidden_units",
        "n_fft",
        "n_layers",
        "stride",
    }
)


def _numeric_architecture(config: dict[str, Any]) -> dict[str, Any]:
    """Cast numeric YAML strings. ``1e-5`` is a string in YAML 1.1."""

    cleaned = dict(config)
    for key, value in cleaned.items():
        if not isinstance(value, str):
            continue
        try:
            if key in _INTEGER_ARCHITECTURE:
                cleaned[key] = int(value)
            elif key == "eps":
                cleaned[key] = float(value)
        except ValueError as exc:
            raise BuildError(f"architecture {key} must be numeric, got {value!r}") from exc
    return cleaned
