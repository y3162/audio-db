"""DNS TF-Locoformer for the experiment framework.

The module is ``src.se.build_model("tf_locoformer", architecture, variant="dns")``.
ESPnet ``xavier_uniform`` initialization runs inside ``seed_model_init`` before
this constructor returns, which is before the builder creates an optimizer.
The generator is the only submodule, so its parameters are this module's
parameters. There is no second copy for inference.

The network still normalizes mixture variance before the STFT. That flag is
independent of the training loss, which leaves variance normalization off.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from src.experiments.errors import BuildError, RuntimeFailure
from src.se import build_model
from src.se.models.tf_locoformer.model import TFLocoformer, _DNS_SEPARATOR
from src.experiments.calc.rng import seed_model_init
from src.experiments.calc.errors import TrainingError
from src.experiments.calc.initialization import apply_espnet_initialization


class TFLocoformerModel(torch.nn.Module):
    """Waveform-to-waveform DNS TF-Locoformer with one initialized generator."""

    def __init__(
        self,
        *,
        seed: int,
        architecture: Mapping[str, Any] | None = None,
        variant: str = "dns",
        init: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        if variant != "dns":
            raise BuildError(f"TFLocoformerModel implements variant dns, got {variant!r}")
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
                generator = build_model("tf_locoformer", config, variant="dns")
                apply_espnet_initialization(generator, init)
        except TrainingError as exc:
            raise BuildError(str(exc)) from exc
        except (TypeError, ValueError, AssertionError, AttributeError) as exc:
            raise BuildError(f"failed to build tf_locoformer: {exc}") from exc
        if not isinstance(generator, TFLocoformer):
            raise BuildError(f"tf_locoformer builder returned {type(generator).__name__}, expected TFLocoformer")
        self.generator = generator
        self.variant = "dns"
        self.init = init
        self.seed = seed
        self.architecture = config
        self.n_fft = int(generator.n_fft)
        self.hop_length = int(generator.hop_length)
        self.normalize_variance = bool(generator.normalize_variance)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return an enhanced waveform with the same shape as ``waveform``."""

        if not isinstance(waveform, torch.Tensor):
            raise RuntimeFailure("TF-Locoformer input must be a tensor")
        if waveform.ndim != 2:
            raise RuntimeFailure(f"TF-Locoformer input must be [batch, time], got rank {waveform.ndim}")
        if waveform.shape[-1] <= self.n_fft // 2:
            raise RuntimeFailure(
                f"waveform length {int(waveform.shape[-1])} is too short for n_fft {self.n_fft}"
            )
        try:
            enhanced = self.generator(waveform)
        except ValueError as exc:
            raise RuntimeFailure(str(exc)) from exc
        if enhanced.shape != waveform.shape:
            raise RuntimeFailure(
                f"TF-Locoformer length {tuple(enhanced.shape)} does not match the input {tuple(waveform.shape)}"
            )
        return enhanced

    def export_for_inference(self) -> dict[str, torch.Tensor]:
        """Full generator state.

        ``TFLocoformer`` keeps every parameter under the ``separator``
        submodule, so the keys already start with ``separator.``.
        ``src.se.models.tf_locoformer.inference.load`` strips that prefix and
        loads the remainder into ``separator``. A dictionary whose keys do
        not use the prefix is loaded into ``separator`` directly. Integer
        tensors keep their dtype.
        """

        exported: dict[str, torch.Tensor] = {}
        for name, value in self.generator.state_dict().items():
            if not torch.is_tensor(value):
                raise RuntimeFailure(f"parameter {name} is not a tensor")
            exported[name] = value.detach().cpu().clone()
        return exported


_LOCO_STFT = {
    "hop_length": 128,
    "n_fft": 256,
    "normalize_variance": True,
}


def _effective_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill omitted DNS keys from the published separator preset."""

    merged: dict[str, Any] = dict(_LOCO_STFT)
    for key, value in _DNS_SEPARATOR.items():
        merged[key] = list(value) if isinstance(value, list) else value
    unknown = set(config) - set(merged)
    if unknown:
        raise BuildError(f"unknown tf_locoformer architecture keys: {sorted(unknown)}")
    for key, value in config.items():
        merged[key] = list(value) if isinstance(value, list) else value
    if isinstance(merged["num_spk"], bool) or int(merged["num_spk"]) != 1:
        raise BuildError("tf_locoformer num_spk must be 1")
    emb_dim = merged["emb_dim"]
    num_groups = merged["num_groups"]
    attention_dim = merged["attention_dim"]
    n_heads = merged["n_heads"]
    for name, value in (
        ("emb_dim", emb_dim),
        ("num_groups", num_groups),
        ("attention_dim", attention_dim),
        ("n_heads", n_heads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BuildError(f"{name} must be a positive integer")
    if emb_dim % num_groups != 0:
        raise BuildError("emb_dim must be divisible by num_groups")
    if attention_dim % n_heads != 0:
        raise BuildError("attention_dim must be divisible by n_heads")
    ffn_type = merged["ffn_type"]
    ffn_hidden = merged["ffn_hidden_dim"]
    if isinstance(ffn_type, list) or isinstance(ffn_hidden, list):
        if not isinstance(ffn_type, list) or not isinstance(ffn_hidden, list) or len(ffn_type) != len(ffn_hidden):
            raise BuildError("ffn_type and ffn_hidden_dim must be the same length")
    if ffn_type not in ("swiglu_conv1d", ["swiglu_conv1d"], ["swiglu_conv1d", "swiglu_conv1d"]) and not (
        isinstance(ffn_type, list) and ffn_type and all(item == "swiglu_conv1d" for item in ffn_type)
    ):
        raise BuildError("ffn_type must be swiglu_conv1d")
    return merged


_INTEGER_KEYS = frozenset(
    {
        "attention_dim",
        "conv1d_kernel",
        "conv1d_shift",
        "emb_dim",
        "ffn_hidden_dim",
        "hop_length",
        "n_fft",
        "n_heads",
        "n_layers",
        "num_groups",
        "num_spk",
    }
)
_FLOAT_KEYS = frozenset({"dropout", "eps"})


def _numeric_architecture(config: dict[str, Any]) -> dict[str, Any]:
    """Cast numeric YAML strings. ``1e-5`` is a string in YAML 1.1."""

    return {key: _coerce_architecture(key, value) for key, value in config.items()}


def _coerce_architecture(key: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_coerce_architecture(key, item) for item in value]
    if not isinstance(value, str):
        return value
    if key not in _INTEGER_KEYS and key not in _FLOAT_KEYS:
        return value
    try:
        if key in _INTEGER_KEYS:
            return int(value)
        return float(value)
    except ValueError as exc:
        raise BuildError(f"architecture {key} must be numeric, got {value!r}") from exc
