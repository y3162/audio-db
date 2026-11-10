"""ESPnet-style parameter initialization for speech-enhancement tasks.

``xavier_uniform`` follows ``espnet2.torch_utils.initialize``: every parameter
with rank greater than 1 is drawn with ``xavier_uniform_``, one-dimensional
parameters whose name contains ``.bias`` are set to zero, then ``Embedding``,
``LayerNorm``, and ``GroupNorm`` are reset to their module defaults. The
caller seeds the CPU generator. This is not applied by the inference API.
"""

from __future__ import annotations

import torch
from torch import nn

from src.experiments.calc.errors import TrainingError


def apply_espnet_initialization(model: nn.Module, init: str) -> None:
    if init != "xavier_uniform":
        raise TrainingError(f"unsupported initialization {init!r}")
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter.data)
    for name, parameter in model.named_parameters():
        if ".bias" in name and parameter.dim() == 1:
            parameter.data.zero_()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm, nn.GroupNorm)):
            module.reset_parameters()
        hook = getattr(module, "espnet_initialization_fn", None)
        if hook is not None:
            hook()
