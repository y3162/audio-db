"""Average the top validation-loss checkpoints the way ESPnet averages n-best models.

ESPnet ``espnet2/main_funcs/average_nbest_models.py`` sorts epochs by the
criterion value with a stable sort, sums the state dictionaries, divides
floating tensors by n, and leaves integer tensors (``num_batches_tracked``)
as the sum. The DNS recipe asks for ``valid.loss.ave_5best``. When fewer than
n checkpoints exist, that file is not an average of the shorter list: ESPnet
falls back to an ``ave_1best`` symlink. This module only averages when at
least n files are supplied. The saved suffix is ``.pt``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.errors import TrainingError


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Mean of floating tensors and sum of integer tensors. Keys must match."""
    if len(states) < 2:
        raise TrainingError("n-best averaging needs at least two checkpoints")
    keys = list(states[0])
    for state in states[1:]:
        if list(state) != keys:
            raise TrainingError("n-best checkpoints do not have the same keys in the same order")
    averaged: dict[str, torch.Tensor] = {}
    count = len(states)
    for key in keys:
        values = []
        for state in states:
            value = state[key]
            if not torch.is_tensor(value):
                raise TrainingError(f"n-best value {key} is not a tensor")
            values.append(value.detach())
        total = values[0].clone()
        for value in values[1:]:
            if value.shape != total.shape or value.dtype != total.dtype:
                raise TrainingError(f"n-best value {key} does not match across checkpoints")
            total = total + value
        if total.dtype.is_floating_point:
            total = total / count
        averaged[key] = total
    return averaged


def average_checkpoint_files(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Load epoch files and average them. A missing file is an error."""
    states = []
    for path in paths:
        if not path.is_file():
            raise TrainingError(f"n-best checkpoint is missing: {path.name}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise TrainingError(f"could not read n-best checkpoint {path.name}") from exc
        if not isinstance(payload, dict):
            raise TrainingError(f"n-best checkpoint {path.name} is not a state dictionary")
        states.append(payload)
    return average_state_dicts(states)
