"""Checkpoint access; always load state dictionaries strictly."""
from collections.abc import Mapping
from pathlib import Path
import torch


def checkpoint_paths(checkpoint, required):
    if isinstance(checkpoint, (str, Path)):
        if len(required) != 1:
            raise ValueError(f'expected named checkpoints: {", ".join(required)}')
        paths = {required[0]: checkpoint}
    elif isinstance(checkpoint, Mapping):
        paths = dict(checkpoint)
    else:
        raise TypeError('checkpoint must be a path or mapping of names to paths')
    if set(paths) != set(required):
        raise ValueError(f'checkpoint names must be exactly {required}; got {tuple(paths)}')
    return {key: Path(value) for key, value in paths.items()}


def load_state(model, path, key=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if key is not None:
        if not isinstance(state, Mapping) or key not in state:
            raise ValueError(f'checkpoint {path} has no {key!r} state dictionary')
        state = state[key]
    model.load_state_dict(state, strict=True)
