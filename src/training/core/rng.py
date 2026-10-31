"""Separate model, dropout, and data random streams.

Crops and sample order are pure functions of a seed and a sample id. They do
not consume the global PyTorch generator, so their results do not depend on
how many random numbers the model drew earlier.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from src.training.errors import TrainingError


def mix_seed(*parts: object) -> int:
    """Stable 63-bit seed. This does not use Python's salted ``hash``."""
    payload = "\n".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**63 - 1) or 1


def set_determinism(enabled: bool, *, cuda: bool = False) -> None:
    """Apply the process-wide deterministic settings used by training.

    ``CUBLAS_WORKSPACE_CONFIG`` is set when it is still unset. It has an effect
    only if CUDA has not been initialized yet, so the training command sets it
    before the first CUDA call. The CUDA attention toggles run only when the
    caller is actually training on a GPU, so a CPU run does not initialize one.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cuda.matmul.allow_tf32 = not enabled
    torch.backends.cudnn.allow_tf32 = not enabled
    if not enabled:
        torch.use_deterministic_algorithms(False)
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    if cuda:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


@contextmanager
def seed_model_init(seed: int) -> Iterator[None]:
    """Seed parameter initialization, then restore the previous generators.

    Construction draws from the CPU generator and this seed alone. Modules are
    created on CPU and then moved, so the CUDA generator is left untouched. A
    later call with the same seed rebuilds the same initial weights even if
    other code used random numbers in between.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingError("seed must be an integer")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def seed_training_stream(seed: int, *, cuda: bool) -> None:
    """Seed dropout and other training-time global generators.

    The value is mixed so it is not the same sequence as model initialization.
    CPU training does not touch the CUDA generator, which would initialize a
    CUDA context on every visible GPU.
    """
    stream = mix_seed("dropout", seed)
    random.seed(stream)
    np.random.seed(stream % (2**32))
    torch.manual_seed(stream)
    if cuda:
        if not torch.cuda.is_available():
            raise TrainingError("CUDA seeding was requested but CUDA is not available")
        torch.cuda.manual_seed_all(stream)


def capture_rng(*, cuda: bool) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if cuda:
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict, *, cuda: bool) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if cuda:
        saved = state.get("cuda")
        if saved is None:
            raise TrainingError("checkpoint has no CUDA generator state")
        torch.cuda.set_rng_state_all(saved)
