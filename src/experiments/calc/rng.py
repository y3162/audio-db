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

from src.experiments.calc.errors import TrainingError


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
    torch.random.default_generator.manual_seed(seed)
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
    torch.random.default_generator.manual_seed(stream)
    if cuda:
        if not torch.cuda.is_available():
            raise TrainingError("CUDA seeding was requested but CUDA is not available")
        torch.cuda.manual_seed_all(stream)


def capture_rng(*, cuda: bool) -> dict:
    """Process-local snapshot, including every visible CUDA device.

    This is the single-process checkpoint field. A multi-process run must not
    restore it on every rank: ``get_rng_state_all`` includes devices this
    process does not train on, and those slots are not the other ranks'
    generators. Use :func:`capture_rank_rng` for that.
    """
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
    # A checkpoint loaded with map_location=cuda moves even the CPU RNG byte
    # tensor to CUDA; both RNG setters require CPU byte tensors.
    torch.random.set_rng_state(state["torch"].cpu())
    if cuda:
        saved = state.get("cuda")
        if saved is None:
            raise TrainingError("checkpoint has no CUDA generator state")
        torch.cuda.set_rng_state_all([item.cpu() for item in saved])


def _numpy_state_copy(state: tuple) -> tuple:
    """Copy NumPy's MT state. ``get_state`` shares the live key array."""
    kind, keys, position, has_gauss, cached = state
    return (kind, np.array(keys, copy=True), int(position), int(has_gauss), float(cached))


def capture_rank_rng(*, cuda: bool) -> dict:
    """RNG for the device this process actually trains on.

    Only the current CUDA device is stored. Other visible devices stay at
    the seed in this process and are not another rank's generator.
    """
    state = {
        "python": random.getstate(),
        "numpy": _numpy_state_copy(np.random.get_state()),
        "torch": torch.random.get_rng_state().clone(),
    }
    if cuda:
        index = int(torch.cuda.current_device())
        state["cuda_device"] = index
        state["cuda"] = torch.cuda.get_rng_state(index).clone()
    return state


def restore_rank_rng(state: dict, *, cuda: bool) -> None:
    """Restore one rank's generators onto this process's current device."""
    random.setstate(state["python"])
    np.random.set_state(_numpy_state_copy(state["numpy"]))
    torch.random.set_rng_state(state["torch"].cpu())
    if not cuda:
        return
    saved = state.get("cuda")
    if saved is None or "cuda_device" not in state:
        raise TrainingError("checkpoint rank is missing its CUDA generator")
    index = int(torch.cuda.current_device())
    saved_index = int(state["cuda_device"])
    if saved_index != index:
        raise TrainingError(
            f"checkpoint CUDA RNG device {saved_index} does not match current device {index}"
        )
    torch.cuda.set_rng_state(saved.detach().cpu(), index)


def resume_rng_plan(saved_rng: object, rng_by_rank: object, *, rank: int, world_size: int) -> tuple[str, dict]:
    """Choose the RNG blob for this rank.

    One process keeps the historical ``rng`` snapshot, which lists every
    visible CUDA device. Two or more processes require ``rng_by_rank``.
    Copying rank 0's device list onto the other ranks rewinds their current
    device to the untouched seed slot.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise TrainingError(f"invalid world size {world_size}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
        raise TrainingError(f"invalid rank {rank} for world size {world_size}")
    if world_size == 1:
        if not isinstance(saved_rng, dict):
            raise TrainingError("checkpoint has no RNG state")
        return "legacy", saved_rng
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != world_size:
        raise TrainingError(
            "multi-process checkpoint has no per-rank RNG; "
            "refusing to copy one rank's CUDA generator onto the other ranks"
        )
    state = rng_by_rank[rank]
    if not isinstance(state, dict):
        raise TrainingError(f"rank {rank} RNG state is missing")
    for key in ("python", "numpy", "torch"):
        if key not in state:
            raise TrainingError(f"rank {rank} RNG is missing {key}")
    return "rank", state
