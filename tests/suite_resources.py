"""CPU and GPU budgets for the test runner.

Host names, device numbers, and a measured thread count are not constants.
``AUDIO_DB_TEST_WORKERS``, ``AUDIO_DB_TEST_THREADS``, ``AUDIO_DB_TEST_GPUS``,
and ``AUDIO_DB_DFN3_WORKERS`` override the defaults.
"""

from __future__ import annotations

import os

# Same cutoff as ``src.evaluation.gpu.BUSY_MEMORY_MIB``. Kept here so this
# module does not import the evaluation package (that import loads torch).
DEFAULT_BUSY_MEMORY_MIB = 1024
DEFAULT_WORKER_CAP = 8
DEFAULT_THREAD_CAP = 8
DEFAULT_DISTRIBUTION_CAP = 32


def _cpus(cpus: int | None) -> int:
    if cpus is None:
        cpus = os.cpu_count() or 1
    return max(1, int(cpus))


def module_workers(cpus: int | None = None) -> int:
    """How many test modules to run at once."""
    cpus = _cpus(cpus)
    override = os.environ.get("AUDIO_DB_TEST_WORKERS", "").strip()
    if override:
        return max(1, int(override))
    return max(1, min(DEFAULT_WORKER_CAP, max(1, cpus // 2)))


def intra_threads(workers: int, cpus: int | None = None) -> int:
    """Intra-op threads for one test process.

    The cap exists because one process using every core was slower than a
    small fixed width on the real-checkpoint smoke. ``deterministic: true``
    still calls ``torch.set_num_threads(1)`` after this.
    """
    override = os.environ.get("AUDIO_DB_TEST_THREADS", "").strip()
    if override:
        return max(1, int(override))
    cpus = _cpus(cpus)
    share = max(1, cpus // max(1, int(workers)))
    return max(1, min(DEFAULT_THREAD_CAP, share))


def apply_intra_threads() -> int:
    """Shrink an uncapped PyTorch default. An explicit override is applied as given.

    Call this from a test that loads a real model in-process. The runner sets
    ``OMP_NUM_THREADS`` before import, and this does not raise that limit.
    """
    import torch

    if os.environ.get("AUDIO_DB_TEST_THREADS", "").strip():
        chosen = intra_threads(1)
    else:
        chosen = min(int(torch.get_num_threads()), intra_threads(1))
    if int(torch.get_num_threads()) != chosen:
        torch.set_num_threads(chosen)
    return chosen


def distribution_workers(cpus: int | None = None) -> int:
    """Processes for independent DeepFilterNet3 distribution draws.

    A direct run may use up to ``DEFAULT_DISTRIBUTION_CAP`` cores. When the
    suite exports ``AUDIO_DB_TEST_WORKERS``, the draws share the machine with
    those modules.
    """
    cpus = _cpus(cpus)
    override = os.environ.get("AUDIO_DB_DFN3_WORKERS", "").strip()
    if override:
        return max(1, int(override))
    sibling = os.environ.get("AUDIO_DB_TEST_WORKERS", "").strip()
    share = cpus if not sibling else max(1, cpus // max(1, int(sibling)))
    return max(1, min(DEFAULT_DISTRIBUTION_CAP, share))


def gpu_test_environment(devices: list[int]) -> dict[str, str]:
    """Physical devices for one GPU lane.

    In-process tests see ``CUDA_VISIBLE_DEVICES``. ``tests/experiments/test_cuda_amp.py``
    copies ``AUDIO_DB_CUDA_GPU0`` and ``AUDIO_DB_CUDA_GPU1`` back into the child
    environment, so those values are the same physical indexes. Visible index 0
    inside that mask is not reused, or the child would leave the selected devices.
    """
    physical = [str(int(device)) for device in devices]
    if not physical:
        raise ValueError("GPU lane devices are empty")
    second = physical[1] if len(physical) > 1 else physical[0]
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(physical),
        "AUDIO_DB_CUDA_GPU0": physical[0],
        "AUDIO_DB_CUDA_GPU1": second,
    }


def thread_environment(threads: int) -> dict[str, str]:
    value = str(max(1, int(threads)))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }


def select_free_gpus(
    rows: list[dict[str, object]],
    busy_uuids: set[str],
    *,
    busy_mib: int = DEFAULT_BUSY_MEMORY_MIB,
) -> list[int]:
    """Physical indexes with low memory and no compute process.

    Memory at or above ``busy_mib`` matches the evaluation idle check. A
    compute process also counts as busy when its memory is still under that
    cutoff, so a small leftover allocation is not treated as a free device.
    """
    free: list[int] = []
    for row in rows:
        memory = int(row["memory_used_mib"])
        uuid = str(row["uuid"])
        if memory >= busy_mib or uuid in busy_uuids:
            continue
        free.append(int(row["index"]))
    return free


def parse_gpu_rows(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi gpu row: {line}")
        try:
            memory = int(parts[2])
        except ValueError:
            # MIG parents can report memory as N/A. Do not treat that as free.
            memory = DEFAULT_BUSY_MEMORY_MIB
        rows.append({"index": int(parts[0]), "uuid": parts[1], "memory_used_mib": memory})
    return rows


def parse_busy_uuids(text: str) -> set[str]:
    busy: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        uuid, _, _pid = line.partition(",")
        token = uuid.strip()
        if token:
            busy.add(token)
    return busy
