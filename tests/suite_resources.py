"""CPU and GPU budgets for the test runner.

Host names, device numbers, and a measured thread count are not constants.
``AUDIO_DB_TEST_WORKERS``, ``AUDIO_DB_TEST_THREADS``, ``AUDIO_DB_TEST_GPUS``,
and ``AUDIO_DB_DFN3_WORKERS`` override the defaults. CPU defaults use the
CPUs this process can run on. GPU defaults stay inside the parent
``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Same cutoff as ``src.evaluation.gpu.BUSY_MEMORY_MIB``. Kept here so this
# module does not import the evaluation package (that import loads torch).
DEFAULT_BUSY_MEMORY_MIB = 1024
DEFAULT_WORKER_CAP = 8
DEFAULT_THREAD_CAP = 8
DEFAULT_DISTRIBUTION_CAP = 32


def _affinity_cpus() -> int | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return max(1, len(getter(0)))
    except OSError:
        return None


def _quota_cpus(quota: str, period: str) -> int | None:
    try:
        allowed = int(str(quota).strip())
        window = int(str(period).strip())
    except ValueError:
        return None
    if allowed <= 0 or window <= 0:
        return None
    return max(1, allowed // window)


def _parse_cpu_max(text: str) -> int | None:
    """cgroup v2 ``cpu.max``. ``max`` is unlimited and contributes no cap."""
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    return _quota_cpus(parts[0], parts[1])


def _parse_cfs_quota(quota: str, period: str) -> int | None:
    """cgroup v1 CFS quota. ``-1`` is unlimited."""
    if str(quota).strip() == "-1":
        return None
    return _quota_cpus(quota, period)


def _read_cgroup_quota(directory: Path) -> int | None:
    cpu_max = directory / "cpu.max"
    if cpu_max.is_file():
        try:
            return _parse_cpu_max(cpu_max.read_text(encoding="utf-8"))
        except OSError:
            return None
    quota = directory / "cpu.cfs_quota_us"
    period = directory / "cpu.cfs_period_us"
    if quota.is_file() and period.is_file():
        try:
            return _parse_cfs_quota(quota.read_text(encoding="utf-8"), period.read_text(encoding="utf-8"))
        except OSError:
            return None
    return None


def _tightest_quota(start: Path, root: Path) -> int | None:
    """Smallest finite quota from ``start`` up through ``root``."""
    limit: int | None = None
    current = start
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        value = _read_cgroup_quota(current)
        if value is not None:
            limit = value if limit is None else min(limit, value)
        if current == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return limit


def _cgroup_cpus() -> int | None:
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    limits: list[int] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, relative = parts
        relative = relative.strip() or "/"
        value: int | None = None
        if hierarchy == "0" and controllers == "":
            root = Path("/sys/fs/cgroup")
            value = _tightest_quota(root / relative.lstrip("/"), root)
        elif "cpu" in controllers.split(","):
            for root in (Path("/sys/fs/cgroup/cpu,cpuacct"), Path("/sys/fs/cgroup/cpu")):
                if root.is_dir():
                    value = _tightest_quota(root / relative.lstrip("/"), root)
                    break
        if value is not None:
            limits.append(value)
    if not limits:
        return None
    return min(limits)


def available_cpus() -> int:
    """CPUs this process can use.

    Affinity and a cgroup CPU quota both apply. ``os.cpu_count()`` is the
    fallback when neither is available. Explicit worker and thread settings
    do not pass through here.
    """
    limits: list[int] = []
    affinity = _affinity_cpus()
    if affinity is not None:
        limits.append(affinity)
    quota = _cgroup_cpus()
    if quota is not None:
        limits.append(quota)
    if not limits:
        limits.append(os.cpu_count() or 1)
    return max(1, min(limits))


def _cpus(cpus: int | None) -> int:
    if cpus is None:
        cpus = available_cpus()
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


def process_threads(cpu_workers: int, extra_processes: int = 0, cpus: int | None = None) -> int:
    """Threads for one test process.

    ``AUDIO_DB_TEST_THREADS`` is used as given for every process. Without
    that override, CPU workers and ``extra_processes`` (the GPU lane, when
    it runs) share the available CPUs. The per-process cap stays
    ``DEFAULT_THREAD_CAP``.
    """
    group = max(1, int(cpu_workers)) + max(0, int(extra_processes))
    return intra_threads(group, cpus)


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

    A direct run may use up to ``DEFAULT_DISTRIBUTION_CAP`` of the CPUs this
    process can use. When the suite exports ``AUDIO_DB_TEST_WORKERS``, each
    draw is one thread and the count is those CPUs divided by the workers.
    ``AUDIO_DB_DFN3_WORKERS`` replaces that calculation. The renderer sets
    the OpenMP family to 1 before spawning, so the draws do not multiply
    ``AUDIO_DB_TEST_THREADS``.
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
    One physical device is repeated in ``AUDIO_DB_CUDA_GPU1`` so both variables
    stay defined. ``CUDA_VISIBLE_DEVICES`` still lists that device once, and
    tests that need two devices skip with NOT_RUN.
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


def _visible_tokens(value: str | None) -> list[str] | None:
    """Tokens in ``CUDA_VISIBLE_DEVICES``. None means the variable is unset."""
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _token_kind(token: str) -> str:
    folded = token.upper()
    if folded.startswith("MIG-"):
        return "mig"
    if folded.startswith("GPU-"):
        return "uuid"
    if token.isascii() and token.isdigit():
        return "index"
    return "other"


def parse_physical_gpu_list(text: str, *, label: str) -> list[int]:
    """Comma-separated physical indexes. UUIDs and MIG ids are rejected."""
    indexes: list[int] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        if _token_kind(token) != "index":
            raise ValueError(f"{label} must be physical GPU indexes, got {text!r}")
        indexes.append(int(token))
    if not indexes:
        raise ValueError(f"{label} is empty")
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"{label} repeats a GPU: {text!r}")
    return indexes


def gpu_inventory_required(visible: str | None, requested: str | None) -> bool:
    """True when the lane must read nvidia-smi before it can choose.

    An empty value, a MIG id, or another unsupported token is decided
    without a query. A GPU UUID needs the UUID map. Automatic selection
    also needs the idle check. An explicit physical list inside an integer
    mask does not.
    """
    tokens = _visible_tokens(visible)
    if tokens is not None:
        if not tokens:
            return False
        for token in tokens:
            kind = _token_kind(token)
            if kind in {"mig", "other"}:
                return False
            if kind == "uuid":
                return True
    return requested is None


def select_test_gpus(
    *,
    visible: str | None,
    requested: str | None,
    rows: list[dict[str, object]] | None,
    busy_uuids: set[str] | None = None,
    query_error: str | None = None,
) -> tuple[list[int], str]:
    """Physical indexes for the GPU lane, and why those indexes were chosen.

    An empty list means the lane is NOT_RUN. ``visible`` is the parent
    ``CUDA_VISIBLE_DEVICES`` text, or None when that variable is unset.
    ``requested`` is ``AUDIO_DB_TEST_GPUS``, or None to choose automatically.
    Indexes are nvidia-smi physical indexes. A ``GPU-`` UUID is resolved
    from ``rows``. A MIG id or any other token is not turned into a
    different GPU. An explicit list skips the idle check and still cannot
    name a device outside the parent mask.
    """
    busy = set() if busy_uuids is None else busy_uuids
    tokens = _visible_tokens(visible)
    wanted = None if requested is None else parse_physical_gpu_list(requested, label="AUDIO_DB_TEST_GPUS")

    if tokens is not None and not tokens:
        if wanted is not None:
            return [], "AUDIO_DB_TEST_GPUS is outside empty CUDA_VISIBLE_DEVICES"
        return [], "CUDA_VISIBLE_DEVICES is empty"

    if tokens is not None:
        bad = next((token for token in tokens if _token_kind(token) in {"mig", "other"}), None)
        if bad is not None:
            return [], (
                f"CUDA_VISIBLE_DEVICES token {bad!r} is not a physical index or GPU UUID; "
                "it was not mapped to another GPU"
            )
        if any(_token_kind(token) == "uuid" for token in tokens) and rows is None:
            detail = query_error or "nvidia-smi is not available"
            return [], f"CUDA_VISIBLE_DEVICES uses a GPU UUID; {detail}; it was not mapped to another GPU"
        present = None if rows is None else {int(row["index"]) for row in rows}
        uuids = {} if rows is None else {str(row["uuid"]).casefold(): int(row["index"]) for row in rows}
        listed: list[int] = []
        missing: list[str] = []
        for token in tokens:
            if _token_kind(token) == "index":
                index = int(token)
                if present is not None and index not in present:
                    missing.append(token)
                    continue
                listed.append(index)
            else:
                index = uuids.get(token.casefold())
                if index is None:
                    missing.append(token)
                    continue
                listed.append(index)
        if not listed:
            shown = ", ".join(missing) if missing else str(visible)
            return [], f"CUDA_VISIBLE_DEVICES ({shown}) did not match an nvidia-smi GPU"
        if len(listed) != len(set(listed)):
            return [], "CUDA_VISIBLE_DEVICES lists the same GPU more than once"
        ignored = f"; ignored unmatched {', '.join(missing)}" if missing else ""
        if wanted is not None:
            outside = [index for index in wanted if index not in set(listed)]
            if outside:
                visible_text = ", ".join(str(index) for index in listed)
                return [], f"AUDIO_DB_TEST_GPUS {outside} is outside CUDA_VISIBLE_DEVICES ({visible_text})"
            return wanted, "AUDIO_DB_TEST_GPUS" + ignored
        if rows is None:
            return [], query_error or "nvidia-smi is not available"
        free = set(select_free_gpus(rows, busy))
        chosen = [index for index in listed if index in free]
        if not chosen:
            visible_text = ", ".join(str(index) for index in listed)
            return [], f"no idle GPU inside CUDA_VISIBLE_DEVICES ({visible_text})"
        return chosen, "idle within CUDA_VISIBLE_DEVICES" + ignored

    if wanted is not None:
        return wanted, "AUDIO_DB_TEST_GPUS"
    if rows is None:
        return [], query_error or "nvidia-smi is not available"
    free_indexes = select_free_gpus(rows, busy)
    if not free_indexes:
        return [], "no GPU is under 1024 MiB and free of compute processes"
    return free_indexes, "idle"
