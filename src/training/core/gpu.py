"""Refuse a CUDA device that already appears busy.

``CUDA_VISIBLE_DEVICES`` renumbers the devices PyTorch sees. The memory check
uses the physical index from that variable, not a number stored in the config.
"""

from __future__ import annotations

import os
import subprocess

from src.evaluation.gpu import BUSY_MEMORY_MIB, parse_nvidia_smi
from src.training.errors import TrainingError


def physical_index(local_rank: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return local_rank
    try:
        indexes = [int(part) for part in visible.split(",") if part.strip()]
    except ValueError as exc:
        raise TrainingError(f"CUDA_VISIBLE_DEVICES is invalid: {visible!r}") from exc
    if local_rank >= len(indexes):
        raise TrainingError(
            f"local_rank {local_rank} is outside CUDA_VISIBLE_DEVICES={visible}"
        )
    return indexes[local_rank]


def assert_gpu_idle(local_rank: int, *, allow_busy: bool) -> None:
    if allow_busy:
        return
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise TrainingError("CUDA was requested but nvidia-smi is not available") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise TrainingError(f"nvidia-smi failed: {detail}")
    rows = parse_nvidia_smi(completed.stdout)
    index = physical_index(local_rank)
    chosen = next((row for row in rows if row["index"] == index), None)
    if chosen is None:
        visible = [row["index"] for row in rows]
        raise TrainingError(f"GPU {index} is not visible to nvidia-smi (visible: {visible})")
    if chosen["memory_used_mib"] >= BUSY_MEMORY_MIB:
        raise TrainingError(
            f"GPU {index} is in use ({chosen['memory_used_mib']} MiB, "
            f"utilization {chosen['utilization']}%). Set CUDA_VISIBLE_DEVICES to a free GPU."
        )
