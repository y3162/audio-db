"""Refuse CUDA devices that already appear busy."""

from __future__ import annotations

import subprocess

from src.evaluation.errors import ConfigError

BUSY_MEMORY_MIB = 1024


def parse_nvidia_smi(text: str) -> list[dict[str, int]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ConfigError(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used_mib": int(parts[1]),
                "utilization": int(parts[2]),
            }
        )
    return rows


def device_index(device: str) -> int:
    if device == "cuda":
        return 0
    _, _, index = device.partition(":")
    if not index:
        raise ConfigError(f"invalid CUDA device: {device}")
    return int(index)


def assert_gpu_idle(device: str, *, allow_busy: bool = False) -> None:
    """Check nvidia-smi before using a GPU. CPU devices skip this."""
    if not str(device).startswith("cuda"):
        return
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
        raise ConfigError("CUDA was requested but nvidia-smi is not available") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ConfigError(f"nvidia-smi failed: {detail}")
    rows = parse_nvidia_smi(completed.stdout)
    index = device_index(device)
    chosen = next((row for row in rows if row["index"] == index), None)
    if chosen is None:
        visible = [row["index"] for row in rows]
        raise ConfigError(f"GPU {index} is not visible to nvidia-smi (visible: {visible})")
    if chosen["memory_used_mib"] >= BUSY_MEMORY_MIB:
        raise ConfigError(
            f"GPU {index} is in use ({chosen['memory_used_mib']} MiB, "
            f"utilization {chosen['utilization']}%). Choose a free GPU, or set "
            "runtime.allow_busy_gpu only when that GPU is yours."
        )
