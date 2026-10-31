"""Execution environment captured with each run."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_metadata(root: Path | None = None) -> dict[str, Any]:
    cwd = root or REPO_ROOT
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def capture_environment() -> dict[str, Any]:
    torch_version = None
    torchaudio_version = None
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        pass
    try:
        import torchaudio

        torchaudio_version = torchaudio.__version__
    except ImportError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch_version,
        "torchaudio": torchaudio_version,
        "numpy": _package_version("numpy"),
        "pyyaml": _package_version("pyyaml"),
        "pesq": _package_version("pesq"),
        "pystoi": _package_version("pystoi"),
        "scipy": _package_version("scipy"),
        "duckdb": _package_version("duckdb"),
    }
