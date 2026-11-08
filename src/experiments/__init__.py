"""Research experiment framework.

The training CLI and reproduction pipeline do not import this package.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from src.experiments.api import Experiment, RunResult, build_experiment, load_experiment

__all__ = ["Experiment", "RunResult", "build_experiment", "load_experiment"]
