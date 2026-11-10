"""Research experiment framework.

Importing this package does not import the trainer, checkpoint runtime, or CUDA.
The database CLI can load ``src.experiments.data`` without building an experiment.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__all__ = ["Experiment", "RunResult", "build_experiment", "load_experiment"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from src.experiments.api import Experiment, RunResult, build_experiment, load_experiment

    values = {
        "Experiment": Experiment,
        "RunResult": RunResult,
        "build_experiment": build_experiment,
        "load_experiment": load_experiment,
    }
    return values[name]
