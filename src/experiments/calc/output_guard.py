"""Refuse provenance writes under the published data directories."""

from __future__ import annotations

from pathlib import Path

from src.experiments.calc.errors import TrainingError


def assert_output_allowed(path: Path) -> None:
    """Same directories as experiment checkpoints, reported as ``TrainingError``."""

    from src.experiments.checkpoint import assert_output_allowed as refuse
    from src.experiments.errors import CheckpointError

    try:
        refuse(path)
    except CheckpointError as exc:
        raise TrainingError(str(exc)) from exc
