"""Errors raised by the experiment framework.

Callers can recover from any of these. Import, type, and value failures that
belong to a component are wrapped in :class:`BuildError` instead of leaking.
"""

from __future__ import annotations


class ExperimentError(Exception):
    """Base class for recoverable experiment failures."""


class ConfigError(ExperimentError):
    """The experiment YAML is syntactically or structurally invalid."""


class BuildError(ExperimentError):
    """A component graph cannot be constructed."""


class PrepareError(ExperimentError):
    """A module could not be placed on its runtime."""


class StateError(ExperimentError):
    """State registration or lookup does not match the live graph."""


class CheckpointError(ExperimentError):
    """A checkpoint cannot be written, read, or used for this run."""


class RuntimeFailure(ExperimentError):
    """A learning step cannot continue without diverging across ranks."""


class DataError(ExperimentError):
    """A snapshot, sampler, or batch specification cannot supply this run."""
