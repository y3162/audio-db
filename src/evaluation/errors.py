"""Errors raised by the evaluation pipeline."""


class ConfigError(ValueError):
    """The experiment configuration is invalid."""


class DatasetError(RuntimeError):
    """The requested dataset cannot be read."""
