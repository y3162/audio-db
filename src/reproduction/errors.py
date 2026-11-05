"""Errors for the reproduction database and its snapshots."""


class ReproductionError(ValueError):
    """The reproduction database, staging, or snapshot cannot be used as requested."""
