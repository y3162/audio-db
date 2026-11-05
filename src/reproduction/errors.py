"""Errors for the reproduction database and its snapshots."""


class ReproductionError(ValueError):
    """The reproduction database, staging, or snapshot cannot be used as requested."""


class PreparationBlocked(ReproductionError):
    """Required lists or audio are missing. The process exit status is 3.

    A blocked preparation does not create derived audio, staging, or a database.
    """
