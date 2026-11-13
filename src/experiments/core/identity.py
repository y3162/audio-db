"""Explicit identities for checkpointed policy classes.

IDs are declared on each concrete class, not inherited. A process rejects
collisions; implementations may move while the declared identifier stays put.
"""

from __future__ import annotations

from src.experiments.errors import CheckpointError

_CLASSES: dict[str, type] = {}


def policy_identity(instance: object) -> str:
    cls = type(instance)
    identity = cls.__dict__.get("checkpoint_class_id")
    if (
        not isinstance(identity, str)
        or "." not in identity
        or not all(part.isidentifier() for part in identity.split("."))
    ):
        raise CheckpointError(f"{cls.__name__} must declare a valid checkpoint_class_id")
    previous = _CLASSES.setdefault(identity, cls)
    if previous is not cls:
        raise CheckpointError(f"duplicate checkpoint class identity {identity!r}")
    return identity
