"""Optional tensors a model forward may keep.

``output_features`` names tensors the forward already computes. It does not
attach hooks or build a graph. Names outside that model's list are an error.
An empty list keeps only the model's required outputs.
"""

from __future__ import annotations

from src.experiments.errors import BuildError


def parse_output_features(value: object, *, allowed: frozenset[str]) -> tuple[str, ...]:
    """Return the requested optional names, or ``()`` when none were requested."""

    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise BuildError("output_features must be a list of feature names")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise BuildError("output_features must be a list of feature names")
        names.append(item)
    if len(set(names)) != len(names):
        raise BuildError("output_features repeats a name")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        available = ", ".join(sorted(allowed)) or "none"
        raise BuildError(
            f"output_features {unknown} are not produced by this forward; "
            f"available optional features: {available}"
        )
    return tuple(names)
