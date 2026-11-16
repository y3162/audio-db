"""Dataset identity comparison.

``identity()`` keeps display fields. Resume compares the semantic fields only,
so a display-name edit does not fail a run whose dataset id and snapshot hash
still match. ``TensorPairData`` has no ``dataset_id`` and keeps full equality.
"""

from __future__ import annotations

from typing import Any

DISPLAY_IDENTITY_KEYS = frozenset({"dataset_name", "dataset_version"})


def comparable_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Return the fields resume is allowed to require.

    Identities without ``dataset_id`` are returned unchanged, except that a
    per-rank ``batch_size`` and ``world_size`` become their product. An epoch
    boundary may change the world size when that product stays the same.
    """

    if "dataset_id" not in identity:
        return global_micro_batch_identity(identity)
    return {key: value for key, value in identity.items() if key not in DISPLAY_IDENTITY_KEYS}


def global_micro_batch_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Replace per-rank batch fields with ``batch_size * world_size``."""

    batch = identity.get("batch_size")
    world = identity.get("world_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        return dict(identity)
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        return dict(identity)
    stable = dict(identity)
    stable.pop("batch_size", None)
    stable.pop("world_size", None)
    stable["global_micro_batch"] = batch * world
    return stable
