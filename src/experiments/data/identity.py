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

    Identities without ``dataset_id`` are returned unchanged.
    """

    if "dataset_id" not in identity:
        return dict(identity)
    return {key: value for key, value in identity.items() if key not in DISPLAY_IDENTITY_KEYS}
