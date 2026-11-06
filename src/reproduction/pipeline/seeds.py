"""Seed mixing for data plans.

This is the same SHA-256 mix as ``src.training.core.rng.mix_seed``. The
pipeline does not import ``src.training``, so the function lives here. A test
checks that the two implementations agree.
"""

from __future__ import annotations

import hashlib


def mix_seed(*parts: object) -> int:
    """Stable 63-bit seed. This does not use Python's salted ``hash``."""
    payload = "\n".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**63 - 1) or 1
