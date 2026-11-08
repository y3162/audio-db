"""Select a host-compatible DeepFilterNet3 realfft library for pytest.

The tracked file next to the model stays where it is. When that file does not
load and ``DF_RFFT_LIBRARY`` is unset, use the ``make df-rfft`` output at
``build/df-rfft/release/libdf_rfft.so``. Workers started later inherit the
variable. A missing build is left unset so the tests still fail.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TRACKED = _ROOT / "src/se/models/deepfilternet3/libdf_rfft.so"
_BUILT = _ROOT / "build/df-rfft/release/libdf_rfft.so"


def _loads(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        ctypes.CDLL(str(path))
    except OSError:
        return False
    return True


def pytest_configure(config) -> None:
    del config
    if os.environ.get("DF_RFFT_LIBRARY"):
        return
    if _loads(_TRACKED):
        return
    if _loads(_BUILT):
        os.environ["DF_RFFT_LIBRARY"] = str(_BUILT)
