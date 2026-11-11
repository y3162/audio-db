"""Name the MP-SENet tiny-gradient weight that exceeded 5e-4.

The allowance stays 5e-4. This only records the element under the same tight
flags as /tmp/audio-db-final-two-models/mp_tight.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from tests.numerics.reference import compare_mp_senet as cmp
from tests.numerics.reference.two_model_final import (
    SEEDS,
    _load_bank,
    _mp_bank,
    _mp_compare,
    _waves,
    apply_flags,
)

# two_model_final assigns AUDIO_DB_NUMERICS_OUT on import. Keep this report separate.
OUT = Path("/tmp/audio-db-level1-20261111/mp")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flags = apply_flags("tight")
    device = torch.device("cuda")
    seed = SEEDS[0]
    h = cmp._hyper()
    from tests.numerics.reference import two_model_final as fin

    splits = []
    original = fin._weight_split

    def capture(left, right):
        result = original(left, right)
        splits.append(result)
        return result

    fin._weight_split = capture
    bank = _mp_bank(cmp, h, device, seed)
    clean, noisy = _waves(2, 8000, seed, 0.5, constant_noise=True)
    rows = _mp_compare(cmp, h, bank, clean, noisy, "adamw", "oa_resync", "official", "audio", True, None, seed)
    report = {
        "flags": flags,
        "seed": seed,
        "width": {"dense_channel": 64, "num_tsconformers": 4},
        "batch": 2,
        "samples": 8000,
        "rows": rows,
        "weight_splits": splits,
    }
    path = OUT / "mp_tiny.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps(splits, indent=2), flush=True)


if __name__ == "__main__":
    main()
