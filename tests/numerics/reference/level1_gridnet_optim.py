"""Record TF-GridNet optimizer state at the official chunk.

Rules are fixed in VALIDATION_PLAN.md before this run. Tolerances are not
edited from the result. The trainer comparison already exists and is not
repeated.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-level1-20261111/gridnet")

import torch  # noqa: E402

from tests.numerics.reference.common import environment  # noqa: E402
from tests.numerics.reference.parallel_gridnet_batch4 import (  # noqa: E402
    _bank,
    _cpu_state,
    _official_steps,
    _pair,
)

OUT = Path(os.environ["AUDIO_DB_NUMERICS_OUT"])


def _plain(rows: list[dict]) -> list[dict]:
    plain = []
    for row in rows:
        plain.append(
            {
                "loss_gap": row["loss_gap"],
                "official_loss": row["official_loss"],
                "independent_loss": row["independent_loss"],
                "loss_on_official_wave": row["loss_on_official_wave"],
                "forward": row["forward"],
                "grad_max_abs": row["grad"]["max_abs"],
                "grad_worst": row["grad"]["worst"],
                "large_cosine_min": row["grad"]["large_cosine_min"],
                "weight_max_abs": row["weight"]["max_abs"],
                "weight_worst": row["weight"]["worst"],
                "optim_max_abs": row["optim"]["max_abs"],
                "optim_worst": row["optim"]["worst"],
                "optim_missing_or_step_mismatch": row["optim"]["missing_or_step_mismatch"],
            }
        )
    return plain


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    official, _independent = _pair(device)
    initial = _cpu_state(official)
    clean, noisy = _bank(3, 4, 48000, device)
    _official_done, _independent_done, rows = _official_steps(device, initial, clean, noisy)
    report = {
        "environment": environment(device),
        "official_commit": "0d136c5456d403cb2c0a4c13e76e8136e32e9b7a",
        "official_function": "espnet2 TFGridNet separator + MultiResL1SpecLoss + Adam",
        "samples": 48000,
        "batch_size": 4,
        "steps": 3,
        "rows": _plain(rows),
    }
    path = OUT / "gridnet_48000_batch4_optim.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps(report["rows"], indent=2), flush=True)


if __name__ == "__main__":
    main()
