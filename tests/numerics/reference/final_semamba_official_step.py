"""One official-width SEMamba++ step with TF32 off and cuDNN deterministic.

The fused mixer comparison is ``final_semamba_inner.py``. This runs the
published width (hid 48, 4 TF-Mamba layers, d_state 16, d_conv 4).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-final"))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from tests.numerics.reference.compare_semamba_step import one_step  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = int(os.environ.get("AUDIO_DB_NUMERICS_STEPS", "1"))
    report = one_step(device, 48, 4, 16, 4, steps)
    report["device"] = str(device)
    report["tf32"] = False
    report["cudnn_deterministic"] = True
    report["cublas_workspace"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    report["official_commit"] = "cd369c89a8106072c077b468417b765b227afd74"
    report["width"] = {"hid_feature": 48, "num_tfmamba": 4, "d_state": 16, "d_conv": 4}
    path = OUT / "semamba_official_step.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    brief = []
    for row in report.get("history", []):
        brief.append({
            "step": row["step"],
            "generator_loss": row["losses"]["generator"],
            "grad": row["generator_grad"]["max_abs"],
            "grad_worst": row["generator_grad"]["worst"],
            "large_cosine": row["generator_grad"].get("large_cosine_min"),
            "weight": row["generator_weight"]["max_abs"],
            "weight_worst": row["generator_weight"]["worst"],
        })
    print(path)
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
