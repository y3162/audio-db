"""MP-SENet inference weights. This file is not a resume checkpoint.

Rank 0 writes ``best_generator.pt`` only when the trainer selects a higher
PESQ. Negative PESQ stays in the comparison; this module does not rescore it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def write_best(
    directory: Path,
    exported: dict[str, torch.Tensor],
    decision: dict[str, Any],
    experiment_claim: str,
) -> None:
    payload_path = directory / "best_generator.pt"
    meta_path = directory / "best_generator.json"
    temporary = directory / ".best_generator.pt.tmp"
    torch.save(exported, temporary)
    os.replace(temporary, payload_path)
    meta = {
        "checkpoint": payload_path.name,
        "epoch": decision["epoch"],
        "experiment_claim": experiment_claim,
        "logical_step": decision["logical_step"],
        "metric": "pesq",
        "pesq": decision["pesq"],
        "pesq_policy": "official_include_negative_one",
    }
    meta_temporary = directory / ".best_generator.json.tmp"
    meta_temporary.write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(meta_temporary, meta_path)
