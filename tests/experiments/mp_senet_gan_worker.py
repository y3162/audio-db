"""CPU / Gloo entry points for the MP-SENet GAN tests.

``torchrun --nproc_per_node=2`` starts this module. The YAML path and the
mode are arguments. Each rank writes its own JSON before the process exits.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from src.experiments.api import build_experiment, load_experiment


def _pesq_split(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
    del enhanced, sample_rate
    if int(os.environ.get("RANK", "0")) == 0:
        return None
    return torch.full((clean.shape[0],), 0.25)


def main() -> None:
    directory = Path(sys.argv[1])
    mode = sys.argv[2]
    if mode == "nan":
        os.environ["MP_SENET_NAN_RANK0"] = "1"
    if mode == "pesq_split":
        import src.training.se.losses as losses

        losses.metric_targets = _pesq_split
    experiment = build_experiment(load_experiment(directory / "experiment.yaml"))
    result = experiment.run()
    trainer = experiment.get("trainer")
    payload = {
        "average_calls": int(trainer._average_calls),
        "logical_step": int(result.logical_step),
        "optimizer_updates": dict(result.optimizer_updates),
        "rank": int(experiment._graph.runtime.rank),
        "stop_reason": result.stop_reason,
        "synced_during_generator_backward": bool(trainer._generator_synced_during_backward),
    }
    path = directory / f"worker-rank{payload['rank']}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
