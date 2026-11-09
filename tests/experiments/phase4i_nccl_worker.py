"""Dump each rank's module state after one experiment run.

``torchrun --nproc_per_node=2`` starts this module. The YAML path is the
first argument. ``--resume`` continues an ``audio-db-experiment-v1``
checkpoint. This is a measurement entry point, not a trainer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from src.experiments.api import Experiment
from src.experiments.config import load_experiment
from src.experiments.runtime import PreparedModel


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: phase4i_nccl_worker CONFIG [--resume PATH]")
    config_path = Path(sys.argv[1])
    resume = None
    if "--resume" in sys.argv:
        resume = Path(sys.argv[sys.argv.index("--resume") + 1])
    experiment = Experiment(load_experiment(config_path))
    result = experiment.run(resume=resume)
    rank = int(experiment.config.runtime_kwargs.get("rank", 0))
    runtime = experiment._runtime
    if runtime is not None and hasattr(runtime, "rank"):
        rank = int(runtime.rank)
    modules: dict[str, dict[str, torch.Tensor]] = {}
    sync_batchnorm: dict[str, bool] = {}
    for component_id in experiment.config.components:
        component = experiment.get(component_id)
        if not isinstance(component, PreparedModel):
            continue
        modules[component_id] = {
            name: value.detach().cpu().clone()
            for name, value in component.state_module.state_dict().items()
            if torch.is_tensor(value)
        }
        sync_batchnorm[component_id] = any(
            module.__class__.__name__ == "SyncBatchNorm" for module in component.state_module.modules()
        )
    directory = experiment.config.output_directory
    if not directory.is_absolute():
        directory = (Path.cwd() / directory).resolve()
    torch.save(modules, directory / f"rank{rank}-modules.pt")
    (directory / f"rank{rank}-summary.json").write_text(
        json.dumps(
            {
                "logical_step": result.logical_step,
                "rank": rank,
                "stop_reason": result.stop_reason,
                "sync_batchnorm": sync_batchnorm,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
