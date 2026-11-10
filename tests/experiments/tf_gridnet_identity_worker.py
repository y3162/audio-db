"""Record parameter identity before and after CPU DDP prepare.

``torchrun --nproc_per_node=2`` starts this module. Identifiers are local to
the process, so each rank writes its own comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from src.experiments.official.tf_gridnet.model import TFGridNetModel
from src.experiments.runtime import PreparePolicy, ProcessRuntime

ARCH = {
    "attn_approx_qk_dim": 8,
    "attn_n_head": 1,
    "emb_dim": 4,
    "emb_hs": 1,
    "emb_ks": 1,
    "eps": 1e-5,
    "lstm_hidden_units": 4,
    "n_fft": 16,
    "n_layers": 1,
    "stride": 8,
}


def main() -> None:
    output = Path(sys.argv[1])
    runtime = ProcessRuntime(device="cpu", precision="fp32", timeout_seconds=30)
    try:
        model = TFGridNetModel(seed=3, architecture=ARCH)
        before = [id(parameter) for parameter in model.parameters()]
        prepared = runtime.prepare_model(model, PreparePolicy(trainable=True, distribute="ddp"))
        after = [id(parameter) for parameter in prepared.parameters]
        optimizer = torch.optim.Adam(prepared.parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
        bound = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        payload = {
            "match": before == after == bound,
            "parameters": len(before),
            "rank": runtime.rank,
            "state_is_unwrapped": prepared.state_module is model,
            "world_size": runtime.world_size,
            "wrapped": prepared.module is not model,
        }
        path = output / f"rank{runtime.rank}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        runtime.barrier()
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
