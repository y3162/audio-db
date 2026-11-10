"""Record parameter identity before and after CPU DDP prepare.

``torchrun --nproc_per_node=2`` starts this module. Identifiers are local to
the process, so each rank writes its own comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from src.experiments.official.tf_locoformer.model import TFLocoformerModel
from src.experiments.runtime import PreparePolicy, ProcessRuntime

ARCH = {
    "attention_dim": 4,
    "conv1d_kernel": 4,
    "conv1d_shift": 1,
    "dropout": 0.0,
    "emb_dim": 4,
    "eps": 1e-5,
    "ffn_hidden_dim": [8, 8],
    "ffn_type": ["swiglu_conv1d", "swiglu_conv1d"],
    "flash_attention": False,
    "hop_length": 8,
    "n_fft": 16,
    "n_heads": 1,
    "n_layers": 1,
    "norm_type": "rmsgroupnorm",
    "normalize_variance": True,
    "num_groups": 4,
    "num_spk": 1,
    "pos_enc": "rope",
    "tf_order": "ft",
}


def main() -> None:
    output = Path(sys.argv[1])
    runtime = ProcessRuntime(device="cpu", precision="fp32", timeout_seconds=30)
    try:
        model = TFLocoformerModel(seed=3, architecture=ARCH)
        before = [id(parameter) for parameter in model.parameters()]
        prepared = runtime.prepare_model(model, PreparePolicy(trainable=True, distribute="ddp"))
        after = [id(parameter) for parameter in prepared.parameters]
        optimizer = torch.optim.AdamW(prepared.parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
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
