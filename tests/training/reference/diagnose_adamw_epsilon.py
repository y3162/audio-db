"""Show AdamW turning sub-epsilon gradient sign differences into O(lr) updates.

The gradients are the recorded step-1 values. Both optimizers are
``torch.optim.AdamW`` with the probe hyperparameters. Identical gradients are
applied to two fresh optimizers to separate the optimizer from the gradient.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

OUTPUT = Path("data/results/training/adamw-epsilon-20261103")


def step(lr, betas, weight_decay, grad, weight=1.0):
    param = torch.nn.Parameter(torch.tensor(weight))
    opt = torch.optim.AdamW([param], lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay)
    param.grad = torch.tensor(grad)
    before = param.detach().clone()
    opt.step()
    theoretical = before * (1 - lr * weight_decay) - lr * grad / (abs(grad) + 1e-8)
    return {
        "grad": grad,
        "abs_grad": abs(grad),
        "below_eps": abs(grad) < 1e-8,
        "signed_fraction": grad / (abs(grad) + 1e-8),
        "updated": float(param.detach()),
        "delta": float(param.detach() - before),
        "closed_form": float(theoretical),
        "closed_form_abs_error": float((param.detach() - theoretical).abs()),
    }


def pair(lr, betas, weight_decay, left, right, weight=1.0):
    a = step(lr, betas, weight_decay, left, weight)
    b = step(lr, betas, weight_decay, right, weight)
    same = step(lr, betas, weight_decay, left, weight)
    other = step(lr, betas, weight_decay, left, weight)
    return {
        "left": a,
        "right": b,
        "update_difference": abs(a["delta"] - b["delta"]),
        "same_gradient_update_difference": abs(same["delta"] - other["delta"]),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    report = {
        "mp_senet": {
            "source": "docs/training/VERIFICATION.md 2026-10-31 dense_encoder.dense_block.dense_block.3.1.bias",
            "lr": 5e-4,
            "betas": [0.8, 0.99],
            "weight_decay": 0.01,
            "measured_weight_abs_difference": 4.052668809890747e-4,
            **pair(5e-4, (0.8, 0.99), 0.01, -8.90e-9, 5.14e-9),
        },
        "mossformergan": {
            "source": "followup-20261102-voicebank moss-1s-cont4 generator.dense_encoder.conv_1.0.bias",
            "lr": 5e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "measured_weight_abs_difference": 0.000315,
            **pair(5e-4, (0.9, 0.999), 0.01, 4.31e-9, -4.89e-9),
        },
    }
    report["mp_senet"]["predicted_matches_measured"] = abs(
        report["mp_senet"]["update_difference"] - report["mp_senet"]["measured_weight_abs_difference"]
    )
    report["mossformergan"]["predicted_matches_measured"] = abs(
        report["mossformergan"]["update_difference"] - report["mossformergan"]["measured_weight_abs_difference"]
    )
    (OUTPUT / "adamw.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
