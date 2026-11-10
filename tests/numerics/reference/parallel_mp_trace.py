"""MP-SENet 3-step CUDA trace with the native STFT and a shared deterministic STFT.

The shared run binds the official mag/phase transforms to audio-db's
implementation and enables deterministic algorithms, which selects the
index-reflection STFT. Thresholds are the existing ones: loss 1e-5, gradient
1e-4, first-step weight 5e-4.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-continue"))

from tests.numerics.reference import compare_mp_senet as cmp  # noqa: E402

FLOOR = 1e-6
LR = 5e-4
EPS = 1e-8
WD = 0.01


def _masked(official, independent) -> dict:
    named = dict(independent.named_parameters())
    worst = 0.0
    worst_name = ""
    large = 0.0
    large_name = ""
    for name, left in official.named_parameters():
        right = named[name]
        delta = (left.detach() - right.detach()).abs()
        value = float(delta.max())
        if value >= worst:
            worst = value
            worst_name = name
        if left.grad is None:
            continue
        mask = left.grad.detach().abs() >= FLOOR
        if bool(mask.any()):
            found = float(delta[mask].max())
            if found >= large:
                large = found
                large_name = name
    return {"max_abs": worst, "worst": worst_name, "large_grad_max_abs": large, "large_grad_worst": large_name}


def _element(official, independent, name: str) -> dict:
    left = dict(official.named_parameters())[name]
    right = dict(independent.named_parameters())[name]
    delta = (left.detach() - right.detach()).abs()
    index = int(delta.view(-1).argmax())
    return {
        "name": name,
        "index": index,
        "weight_abs": float(delta.view(-1)[index]),
        "grad_official": None if left.grad is None else float(left.grad.detach().view(-1)[index]),
        "grad_independent": None if right.grad is None else float(right.grad.detach().view(-1)[index]),
        "param": float(left.detach().view(-1)[index]),
    }


def _formula(official, independent) -> dict:
    named = dict(independent.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, left in official.named_parameters():
        right = named[name]
        if left.grad is None or right.grad is None:
            continue
        predicted = (left.detach() * (1 - LR * WD) - LR * left.grad.detach() / (left.grad.detach().abs() + EPS)) - (
            right.detach() * (1 - LR * WD) - LR * right.grad.detach() / (right.grad.detach().abs() + EPS)
        )
        value = float(predicted.abs().max())
        if value > worst:
            worst = value
            worst_name = name
    return {"formula_max_abs": worst, "worst": worst_name}


def _steps(shared: bool) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    warnings_seen: list[str] = []
    if shared:
        torch.use_deterministic_algorithms(True, warn_only=True)
        cmp.official_stft = cmp.mag_pha_stft
        cmp.official_istft = cmp.mag_pha_istft
    h = cmp._hyper()
    torch.manual_seed(11)
    clean = torch.randn(2, 8000, device=device)
    noisy = clean * 0.5 + 0.05
    modules = cmp._build(h, device)
    optimizers = cmp._optimizers(*modules)
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for step in range(3):
            record = cmp.gan_step(modules, optimizers, clean, noisy, h)
            official_g = modules[2]
            independent_g = modules[0]
            weight = _masked(official_g, independent_g)
            item = {
                "step": step + 1,
                "loss_total": record["generator_loss_gap"]["total"],
                "wave": record["forward"]["wave"]["max_abs"],
                "stft": record["forward"]["stft"]["max_abs"],
                "generator_grad": record["generator_grad"]["max_abs"],
                "generator_grad_worst": record["generator_grad"]["worst"],
                "large_cosine_min": record["generator_grad"]["large_cosine_min"],
                "weight": weight,
                "weight_element": _element(official_g, independent_g, str(weight["worst"])),
                "discriminator_grad": record["discriminator_grad"]["max_abs"],
                "discriminator_weight": record["discriminator_weight"]["max_abs"],
            }
            if step == 0:
                item["formula"] = _formula(official_g, independent_g)
            rows.append(item)
            print("mp", "shared" if shared else "native", item["step"], item["loss_total"], item["generator_grad"], weight["max_abs"], flush=True)
        warnings_seen = [str(item.message) for item in caught if "deterministic" in str(item.message).lower()]
    return {"shared_deterministic_stft": shared, "steps": rows, "deterministic_warnings": warnings_seen}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "official_commit": "89932cfe90d1dacb8e170e4a331d762462c21792",
        "thresholds": {"loss": 1e-5, "grad": 1e-4, "weight_existing_one_step": 5e-4},
        "batch_size": 2,
        "samples": 8000,
        "native": _steps(False),
        "shared_deterministic_stft": _steps(True),
    }
    path = OUT / "mp_trace.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
