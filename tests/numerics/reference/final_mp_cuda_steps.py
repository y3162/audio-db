"""Three MP-SENet CUDA steps under the documented cuBLAS workspace, plus AdamW amplification.

Launch::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=<free> \\
        .venvs/se/bin/python tests/numerics/reference/final_mp_cuda_steps.py

The weight allowance already fixed in ``test_mp_senet_official_step`` is 5e-4.
Gradient allowance in ``compare_mp_senet`` is 1e-4. This script does not change either.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.numerics.reference import compare_mp_senet as cmp
from tests.numerics.reference.common import grad_report, optim_report, weight_report

OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-final"))
LR = 5e-4
EPS = 1e-8
WD = 0.01


class _AmaxPool(nn.Module):
    def forward(self, value):
        return value.amax(dim=(-2, -1), keepdim=True)


def _replace(module) -> int:
    found = 0
    for index, layer in enumerate(getattr(module, "layers", [])):
        if isinstance(layer, nn.AdaptiveMaxPool2d):
            module.layers[index] = _AmaxPool()
            found += 1
    return found


def _formula_gap(official, independent) -> dict[str, float | str]:
    named = dict(independent.named_parameters())
    worst = 0.0
    worst_name = ""
    actual = 0.0
    for name, left in official.named_parameters():
        right = named[name]
        if left.grad is None or right.grad is None:
            continue
        predicted = (
            left.detach() * (1 - LR * WD) - LR * left.grad.detach() / (left.grad.detach().abs() + EPS)
        ) - (
            right.detach() * (1 - LR * WD) - LR * right.grad.detach() / (right.grad.detach().abs() + EPS)
        )
        value = float(predicted.abs().max())
        if value > worst:
            worst = value
            worst_name = name
            actual = float((left.detach() - right.detach()).abs().max())
    return {"formula_max_abs": worst, "worst": worst_name, "pre_step_weight_max_abs": actual}


def _bias_trace(official, independent, name: str) -> dict:
    left = dict(official.named_parameters())[name]
    right = dict(independent.named_parameters())[name]
    grad = (left.grad - right.grad).detach()
    index = int(grad.abs().argmax())
    return {
        "name": name,
        "index": index,
        "grad_official": float(left.grad.detach().reshape(-1)[index]),
        "grad_independent": float(right.grad.detach().reshape(-1)[index]),
        "grad_abs_diff": float(grad.reshape(-1)[index].abs()),
        "param": float(left.detach().reshape(-1)[index]),
    }


def _one(device, clean, noisy, h, *, deterministic_pool: bool) -> dict:
    modules = cmp._build(h, device)
    optimizers = cmp._optimizers(*modules)
    replaced = 0
    if deterministic_pool:
        replaced = _replace(modules[1]) + _replace(modules[3])
    record = cmp.gan_step(modules, optimizers, clean, noisy, h)
    generator, _discriminator, official_g, _official_d = modules
    trace_name = record["generator_grad"]["worst"]
    return {
        "replaced_pools": replaced,
        "loss_total": record["generator_loss_gap"]["total"],
        "wave": record["forward"]["wave"]["max_abs"],
        "generator_grad": record["generator_grad"]["max_abs"],
        "generator_grad_worst": trace_name,
        "generator_grad_cosine_min": record["generator_grad"]["cosine_min"],
        "generator_large_cosine_min": record["generator_grad"]["large_cosine_min"],
        "discriminator_grad": record["discriminator_grad"]["max_abs"],
        "formula": _formula_gap(official_g, generator),
        "bias_trace_before_step_consumed": _bias_trace(official_g, generator, trace_name),
        "generator_weight": record["generator_weight"],
        "generator_optim": record["generator_optim"],
        "discriminator_weight": record["discriminator_weight"],
    }


def _self(device, clean, noisy, h) -> dict:
    h_local = h
    modules = cmp._build(h_local, device)
    optimizers = cmp._optimizers(*modules)
    first = cmp.gan_step(modules, optimizers, clean, noisy, h_local)
    weight_a = {name: value.detach().clone() for name, value in modules[2].named_parameters()}
    modules_b = cmp._build(h_local, device)
    optimizers_b = cmp._optimizers(*modules_b)
    cmp.gan_step(modules_b, optimizers_b, clean, noisy, h_local)
    worst = 0.0
    worst_name = ""
    named = dict(modules_b[2].named_parameters())
    for name, value in weight_a.items():
        delta = float((value - named[name].detach()).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    return {"official_versus_official_weight": worst, "worst": worst_name, "cross_loss": first["generator_loss_gap"]["total"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    h = cmp._hyper()
    torch.manual_seed(11)
    clean = torch.randn(2, 8000, device=device)
    noisy = clean * 0.5 + 0.05
    report = {
        "device": str(device),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "official_commit": "89932cfe90d1dacb8e170e4a331d762462c21792",
        "thresholds": {"loss": 1e-5, "grad": 1e-4, "weight_existing_test": 5e-4},
        "condition": "official AdaptiveMaxPool2d, cudnn.deterministic, TF32 off",
    }
    steps = []
    modules = cmp._build(h, device)
    optimizers = cmp._optimizers(*modules)
    pre_step = {}

    def _capture(optim, label):
        original = optim.step

        def step(*args, **kwargs):
            pre_step[label] = [
                (parameter.detach().clone(), None if parameter.grad is None else parameter.grad.detach().clone())
                for group in optim.param_groups
                for parameter in group["params"]
            ]
            return original(*args, **kwargs)

        optim.step = step

    _capture(optimizers[0], "independent")
    _capture(optimizers[2], "official")
    for step in range(3):
        pre_step.clear()
        record = cmp.gan_step(modules, optimizers, clean, noisy, h)
        item = {
            "step": step + 1,
            "loss_total": record["generator_loss_gap"]["total"],
            "generator_grad": record["generator_grad"]["max_abs"],
            "generator_grad_worst": record["generator_grad"]["worst"],
            "large_cosine_min": record["generator_grad"]["large_cosine_min"],
            "generator_weight": record["generator_weight"]["max_abs"],
            "generator_weight_worst": record["generator_weight"]["worst"],
            "generator_optim": record["generator_optim"]["max_abs"],
            "discriminator_grad": record["discriminator_grad"]["max_abs"],
            "discriminator_weight": record["discriminator_weight"]["max_abs"],
        }
        if step == 0 and "official" in pre_step and "independent" in pre_step:
            worst = 0.0
            worst_name = ""
            names = [name for name, _parameter in modules[2].named_parameters()]
            for name, (param_o, grad_o), (param_i, grad_i) in zip(
                names, pre_step["official"], pre_step["independent"], strict=True
            ):
                if grad_o is None or grad_i is None:
                    continue
                predicted = (
                    param_o * (1 - LR * WD) - LR * grad_o / (grad_o.abs() + EPS)
                ) - (
                    param_i * (1 - LR * WD) - LR * grad_i / (grad_i.abs() + EPS)
                )
                value = float(predicted.abs().max())
                if value > worst:
                    worst = value
                    worst_name = name
            item["formula_weight_max_abs"] = worst
            item["formula_worst"] = worst_name
            item["weight_minus_formula"] = abs(item["generator_weight"] - worst)
        steps.append(item)
        print("mp step", item["step"], item["generator_grad"], item["generator_weight"], flush=True)
    report["steps"] = steps
    report["self_official"] = _self(device, clean, noisy, h)
    caught = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            report["deterministic_pool"] = _one(device, clean, noisy, h, deterministic_pool=True)
        except Exception as exc:
            report["deterministic_pool"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        for warning in caught_warnings:
            caught.append(f"{warning.category.__name__}: {warning.message}")
    report["deterministic_warnings"] = caught[:30]
    path = OUT / "mp_cuda_steps.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
