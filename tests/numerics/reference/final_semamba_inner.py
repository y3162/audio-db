"""Isolate official ``mamba_inner_fn`` backward from the SEMamba++ generator.

Official width used by the mixer: d_model=48, d_state=16, d_conv=4, expand=2.
The fused call is the function official ``Mamba.forward`` executes. A second
pair forces the pure-PyTorch ``selective_scan_ref`` shipped in the same
``mamba_ssm`` package. That path is not the official training default.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch

OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-final"))
ROOT = Path(__file__).resolve().parents[3]


def _load_selective():
    path = ROOT / "src/se/models/semambapp/network.py"
    spec = importlib.util.spec_from_file_location("audio_db_semamba_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SelectiveSSM


def _grad_gap(left, right) -> dict[str, float | str]:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, value in left.named_parameters():
        if value.grad is None or named[name].grad is None:
            continue
        delta = float((value.grad - named[name].grad).abs().max())
        if delta > worst:
            worst = delta
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def _replay(fn, args, repeats: int) -> dict:
    grads = []
    for _ in range(repeats):
        cloned = []
        for value in args:
            if torch.is_tensor(value):
                cloned.append(value.detach().clone().requires_grad_(value.is_floating_point()))
            else:
                cloned.append(value)
        output = fn(*cloned)
        output.float().square().mean().backward()
        grads.append(cloned[0].grad.detach().clone())
    worst = 0.0
    for index in range(1, len(grads)):
        worst = max(worst, float((grads[0] - grads[index]).abs().max()))
    return {"repeats": repeats, "xz_grad_max_abs_versus_first": worst}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn, selective_scan_ref
    import mamba_ssm

    SelectiveSSM = _load_selective()
    torch.manual_seed(5)
    official = Mamba(d_model=48, d_state=16, d_conv=4, expand=2, device=device, dtype=torch.float32)
    independent = SelectiveSSM(48, 16, 4, 2, scan_backend="fused").to(device)
    independent.load_state_dict(official.state_dict(), strict=True)
    hidden = torch.randn(4, 81, 48, device=device)
    official.train()
    independent.train()
    torch.manual_seed(9)
    left = official(hidden)
    torch.manual_seed(9)
    right = independent(hidden)
    forward = float((left - right).abs().max())
    left.float().square().mean().backward()
    right.float().square().mean().backward()
    captured = {}
    real = mamba_inner_fn

    def wrapped(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real(*args, **kwargs)

    import mamba_ssm.modules.mamba_simple as simple
    simple.mamba_inner_fn = wrapped
    official.zero_grad(set_to_none=True)
    official(hidden.detach())
    simple.mamba_inner_fn = real
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"),
        "official_commit": "cd369c89a8106072c077b468417b765b227afd74",
        "mixer": {"d_model": 48, "d_state": 16, "d_conv": 4, "expand": 2, "length": 81, "batch": 4},
        "fused_forward_max_abs": forward,
        "fused_grad": _grad_gap(official, independent),
        "official_in_proj_bias": official.in_proj.bias is None,
        "independent_in_proj_bias": independent.in_proj.bias is None,
    }
    if "args" in captured:
        report["mamba_inner_self"] = _replay(real, captured["args"], 4)
        report["mamba_inner_arg_kinds"] = [type(value).__name__ for value in captured["args"]]
    # Deterministic reference scan shipped beside the fused kernel. Not the training default.
    official.zero_grad(set_to_none=True)
    independent.zero_grad(set_to_none=True)
    official.use_fast_path = False
    simple.selective_scan_fn = selective_scan_ref
    torch.manual_seed(9)
    left_ref = official(hidden.detach())
    simple.selective_scan_fn = selective_scan_fn
    independent.scan_backend = "reference"
    torch.manual_seed(9)
    right_ref = independent(hidden.detach())
    left_ref.float().square().mean().backward()
    right_ref.float().square().mean().backward()
    report["reference_scan"] = {
        "note": "official Mamba use_fast_path=False with selective_scan_fn replaced by selective_scan_ref; audio-db reference backend",
        "forward_max_abs": float((left_ref - right_ref).abs().max()),
        "grad": _grad_gap(official, independent),
    }
    path = OUT / "semamba_inner.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps({key: report[key] for key in ("fused_forward_max_abs", "fused_grad", "mamba_inner_self", "reference_scan")}, indent=2))


if __name__ == "__main__":
    main()
