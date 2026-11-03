"""Record selective-scan call layout for one shared SEMamba++ checkpoint.

Official is executed twice and independent once. Forward inputs are compared
with the first official call. This does not replace the fused kernel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.core.rng import capture_rng, restore_rng  # noqa: E402
from tests.training.four_process.compare import seed_probe  # noqa: E402
from tests.training.four_process.real_models import SEMambaPP  # noqa: E402


def _meta(value):
    if not torch.is_tensor(value):
        return {"tensor": False}
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype).replace("torch.", ""),
        "contiguous": bool(value.is_contiguous()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.modules, map_location="cpu", weights_only=False)
    payload["initial"] = checkpoint["modules"]
    adapter = SEMambaPP()
    adapter.configure({"scan_backend": "fused", "scheduler_step": "epoch", "official_width": True})
    clean = payload["clean"].to(device)
    noisy = payload["noisy"].to(device)
    if clean.dim() == 1:
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
    import selective_scan_cuda
    reference = {"fwd": [], "bwd": []}
    current = {"fwd": [], "bwd": []}

    def fwd(*call_args, **kwargs):
        out = original_fwd(*call_args, **kwargs)
        record = {"inputs": [_meta(value) for value in call_args[:7]]}
        index = len(current["fwd"])
        if len(reference["fwd"]) == index:
            reference["fwd"].append([value.detach().clone() if torch.is_tensor(value) else value for value in call_args[:7]])
            record["max_abs"] = [0.0] * 7
        else:
            record["max_abs"] = [
                float((value - saved).abs().max()) if torch.is_tensor(value) else None
                for value, saved in zip(call_args[:7], reference["fwd"][index])
            ]
        current["fwd"].append(record)
        return out

    def bwd(*call_args, **kwargs):
        out = original_bwd(*call_args, **kwargs)
        record = {"dout": _meta(call_args[8]), "inputs": [_meta(value) for value in call_args[:7]]}
        index = len(current["bwd"])
        if len(reference["bwd"]) == index:
            reference["bwd"].append(call_args[8].detach().clone())
            saved_out = [value.detach().clone() if torch.is_tensor(value) else None for value in out]
            reference.setdefault("bwd_out", []).append(saved_out)
            record["dout_max_abs"] = 0.0
            record["grad_max_abs"] = [0.0 if torch.is_tensor(value) else None for value in out]
        else:
            record["dout_max_abs"] = float((call_args[8] - reference["bwd"][index]).abs().max())
            record["grad_max_abs"] = [
                float((value - saved).abs().max()) if torch.is_tensor(value) and saved is not None else None
                for value, saved in zip(out, reference["bwd_out"][index])
            ]
        current["bwd"].append(record)
        return out

    original_fwd = selective_scan_cuda.fwd
    original_bwd = selective_scan_cuda.bwd
    selective_scan_cuda.fwd = fwd
    selective_scan_cuda.bwd = bwd
    runs = []
    try:
        for implementation in ("official", "official", "independent"):
            current["fwd"], current["bwd"] = [], []
            seed_probe(17, False)
            modules = adapter.build(implementation, device)
            for name, module in modules.items():
                module.load_state_dict(payload["initial"][name], strict=True)
            optimizers = adapter.optimizers(modules, adapter.lr)
            schedulers = adapter.schedulers(optimizers)
            seed_probe(17, True)
            rng = capture_rng(cuda=True)
            restore_rng(rng, cuda=True)
            step = adapter.train_step(implementation, modules, optimizers, schedulers, clean, noisy, 0)
            runs.append({
                "implementation": implementation,
                "loss": float(step.losses["generator"]),
                "forward_calls": len(current["fwd"]),
                "backward_calls": len(current["bwd"]),
                "forward": current["fwd"],
                "backward": current["bwd"],
            })
            del modules
            torch.cuda.empty_cache()
    finally:
        selective_scan_cuda.fwd = original_fwd
        selective_scan_cuda.bwd = original_bwd
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"step": checkpoint.get("data_position"), "runs": runs}, indent=2) + "\n")
    def peak(run, section, key):
        values = []
        for row in run[section]:
            item = row[key]
            if isinstance(item, list):
                values.extend(value for value in item if isinstance(value, float))
            elif isinstance(item, float):
                values.append(item)
        return max(values) if values else None
    print(json.dumps({
        "calls": [run["forward_calls"] for run in runs],
        "loss": [run["loss"] for run in runs],
        "fwd_max": [peak(run, "forward", "max_abs") for run in runs],
        "dout_max": [peak(run, "backward", "dout_max_abs") for run in runs],
        "bwd_grad_max": [peak(run, "backward", "grad_max_abs") for run in runs],
        "first_stride": runs[0]["forward"][0]["inputs"][0]["stride"] if runs[0]["forward"] else None,
        "cross_stride": runs[2]["forward"][0]["inputs"][0]["stride"] if runs[2]["forward"] else None,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
