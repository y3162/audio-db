"""Compare the independent SEMamba++ generator with the reference network.

    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/semamba/bin/python tests/numerics/reference/compare_semamba_generator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "src" / "se" / "official" / "semambapp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from model.semambapp import SEMambapp  # noqa: E402
from src.se.models.semambapp.network import SEMambaPP  # noqa: E402


def _cfg(width: int, layers: int, state: int, conv: int, expand: int) -> dict:
    return {
        "model_cfg": {
            "hid_feature": width,
            "unet_expansion": 1,
            "compress_factor": 0.3,
            "num_tfmamba": layers,
            "d_state": state,
            "d_conv": conv,
            "expand": expand,
            "norm_epsilon": 1e-5,
            "beta": 2.0,
            "input_channel": 2,
            "output_channel": 1,
            "dropout": 0.1,
        },
        "stft_cfg": {"sampling_rate": 16000, "n_fft": 400, "hop_size": 100, "win_size": 400},
    }


def _flat(cfg: dict) -> dict:
    values = dict(cfg["model_cfg"])
    values["n_fft"] = cfg["stft_cfg"]["n_fft"]
    return values


def _stats(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    delta = (reference.detach().float() - other.detach().float()).abs()
    left = reference.detach().float().reshape(-1)
    right = other.detach().float().reshape(-1)
    cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "cosine": cosine}


def _run(device: torch.device, width: int, layers: int) -> dict:
    cfg = _cfg(width, layers, 16 if width >= 48 else 4, 4 if width >= 48 else 2, 2)
    official = SEMambapp(cfg).to(device).eval()
    independent = SEMambaPP(_flat(cfg)).to(device).eval()
    official_keys = set(official.state_dict())
    independent_keys = set(independent.state_dict())
    report = {
        "width": width,
        "layers": layers,
        "parameters_official": sum(p.numel() for p in official.parameters()),
        "parameters_independent": sum(p.numel() for p in independent.parameters()),
        "missing": sorted(official_keys - independent_keys)[:40],
        "extra": sorted(independent_keys - official_keys)[:40],
        "missing_count": len(official_keys - independent_keys),
        "extra_count": len(independent_keys - official_keys),
    }
    if report["missing_count"] or report["extra_count"]:
        return report
    independent.load_state_dict(official.state_dict(), strict=True)
    torch.manual_seed(11)
    frames = 16
    magnitude = torch.rand(1, 201, frames, device=device) + 0.1
    phase = torch.randn(1, 201, frames, device=device)
    with torch.no_grad():
        off_mag, off_phase, _ = official(magnitude, phase)
        our_mag, our_phase = independent(magnitude, phase)
    report["magnitude"] = _stats(off_mag, our_mag)
    report["phase"] = _stats(off_phase, our_phase)
    left_mag = magnitude.detach().clone().requires_grad_(True)
    right_mag = magnitude.detach().clone().requires_grad_(True)
    left_phase = phase.detach().clone().requires_grad_(True)
    right_phase = phase.detach().clone().requires_grad_(True)
    official.train()
    independent.train()
    off_mag, off_phase, _ = official(left_mag, left_phase)
    our_mag, our_phase = independent(right_mag, right_phase)
    (off_mag.square().mean() + off_phase.square().mean()).backward()
    (our_mag.square().mean() + our_phase.square().mean()).backward()
    report["train_magnitude"] = _stats(off_mag, our_mag)
    report["train_phase"] = _stats(off_phase, our_phase)
    report["input_grad_magnitude"] = _stats(left_mag.grad, right_mag.grad)
    independent_grads = {name: parameter.grad for name, parameter in independent.named_parameters()}
    worst = 0.0
    worst_name = ""
    compared = 0
    for name, left in official.named_parameters():
        right = independent_grads.get(name)
        if left.grad is None or right is None:
            continue
        gap = _stats(left.grad, right)["max_abs"]
        compared += 1
        if gap > worst:
            worst = gap
            worst_name = name
    report["parameter_grad_max_abs"] = worst
    report["parameter_grad_worst"] = worst_name
    report["parameter_grad_compared"] = compared
    return report


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"device": str(device), "narrow": _run(device, 8, 1), "official_width": _run(device, 48, 4)}, indent=2))


if __name__ == "__main__":
    main()
