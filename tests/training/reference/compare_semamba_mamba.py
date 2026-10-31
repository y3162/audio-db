"""Compare the independent selective scan with the CUDA Mamba kernel.

Run on a Turing GPU. The installed ``mamba_ssm`` and ``causal_conv1d``
binaries contain sm_75 cubins only, so an A100 cannot load them.

    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/semamba/bin/python tests/training/reference/compare_semamba_mamba.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mamba_ssm.modules.mamba_simple import Mamba  # noqa: E402
from src.se.models.semambapp.network import SelectiveSSM  # noqa: E402


def _stats(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    left = reference.detach().float().reshape(-1)
    right = other.detach().float().reshape(-1)
    delta = (left - right).abs()
    cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "cosine": cosine,
    }


def _compare(device: torch.device) -> dict:
    torch.manual_seed(5)
    official = Mamba(d_model=48, d_state=16, d_conv=4, expand=2).to(device)
    independent = SelectiveSSM(48, 16, 4, 2).to(device)
    missing = sorted(set(official.state_dict()) - set(independent.state_dict()))
    extra = sorted(set(independent.state_dict()) - set(official.state_dict()))
    if missing or extra:
        return {"missing": missing, "extra": extra}
    independent.load_state_dict(official.state_dict(), strict=True)
    source = torch.randn(2, 64, 48, device=device)
    left_in = source.detach().clone().requires_grad_(True)
    right_in = source.detach().clone().requires_grad_(True)
    left = official(left_in)
    right = independent(right_in)
    left.sum().backward()
    right.sum().backward()
    gradients = {"input": _stats(left_in.grad, right_in.grad)}
    for name, left_param, right_param in zip(
        [name for name, _ in official.named_parameters()],
        official.parameters(),
        independent.parameters(),
        strict=True,
    ):
        if left_param.grad is None or right_param.grad is None:
            gradients[name] = {"max_abs": None}
            continue
        gradients[name] = _stats(left_param.grad, right_param.grad)
    finite = bool(torch.isfinite(right).all() and torch.isfinite(left).all())
    return {
        "device": str(device),
        "output": _stats(left, right),
        "finite": finite,
        "gradients": gradients,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the fused Mamba kernel")
    device = torch.device("cuda:0")
    print(json.dumps(_compare(device), indent=2))


if __name__ == "__main__":
    main()
