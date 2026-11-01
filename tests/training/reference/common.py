"""Shared helpers for official-versus-independent training comparisons."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "data" / "results" / "training" / "official-compare-20261101"


def save_report(name: str, payload: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def environment(device: torch.device) -> dict[str, Any]:
    visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    info = {
        "host": socket.gethostname(),
        "device": str(device),
        "cuda_visible_devices": visible,
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability(device))
    return info


def load_cached(key: str) -> dict[str, Any]:
    """Waveforms rendered on a host whose audio backend can read the corpus.

    Tensor's torchaudio build has no FLAC backend. The tensors are the same
    recipe render and crop the training dataset would produce.
    """
    path = REPORT_DIR / "speech_cache.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return saved[key]


def load_split(name: str, *, role: str, sample_rate: int, segment_size: int | None, training: bool, seed: int = 1234):
    from src.training.data.dataset import load_recipe_dataset

    return load_recipe_dataset(
        ROOT / "data" / "db" / "training.duckdb",
        name,
        role=role,
        seed=seed,
        sample_rate=sample_rate,
        segment_size=segment_size,
        training=training,
    )


def gap(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    delta = (reference.detach().float() - other.detach().float()).abs()
    left = reference.detach().float().reshape(-1)
    right = other.detach().float().reshape(-1)
    cosine = 1.0 if left.numel() < 2 else float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "cosine": cosine}


def grad_report(official: torch.nn.Module, independent: torch.nn.Module, *, atol: float) -> dict[str, Any]:
    worst = 0.0
    worst_name = ""
    relative_worst = 0.0
    relative_name = ""
    cosine_min = 1.0
    large_cosine = 1.0
    exceeded = 0
    compared = 0
    large = 0
    nonfinite = 0
    named = dict(independent.named_parameters())
    for name, left in official.named_parameters():
        right = named[name]
        if left.grad is None and right.grad is None:
            continue
        if left.grad is None or right.grad is None or not torch.isfinite(left.grad).all() or not torch.isfinite(right.grad).all():
            nonfinite += 1
            continue
        delta = (left.grad.detach().float() - right.grad.detach().float()).abs()
        absolute = float(delta.max())
        scale = float(left.grad.detach().abs().max())
        relative = absolute / max(scale, 1e-12)
        compared += 1
        if absolute > atol:
            exceeded += 1
        if absolute > worst:
            worst = absolute
            worst_name = name
        if relative > relative_worst:
            relative_worst = relative
            relative_name = name
        flat_l = left.grad.detach().float().reshape(-1)
        flat_r = right.grad.detach().float().reshape(-1)
        if flat_l.numel() > 1:
            cosine = float(torch.nn.functional.cosine_similarity(flat_l, flat_r, dim=0))
            cosine_min = min(cosine_min, cosine)
            if scale > 1e-5:
                large += 1
                large_cosine = min(large_cosine, cosine)
    return {
        "max_abs": worst,
        "worst": worst_name,
        "max_relative": relative_worst,
        "relative_worst": relative_name,
        "cosine_min": cosine_min,
        "large_cosine_min": large_cosine,
        "exceeded_atol": exceeded,
        "compared": compared,
        "large_compared": large,
        "nonfinite": nonfinite,
        "atol": atol,
    }


def weight_report(official: torch.nn.Module, independent: torch.nn.Module) -> dict[str, float | str]:
    worst = 0.0
    worst_name = ""
    named = dict(independent.named_parameters())
    for name, left in official.named_parameters():
        value = float((left.detach() - named[name].detach()).abs().max())
        if value > worst:
            worst = value
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def optim_report(official: torch.optim.Optimizer, independent: torch.optim.Optimizer) -> dict[str, Any]:
    worst = 0.0
    worst_name = ""
    missing = 0
    left_params = [parameter for group in official.param_groups for parameter in group["params"]]
    right_params = [parameter for group in independent.param_groups for parameter in group["params"]]
    for index, (left, right) in enumerate(zip(left_params, right_params, strict=True)):
        left_state = official.state.get(left, {})
        right_state = independent.state.get(right, {})
        keys = set(left_state) | set(right_state)
        for key in keys:
            if key == "step":
                if int(left_state.get(key, -1)) != int(right_state.get(key, -2)):
                    missing += 1
                continue
            if key not in left_state or key not in right_state:
                missing += 1
                continue
            if not torch.is_tensor(left_state[key]):
                continue
            value = float((left_state[key].detach().float() - right_state[key].detach().float()).abs().max())
            if value > worst:
                worst = value
                worst_name = f"{index}:{key}"
    return {"max_abs": worst, "worst": worst_name, "missing_or_step_mismatch": missing}


def rng_snapshot() -> dict[str, torch.Tensor]:
    snap = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state()
    return snap


def rng_restore(snap: dict[str, torch.Tensor]) -> None:
    torch.set_rng_state(snap["cpu"])
    if "cuda" in snap:
        torch.cuda.set_rng_state(snap["cuda"])


def reference_scores(clean: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    from src.evaluation.metrics.pesq import score_pesq
    from src.evaluation.metrics.stoi import score_stoi

    return {
        "pesq": score_pesq(clean.detach().cpu(), estimate.detach().cpu(), sample_rate),
        "stoi": score_stoi(clean.detach().cpu(), estimate.detach().cpu(), sample_rate, extended=False),
    }
