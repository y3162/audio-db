"""One step: official batch 8 versus micro 4 x accumulation 2.

Full SEMamba++ width and the same eight waveforms. CPU uses the reference
scan. CUDA uses ``scan_backend: fused``, one model at a time, and compares
the accumulation gap with a second batch-8 run. CUDA bit identity is not
required.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from src.experiments.calc.semambapp import SEMambaTask


def _config(device: str) -> dict:
    architecture = {
        "hid_feature": 48,
        "num_tfmamba": 4,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "norm_epsilon": 1.0e-5,
        "input_channel": 2,
        "output_channel": 1,
        "n_fft": 400,
    }
    if device == "cuda":
        architecture["scan_backend"] = "fused"
    return {
        "seed": 17,
        "deterministic": False,
        "loss": "official",
        "optimizer": {
            "name": "adamw",
            "lr": 0.0002,
            "betas": [0.8, 0.99],
            "weight_decay": 0.01,
            "amsgrad": False,
        },
        "scheduler": {"name": "exponential", "gamma": 0.99},
        "model": {
            "name": "semambapp",
            "sampling_rate": 16000,
            "architecture": architecture,
            "stft": {"hop": 100, "win": 400},
        },
    }


def _task(device: str) -> SEMambaTask:
    task = SEMambaTask(_config(device), torch.device(device))
    task.capture_update = True
    task.capture_gradients = True
    return task


def _weights(task: SEMambaTask) -> dict[str, torch.Tensor]:
    found = {}
    for label, module in (
        ("generator", task.generator),
        ("mrd", task.discriminator_mrd),
        ("band", task.discriminator_band),
    ):
        for name, parameter in module.named_parameters():
            found[f"{label}.{name}"] = parameter.detach().cpu().clone()
    return found


def _relative_l2(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name in left:
        diff = (left[name] - right[name]).detach().double()
        numerator += float(diff.square().sum())
        denominator += float(left[name].detach().double().square().sum())
    return numerator**0.5 / (denominator**0.5 + 1e-12)


def _max_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    shared = sorted(set(left) & set(right))
    if not shared or set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        raise SystemExit(f"parameter names differ: {missing[:8]}")
    return max(float((left[name] - right[name]).abs().max()) for name in shared)


def _clone_tree(value):
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _audio_max(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return float((torch.cat(left, 0) - torch.cat(right, 0)).abs().max())


def _run(task: SEMambaTask, batches: list[dict], accumulation: int) -> tuple[dict, dict, dict]:
    if task.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(task.device)
        torch.cuda.synchronize(task.device)
    started = time.perf_counter()
    losses = task.train_update(batches, accumulation)
    if task.device.type == "cuda":
        torch.cuda.synchronize(task.device)
    seconds = time.perf_counter() - started
    capture = {
        "generator_grad": task.last_capture["generator_grad"],
        "discriminator_grad": task.last_capture["discriminator_grad"],
        "audio": [audio.cpu() for audio in task.last_capture["audio"]],
    }
    stats = {
        "seconds": seconds,
        "losses": losses,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(task.device) if task.device.type == "cuda" else None
        ),
    }
    return _weights(task), capture, stats


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if device not in ("cpu", "cuda"):
        raise SystemExit("device must be cpu or cuda")
    length = 8000
    generator = torch.Generator().manual_seed(17)
    clean = torch.randn(8, length, generator=generator)
    noisy = clean + 0.05 * torch.randn(8, length, generator=generator)
    ids = [f"u{index}" for index in range(8)]
    whole = {"clean": clean, "noisy": noisy, "case_id": ids}
    first = {"clean": clean[:4], "noisy": noisy[:4], "case_id": ids[:4]}
    second = {"clean": clean[4:], "noisy": noisy[4:], "case_id": ids[4:]}

    started = time.perf_counter()
    task = _task(device)
    initial = _clone_tree(task.state_dict())
    build_seconds = time.perf_counter() - started
    weights_a, capture_a, stats_a = _run(task, [whole], 1)
    task.load_state_dict(initial)
    weights_b, capture_b, stats_b = _run(task, [whole], 1)
    task.load_state_dict(initial)
    weights_c, capture_c, stats_c = _run(task, [first, second], 2)

    report = {
        "device": device,
        "scan": "fused" if device == "cuda" else "reference",
        "length": length,
        "deterministic": False,
        "build_seconds": build_seconds,
        "official_seconds": stats_a["seconds"],
        "repeat_seconds": stats_b["seconds"],
        "accumulation_seconds": stats_c["seconds"],
        "peak_allocated_bytes": stats_a["peak_allocated_bytes"],
        "loss_official": stats_a["losses"],
        "loss_repeat": stats_b["losses"],
        "loss_accumulation": stats_c["losses"],
        "self_rerun_weight_max_abs": _max_diff(weights_a, weights_b),
        "accum_weight_max_abs": _max_diff(weights_a, weights_c),
        "self_rerun_weight_relative_l2": _relative_l2(weights_a, weights_b),
        "accum_weight_relative_l2": _relative_l2(weights_a, weights_c),
        "self_rerun_generator_grad_max_abs": _max_diff(capture_a["generator_grad"], capture_b["generator_grad"]),
        "accum_generator_grad_max_abs": _max_diff(capture_a["generator_grad"], capture_c["generator_grad"]),
        "self_rerun_generator_grad_relative_l2": _relative_l2(
            capture_a["generator_grad"], capture_b["generator_grad"]
        ),
        "accum_generator_grad_relative_l2": _relative_l2(
            capture_a["generator_grad"], capture_c["generator_grad"]
        ),
        "self_rerun_discriminator_grad_relative_l2": _relative_l2(
            capture_a["discriminator_grad"], capture_b["discriminator_grad"]
        ),
        "accum_discriminator_grad_relative_l2": _relative_l2(
            capture_a["discriminator_grad"], capture_c["discriminator_grad"]
        ),
        "self_rerun_discriminator_grad_max_abs": _max_diff(
            capture_a["discriminator_grad"], capture_b["discriminator_grad"]
        ),
        "accum_discriminator_grad_max_abs": _max_diff(
            capture_a["discriminator_grad"], capture_c["discriminator_grad"]
        ),
        "self_rerun_audio_max_abs": _audio_max(capture_a["audio"], capture_b["audio"]),
        "accum_audio_max_abs": _audio_max(capture_a["audio"], capture_c["audio"]),
    }
    out = Path("data/results/training/go-20261104")
    out.mkdir(parents=True, exist_ok=True)
    name = "semamba-accum-cuda.json" if device == "cuda" else "semamba-accum-cpu.json"
    (out / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if not str(key).startswith("loss_")}, indent=2))


if __name__ == "__main__":
    main()
