"""MP-SENet STFT forward and backward: native center=True versus index reflection.

Runs the same waveform twice on each implementation. A gap between the two
native calls is CUDA nondeterminism. A gap that remains on CPU is an
inequivalent pad. This does not train MP-SENet.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-continue"))

from src.experiments.calc.stft import centered_stft  # noqa: E402


def _native(waveform: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        waveform,
        400,
        hop_length=100,
        win_length=400,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )


def _index(waveform: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return centered_stft(waveform, 400, 100, window)


def _once(device: torch.device, kind: str) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(11)
    wave = torch.randn(2, 8000, generator=generator, dtype=torch.float32, device="cpu").to(device)
    window = torch.hann_window(400, dtype=torch.float32).to(device)
    wave = wave.detach().requires_grad_(True)
    spectrum = _native(wave, window) if kind == "native" else _index(wave, window)
    loss = spectrum.abs().sum()
    loss.backward()
    return {"loss": float(loss.detach()), "grad": wave.grad.detach().clone(), "spectrum": spectrum.detach().clone()}


def _pair(device: torch.device, kind: str) -> dict:
    first = _once(device, kind)
    second = _once(device, kind)
    return {
        "self_loss_gap": abs(first["loss"] - second["loss"]),
        "self_grad_max_abs": float((first["grad"] - second["grad"]).abs().max()),
        "self_spectrum_max_abs": float((first["spectrum"] - second["spectrum"]).abs().max()),
        "loss": first["loss"],
        "grad": first["grad"],
        "spectrum": first["spectrum"],
    }


def _cross(left: dict, right: dict) -> dict:
    return {
        "loss_gap": abs(left["loss"] - right["loss"]),
        "grad_max_abs": float((left["grad"] - right["grad"]).abs().max()),
        "spectrum_max_abs": float((left["spectrum"] - right["spectrum"]).abs().max()),
    }


def _device_report(device: torch.device) -> dict:
    native = _pair(device, "native")
    index = _pair(device, "index")
    return {
        "native_self": {key: native[key] for key in ("self_loss_gap", "self_grad_max_abs", "self_spectrum_max_abs")},
        "index_self": {key: index[key] for key in ("self_loss_gap", "self_grad_max_abs", "self_spectrum_max_abs")},
        "native_versus_index": _cross(native, index),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {"cpu": _device_report(torch.device("cpu"))}
    if torch.cuda.is_available():
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            report["deterministic_algorithms"] = f"{type(exc).__name__}: {exc}"
        report["cuda"] = _device_report(torch.device("cuda:0"))
    else:
        report["cuda"] = {"status": "NOT_RUN", "reason": "cuda unavailable"}
    path = OUT / "mp_stft_backward.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
