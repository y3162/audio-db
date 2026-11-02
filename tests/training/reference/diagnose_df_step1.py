"""Locate the first DeepFilterNet3 official/independent gradient difference.

Uses the saved four-process input so the waveform and initial weights match
the 2026-11-02 CPU run. Does not write over that directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.training.four_process.real_models import DeepFilterNet3


def main() -> None:
    source = Path(sys.argv[1])
    prepared = torch.load(source / "input.pt", map_location="cpu", weights_only=False)
    clean = prepared["clean"]
    noisy = prepared["noisy"]
    if clean.ndim == 1:
        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
    torch.set_num_threads(1)
    adapter = DeepFilterNet3()
    adapter.configure({})
    official = adapter.build("official", torch.device("cpu"))
    independent = DeepFilterNet3()
    independent.configure({})
    modules = independent.build("independent", torch.device("cpu"))
    for name, module in modules.items():
        module.load_state_dict(prepared["initial"][name], strict=True)
    for name, module in official.items():
        module.load_state_dict(prepared["initial"][name], strict=True)

    wave_o, loss_o = _forward(adapter, "official", official, clean, noisy)
    wave_i, loss_i = _forward(independent, "independent", modules, clean, noisy)
    print("loss", float(loss_o), float(loss_i), "wave", float((wave_o - wave_i).abs().max()))

    loss_o.backward()
    loss_i.backward()
    rows = []
    for (name_o, param_o), (name_i, param_i) in zip(
        official["generator"].named_parameters(), modules["generator"].named_parameters(), strict=True
    ):
        if name_o != name_i:
            raise SystemExit(f"name mismatch {name_o} {name_i}")
        grad_o = param_o.grad
        grad_i = param_i.grad
        if grad_o is None and grad_i is None:
            continue
        diff = (grad_o - grad_i).abs()
        rows.append((float(diff.max()), float(grad_o.abs().max()), float(grad_i.abs().max()), name_o))
    rows.sort(reverse=True)
    print("top parameter gradient differences")
    for row in rows[:12]:
        print(f"{row[0]:.6e} official_max {row[1]:.6e} independent_max {row[2]:.6e} {row[3]}")

    # Same detached waveform through both spectral losses.
    for module in list(official.values()) + list(modules.values()):
        module.zero_grad(set_to_none=True)
    shared = wave_o.detach().requires_grad_(True)
    spectral_o = adapter._official_loss["spectral"](shared, wave_o.detach())
    spectral_o.backward()
    grad_official = shared.grad.detach().clone()
    shared.grad = None
    from src.training.se.deepfilternet import _multi_resolution_spectral
    spectral_i = _multi_resolution_spectral(shared, wave_o.detach())
    spectral_i.backward()
    delta = (grad_official - shared.grad).abs()
    print(
        "spectral-only waveform grad",
        "loss", float(spectral_o), float(spectral_i),
        "max", float(delta.max()),
        "rel", float(delta.norm() / grad_official.norm()),
    )


def _forward(adapter, implementation, modules, clean, noisy):
    from tests.training.four_process.real_models import _df_loss
    return _df_loss(
        implementation,
        modules["generator"],
        adapter._stream,
        adapter._params,
        clean,
        noisy,
        adapter._official_loss,
    )


if __name__ == "__main__":
    main()
