"""Explain the TF-Locoformer float32 loss gap at DNS width.

Thresholds and the 1 ULP rule are fixed in VALIDATION_PLAN.md. This script
records measurements. It does not change those rules.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-level1-20261111/locoformer"))

from src.experiments.calc.spectral import multi_resolution_l1  # noqa: E402
from src.se.models.tf_locoformer.model import TFLocoformer  # noqa: E402
from tests.numerics.reference.common import environment, gap, grad_report, optim_report, weight_report  # noqa: E402
from tests.numerics.reference.compare_tf_gridnet import _official_loss_class  # noqa: E402
from tests.se.test_tf_locoformer import OFFICIAL  # noqa: E402

WINDOWS = (256, 512, 768, 1024)
DNS = dict(
    num_spk=1,
    n_layers=6,
    emb_dim=128,
    norm_type="rmsgroupnorm",
    num_groups=4,
    tf_order="ft",
    n_heads=4,
    flash_attention=False,
    attention_dim=128,
    pos_enc="rope",
    ffn_type=["swiglu_conv1d", "swiglu_conv1d"],
    ffn_hidden_dim=[384, 384],
    conv1d_kernel=4,
    conv1d_shift=1,
    dropout=0.0,
    eps=1.0e-5,
    n_fft=256,
    hop_length=128,
    normalize_variance=True,
)


def _separator_class():
    root = str(OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    from standalone.tflocoformer_separator import TFLocoformerSeparator

    return TFLocoformerSeparator


def _audio() -> TFLocoformer:
    options = {key: value for key, value in DNS.items() if key not in ("n_fft", "hop_length", "normalize_variance")}
    return TFLocoformer(
        n_fft=DNS["n_fft"],
        hop_length=DNS["hop_length"],
        normalize_variance=DNS["normalize_variance"],
        **options,
    )


def _save(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)


def _cgap(left: torch.Tensor, right: torch.Tensor) -> dict:
    if torch.is_complex(left):
        return gap(torch.view_as_real(left), torch.view_as_real(right))
    return gap(left, right)


def _set_sdp(module: torch.nn.Module, *, math_only: bool) -> None:
    for child in module.modules():
        config = getattr(child, "flash_attention_config", None)
        if not isinstance(config, dict) or config.get("enable_flash"):
            continue
        child.flash_attention_config = {
            "enable_flash": False,
            "enable_math": True,
            "enable_mem_efficient": not math_only,
        }


def _spectrum(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.std(waveform, dim=1, keepdim=True)
    window = torch.hann_window(256, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform / scale, 256, 128, 256, window, center=True, return_complex=True
    ).transpose(1, 2)
    return spectrum, scale


def _wave_from(separator: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    scale = torch.std(waveform, dim=1, keepdim=True)
    window = torch.hann_window(256, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform / scale, 256, 128, 256, window, center=True, return_complex=True
    ).transpose(1, 2)
    enhanced = separator(spectrum)[:, 0].transpose(1, 2)
    restored = torch.istft(enhanced, 256, 128, 256, window, center=True, length=waveform.shape[-1])
    return restored * scale


def _block_gaps(left: torch.nn.Module, right: torch.nn.Module, spectrum: torch.Tensor) -> list[dict]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def hook(name: str):
        def remember(_module, _inputs, output):
            captured[name] = output.detach()

        return remember

    hooks.append(left.conv.register_forward_hook(hook("l_conv")))
    hooks.append(right.conv.register_forward_hook(hook("r_conv")))
    for index, (block_l, block_r) in enumerate(zip(left.blocks, right.blocks, strict=True)):
        hooks.append(block_l.register_forward_hook(hook(f"l_b{index}")))
        hooks.append(block_r.register_forward_hook(hook(f"r_b{index}")))
    hooks.append(left.deconv.register_forward_hook(hook("l_deconv")))
    hooks.append(right.deconv.register_forward_hook(hook("r_deconv")))
    with torch.no_grad():
        out_l = left(spectrum)
        out_r = right(spectrum)
    for item in hooks:
        item.remove()
    names = ["conv", *[f"b{index}" for index in range(len(left.blocks))], "deconv"]
    rows = [{"name": "separator_out", **_cgap(out_l, out_r)}]
    for name in names:
        rows.append({"name": name, **_cgap(captured[f"l_{name}"], captured[f"r_{name}"])})
    return rows


def _float_ulp(value: float) -> float:
    number = np.float32(value)
    return float(np.nextafter(number, np.float32(np.inf)) - number)


def _terms(target: torch.Tensor, estimate: torch.Tensor, dtype: torch.dtype) -> dict[str, float]:
    reference = target.detach().to(device="cpu", dtype=dtype)
    predicted = estimate.detach().to(device="cpu", dtype=dtype)
    scale = torch.sum(predicted * reference, dim=-1, keepdim=True) / (
        torch.sum(predicted.square(), dim=-1, keepdim=True) + 1e-8
    )
    time_loss = (predicted * scale - reference).abs().sum(dim=-1)
    spectral = torch.zeros_like(time_loss)
    per_window = {}
    for window in WINDOWS:
        hop = window // 2
        pad = window // 2
        def stft(wave: torch.Tensor) -> torch.Tensor:
            left = wave[..., 1 : pad + 1].flip(-1)
            right = wave[..., -pad - 1 : -1].flip(-1)
            framed = torch.cat((left, wave, right), dim=-1)
            return torch.stft(
                framed, window, hop_length=hop, win_length=window, window=None, center=False, return_complex=True
            )

        delta = (stft(predicted * scale).abs() - stft(reference).abs()).abs().sum(dim=(-2, -1))
        per_window[str(window)] = float(delta.mean())
        spectral = spectral + delta
    total = (time_loss * 0.5 + 0.5 * spectral / len(WINDOWS)).mean()
    return {"time": float(time_loss.mean()), **per_window, "total": float(total)}


def _loss_pair(criterion, clean: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> dict:
    def official(wave: torch.Tensor) -> float:
        return float(criterion(clean, wave).mean().detach())

    def port(wave: torch.Tensor) -> float:
        return float(multi_resolution_l1(clean, wave, time_domain_weight=0.5, normalize_variance=False).detach())

    official_left = official(left)
    official_right = official(right)
    port_left = port(left)
    port_right = port(right)
    return {
        "official_on_left": official_left,
        "official_on_right": official_right,
        "port_on_left": port_left,
        "port_on_right": port_right,
        "identity_on_left": abs(official_left - port_left),
        "identity_on_right": abs(official_right - port_right),
        "official_gap": abs(official_left - official_right),
        "port_gap": abs(port_left - port_right),
        "ulp_at_official_left": _float_ulp(official_left),
        "float64_terms_left": _terms(clean, left, torch.float64),
        "float64_terms_right": _terms(clean, right, torch.float64),
    }


def _formula(left, right, pre_l, pre_r, grad_l, grad_r, lr: float, decay: float) -> dict:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, param in left.named_parameters():
        if grad_l.get(name) is None or grad_r.get(name) is None:
            continue
        scale = 1.0 - lr * decay
        pred_l = pre_l[name] * scale - lr * grad_l[name] / (grad_l[name].abs() + 1e-8)
        pred_r = pre_r[name] * scale - lr * grad_r[name] / (grad_r[name].abs() + 1e-8)
        actual = param.detach() - named[name].detach()
        value = float((actual - (pred_l - pred_r)).abs().max())
        if value >= worst:
            worst = value
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def _clone(module: torch.nn.Module) -> tuple[dict, dict]:
    weights = {}
    grads = {}
    for name, param in module.named_parameters():
        weights[name] = param.detach().clone()
        grads[name] = None if param.grad is None else param.grad.detach().clone()
    return weights, grads


def _tone(step: int, length: int, device: torch.device, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.arange(length, device=device) / 16000
    rows_clean = []
    rows_noisy = []
    for index in range(batch):
        clean = 0.2 * torch.sin(2 * torch.pi * (120 + 10 * step + index) * time)
        noisy = clean * 0.8 + 0.05 * torch.sin(2 * torch.pi * (380 + 15 * step + index) * time)
        rows_clean.append(clean)
        rows_noisy.append(noisy)
    return torch.stack(rows_clean), torch.stack(rows_noisy)


def _copy_weights(source: torch.nn.Module, dest: torch.nn.Module) -> None:
    dest.load_state_dict(source.state_dict(), strict=True)


def _train_pair(device, criterion, steps: int, *, math_only: bool, resync: bool, kind: str) -> list[dict]:
    official_cls = _separator_class()
    left = official_cls(**{key: value for key, value in DNS.items() if key not in ("n_fft", "hop_length", "normalize_variance")}).to(device)
    if kind == "oo":
        right = official_cls(**{key: value for key, value in DNS.items() if key not in ("n_fft", "hop_length", "normalize_variance")}).to(device)
        _copy_weights(left, right)
    else:
        audio = _audio().to(device)
        _copy_weights(left, audio.separator)
        right = audio.separator
    _set_sdp(left, math_only=math_only)
    _set_sdp(right, math_only=math_only)
    opt_l = torch.optim.AdamW(left.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    opt_r = torch.optim.AdamW(right.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    rows = []
    for step in range(steps):
        clean, noisy = _tone(step, 64000, device)
        left.train()
        right.train()
        opt_l.zero_grad(set_to_none=True)
        opt_r.zero_grad(set_to_none=True)
        wave_l = _wave_from(left, noisy)
        loss_l = multi_resolution_l1(clean, wave_l, time_domain_weight=0.5, normalize_variance=False)
        loss_l.backward()
        torch.nn.utils.clip_grad_norm_(left.parameters(), 5.0)
        wave_r = _wave_from(right, noisy)
        loss_r = multi_resolution_l1(clean, wave_r, time_domain_weight=0.5, normalize_variance=False)
        loss_r.backward()
        torch.nn.utils.clip_grad_norm_(right.parameters(), 5.0)
        pre_l, grad_l = _clone(left)
        pre_r, grad_r = _clone(right)
        opt_l.step()
        opt_r.step()
        with torch.no_grad():
            official_gap = abs(float(criterion(clean, wave_l.detach()).mean()) - float(criterion(clean, wave_r.detach()).mean()))
        row = {
            "step": step + 1,
            "port_loss_gap": abs(float(loss_l.detach()) - float(loss_r.detach())),
            "official_loss_gap": official_gap,
            "ulp": _float_ulp(float(loss_l.detach())),
            "wave": gap(wave_l, wave_r),
            "grad": grad_report(left, right, atol=1e-4),
            "weight": weight_report(left, right),
            "optim": optim_report(opt_l, opt_r),
            "formula": _formula(left, right, pre_l, pre_r, grad_l, grad_r, 1e-3, 0.01) if step == 0 else None,
        }
        rows.append(row)
        print(kind, "math" if math_only else "native", row["step"], row["port_loss_gap"], row["wave"]["max_abs"], row["weight"]["max_abs"], flush=True)
        if resync:
            _copy_weights(left, right)
            opt_r.load_state_dict(opt_l.state_dict())
        del wave_l, wave_r, loss_l, loss_r, pre_l, pre_r, grad_l, grad_r
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report: dict = {
        "environment": environment(device),
        "official_commit": "7a615460d347ff7334a13dbb831d16280da72cdc",
        "width": DNS,
        "official_batch_size": 4,
        "samples": 64000,
        "native_sdp": "enable_flash=False, enable_math=True, enable_mem_efficient=True",
        "tight_sdp": "enable_mem_efficient=False; control only, not the official native setting",
    }
    criterion = _official_loss_class()(
        window_sz=list(WINDOWS),
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        normalize_variance=False,
        reduction="sum",
    ).to(device)
    official_cls = _separator_class()
    options = {key: value for key, value in DNS.items() if key not in ("n_fft", "hop_length", "normalize_variance")}
    torch.manual_seed(3)
    official = official_cls(**options).to(device)
    other = official_cls(**options).to(device)
    _copy_weights(official, other)
    audio = _audio().to(device)
    _copy_weights(official, audio.separator)
    init_gap = weight_report(official, audio.separator)
    report["init_weight_max_abs"] = init_gap
    clean, noisy = _tone(0, 64000, device)
    spectrum, _scale = _spectrum(noisy)
    # Loss identity does not need the network.
    probe_clean = torch.randn(1, 4096, device=device)
    probe_est = torch.randn(1, 4096, device=device)
    report["loss_identity_random_4096"] = abs(
        float(criterion(probe_clean, probe_est).mean().detach())
        - float(multi_resolution_l1(probe_clean, probe_est, time_domain_weight=0.5, normalize_variance=False).detach())
    )
    print("identity", report["loss_identity_random_4096"], flush=True)
    _save("partial.json", report)

    for math_only, label in ((False, "native"), (True, "math")):
        _set_sdp(official, math_only=math_only)
        _set_sdp(other, math_only=math_only)
        _set_sdp(audio.separator, math_only=math_only)
        oo = _block_gaps(official, other, spectrum)
        oa = _block_gaps(official, audio.separator, spectrum)
        with torch.no_grad():
            wave_o = _wave_from(official, noisy)
            wave_a = _wave_from(audio.separator, noisy)
            wave_oo = _wave_from(other, noisy)
        report[label] = {
            "separator_oo": oo,
            "separator_oa": oa,
            "wave_oo": gap(wave_o, wave_oo),
            "wave_oa": gap(wave_o, wave_a),
            "loss_oa": _loss_pair(criterion, clean, wave_o, wave_a),
            "loss_oo": _loss_pair(criterion, clean, wave_o, wave_oo),
        }
        print(label, "wave_oa", report[label]["wave_oa"], "port_gap", report[label]["loss_oa"]["port_gap"], flush=True)
        _save("partial.json", report)
        del wave_o, wave_a, wave_oo
        torch.cuda.empty_cache()

    try:
        clean4, noisy4 = _tone(0, 64000, device, batch=4)
        spectrum4, _scale4 = _spectrum(noisy4)
        _set_sdp(official, math_only=False)
        _set_sdp(audio.separator, math_only=False)
        report["batch4_native_oa"] = _block_gaps(official, audio.separator, spectrum4)
        print("batch4", report["batch4_native_oa"][0], flush=True)
    except RuntimeError as exc:
        report["batch4_native_oa"] = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
        torch.cuda.empty_cache()
        print("batch4 blocked", exc, flush=True)
    _save("partial.json", report)
    del official, other, audio, spectrum
    torch.cuda.empty_cache()

    report["steps_native_oa"] = _train_pair(device, criterion, 3, math_only=False, resync=False, kind="oa")
    _save("partial.json", report)
    report["steps_native_oo"] = _train_pair(device, criterion, 3, math_only=False, resync=False, kind="oo")
    _save("partial.json", report)
    report["steps_math_oa_resync"] = _train_pair(device, criterion, 3, math_only=True, resync=True, kind="oa")
    _save("partial.json", report)
    report["steps_math_oo_resync"] = _train_pair(device, criterion, 3, math_only=True, resync=True, kind="oo")
    _save("locoformer_ulp.json", report)


if __name__ == "__main__":
    main()
