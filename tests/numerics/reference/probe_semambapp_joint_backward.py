"""Separate joint-backward waveform differences for SEMamba++.

Diagnostic only. Official sources and production modules are not modified.
Run on a CUDA device:

    CUDA_VISIBLE_DEVICES=<gpu> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/se/bin/python tests/numerics/reference/probe_semambapp_joint_backward.py micro
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _out() -> Path:
    path = Path(os.environ.get("AUDIO_DB_WARP_OUT", "/tmp/audio-db-joint-grad"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _flags(strict: bool = False) -> None:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    warnings.filterwarnings("ignore", message="Deterministic mode:")


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    if tuple(left.shape) != tuple(right.shape):
        return {"bitwise": False, "max_abs": None, "mismatch": None, "shape": [tuple(left.shape), tuple(right.shape)]}
    delta = (left.detach().float() - right.detach().float()).abs()
    mismatch = left != right
    length = int(left.shape[-1])
    head = min(200, length)
    regions = {
        "head200": int(mismatch[..., :head].sum()),
        "tail200": int(mismatch[..., -head:].sum()) if length else 0,
        "interior": int(mismatch[..., head:-head].sum()) if length > 2 * head else 0,
    }
    indices = mismatch.reshape(-1).nonzero(as_tuple=False).flatten()[:80].tolist()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    return {
        "bitwise": bool(torch.equal(left, right)),
        "max_abs": max_abs,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "mismatch": int(mismatch.sum()),
        "numel": int(left.numel()),
        "regions": regions,
        "indices": indices,
    }


def _index_pad(waveform: torch.Tensor, pad: int) -> torch.Tensor:
    left = waveform[..., 1 : pad + 1].flip(-1)
    right = waveform[..., -pad - 1 : -1].flip(-1)
    return torch.cat((left, waveform, right), dim=-1)


class _FusedReflect(torch.autograd.Function):
    """Index reflection whose backward adds both contributions before returning."""

    @staticmethod
    def forward(ctx, waveform: torch.Tensor, pad: int, order: str) -> torch.Tensor:
        ctx.pad = int(pad)
        ctx.order = order
        if pad == 0:
            return waveform
        return _index_pad(waveform, int(pad))

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        pad = int(ctx.pad)
        if pad == 0:
            return grad, None, None
        center = grad[..., pad:-pad]
        left = grad[..., :pad].flip(-1)
        right = grad[..., -pad:].flip(-1)
        if ctx.order == "cr":
            folded = center.clone()
            folded[..., 1 : pad + 1] = folded[..., 1 : pad + 1] + left
            folded[..., -pad - 1 : -1] = folded[..., -pad - 1 : -1] + right
        elif ctx.order == "rc":
            folded = torch.zeros_like(center)
            folded[..., 1 : pad + 1] = left
            folded[..., -pad - 1 : -1] = right
            folded = folded + center
        else:
            raise RuntimeError(ctx.order)
        return folded, None, None


def _stft(framed: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, dtype=framed.dtype).to(framed.device)
    return torch.stft(
        framed,
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=False,
        return_complex=True,
    )


def _analysis(waveform: torch.Tensor, mode: str, n_fft: int = 400, hop: int = 100) -> torch.Tensor:
    pad = n_fft // 2
    if mode == "index":
        framed = _index_pad(waveform, pad)
    elif mode == "fpad":
        framed = F.pad(waveform, (pad, pad), mode="reflect")
    elif mode == "fused_cr":
        framed = _FusedReflect.apply(waveform, pad, "cr")
    elif mode == "fused_rc":
        framed = _FusedReflect.apply(waveform, pad, "rc")
    elif mode == "center":
        return torch.stft(
            waveform,
            n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=torch.hann_window(n_fft, dtype=waveform.dtype).to(waveform.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
    else:
        raise RuntimeError(mode)
    return _stft(framed, n_fft, hop)


def _grad(mode: str, waveform: torch.Tensor, kind: str) -> torch.Tensor:
    value = waveform.detach().requires_grad_(True)
    spec = _analysis(value, mode)
    extra = _analysis(value, mode, n_fft=256, hop=64)
    if kind == "single":
        loss = spec.abs().mean()
    elif kind == "with_square":
        loss = spec.abs().mean() + value.square().mean()
    elif kind == "two_stft":
        loss = spec.abs().mean() + extra.abs().mean()
    elif kind == "three":
        third = _analysis(value, mode, n_fft=512, hop=128)
        loss = spec.abs().mean() + extra.abs().mean() + third.abs().mean() + value.square().mean()
    else:
        raise RuntimeError(kind)
    loss.backward()
    return value.grad.detach()


def _repeat_spread(rows: list[torch.Tensor]) -> dict:
    if len(rows) < 2:
        return {"bitwise": True, "max_abs": 0.0}
    pairs = [_compare(rows[0], item) for item in rows[1:]]
    return {
        "bitwise": all(item["bitwise"] for item in pairs),
        "max_abs": max(item["max_abs"] for item in pairs),
        "mismatch": max(item["mismatch"] for item in pairs),
    }


def run_micro(device: torch.device) -> dict:
    torch.manual_seed(23)
    waveform = torch.randn(1, 8000, device=device)
    modes = ("index", "fpad", "fused_cr", "fused_rc", "center")
    kinds = ("single", "with_square", "two_stft", "three")
    report: dict = {"device": str(device)}
    stored = {}
    for kind in kinds:
        stored[kind] = {}
        for mode in modes:
            grads = [_grad(mode, waveform, kind) for _ in range(3)]
            stored[kind][mode] = grads[0]
            report.setdefault("spread", {}).setdefault(kind, {})[mode] = _repeat_spread(grads)
        report.setdefault("versus_fpad", {})[kind] = {
            mode: _compare(stored[kind][mode], stored[kind]["fpad"]) for mode in modes if mode != "fpad"
        }
        report.setdefault("versus_center", {})[kind] = {
            mode: _compare(stored[kind][mode], stored[kind]["center"]) for mode in modes if mode != "center"
        }
    # Same upstream into the pad only: this is the isolated pad backward.
    pad = 200
    upstream = torch.randn(1, 8000 + 2 * pad, device=device)
    isolated = {}
    for mode in modes:
        value = waveform.detach().requires_grad_(True)
        framed = _analysis_frame(value, mode, pad)
        torch.autograd.backward(framed, upstream)
        isolated[mode] = value.grad.detach()
    report["isolated_pad_versus_fpad"] = {
        mode: _compare(isolated[mode], isolated["fpad"]) for mode in modes if mode != "fpad"
    }
    # Joint versus the sum of independent backwards, index and fpad.
    report["joint_versus_sum"] = {}
    for mode in ("index", "fpad", "fused_cr", "center"):
        value = waveform.detach().requires_grad_(True)
        first = _analysis(value, mode).abs().mean()
        second = value.square().mean()
        third = _analysis(value, mode, 256, 64).abs().mean()
        (first + second + third).backward()
        joint = value.grad.detach().clone()
        pieces = []
        for term in ("stft", "square", "stft256"):
            value = waveform.detach().requires_grad_(True)
            if term == "stft":
                loss = _analysis(value, mode).abs().mean()
            elif term == "square":
                loss = value.square().mean()
            else:
                loss = _analysis(value, mode, 256, 64).abs().mean()
            loss.backward()
            pieces.append(value.grad.detach())
        folded = pieces[0] + pieces[1] + pieces[2]
        report["joint_versus_sum"][mode] = _compare(joint, folded)
    return report


def _analysis_frame(waveform: torch.Tensor, mode: str, pad: int) -> torch.Tensor:
    if mode == "index":
        return _index_pad(waveform, pad)
    if mode == "fpad":
        return F.pad(waveform, (pad, pad), mode="reflect")
    if mode == "fused_cr":
        return _FusedReflect.apply(waveform, pad, "cr")
    if mode == "fused_rc":
        return _FusedReflect.apply(waveform, pad, "rc")
    if mode == "center":
        # center=True pads internally; compare the unpadded grad, so return a
        # padded tensor built the same way torch.stft does.
        return F.pad(waveform, (pad, pad), mode="reflect")
    raise RuntimeError(mode)


def main_micro() -> None:
    _flags(False)
    cuda = torch.device("cuda:0")
    cpu = torch.device("cpu")
    report = {"cuda": run_micro(cuda), "cpu": run_micro(cpu)}
    destination = _out() / "micro.json"
    destination.write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps({"file": str(destination)}), flush=True)
    for device_name, body in report.items():
        for kind, rows in body["versus_fpad"].items():
            for mode, stats in rows.items():
                print(json.dumps({"device": device_name, "kind": kind, "mode": mode, **stats}), flush=True)


WEIGHTS = {
    "adversarial": 1.0,
    "matching": 1.0,
    "mel": 0.1,
    "magnitude": 0.9,
    "phase": 0.3,
    "complex": 0.1,
    "consistency": 0.1,
}
ALL_TERMS = ("adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency")
WAVE_TERMS = ("adversarial", "matching", "mel", "consistency")


def _fold(losses: dict, names: tuple[str, ...], *, multiply_one: bool) -> torch.Tensor:
    total = None
    for name in names:
        term = losses[name]
        weight = WEIGHTS[name]
        if multiply_one or weight != 1.0:
            term = term * weight
        total = term if total is None else total + term
    return total


def _subsets() -> list[tuple[str, ...]]:
    found = []
    for name in ALL_TERMS:
        found.append((name,))
    wave = WAVE_TERMS
    for size in (2, 3):
        found.extend(itertools_combinations(wave, size))
    found.append(wave)
    found.append(ALL_TERMS)
    for name in ALL_TERMS:
        found.append(tuple(item for item in ALL_TERMS if item != name))
    return found


def itertools_combinations(items: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    chosen = []
    def walk(start: int, prefix: tuple[str, ...]) -> None:
        if len(prefix) == size:
            chosen.append(prefix)
            return
        for index in range(start, len(items)):
            walk(index + 1, prefix + (items[index],))
    walk(0, ())
    return chosen


def _post(spectrum: torch.Tensor, addeps: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if addeps:
        epsilon = 1e-10
        magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + epsilon)
        phase = torch.atan2(spectrum.imag + epsilon, spectrum.real + epsilon)
    else:
        magnitude = spectrum.abs()
        phase = spectrum.angle()
    magnitude = magnitude.pow(0.3)
    compressed = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    return magnitude, phase, compressed


def _analysis_mag(waveform: torch.Tensor, addeps: bool, mode: str):
    from src.experiments.calc.stft import semamba_mag_phase_stft
    from tests.numerics.reference.compare_semamba_step import mag_phase_stft

    if mode == "audio":
        return semamba_mag_phase_stft(waveform, 400, 100, 400, 0.3, addeps=addeps)
    if mode == "official":
        return mag_phase_stft(waveform, 400, 100, 400, 0.3, addeps=addeps)
    pad = 200
    window = torch.hann_window(400, dtype=waveform.dtype).to(waveform.device)
    if mode == "fpad":
        framed = F.pad(waveform, (pad, pad), mode="reflect")
    elif mode == "fused_cr":
        framed = _FusedReflect.apply(waveform, pad, "cr")
    else:
        raise RuntimeError(mode)
    spectrum = torch.stft(
        framed, 400, hop_length=100, win_length=400, window=window,
        center=False, normalized=False, return_complex=True,
    )
    return _post(spectrum, addeps)


def _inverse(magnitude: torch.Tensor, phase: torch.Tensor, length: int, mode: str) -> torch.Tensor:
    from src.experiments.calc.stft import semamba_mag_phase_istft
    from tests.numerics.reference.compare_semamba_step import mag_phase_istft

    if mode == "audio":
        return semamba_mag_phase_istft(magnitude, phase, 400, 100, 400, 0.3, length=length)
    audio = mag_phase_istft(magnitude, phase, 400, 100, 400, 0.3)
    return audio[..., :length]


def _build_parts(generator, clean, noisy, analysis: str, inverse: str) -> dict:
    official_generator = generator.__class__.__name__ == "SEMambapp"
    clean_mag, clean_phase, clean_complex = _analysis_mag(clean, False, analysis)
    noisy_mag, noisy_phase, _unused = _analysis_mag(noisy, False, analysis)
    generated = generator(noisy_mag, noisy_phase)
    if official_generator:
        magnitude, phase, complex_spectrum = generated
    else:
        magnitude, phase = generated
        complex_spectrum = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    audio = _inverse(magnitude, phase, int(clean.shape[-1]), inverse)
    _unused_mag, _unused_phase, recomputed = _analysis_mag(audio, True, analysis)
    return {
        "clean_mag": clean_mag,
        "clean_phase": clean_phase,
        "clean_complex": clean_complex,
        "noisy_mag": noisy_mag,
        "noisy_phase": noisy_phase,
        "mag": magnitude,
        "phase": phase,
        "complex": complex_spectrum,
        "recomputed": recomputed,
        "audio": audio,
    }


def _decoder_grads(generator, scalar) -> dict[str, torch.Tensor]:
    named = [(name, param) for name, param in generator.named_parameters() if param.requires_grad]
    grads = torch.autograd.grad(scalar, [param for _name, param in named], retain_graph=True, allow_unused=True)
    result = {}
    for (name, param), grad in zip(named, grads, strict=True):
        if grad is None:
            continue
        if name.startswith("mask_decoder") or name.startswith("phase_decoder") or name.startswith("dense_encoder") or name.startswith("TSMamba"):
            result[name] = grad.detach().cpu()
    return result


def _audio_grad(scalar, audio: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(scalar, audio, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(audio)
    return grad.detach()


def run_model() -> None:
    from tests.numerics.reference import compare_semamba_warp as warp

    destination_dir = _out()
    if not (destination_dir / "init.pt").exists():
        warp.prepare()
    warp._flags()
    mode = os.environ.get("AUDIO_DB_RMS_WARP", "1")
    warp.configure(mode, False)
    payload = warp._load()
    device = warp._device()
    official_pack = warp._build_official()
    audio_pack = warp._build_audio("fused")
    report: dict = {"warp": mode, "width": list(warp.WIDTH)}
    bundles = {}
    for label, pack, analysis, inverse, loss_kind in (
        ("official", official_pack, "official", "official", "official"),
        ("official_repeat", official_pack, "official", "official", "official"),
        ("audio", audio_pack, "audio", "audio", "audio"),
        ("audio_repeat", audio_pack, "audio", "audio", "audio"),
        ("audio_fpad", audio_pack, "fpad", "audio", "audio"),
        ("audio_fused", audio_pack, "fused_cr", "audio", "audio"),
        ("audio_official_stft", audio_pack, "official", "audio", "audio"),
        ("audio_official_istft", audio_pack, "audio", "official", "audio"),
    ):
        print(f"case {label}", flush=True)
        cfg, generator, cqt, mrd, mel = pack
        clean, noisy = warp._move(payload, generator, cqt, mrd, mel, device)
        parts = _build_parts(generator, clean, noisy, analysis, inverse)
        losses = warp._losses_for(loss_kind, "family", parts, clean, cqt, mrd, mel, cfg)
        scalar = losses["generator"]
        audio_grad = _audio_grad(scalar, parts["audio"])
        decoded = _decoder_grads(generator, scalar)
        bundles[label] = {
            "audio_grad": audio_grad.cpu(),
            "decoded": decoded,
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "forward": {
                "audio": parts["audio"].detach().cpu(),
                "mag": parts["mag"].detach().cpu(),
                "phase": parts["phase"].detach().cpu(),
                "recomputed": parts["recomputed"].detach().cpu(),
            },
        }
        del parts, losses, scalar
        torch.cuda.empty_cache()
    reference = bundles["official"]
    pairs = {}
    for label, bundle in bundles.items():
        if label == "official":
            continue
        pairs[label] = {
            "forward_audio": _compare(reference["forward"]["audio"], bundle["forward"]["audio"]),
            "forward_mag": _compare(reference["forward"]["mag"], bundle["forward"]["mag"]),
            "forward_phase": _compare(reference["forward"]["phase"], bundle["forward"]["phase"]),
            "audio_grad": _compare(reference["audio_grad"], bundle["audio_grad"]),
            "decoder": warp._grouped(reference["decoded"], bundle["decoded"])["buckets"],
            "loss_abs": {name: abs(reference["losses"][name] - bundle["losses"][name]) for name in reference["losses"] if name in bundle["losses"]},
        }
        print(json.dumps({"case": label, "audio_grad": pairs[label]["audio_grad"], "phase_decoder": pairs[label]["decoder"].get("phase_decoder"), "magnitude_decoder": pairs[label]["decoder"].get("magnitude_decoder")}), flush=True)
    # Shared-graph loss subsets. One forward per implementation.
    subset_rows = []
    shared_grads = {}
    for label, pack, analysis, inverse, loss_kind, multiply_one in (
        ("official", official_pack, "official", "official", "official", True),
        ("audio", audio_pack, "audio", "audio", "audio", False),
    ):
        cfg, generator, cqt, mrd, mel = pack
        clean, noisy = warp._move(payload, generator, cqt, mrd, mel, device)
        parts = _build_parts(generator, clean, noisy, analysis, inverse)
        losses = warp._losses_for(loss_kind, "family", parts, clean, cqt, mrd, mel, cfg)
        shared_grads[label] = {}
        for names in _subsets():
            grad = _audio_grad(_fold(losses, names, multiply_one=multiply_one), parts["audio"]).cpu()
            shared_grads[label]["+".join(names)] = grad
        shared_grads[label]["__generator__"] = _audio_grad(losses["generator"], parts["audio"]).cpu()
        del parts, losses
        torch.cuda.empty_cache()
    for key in shared_grads["official"]:
        stats = _compare(shared_grads["official"][key], shared_grads["audio"][key])
        subset_rows.append({"subset": key, **stats})
        print(json.dumps({"subset": key, "bitwise": stats["bitwise"], "max_abs": stats["max_abs"], "mismatch": stats["mismatch"], "regions": stats["regions"]}), flush=True)
    # Sum of independent waveform-term gradients versus the joint generator gradient.
    independent = {}
    for label, pack, analysis, inverse, loss_kind, multiply_one in (
        ("official", official_pack, "official", "official", "official", True),
        ("audio", audio_pack, "audio", "audio", "audio", False),
    ):
        pieces = []
        for name in WAVE_TERMS:
            print(f"independent {label} {name}", flush=True)
            cfg, generator, cqt, mrd, mel = pack
            clean, noisy = warp._move(payload, generator, cqt, mrd, mel, device)
            parts = _build_parts(generator, clean, noisy, analysis, inverse)
            losses = warp._losses_for(loss_kind, "family", parts, clean, cqt, mrd, mel, cfg)
            pieces.append(_audio_grad(_fold(losses, (name,), multiply_one=multiply_one), parts["audio"]).cpu())
            del parts, losses
        folded = pieces[0]
        for piece in pieces[1:]:
            folded = folded + piece
        independent[label] = {
            "versus_joint": _compare(folded, shared_grads[label]["__generator__"]),
            "versus_wave_joint": _compare(folded, shared_grads[label]["+".join(WAVE_TERMS)]),
        }
        print(json.dumps({"independent": label, **independent[label]["versus_joint"]}), flush=True)
    # Order of the scalar sum on the shared audio graph.
    order_rows = {}
    cfg, generator, cqt, mrd, mel = audio_pack
    clean, noisy = warp._move(payload, generator, cqt, mrd, mel, device)
    parts = _build_parts(generator, clean, noisy, "audio", "audio")
    losses = warp._losses_for("audio", "family", parts, clean, cqt, mrd, mel, cfg)
    orders = {
        "forward": ALL_TERMS,
        "reversed": tuple(reversed(ALL_TERMS)),
        "wave_last": ("magnitude", "phase", "complex", "adversarial", "matching", "mel", "consistency"),
        "consistency_first": ("consistency", "adversarial", "matching", "mel", "magnitude", "phase", "complex"),
    }
    order_grads = {name: _audio_grad(_fold(losses, order, multiply_one=False), parts["audio"]).cpu() for name, order in orders.items()}
    order_rows = {name: _compare(order_grads["forward"], grad) for name, grad in order_grads.items()}
    def adam_decoder(pack, analysis: str, inverse: str, loss_kind: str) -> dict:
        cfg, generator, cqt, mrd, mel = pack
        clean, noisy = warp._move(payload, generator, cqt, mrd, mel, device)
        optimizer = torch.optim.AdamW(
            generator.parameters(), lr=warp.LR, betas=warp.BETAS, weight_decay=warp.WEIGHT_DECAY
        )
        parts = _build_parts(generator, clean, noisy, analysis, inverse)
        losses = warp._losses_for(loss_kind, "family", parts, clean, cqt, mrd, mel, cfg)
        losses["generator"].backward()
        optimizer.step()
        weights = {}
        moments = {}
        for name, param in generator.named_parameters():
            if not (name.startswith("mask_decoder") or name.startswith("phase_decoder")):
                continue
            weights[name] = param.detach().cpu()
            slot = optimizer.state[param]
            moments[name + ".exp_avg"] = slot["exp_avg"].detach().cpu()
            moments[name + ".exp_avg_sq"] = slot["exp_avg_sq"].detach().cpu()
        return {"weights": weights, "moments": moments}

    print("adam", flush=True)
    official_adam = adam_decoder(official_pack, "official", "official", "official")
    audio_adam = adam_decoder(audio_pack, "audio", "audio", "audio")
    report["pairs"] = pairs
    report["subsets"] = subset_rows
    report["independent"] = independent
    report["sum_order"] = order_rows
    report["adam"] = {
        "weights": warp._grouped(official_adam["weights"], audio_adam["weights"])["buckets"],
        "moments": warp._grouped(official_adam["moments"], audio_adam["moments"])["buckets"],
    }
    path = destination_dir / f"model_warp{mode}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    torch.save(
        {"official_audio_grad": bundles["official"]["audio_grad"], "audio_audio_grad": bundles["audio"]["audio_grad"]},
        destination_dir / "baseline_audio_grads.pt",
    )
    print(json.dumps({"file": str(path)}), flush=True)


def run_prod() -> None:
    """Production analysis STFT versus official, with a second waveform term."""
    from src.experiments.calc.stft import semamba_mag_phase_stft
    from tests.numerics.reference.compare_semamba_step import mag_phase_stft

    _flags(False)
    device = torch.device("cuda:0")
    torch.manual_seed(23)
    waveform = torch.randn(1, 8000, device=device)

    def grad(function) -> torch.Tensor:
        value = waveform.detach().requires_grad_(True)
        mag, phase, compressed = function(value, 400, 100, 400, 0.3, addeps=False)
        (mag.square().mean() + phase.square().mean() + compressed.square().mean() + value.square().mean()).backward()
        return value.grad.detach()

    official = grad(mag_phase_stft)
    audio = grad(semamba_mag_phase_stft)
    stats = _compare(official, audio)
    print(json.dumps({"prod_joint": stats}), flush=True)
    (_out() / "prod.json").write_text(json.dumps(stats), encoding="utf-8")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "micro"
    if command == "micro":
        main_micro()
    elif command == "prod":
        run_prod()
    elif command == "model":
        run_model()
    else:
        raise SystemExit(command)
