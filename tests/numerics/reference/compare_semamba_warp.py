"""Warp-controlled SEMamba++ numerics. Diagnostic only.

Triton RMSNorm warps are pinned by replacing the autotuner config list in this
process. Official sources and the production scan path are not modified.

Import ``compare_semamba_step`` before enabling deterministic algorithms so
the RMSNorm autotuner keeps the published config list. Deterministic mode is
then enabled the same way as ``compare_semamba_scan_variance.py``.

    CUDA_VISIBLE_DEVICES=<gpu> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/se/bin/python tests/numerics/reference/compare_semamba_warp.py prepare
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.numerics.reference.compare_semamba_step import (  # noqa: E402
    BETAS,
    LR,
    WEIGHT_DECAY,
    _cfg,
    _flat,
    _independent_losses,
    _official_losses,
)
from tests.numerics.reference.compare_semamba_step import mag_phase_istft, mag_phase_stft  # noqa: E402
from src.experiments.calc.spectral import (  # noqa: E402
    compressed_complex_losses,
    feature_matching,
    least_squares_discriminator,
    least_squares_generator,
    summed_slaney_log_mel,
)
from src.experiments.calc.stft import semamba_mag_phase_istft, semamba_mag_phase_stft  # noqa: E402

WIDTH = (48, 4, 16, 4)
GAMMA = 0.99
MODES = ("autotune", "1", "2")


def _out() -> Path:
    path = Path(os.environ.get("AUDIO_DB_WARP_OUT", "/tmp/audio-db-warp-numerics"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    return torch.device("cuda:0")


def _flags() -> None:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", message="Deterministic mode:")


def _kernels():
    from mamba_ssm.ops.triton import layer_norm as layer_norm

    return layer_norm._layer_norm_fwd_1pass_kernel, layer_norm._layer_norm_bwd_kernel


_SNAPSHOT: list | None = None


def _snapshot() -> list:
    global _SNAPSHOT
    if _SNAPSHOT is None:
        _SNAPSHOT = [list(kernel.configs) for kernel in _kernels()]
    return _SNAPSHOT


def configure(mode: str, trace: bool, reset_cache: bool = True) -> list[dict]:
    """Pin both RMSNorm autotuners. ``autotune`` restores the imported list.

    ``reset_cache=False`` keeps the selected autotune entry so a second call
    does not benchmark again. The launch wrapper is removed unless ``trace``.
    """

    import triton

    kernels = _kernels()
    original = _snapshot()
    if mode == "autotune":
        chosen = original
    else:
        warp = int(mode)
        chosen = [[triton.Config({}, num_warps=warp)] for _kernel in kernels]
    traces: list[dict] = []
    for kernel, configs in zip(kernels, chosen, strict=True):
        if not hasattr(kernel, "_audio_db_run"):
            kernel._audio_db_run = kernel.run
        kernel.run = kernel._audio_db_run
        if reset_cache:
            kernel.configs = list(configs)
            kernel.cache.clear()
            kernel.best_config = None
        if not trace:
            continue
        name = kernel.base_fn.__name__

        def wrapped(*args, _kernel=kernel, _run=kernel._audio_db_run, _name=name, **kwargs):
            result = _run(*args, **kwargs)
            tensor = args[0]
            config = _kernel.best_config
            traces.append(
                {
                    "kernel": _name,
                    "num_warps": None if config is None else int(config.num_warps),
                    "num_stages": None if config is None else int(getattr(config, "num_stages", -1)),
                    "rows": None if not torch.is_tensor(tensor) else int(tensor.shape[0]),
                    "width": None if not torch.is_tensor(tensor) else int(tensor.shape[-1]),
                    "stride": None if not torch.is_tensor(tensor) else [int(item) for item in tensor.stride()],
                    "dtype": None if not torch.is_tensor(tensor) else str(tensor.dtype),
                }
            )
            return result

        kernel.run = wrapped
    return traces


def _configured() -> dict[str, list[int]]:
    names = ("fwd", "bwd")
    return {
        name: [int(config.num_warps) for config in kernel.configs]
        for name, kernel in zip(names, _kernels(), strict=True)
    }


def _selected_warps() -> dict[str, int | None]:
    selected = {}
    for name, kernel in zip(("fwd", "bwd"), _kernels(), strict=True):
        config = getattr(kernel, "best_config", None)
        selected[name] = None if config is None else int(config.num_warps)
    return selected


def _warp_ok(mode: str, traces: list[dict]) -> bool:
    selected = set(_selected_warps().values())
    observed = {item["num_warps"] for item in traces} or selected
    if None in observed or not observed:
        return False
    if mode == "autotune":
        configured = {warp for warps in _configured().values() for warp in warps}
        return observed <= configured and len(configured) > 1
    return observed == {int(mode)} and _configured()["fwd"] == [int(mode)] and _configured()["bwd"] == [int(mode)]


def _meta(tensor: torch.Tensor) -> dict:
    return {
        "shape": [int(item) for item in tensor.shape],
        "stride": [int(item) for item in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _digest(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _hash_named(named: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(named):
        tensor = named[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _versions() -> dict:
    import triton
    import mamba_ssm

    official = ROOT / "src/official/semambapp"
    commit = __import__("subprocess").check_output(
        ["git", "-C", str(official), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version()),
        "triton": triton.__version__,
        "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"),
        "official_semambapp": commit,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _rng() -> dict:
    return {
        "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda": [
            hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ],
    }


def _load():
    return torch.load(_out() / "init.pt", map_location="cpu", weights_only=False)


def _build_official():
    from model.discriminator import MultiResolutionDiscriminator as OfficialMRD
    from model.discriminator import MultiScaleSubbandCQTDiscriminator as OfficialCQT
    from model.loss import MultiScaleMelSpectrogramLoss
    from model.semambapp import SEMambapp

    cfg = _cfg(*WIDTH)
    return cfg, SEMambapp(cfg), OfficialCQT(), OfficialMRD(), MultiScaleMelSpectrogramLoss(sampling_rate=16000)


def _build_audio(scan: str = "fused"):
    from model.loss import MultiScaleMelSpectrogramLoss
    from src.models.semambapp.discriminator import MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator
    from src.models.semambapp.network import SEMambaPP

    cfg = _cfg(*WIDTH)
    return (
        cfg,
        SEMambaPP(_flat(cfg, scan)),
        MultiScaleSubbandCQTDiscriminator(),
        MultiResolutionDiscriminator(),
        MultiScaleMelSpectrogramLoss(sampling_rate=16000),
    )


def _move(payload, generator, cqt, mrd, mel, device):
    generator.load_state_dict(payload["generator"], strict=True)
    cqt.load_state_dict(payload["cqt"], strict=True)
    mrd.load_state_dict(payload["mrd"], strict=True)
    for module in (generator, cqt, mrd, mel):
        module.to(device)
        module.train()
    return payload["clean"].to(device), payload["noisy"].to(device)


def prepare() -> None:
    _cfg_model, generator, cqt, mrd, _mel = _build_official()
    torch.manual_seed(23)
    clean = torch.randn(1, 8000)
    noisy = clean * 0.6 + 0.05 * torch.randn_like(clean)
    rms_x = torch.randn(100, 81, 48)
    rms_weight = torch.randn(48)
    rms_dy = torch.randn_like(rms_x)
    block_dy = torch.randn_like(rms_x)
    payload = {
        "generator": generator.state_dict(),
        "cqt": cqt.state_dict(),
        "mrd": mrd.state_dict(),
        "clean": clean,
        "noisy": noisy,
        "rms_x": rms_x,
        "rms_weight": rms_weight,
        "rms_dy": rms_dy,
        "block_dy": block_dy,
        "cfg_width": list(WIDTH),
    }
    destination = _out() / "init.pt"
    torch.save(payload, destination)
    manifest = {
        "generator": _hash_named(payload["generator"]),
        "cqt": _hash_named(payload["cqt"]),
        "mrd": _hash_named(payload["mrd"]),
        "clean": _digest(clean),
        "noisy": _digest(noisy),
        "rms_x": _digest(rms_x),
        "rms_weight": _digest(rms_weight),
        "rms_dy": _digest(rms_dy),
        "rms_x_meta": _meta(rms_x),
        "note": "Model construction consumes RNG before the waveform seed. Reload this file; do not rebuild to reproduce weights.",
    }
    (_out() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "bitwise": False,
            "max_abs": None,
            "mean_abs": None,
            "relative": None,
            "mismatch": None,
            "numel": int(left.numel()),
            "nonfinite": True,
            "shape": [_meta(left)["shape"], _meta(right)["shape"]],
        }
    bitwise = bool(torch.equal(left, right))
    delta = (left.detach().float() - right.detach().float()).abs()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    scale = float(left.detach().float().abs().max()) if left.numel() else 0.0
    return {
        "bitwise": bitwise,
        "max_abs": max_abs,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "relative": max_abs / max(scale, 1e-12),
        "mismatch": int((left != right).sum()),
        "numel": int(left.numel()),
        "nonfinite": not bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def _bucket(name: str) -> str:
    if name.startswith("dense_encoder"):
        return "dense_encoder"
    if name.startswith("TSMamba"):
        return "tsmamba"
    if name.startswith("mask_decoder"):
        return "magnitude_decoder"
    if name.startswith("phase_decoder"):
        return "phase_decoder"
    return "other"


def _grouped(left: dict, right: dict) -> dict:
    names = sorted(set(left) & set(right))
    buckets: dict[str, dict] = {}
    for name in names:
        bucket = _bucket(name)
        row = buckets.setdefault(
            bucket, {"bitwise": True, "max_abs": 0.0, "mean_abs": 0.0, "mismatch": 0, "numel": 0, "worst": "", "nonfinite": False}
        )
        stats = _compare(left[name], right[name])
        row["bitwise"] = bool(row["bitwise"] and stats["bitwise"])
        row["nonfinite"] = bool(row["nonfinite"] or stats["nonfinite"])
        row["mismatch"] += int(stats["mismatch"] or 0)
        row["numel"] += int(stats["numel"])
        row["scale"] = max(float(row.get("scale", 0.0)), float(left[name].detach().float().abs().max()))
        if stats["max_abs"] is not None and stats["max_abs"] >= row["max_abs"]:
            row["max_abs"] = stats["max_abs"]
            row["worst"] = name
        row["mean_abs"] += float(stats["mean_abs"] or 0.0) * int(stats["numel"])
    for row in buckets.values():
        row["mean_abs"] = row["mean_abs"] / row["numel"] if row["numel"] else 0.0
        row["relative"] = row["max_abs"] / max(float(row.pop("scale")), 1e-12)
    missing = sorted(set(left) ^ set(right))
    return {"buckets": buckets, "missing": missing}


def run_rms(kind: str, mode: str, trial: str, trace: bool) -> None:
    _flags()
    payload = _load()
    device = _device()
    traces = configure(mode, trace)
    weight = payload["rms_weight"].to(device)
    upstream = payload["rms_dy"].to(device)
    source = payload["rms_x"].to(device).detach()
    if kind == "official":
        from mamba_ssm.ops.triton.layer_norm import RMSNorm

        module = RMSNorm(weight.numel(), eps=1e-5).to(device)
        module.weight.data.copy_(weight)

        def apply(value):
            return module(value)
    elif kind == "audiodb":
        from src.models.semambapp.network import RMSNorm

        module = RMSNorm(int(weight.numel()), eps=1e-5).to(device)
        module.weight.data.copy_(weight)

        def apply(value):
            return module(value, fused=True)
    else:
        raise SystemExit(kind)
    module.train()
    left = source.detach().requires_grad_(True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = apply(left)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    torch.cuda.synchronize()
    started = time.perf_counter()
    output.backward(upstream)
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    saved_output = output.detach().clone()
    saved_input_grad = left.grad.detach().clone()
    saved_weight_grad = module.weight.grad.detach().clone()
    module.zero_grad(set_to_none=True)
    again = source.detach().requires_grad_(True)
    repeated = apply(again)
    repeated.backward(upstream)
    record = {
        "kind": kind,
        "mode": mode,
        "trial": trial,
        "trace": trace,
        "configured": _configured(),
        "selected": _selected_warps(),
        "traces": list(traces),
        "warp_ok": _warp_ok(mode, traces),
        "widths": sorted({item["width"] for item in traces}),
        "repeat_bitwise": bool(
            torch.equal(saved_output, repeated.detach())
            and torch.equal(saved_input_grad, again.grad.detach())
            and torch.equal(saved_weight_grad, module.weight.grad.detach())
        ),
        "repeat_output": _compare(saved_output, repeated.detach()),
        "repeat_input_grad": _compare(saved_input_grad, again.grad.detach()),
        "y": saved_output.cpu(),
        "input_grad": saved_input_grad.cpu(),
        "weight_grad": saved_weight_grad.cpu(),
        "meta": _meta(output),
        "input_meta": _meta(source),
        "hashes": {"x": _digest(source), "weight": _digest(module.weight.detach()), "dy": _digest(upstream)},
        "seconds": {"forward": forward_seconds, "backward": backward_seconds},
        "versions": _versions(),
        "rng": _rng(),
    }
    # The weight-grad comparison above is a tensor with itself. Recompute against a fresh module.
    if trace:
        configure(mode, False, reset_cache=False)
        if kind == "official":
            from mamba_ssm.ops.triton.layer_norm import RMSNorm

            quiet = RMSNorm(weight.numel(), eps=1e-5).to(device)
            quiet.weight.data.copy_(weight)

            def apply_quiet(value):
                return quiet(value)
        else:
            from src.models.semambapp.network import RMSNorm

            quiet = RMSNorm(int(weight.numel()), eps=1e-5).to(device)
            quiet.weight.data.copy_(weight)

            def apply_quiet(value):
                return quiet(value, fused=True)
        probe = source.detach().requires_grad_(True)
        quiet_out = apply_quiet(probe)
        quiet_out.backward(upstream)
        record["trace_matches_quiet"] = {
            "output": _compare(saved_output, quiet_out.detach()),
            "input_grad": _compare(saved_input_grad, probe.grad.detach()),
            "weight_grad": _compare(saved_weight_grad, quiet.weight.grad.detach()),
        }
    if not record["warp_ok"]:
        raise SystemExit(f"warp control was not observed: {traces[:4]}")
    suffix = "" if trace else "_notrace"
    destination = _out() / f"rms_{kind}_{mode}_{trial}{suffix}.pt"
    torch.save(record, destination)
    print(json.dumps({
        "file": str(destination),
        "warp_ok": record["warp_ok"],
        "configured": record["configured"],
        "observed": sorted({item["num_warps"] for item in traces}),
        "widths": record["widths"],
        "repeat_bitwise": record["repeat_bitwise"],
        "trace_matches_quiet": record.get("trace_matches_quiet"),
        "seconds": record["seconds"],
    }))


def _spectrum(kind: str, waveform: torch.Tensor, addeps: bool):
    if kind == "official":
        return mag_phase_stft(waveform, 400, 100, 400, 0.3, addeps=addeps)
    return semamba_mag_phase_stft(waveform, 400, 100, 400, 0.3, addeps=addeps)


def _waveform(kind: str, magnitude: torch.Tensor, phase: torch.Tensor, length: int) -> torch.Tensor:
    if kind == "official":
        audio = mag_phase_istft(magnitude, phase, 400, 100, 400, 0.3)
        return audio[..., :length]
    return semamba_mag_phase_istft(magnitude, phase, 400, 100, 400, 0.3, length=length)


def _parts(generator, clean, noisy, stft_kind: str) -> dict:
    official_generator = generator.__class__.__name__ == "SEMambapp"
    clean_mag, clean_phase, clean_complex = _spectrum(stft_kind, clean, False)
    noisy_mag, noisy_phase, _unused = _spectrum(stft_kind, noisy, False)
    generated = generator(noisy_mag, noisy_phase)
    if official_generator:
        magnitude, phase, complex_spectrum = generated
    else:
        magnitude, phase = generated
        complex_spectrum = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    audio = _waveform(stft_kind, magnitude, phase, int(clean.shape[-1]))
    _unused_mag, _unused_phase, recomputed = _spectrum(stft_kind, audio, True)
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


def _audio_losses(parts, clean, cqt, mrd, order: str) -> dict:
    # The Trainer passes the pair together, interleaving real/fake forwards.
    real_q, fake_q, real_qf, fake_qf = cqt(clean, parts["audio"])
    real_r, fake_r, real_rf, fake_rf = mrd(clean, parts["audio"])
    if order == "flat":
        discriminator = sum(
            least_squares_discriminator(real, fake)
            for real, fake in zip(real_q + real_r, fake_q + fake_r, strict=True)
        )
        adversarial = sum(least_squares_generator(score) for score in fake_q + fake_r)
        matching = feature_matching(real_qf + real_rf, fake_qf + fake_rf, scale=2.0, reduction="sum")
    elif order == "family":
        discriminator = sum(
            least_squares_discriminator(real, fake) for real, fake in zip(real_q, fake_q, strict=True)
        ) + sum(least_squares_discriminator(real, fake) for real, fake in zip(real_r, fake_r, strict=True))
        adversarial = sum(least_squares_generator(score) for score in fake_q) + sum(
            least_squares_generator(score) for score in fake_r
        )
        matching = feature_matching(real_qf, fake_qf, scale=2.0, reduction="sum") + feature_matching(
            real_rf, fake_rf, scale=2.0, reduction="sum"
        )
    else:
        raise SystemExit(order)
    spectral = compressed_complex_losses(
        parts["clean_mag"],
        parts["mag"],
        parts["clean_phase"],
        parts["phase"],
        parts["clean_complex"],
        parts["complex"],
        parts["recomputed"],
        phase_mode="matrix",
    )
    mel_loss = summed_slaney_log_mel(clean, parts["audio"], 16000)
    generator = (
        adversarial
        + matching
        + mel_loss * 0.1
        + spectral["magnitude"] * 0.9
        + spectral["phase"] * 0.3
        + spectral["complex"] * 0.1
        + spectral["consistency"] * 0.1
    )
    return {
        "discriminator": discriminator,
        "generator": generator,
        "adversarial": adversarial,
        "matching": matching,
        "mel": mel_loss,
        **spectral,
    }


def _cpu_grads(module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _attach(generator) -> dict:
    captured: dict[str, torch.Tensor] = {}

    def save(name):
        def hook(_module, inputs, output):
            captured[name] = output.detach()
            if name == "dense_in" and inputs and torch.is_tensor(inputs[0]) and inputs[0].requires_grad:
                inputs[0].retain_grad()
                captured["dense_in"] = inputs[0]
        return hook

    # audio-db calls the encoder submodules directly, so the encoder module
    # itself does not run. dense_conv_2 is the encoder output on both paths.
    handles = [
        generator.dense_encoder.dense_conv_1.register_forward_hook(save("dense_in")),
        generator.dense_encoder.dense_conv_2.register_forward_hook(save("dense")),
    ]
    for index, block in enumerate(generator.TSMamba):
        handles.append(block.register_forward_hook(save(f"block{index}")))
    return {"captured": captured, "handles": handles}


def _losses_for(kind: str, order: str, parts, clean, cqt, mrd, mel, cfg):
    if kind == "official":
        return _official_losses(parts, clean, cqt, mrd, mel, cfg)
    return _audio_losses(parts, clean, cqt, mrd, order)


def _case(label, generator_kind, stft_kind, loss_kind, order, payload, device, mode):
    if generator_kind == "official":
        cfg, generator, _cqt, _mrd, _mel = _build_official()
    else:
        cfg, generator, _cqt, _mrd, _mel = _build_audio()
    if loss_kind == "official":
        _cfg_loss, _generator, cqt, mrd, mel = _build_official()
    else:
        _cfg_loss, _generator, cqt, mrd, mel = _build_audio()
    clean, noisy = _move(payload, generator, cqt, mrd, mel, device)
    attachment = _attach(generator)
    parts = _parts(generator, clean, noisy, stft_kind)
    losses = _losses_for(loss_kind, order, parts, clean, cqt, mrd, mel, cfg)
    for module in (generator, cqt, mrd):
        module.zero_grad(set_to_none=True)
    losses["generator"].backward()
    for handle in attachment["handles"]:
        handle.remove()
    captured = attachment["captured"]
    dense_in = captured.pop("dense_in")
    stages = {
        "dense": captured["dense"].cpu(),
        "mag": parts["mag"].detach().cpu(),
        "phase": parts["phase"].detach().cpu(),
        "audio": parts["audio"].detach().cpu(),
    }
    for index in range(len(generator.TSMamba)):
        stages[f"block{index}"] = captured[f"block{index}"].cpu()
    return {
        "label": label,
        "stages": stages,
        "grads": _cpu_grads(generator),
        "dense_in_grad": None if dense_in.grad is None else dense_in.grad.detach().cpu(),
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "weight_hash": _hash_named({name: value.detach().cpu() for name, value in generator.state_dict().items()}),
    }


def _stage_report(left: dict, right: dict) -> dict:
    report = {}
    for name in left["stages"]:
        report[name] = _compare(left["stages"][name], right["stages"][name])
    report["grads"] = _grouped(left["grads"], right["grads"])
    if left["dense_in_grad"] is not None and right["dense_in_grad"] is not None:
        report["dense_in_grad"] = _compare(left["dense_in_grad"], right["dense_in_grad"])
    report["losses"] = {
        name: abs(left["losses"][name] - right["losses"][name]) for name in left["losses"] if name in right["losses"]
    }
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    raise TypeError(type(value))


def ablate(mode: str) -> None:
    _flags()
    traces = configure(mode, True)
    payload = _load()
    device = _device()
    specs = [
        ("official", "official", "official", "official", "family"),
        ("official_repeat", "official", "official", "official", "family"),
        ("audio", "audio", "audio", "audio", "flat"),
        ("audio_repeat", "audio", "audio", "audio", "flat"),
        ("audio_gen_official_path", "audio", "official", "official", "family"),
        ("audio_gen_official_stft", "audio", "official", "audio", "flat"),
        ("audio_family_order", "audio", "audio", "audio", "family"),
    ]
    rows = {}
    for label, generator_kind, stft_kind, loss_kind, order in specs:
        print(f"ablate {label}", flush=True)
        rows[label] = _case(label, generator_kind, stft_kind, loss_kind, order, payload, device, mode)
    pairs = {
        "official_repeat": _stage_report(rows["official"], rows["official_repeat"]),
        "audio_repeat": _stage_report(rows["audio"], rows["audio_repeat"]),
        "native": _stage_report(rows["official"], rows["audio"]),
        "generator_only": _stage_report(rows["official"], rows["audio_gen_official_path"]),
        "official_stft_audio_loss": _stage_report(rows["official"], rows["audio_gen_official_stft"]),
        "family_order_vs_official": _stage_report(rows["official"], rows["audio_family_order"]),
        "family_order_vs_audio": _stage_report(rows["audio"], rows["audio_family_order"]),
    }
    # Identical discriminator activations, two summation orders, gradient into the waveform.
    clean = payload["clean"].to(device)
    noisy = payload["noisy"].to(device)
    _cfg_model, generator, cqt, mrd, _mel = _build_audio()
    _move(payload, generator, cqt, mrd, _mel, device)
    with torch.no_grad():
        sample = _parts(generator, clean, noisy, "audio")["audio"].detach()
    sample = sample.requires_grad_(True)
    real_q, real_qf = cqt(clean)
    fake_q, fake_qf = cqt(sample)
    real_r, real_rf = mrd(clean)
    fake_r, fake_rf = mrd(sample)

    def gan(order: str) -> torch.Tensor:
        if order == "flat":
            adversarial = sum(least_squares_generator(score) for score in fake_q + fake_r)
            matching = feature_matching(real_qf + real_rf, fake_qf + fake_rf, scale=2.0, reduction="sum")
        else:
            adversarial = sum(least_squares_generator(score) for score in fake_q) + sum(
                least_squares_generator(score) for score in fake_r
            )
            matching = feature_matching(real_qf, fake_qf, scale=2.0, reduction="sum") + feature_matching(
                real_rf, fake_rf, scale=2.0, reduction="sum"
            )
        return adversarial + matching

    left = gan("flat")
    left.backward(retain_graph=True)
    flat_grad = sample.grad.detach().clone()
    sample.grad = None
    right = gan("family")
    right.backward()
    family_grad = sample.grad.detach().clone()
    order_only = {
        "loss_abs": abs(float(left.detach()) - float(right.detach())),
        "audio_grad": _compare(flat_grad, family_grad),
    }
    report = {
        "mode": mode,
        "configured": _configured(),
        "warp_ok": _warp_ok(mode, []),
        "weight_hashes": {label: rows[label]["weight_hash"] for label in rows},
        "init_generator": json.loads((_out() / "manifest.json").read_text(encoding="utf-8"))["generator"],
        "pairs": pairs,
        "order_only": order_only,
        "observed": sorted({item["num_warps"] for item in traces}),
        "widths": sorted({item["width"] for item in traces if item["width"] is not None}),
        "versions": _versions(),
    }
    report["warp_ok"] = _warp_ok(mode, traces)
    destination = _out() / f"ablate_{mode}.json"
    destination.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(destination)


def _pad_pair(waveform: torch.Tensor, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    left = waveform[..., 1 : pad + 1].flip(-1)
    right = waveform[..., -pad - 1 : -1].flip(-1)
    indexed = torch.cat((left, waveform, right), dim=-1)
    native = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")
    return indexed, native


def run_stft() -> None:
    _flags()
    payload = _load()
    device = _device()
    waveform = payload["noisy"].to(device)

    def backward(kind: str, addeps: bool):
        value = waveform.detach().requires_grad_(True)
        magnitude, phase, compressed = _spectrum(kind, value, addeps)
        loss = magnitude.square().mean() + phase.square().mean() + compressed.square().mean()
        loss.backward()
        return magnitude.detach(), phase.detach(), value.grad.detach()

    official = [backward("official", False) for _index in range(5)]
    audio = [backward("audio", False) for _index in range(5)]
    official_eps = [backward("official", True) for _index in range(3)]
    audio_eps = [backward("audio", True) for _index in range(3)]
    pad = 200
    indexed_grads = []
    native_grads = []
    indexed_forward = None
    native_forward = None
    for _index in range(5):
        value = waveform.detach().requires_grad_(True)
        indexed, native = _pad_pair(value, pad)
        (indexed.square().mean()).backward()
        indexed_grads.append(value.grad.detach().clone())
        if indexed_forward is None:
            indexed_forward = indexed.detach()
        value = waveform.detach().requires_grad_(True)
        indexed, native = _pad_pair(value, pad)
        (native.square().mean()).backward()
        native_grads.append(value.grad.detach().clone())
        if native_forward is None:
            native_forward = native.detach()

    def spread(rows):
        pairs = [_compare(rows[0], rows[index]) for index in range(1, len(rows))]
        return {"max_abs": max(item["max_abs"] for item in pairs), "bitwise": all(item["bitwise"] for item in pairs)}

    # iSTFT backward from a shared spectrum.
    magnitude, phase, _compressed = _spectrum("audio", waveform.detach(), False)
    magnitude = magnitude.detach()
    phase = phase.detach()

    def istft_grad(kind: str):
        mag = magnitude.detach().requires_grad_(True)
        pha = phase.detach().requires_grad_(True)
        audio = _waveform(kind, mag, pha, int(waveform.shape[-1]))
        audio.square().mean().backward()
        return audio.detach(), mag.grad.detach(), pha.grad.detach()

    official_istft = [istft_grad("official") for _index in range(3)]
    audio_istft = [istft_grad("audio") for _index in range(3)]
    report = {
        "forward_mag": _compare(official[0][0], audio[0][0]),
        "forward_phase": _compare(official[0][1], audio[0][1]),
        "official_grad_spread": spread([item[2] for item in official]),
        "audio_grad_spread": spread([item[2] for item in audio]),
        "official_vs_audio_grad": _compare(official[0][2], audio[0][2]),
        "addeps_forward_mag": _compare(official_eps[0][0], audio_eps[0][0]),
        "addeps_grad": _compare(official_eps[0][2], audio_eps[0][2]),
        "addeps_official_spread": spread([item[2] for item in official_eps]),
        "reflect_forward": _compare(indexed_forward, native_forward),
        "reflect_indexed_spread": spread(indexed_grads),
        "reflect_native_spread": spread(native_grads),
        "reflect_indexed_vs_native": _compare(indexed_grads[0], native_grads[0]),
        "istft_audio": _compare(official_istft[0][0], audio_istft[0][0]),
        "istft_mag_grad": _compare(official_istft[0][1], audio_istft[0][1]),
        "istft_phase_grad": _compare(official_istft[0][2], audio_istft[0][2]),
        "istft_official_audio_spread": spread([item[0] for item in official_istft]),
        "versions": _versions(),
    }
    destination = _out() / "stft.json"
    destination.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(destination)


def _copy_mixer(official_block, audio_block) -> None:
    audio_block.norm.weight.data.copy_(official_block.norm.weight.detach())
    pairs = (
        ("in_proj", "weight"),
        ("conv1d", "weight"),
        ("conv1d", "bias"),
        ("x_proj", "weight"),
        ("dt_proj", "weight"),
        ("dt_proj", "bias"),
        ("out_proj", "weight"),
    )
    for module_name, field in pairs:
        source = getattr(getattr(official_block.mixer, module_name), field)
        target = getattr(getattr(audio_block.mixer, module_name), field)
        if source.shape != target.shape:
            raise SystemExit(f"shape mismatch {module_name}.{field}: {tuple(source.shape)} vs {tuple(target.shape)}")
        target.data.copy_(source.detach())
    audio_block.mixer.A_log.data.copy_(official_block.mixer.A_log.detach())
    audio_block.mixer.D.data.copy_(official_block.mixer.D.detach())


def run_mamba(mode: str) -> None:
    _flags()
    traces = configure(mode, True)
    from model.semambapp import create_block
    from src.models.semambapp.network import MambaResidual
    import mamba_ssm.modules.mamba_simple as mamba_simple
    import mamba_ssm.ops.selective_scan_interface as interface

    payload = _load()
    device = _device()
    cfg = _cfg(*WIDTH)
    official = create_block(48, cfg).to(device).train()
    audio = MambaResidual(48, 16, 4, 2, 1e-5, scan_backend="fused").to(device).train()
    _copy_mixer(official, audio)
    hidden = payload["rms_x"].to(device)
    upstream = payload["block_dy"].to(device)
    captured: dict[str, dict] = {}
    original = interface.mamba_inner_fn

    def wrapped(tag):
        def inner(*args, **kwargs):
            xz = args[0]
            if tag in captured:
                return original(*args, **kwargs)
            captured[tag] = {
                "shape": [int(item) for item in xz.shape],
                "stride": [int(item) for item in xz.stride()],
                "dtype": str(xz.dtype),
                "xz": xz.detach(),
                "conv": args[1].detach(),
                "dt": kwargs["delta_bias"].detach(),
            }
            return original(*args, **kwargs)
        return inner

    current = {"tag": "official"}

    def inner(*args, **kwargs):
        return wrapped(current["tag"])(*args, **kwargs)

    mamba_simple.mamba_inner_fn = inner
    interface.mamba_inner_fn = inner
    official_hidden = hidden.detach().requires_grad_(True)
    official_out, official_residual = official(official_hidden, None)
    official_out.backward(upstream)
    official_grads = {
        "norm": official.norm.weight.grad.detach().clone(),
        "in_proj": official.mixer.in_proj.weight.grad.detach().clone(),
        "conv": official.mixer.conv1d.weight.grad.detach().clone(),
        "conv_bias": official.mixer.conv1d.bias.grad.detach().clone(),
        "x_proj": official.mixer.x_proj.weight.grad.detach().clone(),
        "dt_proj": official.mixer.dt_proj.weight.grad.detach().clone(),
        "dt_bias": official.mixer.dt_proj.bias.grad.detach().clone(),
        "A_log": official.mixer.A_log.grad.detach().clone(),
        "D": official.mixer.D.grad.detach().clone(),
        "out_proj": official.mixer.out_proj.weight.grad.detach().clone(),
    }
    official_input_grad = official_hidden.grad.detach().clone() if official_hidden.grad is not None else None
    current["tag"] = "audio"
    audio_hidden = hidden.detach().requires_grad_(True)
    audio_out, audio_residual = audio(audio_hidden, None)
    audio_out.backward(upstream)
    audio_grads = {
        "norm": audio.norm.weight.grad.detach().clone(),
        "in_proj": audio.mixer.in_proj.weight.grad.detach().clone(),
        "conv": audio.mixer.conv1d.weight.grad.detach().clone(),
        "conv_bias": audio.mixer.conv1d.bias.grad.detach().clone(),
        "x_proj": audio.mixer.x_proj.weight.grad.detach().clone(),
        "dt_proj": audio.mixer.dt_proj.weight.grad.detach().clone(),
        "dt_bias": audio.mixer.dt_proj.bias.grad.detach().clone(),
        "A_log": audio.mixer.A_log.grad.detach().clone(),
        "D": audio.mixer.D.grad.detach().clone(),
        "out_proj": audio.mixer.out_proj.weight.grad.detach().clone(),
    }
    interface.mamba_inner_fn = original
    mamba_simple.mamba_inner_fn = original
    grad_report = {name: _compare(official_grads[name], audio_grads[name]) for name in official_grads}
    report = {
        "mode": mode,
        "configured": _configured(),
        "warp_ok": _warp_ok(mode, traces),
        "observed": sorted({item["num_warps"] for item in traces}),
        "widths": sorted({item["width"] for item in traces}),
        "output": _compare(official_out.detach(), audio_out.detach()),
        "residual": _compare(official_residual.detach(), audio_residual.detach()),
        "xz": _compare(captured["official"]["xz"], captured["audio"]["xz"]),
        "xz_meta": {"official": {key: captured["official"][key] for key in ("shape", "stride", "dtype")},
                    "audio": {key: captured["audio"][key] for key in ("shape", "stride", "dtype")}},
        "conv_weight": _compare(captured["official"]["conv"], captured["audio"]["conv"]),
        "grads": grad_report,
        "input_grad": None if official_input_grad is None else _compare(official_input_grad, audio_hidden.grad.detach()),
        "versions": _versions(),
    }
    destination = _out() / f"mamba_{mode}.json"
    destination.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    if not report["warp_ok"]:
        raise SystemExit("mamba warp control was not observed")
    print(destination)


def _optimizers(generator, cqt, mrd):
    optim_g = torch.optim.AdamW(generator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    optim_d = torch.optim.AdamW(list(cqt.parameters()) + list(mrd.parameters()), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=GAMMA)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=GAMMA)
    return optim_g, optim_d, scheduler_g, scheduler_d


def _optim_state(optim, named) -> dict:
    rows = {}
    for name, parameter in named:
        value = optim.state.get(parameter)
        if not value:
            rows[name] = None
            continue
        rows[name] = {
            field: item.detach().cpu() if torch.is_tensor(item) else item
            for field, item in value.items()
        }
    return rows


def run_step(kind: str, mode: str, trial: str) -> None:
    _flags()
    traces = configure(mode, True)
    payload = _load()
    device = _device()
    if kind == "official":
        cfg, generator, cqt, mrd, mel = _build_official()
        loss_kind = "official"
    elif kind == "audiodb":
        cfg, generator, cqt, mrd, mel = _build_audio("fused")
        loss_kind = "audio"
    else:
        raise SystemExit(kind)
    clean, noisy = _move(payload, generator, cqt, mrd, mel, device)
    noisy = noisy.detach().requires_grad_(True)
    loaded_hash = _hash_named({name: tensor.detach().cpu() for name, tensor in generator.state_dict().items()})
    init_hash = json.loads((_out() / "manifest.json").read_text(encoding="utf-8"))["generator"]
    optim_g, optim_d, scheduler_g, scheduler_d = _optimizers(generator, cqt, mrd)
    before = {name: tensor.detach().cpu().clone() for name, tensor in generator.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    attachment = _attach(generator)
    parts = _parts(generator, clean, noisy, "official" if kind == "official" else "audio")
    detached = {**parts, "audio": parts["audio"].detach()}
    for optim in (optim_d, optim_g):
        optim.zero_grad(set_to_none=True)
    disc_pack = _losses_for(loss_kind, "family", detached, clean, cqt, mrd, mel, cfg)
    disc_pack["discriminator"].backward()
    cqt_grad = _cpu_grads(cqt)
    mrd_grad = _cpu_grads(mrd)
    optim_d.step()
    optim_g.zero_grad(set_to_none=True)
    noisy.grad = None
    for handle in attachment["handles"]:
        handle.remove()
    attachment = _attach(generator)
    torch.cuda.synchronize()
    started = time.perf_counter()
    again = _parts(generator, clean, noisy, "official" if kind == "official" else "audio")
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    gen_pack = _losses_for(loss_kind, "family", again, clean, cqt, mrd, mel, cfg)
    torch.cuda.synchronize()
    started = time.perf_counter()
    gen_pack["generator"].backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    generator_grad = _cpu_grads(generator)
    optim_g.step()
    torch.cuda.synchronize()
    for handle in attachment["handles"]:
        handle.remove()
    captured = attachment["captured"]
    dense_in = captured.pop("dense_in", None)
    if not _warp_ok(mode, traces):
        raise SystemExit(f"step warp control was not observed for {kind} {mode}")
    record = {
        "kind": kind,
        "mode": mode,
        "trial": trial,
        "configured": _configured(),
        "traces": traces,
        "warp_ok": True,
        "observed": sorted({item["num_warps"] for item in traces}),
        "widths": sorted({item["width"] for item in traces}),
        "weight_hash_matches_init": loaded_hash == init_hash,
        "loaded_hash": loaded_hash,
        "init_hash": init_hash,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
        "audio": again["audio"].detach().float().cpu(),
        "mag": again["mag"].detach().cpu(),
        "phase": again["phase"].detach().cpu(),
        "dense": captured["dense"].cpu(),
        "blocks": [captured[f"block{index}"].cpu() for index in range(len(generator.TSMamba))],
        "losses": {name: float(value.detach()) for name, value in gen_pack.items()},
        "discriminator_loss": float(disc_pack["discriminator"].detach()),
        "generator_grad": generator_grad,
        "cqt_grad": cqt_grad,
        "mrd_grad": mrd_grad,
        "input_grad": None if noisy.grad is None else noisy.grad.detach().cpu(),
        "dense_in_grad": None if dense_in is None or dense_in.grad is None else dense_in.grad.detach().cpu(),
        "before": before,
        "after": {name: tensor.detach().cpu() for name, tensor in generator.state_dict().items()},
        "after_cqt": {name: tensor.detach().cpu() for name, tensor in cqt.state_dict().items()},
        "after_mrd": {name: tensor.detach().cpu() for name, tensor in mrd.state_dict().items()},
        "optim_g": _optim_state(optim_g, list(generator.named_parameters())),
        "optim_d": _optim_state(
            optim_d,
            [(f"cqt.{name}", value) for name, value in cqt.named_parameters()]
            + [(f"mrd.{name}", value) for name, value in mrd.named_parameters()],
        ),
        "scheduler_equal_init": scheduler_g.last_epoch == 0 and scheduler_d.last_epoch == 0,
        "meta": {"audio": _meta(again["audio"]), "mag": _meta(again["mag"]), "phase": _meta(again["phase"])},
        "versions": _versions(),
        "rng": _rng(),
        "finite": bool(torch.isfinite(again["audio"]).all()),
    }
    destination = _out() / f"step_{kind}_{mode}_{trial}.pt"
    torch.save(record, destination)
    print(json.dumps({
        "file": str(destination),
        "observed": record["observed"],
        "widths": record["widths"],
        "weight_hash_matches_init": record["weight_hash_matches_init"],
        "losses": record["losses"],
        "seconds": {"forward": forward_seconds, "backward": backward_seconds},
        "peak_allocated": record["peak_allocated"],
    }))


def microbench() -> None:
    _flags()
    payload = _load()
    device = _device()
    from mamba_ssm.ops.triton.layer_norm import RMSNorm

    rows = []
    source = payload["rms_x"].to(device)
    upstream = payload["rms_dy"].to(device)
    weight = payload["rms_weight"].to(device)
    for mode in ("autotune", "1", "2", "4", "8", "16"):
        configure(mode, False)
        module = RMSNorm(weight.numel(), eps=1e-5).to(device)
        module.weight.data.copy_(weight)
        value = source.detach().requires_grad_(True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = module(value)
        output.backward(upstream)
        torch.cuda.synchronize()
        first = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _index in range(30):
            value = source.detach().requires_grad_(True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = module(value)
            output.backward(upstream)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
        ordered = sorted(samples)
        rows.append(
            {
                "mode": mode,
                "configured": _configured(),
                "first_seconds": first,
                "median_seconds": ordered[len(ordered) // 2],
                "peak_allocated": int(torch.cuda.max_memory_allocated()),
            }
        )
        del module
        torch.cuda.empty_cache()
    destination = _out() / "microbench.json"
    destination.write_text(json.dumps({"rows": rows, "versions": _versions()}, indent=2), encoding="utf-8")
    print(destination)


def _optim_gap(left: dict, right: dict) -> dict:
    worst = 0.0
    bitwise = set(left) == set(right)
    mismatch = 0
    for key in sorted(set(left) & set(right)):
        value = left[key]
        other = right[key]
        if value is None or other is None:
            bitwise = bitwise and value is None and other is None
            continue
        for field, item in value.items():
            if not torch.is_tensor(item):
                bitwise = bitwise and item == other[field]
                continue
            stats = _compare(item, other[field])
            worst = max(worst, float(stats["max_abs"]))
            bitwise = bitwise and stats["bitwise"]
            mismatch += int(stats["mismatch"] or 0)
    return {"max_abs": worst, "bitwise": bitwise, "mismatch": mismatch, "missing": sorted(set(left) ^ set(right))}


def _pair_steps(left: dict, right: dict) -> dict:
    report = {
        "audio": _compare(left["audio"], right["audio"]),
        "mag": _compare(left["mag"], right["mag"]),
        "phase": _compare(left["phase"], right["phase"]),
        "dense": _compare(left["dense"], right["dense"]),
        "blocks": [_compare(a, b) for a, b in zip(left["blocks"], right["blocks"], strict=True)],
        "losses": {name: abs(left["losses"][name] - right["losses"][name]) for name in left["losses"]},
        "discriminator_loss": abs(left["discriminator_loss"] - right["discriminator_loss"]),
        "generator_grad": _grouped(left["generator_grad"], right["generator_grad"]),
        "cqt_grad": _grouped(left["cqt_grad"], right["cqt_grad"]),
        "mrd_grad": _grouped(left["mrd_grad"], right["mrd_grad"]),
        "after": _grouped(left["after"], right["after"]),
        "after_cqt": _grouped(left["after_cqt"], right["after_cqt"]) if "after_cqt" in left and "after_cqt" in right else None,
        "after_mrd": _grouped(left["after_mrd"], right["after_mrd"]) if "after_mrd" in left and "after_mrd" in right else None,
        "optim_g": _optim_gap(left["optim_g"], right["optim_g"]),
        "optim_d": _optim_gap(left["optim_d"], right["optim_d"]),
    }
    if left["input_grad"] is not None and right["input_grad"] is not None:
        report["input_grad"] = _compare(left["input_grad"], right["input_grad"])
    if left["dense_in_grad"] is not None and right["dense_in_grad"] is not None:
        report["dense_in_grad"] = _compare(left["dense_in_grad"], right["dense_in_grad"])
    return report


def _load_step(kind: str, mode: str, trial: int) -> dict:
    return torch.load(_out() / f"step_{kind}_{mode}_{trial}.pt", map_location="cpu", weights_only=False)


def _envelope(pairs: list[dict], key_path) -> dict:
    values = []
    for pair in pairs:
        cursor = pair
        for key in key_path:
            cursor = cursor[key]
        values.append(float(cursor))
    return {"max": max(values), "min": min(values), "mean": sum(values) / len(values)}


def summarize() -> None:
    rms_rows = []
    for path in sorted(_out().glob("rms_*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        rms_rows.append(
            {
                "file": path.name,
                "kind": record["kind"],
                "mode": record["mode"],
                "trial": record["trial"],
                "trace": record["trace"],
                "warp_ok": record["warp_ok"],
                "configured": record["configured"],
                "observed": sorted({item["num_warps"] for item in record["traces"]})
                or sorted({value for value in record.get("selected", {}).values() if value is not None}),
                "widths": record["widths"],
                "repeat_bitwise": record["repeat_bitwise"],
                "trace_matches_quiet": record.get("trace_matches_quiet"),
                "meta": record["meta"],
                "hashes": record["hashes"],
            }
        )
    comparisons = []
    files = {(row["kind"], row["mode"], str(row["trial"]), row["trace"]): row for row in []}
    loaded = {}
    for path in sorted(_out().glob("rms_*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        loaded[path.name] = record
    names = sorted(loaded)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = loaded[left_name]
            right = loaded[right_name]
            if bool(left["trace"]) != bool(right["trace"]):
                same = left["kind"] == right["kind"] and left["mode"] == right["mode"] and str(left["trial"]) == str(right["trial"])
                if not same:
                    continue
            elif not left["trace"] or not right["trace"]:
                continue
            comparisons.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "output": _compare(left["y"], right["y"]),
                    "input_grad": _compare(left["input_grad"], right["input_grad"]),
                    "weight_grad": _compare(left["weight_grad"], right["weight_grad"]),
                }
            )
    step_pairs = []
    for mode in MODES:
        official = []
        audio = []
        for trial in range(3):
            official_path = _out() / f"step_official_{mode}_{trial}.pt"
            audio_path = _out() / f"step_audiodb_{mode}_{trial}.pt"
            if official_path.exists():
                official.append(torch.load(official_path, map_location="cpu", weights_only=False))
            if audio_path.exists():
                audio.append(torch.load(audio_path, map_location="cpu", weights_only=False))
        for left_index in range(len(official)):
            for right_index in range(left_index + 1, len(official)):
                step_pairs.append({"relation": "official_repeat", "mode": mode, "left": left_index, "right": right_index, **_pair_steps(official[left_index], official[right_index])})
        for left_index in range(len(audio)):
            for right_index in range(left_index + 1, len(audio)):
                step_pairs.append({"relation": "audiodb_repeat", "mode": mode, "left": left_index, "right": right_index, **_pair_steps(audio[left_index], audio[right_index])})
        for left in audio:
            for right in official:
                step_pairs.append({"relation": "audiodb_vs_official", "mode": mode, "left": left["trial"], "right": right["trial"], **_pair_steps(left, right)})
    cross = []
    if (_out() / "step_official_1_0.pt").exists() and (_out() / "step_official_2_0.pt").exists():
        cross.append({"relation": "official_warp1_vs_warp2", **_pair_steps(_load_step("official", "1", 0), _load_step("official", "2", 0))})
    if (_out() / "step_audiodb_1_0.pt").exists() and (_out() / "step_audiodb_2_0.pt").exists():
        cross.append({"relation": "audiodb_warp1_vs_warp2", **_pair_steps(_load_step("audiodb", "1", 0), _load_step("audiodb", "2", 0))})
    report = {
        "rms": rms_rows,
        "rms_pairs": comparisons,
        "step_pairs": step_pairs,
        "cross_warp": cross,
    }
    destination = _out() / "summary.json"
    destination.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(destination)
    del files


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if command == "prepare":
        prepare()
    elif command == "rms":
        _kind, mode, trial = sys.argv[2], sys.argv[3], sys.argv[4]
        trace = sys.argv[5] != "notrace" if len(sys.argv) > 5 else True
        run_rms(_kind, mode, trial, trace)
    elif command == "step":
        run_step(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == "ablate":
        ablate(sys.argv[2])
    elif command == "stft":
        run_stft()
    elif command == "mamba":
        run_mamba(sys.argv[2])
    elif command == "microbench":
        microbench()
    elif command == "summarize":
        summarize()
    else:
        raise SystemExit(command)


if __name__ == "__main__":
    main()
