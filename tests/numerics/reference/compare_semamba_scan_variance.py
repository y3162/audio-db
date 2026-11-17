"""Official-versus-official spread, then audio-db fused and reference, at official width.

Each trial is a separate process. Initial weights and the waveform are loaded
from one CPU file. One logical step updates the discriminator, then the
generator. ExponentialLR is constructed and not stepped: the official schedule
moves at an epoch boundary.

Outputs stay under ``AUDIO_DB_NUMERICS_OUT`` (default ``/tmp/audio-db-scan-variance``).
This script does not write ``data/``.

    CUDA_VISIBLE_DEVICES=<free> .venvs/se/bin/python \\
        tests/numerics/reference/compare_semamba_scan_variance.py all
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

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
    _gap,
    _independent_losses,
    _independent_parts,
    _official_losses,
    _official_parts,
)

WIDTH = (48, 4, 16, 4)
GAMMA = 0.99
TRIALS = 3


def _out() -> Path:
    path = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-scan-variance"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _physical_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return visible or "0"


def _device_used_mib() -> int | None:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", f"--id={_physical_gpu()}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(text.strip().splitlines()[0])


def _watch_device(samples: list[int], stop: threading.Event) -> None:
    while not stop.is_set():
        used = _device_used_mib()
        if used is not None:
            samples.append(used)
        stop.wait(0.05)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    return torch.device("cuda:0")


def _versions() -> dict:
    official = ROOT / "src/official/semambapp"
    commit = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    info = {
        "official_semambapp": commit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        import mamba_ssm

        info["mamba_ssm"] = getattr(mamba_ssm, "__version__", "unknown")
    except ImportError:
        info["mamba_ssm"] = None
    try:
        import causal_conv1d

        info["causal_conv1d"] = getattr(causal_conv1d, "__version__", "unknown")
    except ImportError:
        info["causal_conv1d"] = None
    native = ROOT / "build/native-src/mamba"
    if (native / ".git").exists() or (native / ".git").is_file():
        info["mamba_source"] = subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
    causal = ROOT / "build/native-src/causal-conv1d"
    if (causal / ".git").exists() or (causal / ".git").is_file():
        info["causal_conv1d_source"] = subprocess.check_output(
            ["git", "-C", str(causal), "rev-parse", "HEAD"], text=True
        ).strip()
    return info


def _build_official():
    from model.discriminator import MultiResolutionDiscriminator as OfficialMRD
    from model.discriminator import MultiScaleSubbandCQTDiscriminator as OfficialCQT
    from model.loss import MultiScaleMelSpectrogramLoss
    from model.semambapp import SEMambapp

    cfg = _cfg(*WIDTH)
    return cfg, SEMambapp(cfg), OfficialCQT(), OfficialMRD(), MultiScaleMelSpectrogramLoss(sampling_rate=16000)


def _build_audio(scan: str):
    from model.loss import MultiScaleMelSpectrogramLoss
    from src.models.semambapp.discriminator import MultiResolutionDiscriminator, MultiScaleSubbandCQTDiscriminator
    from src.models.semambapp.network import SEMambaPP

    cfg = _cfg(*WIDTH)
    return cfg, SEMambaPP(_flat(cfg, scan)), MultiScaleSubbandCQTDiscriminator(), MultiResolutionDiscriminator(), MultiScaleMelSpectrogramLoss(sampling_rate=16000)


def prepare() -> None:
    cfg, generator, cqt, mrd, _mel = _build_official()
    torch.manual_seed(23)
    clean = torch.randn(1, 8000)
    noisy = clean * 0.6 + 0.05 * torch.randn_like(clean)
    payload = {
        "generator": generator.state_dict(),
        "cqt": cqt.state_dict(),
        "mrd": mrd.state_dict(),
        "clean": clean,
        "noisy": noisy,
        "cfg_width": list(WIDTH),
    }
    destination = _out() / "init.pt"
    torch.save(payload, destination)
    print(destination)


def _load_init():
    payload = torch.load(_out() / "init.pt", map_location="cpu", weights_only=True)
    return payload


def _optimizers(generator, cqt, mrd):
    optim_g = torch.optim.AdamW(generator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    optim_d = torch.optim.AdamW(
        list(cqt.parameters()) + list(mrd.parameters()), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=GAMMA)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=GAMMA)
    return optim_g, optim_d, scheduler_g, scheduler_d


def _cpu_state(module) -> dict:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _cpu_grads(module) -> dict:
    grads = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        grads[name] = parameter.grad.detach().cpu()
    return grads


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


def _step(kind: str, scan: str | None) -> dict:
    device = _device()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    payload = _load_init()
    clean = payload["clean"].to(device)
    noisy = payload["noisy"].to(device)
    fused_calls = {"n": 0}
    if kind == "official":
        cfg, generator, cqt, mrd, mel = _build_official()
        parts = lambda: _official_parts(generator, clean, noisy, cfg)  # noqa: E731
        losses = lambda tensors, disc_cqt, disc_mrd: _official_losses(tensors, clean, disc_cqt, disc_mrd, mel, cfg)  # noqa: E731
    else:
        if scan not in ("fused", "reference"):
            raise SystemExit("audio-db scan must be fused or reference")
        cfg, generator, cqt, mrd, mel = _build_audio(scan)
        if scan == "fused":
            import mamba_ssm.ops.selective_scan_interface as interface

            original = interface.mamba_inner_fn

            def _counted(*args, **kwargs):
                fused_calls["n"] += 1
                return original(*args, **kwargs)

            interface.mamba_inner_fn = _counted
        parts = lambda: _independent_parts(generator, clean, noisy)  # noqa: E731
        losses = lambda tensors, disc_cqt, disc_mrd: _independent_losses(tensors, clean, disc_cqt, disc_mrd)  # noqa: E731
    generator.load_state_dict(payload["generator"], strict=True)
    cqt.load_state_dict(payload["cqt"], strict=True)
    mrd.load_state_dict(payload["mrd"], strict=True)
    generator.to(device)
    cqt.to(device)
    mrd.to(device)
    mel.to(device)
    for module in (generator, cqt, mrd):
        module.train()
    optim_g, optim_d, scheduler_g, scheduler_d = _optimizers(generator, cqt, mrd)
    torch.cuda.reset_peak_memory_stats()
    device_samples: list[int] = []
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_device, args=(device_samples, stop), daemon=True)
    watcher.start()
    forward = parts()
    detached = {**forward, "audio": forward["audio"].detach()}
    for optim in (optim_d, optim_g):
        optim.zero_grad(set_to_none=True)
    disc = losses(detached, cqt, mrd)["discriminator"]
    disc.backward()
    disc_grad_g = _cpu_grads(cqt)
    optim_d.step()
    optim_g.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    again = parts()
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    gen_pack = losses(again, cqt, mrd)
    gen = gen_pack["generator"]
    started = time.perf_counter()
    gen.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    gen_grad = _cpu_grads(generator)
    optim_g.step()
    torch.cuda.synchronize()
    stop.set()
    watcher.join(timeout=1)
    audio = again["audio"].detach().float().cpu()
    record = {
        "kind": kind,
        "scan": scan or "official",
        "fused_calls": fused_calls["n"],
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
        "peak_device_mib": max(device_samples) if device_samples else None,
        "audio": audio,
        "losses": {
            "generator": float(gen.detach()),
            "discriminator": float(disc.detach()),
        },
        "generator": _cpu_state(generator),
        "cqt": _cpu_state(cqt),
        "mrd": _cpu_state(mrd),
        "generator_grad": gen_grad,
        "cqt_grad": disc_grad_g,
        "optim_g": _optim_state(optim_g, generator.named_parameters()),
        "optim_d": _optim_state(optim_d, list(cqt.named_parameters()) + list(mrd.named_parameters())),
        "scheduler_g": dict(scheduler_g.state_dict()),
        "scheduler_d": dict(scheduler_d.state_dict()),
    }
    for name in ("adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency"):
        record["losses"][name] = float(gen_pack[name].detach())
    return record


def _save_trial(name: str) -> None:
    kind, _, scan = name.partition(":")
    record = _step(kind, scan or None)
    destination = _out() / f"{name.replace(':', '_')}.pt"
    torch.save(record, destination)
    summary = {
        "kind": record["kind"],
        "scan": record["scan"],
        "fused_calls": record["fused_calls"],
        "losses": record["losses"],
        "forward_seconds": record["forward_seconds"],
        "backward_seconds": record["backward_seconds"],
        "peak_allocated": record["peak_allocated"],
        "peak_reserved": record["peak_reserved"],
        "peak_device_mib": record["peak_device_mib"],
    }
    print(json.dumps(summary))
    print(destination)


def _named_gap(left: dict, right: dict) -> dict:
    keys = sorted(set(left) & set(right))
    missing = sorted(set(left) ^ set(right))
    if not keys:
        return {"max_abs": None, "mean_abs": None, "p95": None, "cosine": None, "relative": None, "bitwise": False, "missing": missing}
    flat_left = torch.cat([left[key].float().reshape(-1) for key in keys])
    flat_right = torch.cat([right[key].float().reshape(-1) for key in keys])
    report = _gap(flat_left, flat_right)
    delta = (flat_left - flat_right).abs()
    report["p95"] = float(torch.quantile(delta, 0.95))
    report["relative"] = float(delta.max()) / max(float(flat_left.abs().max()), 1e-12)
    report["bitwise"] = bool(torch.equal(flat_left, flat_right)) and not missing
    report["missing"] = missing
    return report


def _worst_parameters(left: dict, right: dict, bucket: str, limit: int = 8) -> list:
    rows = []
    for key in sorted(set(left[bucket]) & set(right[bucket])):
        delta = (left[bucket][key].float() - right[bucket][key].float()).abs()
        rows.append({"name": key, "max_abs": float(delta.max()), "mean_abs": float(delta.mean())})
    rows.sort(key=lambda item: item["max_abs"], reverse=True)
    return rows[:limit]


def _pair_stats(left: dict, right: dict) -> dict:
    audio = _gap(left["audio"], right["audio"])
    audio_delta = (left["audio"].float() - right["audio"].float()).abs()
    audio["p95"] = float(torch.quantile(audio_delta, 0.95))
    audio["bitwise"] = bool(torch.equal(left["audio"], right["audio"]))
    audio["relative"] = audio["max_abs"] / max(float(left["audio"].abs().max()), 1e-12)
    losses = {
        name: abs(left["losses"][name] - right["losses"][name])
        for name in left["losses"]
    }
    weight = {bucket: _named_gap(left[bucket], right[bucket]) for bucket in ("generator", "cqt", "mrd")}
    grad = {bucket: _named_gap(left[bucket], right[bucket]) for bucket in ("generator_grad", "cqt_grad")}
    optim = {}
    for name in ("optim_g", "optim_d"):
        worst = 0.0
        bitwise = set(left[name]) == set(right[name])
        for key in sorted(set(left[name]) & set(right[name])):
            value = left[name][key]
            other = right[name][key]
            if value is None or other is None:
                bitwise = bitwise and value is None and other is None
                continue
            for field, item in value.items():
                if not torch.is_tensor(item):
                    bitwise = bitwise and item == other[field]
                    continue
                delta = (item.float() - other[field].float()).abs()
                worst = max(worst, float(delta.max()))
                bitwise = bitwise and bool(torch.equal(item, other[field]))
        optim[name] = {"max_abs": worst, "bitwise": bitwise, "missing": sorted(set(left[name]) ^ set(right[name]))}
    scheduler = {
        "g_equal": left["scheduler_g"] == right["scheduler_g"],
        "d_equal": left["scheduler_d"] == right["scheduler_d"],
    }
    return {"audio": audio, "losses": losses, "weight": weight, "grad": grad, "optim": optim, "scheduler": scheduler}


def _load_named(name: str) -> dict:
    return torch.load(_out() / f"{name}.pt", map_location="cpu", weights_only=False)


def summarize() -> None:
    official = [f"official_{index}" for index in range(TRIALS)]
    loaded = {name: _load_named(name) for name in official}
    loaded["fused"] = _load_named("audiodb_fused")
    loaded["reference"] = _load_named("audiodb_reference")
    pairs = []
    for left in range(TRIALS):
        for right in range(left + 1, TRIALS):
            pairs.append({"left": official[left], "right": official[right], **_pair_stats(loaded[official[left]], loaded[official[right]])})
    report = {
        "versions": _versions(),
        "trials": TRIALS,
        "note": "Three official processes fix the spread. That is a small sample, not a confidence interval.",
        "official_pairs": pairs,
        "fused_vs_official_0": _pair_stats(loaded["fused"], loaded[official[0]]),
        "reference_vs_official_0": _pair_stats(loaded["reference"], loaded[official[0]]),
        "worst_generator_weights": {
            "official_0_vs_1": _worst_parameters(loaded[official[0]], loaded[official[1]], "generator"),
            "fused_vs_official_0": _worst_parameters(loaded["fused"], loaded[official[0]], "generator"),
            "reference_vs_official_0": _worst_parameters(loaded["reference"], loaded[official[0]], "generator"),
        },
        "worst_generator_grads": {
            "official_0_vs_1": _worst_parameters(loaded[official[0]], loaded[official[1]], "generator_grad"),
            "fused_vs_official_0": _worst_parameters(loaded["fused"], loaded[official[0]], "generator_grad"),
            "reference_vs_official_0": _worst_parameters(loaded["reference"], loaded[official[0]], "generator_grad"),
        },
        "fused_calls": loaded["fused"]["fused_calls"],
        "reference_fused_calls": loaded["reference"]["fused_calls"],
        "memory": {
            name: {
                "peak_allocated": loaded[name]["peak_allocated"],
                "peak_reserved": loaded[name]["peak_reserved"],
                "peak_device_mib": loaded[name]["peak_device_mib"],
                "forward_seconds": loaded[name]["forward_seconds"],
                "backward_seconds": loaded[name]["backward_seconds"],
            }
            for name in ("fused", "reference", official[0])
        },
    }
    destination = _out() / "summary.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(destination)


def bench() -> None:
    """Warm one step, then time three generator updates. Official width, one batch."""

    device = _device()
    payload = _load_init()
    rows = []
    for scan in ("reference", "fused"):
        cfg, generator, cqt, mrd, mel = _build_audio(scan)
        generator.load_state_dict(payload["generator"], strict=True)
        cqt.load_state_dict(payload["cqt"], strict=True)
        mrd.load_state_dict(payload["mrd"], strict=True)
        for module in (generator, cqt, mrd, mel):
            module.to(device)
            module.train()
        clean = payload["clean"].to(device)
        noisy = payload["noisy"].to(device)
        optim_g, optim_d, _scheduler_g, _scheduler_d = _optimizers(generator, cqt, mrd)
        calls = {"n": 0}
        if scan == "fused":
            import mamba_ssm.ops.selective_scan_interface as interface

            original = interface.mamba_inner_fn

            def _counted(*args, **kwargs):
                calls["n"] += 1
                return original(*args, **kwargs)

            interface.mamba_inner_fn = _counted

        def one() -> None:
            forward = _independent_parts(generator, clean, noisy)
            detached = {**forward, "audio": forward["audio"].detach()}
            optim_d.zero_grad(set_to_none=True)
            optim_g.zero_grad(set_to_none=True)
            _independent_losses(detached, clean, cqt, mrd)["discriminator"].backward()
            optim_d.step()
            optim_g.zero_grad(set_to_none=True)
            again = _independent_parts(generator, clean, noisy)
            loss = _independent_losses(again, clean, cqt, mrd)["generator"]
            if not torch.isfinite(loss):
                raise SystemExit(f"{scan} loss is not finite")
            loss.backward()
            optim_g.step()

        torch.cuda.synchronize()
        one()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        device_samples: list[int] = []
        stop = threading.Event()
        watcher = threading.Thread(target=_watch_device, args=(device_samples, stop), daemon=True)
        watcher.start()
        samples = []
        for _index in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            one()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
        stop.set()
        watcher.join(timeout=1)
        rows.append(
            {
                "scan": scan,
                "fused_calls": calls["n"],
                "warmup_steps": 1,
                "timed_steps": 3,
                "seconds": samples,
                "median_seconds": sorted(samples)[1],
                "steps_per_second": 1.0 / sorted(samples)[1],
                "peak_allocated": int(torch.cuda.max_memory_allocated()),
                "peak_reserved": int(torch.cuda.max_memory_reserved()),
                "peak_device_mib": max(device_samples) if device_samples else None,
                "finite": True,
            }
        )
        del generator, cqt, mrd, mel
        torch.cuda.empty_cache()
    destination = _out() / "bench.json"
    destination.write_text(json.dumps({"versions": _versions(), "rows": rows}, indent=2), encoding="utf-8")
    print(destination)


def _spawn(argument: str) -> None:
    env = os.environ.copy()
    env["AUDIO_DB_NUMERICS_OUT"] = str(_out())
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run([sys.executable, __file__, argument], check=True, env=env)


def all_trials() -> None:
    prepare()
    for index in range(TRIALS):
        _spawn(f"save:official_{index}")
    _spawn("save:audiodb:fused")
    _spawn("save:audiodb:reference")
    summarize()
    _spawn("bench")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    if command == "prepare":
        prepare()
    elif command == "all":
        all_trials()
    elif command == "summarize":
        summarize()
    elif command == "bench":
        bench()
    elif command.startswith("save:"):
        name = command.split(":", 1)[1]
        if name.startswith("official_"):
            _save_trial("official:")
            (_out() / "official_.pt").rename(_out() / f"{name}.pt")
        elif name in {"audiodb:fused", "audiodb:reference"}:
            _save_trial(name)
        else:
            raise SystemExit(command)
    else:
        raise SystemExit(command)


if __name__ == "__main__":
    main()
