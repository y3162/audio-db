"""Declared-protocol measurement for SEMamba++ and MP-SENet.

The decision thresholds live in ``two_model_decision`` and are applied after
the runs. This file does not change them to fit a result. Official checkouts
and ``data/`` are not modified. Reports go to ``/tmp/audio-db-final-two-models``.

    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/semambapp/bin/python -u tests/numerics/reference/two_model_final.py semamba
    CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/se/bin/python -u tests/numerics/reference/two_model_final.py mp
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["AUDIO_DB_NUMERICS_OUT"] = "/tmp/audio-db-final-two-models"

import torch

TF32_MATMUL_DEFAULT = bool(torch.backends.cuda.matmul.allow_tf32)
TF32_CUDNN_DEFAULT = bool(torch.backends.cudnn.allow_tf32)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path("/tmp/audio-db-final-two-models")

from tests.numerics.reference.common import rng_restore, rng_snapshot  # noqa: E402
from tests.numerics.reference.two_model_decision import (  # noqa: E402
    MP_ORDER,
    MP_TRAIN,
    MP_TINY_WEIGHT_ATOL,
    SEEDS,
    SEMAMBA_ORDER,
    SEMAMBA_TRAIN,
    SGD_LRS,
    SGD_MOMENTUM,
    SGD_WEIGHT_DECAY,
    STEPS,
    TINY_GRAD,
    evaluate_campaign,
    source_inventory,
    strict_reasons,
    trainer_status,
)
from tests.numerics.reference.two_model_fairness import (  # noqa: E402
    _Retain,
    _bias_values,
    _copy_optim,
    _cpu_state,
    _grad_stats,
    _load_state,
    _null_bias_names,
    _optim_gap,
    _tensor_gap,
    environment,
)

EDGES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _git(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _flags() -> dict:
    return {
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def apply_flags(mode: str) -> dict:
    """Official train.py enables cuDNN benchmark and does not set TF32 or clip."""

    torch.backends.cuda.matmul.allow_tf32 = TF32_MATMUL_DEFAULT
    torch.backends.cudnn.allow_tf32 = TF32_CUDNN_DEFAULT
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    if mode == "tight":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif mode == "deterministic_warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif mode == "deterministic_strict":
        torch.use_deterministic_algorithms(True, warn_only=False)
    elif mode != "official_native":
        raise ValueError(mode)
    return _flags()


def _adam_defaults() -> dict:
    parameter = torch.nn.Parameter(torch.zeros(1))
    probe = torch.optim.AdamW([parameter], lr=1e-3)
    return {
        "weight_decay": float(probe.defaults["weight_decay"]),
        "eps": float(probe.defaults["eps"]),
        "amsgrad": bool(probe.defaults["amsgrad"]),
        "zero_grad_set_to_none_default": True,
    }


def protocol() -> dict:
    semamba = SEMAMBA_TRAIN.read_text(encoding="utf-8") if SEMAMBA_TRAIN.is_file() else ""
    mp = MP_TRAIN.read_text(encoding="utf-8") if MP_TRAIN.is_file() else ""
    return {
        "written_before_measurement": True,
        "audio_db_head": _git(ROOT),
        "semambapp_official": _git(ROOT / "src" / "se" / "official" / "semambapp"),
        "mp_senet_official": _git(ROOT / "src" / "se" / "official" / "mp-senet"),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "data_seed_offset": 10007,
        "semamba": {
            "width": {"hid_feature": 48, "num_tfmamba": 4, "d_state": 16, "d_conv": 4},
            "batch": 1,
            "samples": 8000,
            "scan": "fused",
            "optimizer": "AdamW",
            "lr": 2e-4,
            "betas": [0.8, 0.99],
            "weight_decay": 0.01,
            "eps": 1e-8,
            "grad_clip": None,
            "amp": None,
            "scheduler": "ExponentialLR gamma 0.99 once per finished epoch; not stepped in these 3 updates",
            "update": "discriminator on detached audio, then generator on the same forward",
            "official_lines": "src/se/official/semambapp/train.py train() waveform step, about 239-307",
            "call_label": "OFFICIAL_COMPONENT_STEP",
            "source": source_inventory(semamba, SEMAMBA_ORDER) if semamba else None,
        },
        "mp": {
            "width": {"dense_channel": 64, "num_tsconformers": 4},
            "batch": 2,
            "samples": 8000,
            "stft": "native mag_pha_stft / mag_pha_istft; index reflection is not a GO basis",
            "optimizer": "AdamW",
            "lr": 5e-4,
            "betas": [0.8, 0.99],
            "weight_decay": 0.01,
            "eps": 1e-8,
            "grad_clip": None,
            "amp": None,
            "scheduler": "ExponentialLR gamma 0.99 once per finished epoch; not stepped in these 3 updates",
            "update": "discriminator then generator, one generator forward",
            "official_lines": "src/se/official/mp-senet/train.py train() step, about 121-184",
            "call_label": "OFFICIAL_COMPONENT_STEP",
            "tiny_weight_atol": MP_TINY_WEIGHT_ATOL,
            "source": source_inventory(mp, MP_ORDER) if mp else None,
        },
        "sgd": {
            "momentum": SGD_MOMENTUM,
            "weight_decay": SGD_WEIGHT_DECAY,
            "lrs": SGD_LRS,
            "note": "SGD is a diagnostic control. A pass does not prove the official AdamW update.",
        },
        "torch_adamw_defaults": _adam_defaults(),
        "tf32_process_default": {"matmul": TF32_MATMUL_DEFAULT, "cudnn": TF32_CUDNN_DEFAULT},
        "primary_flags": "official_native",
        "strict_label_is_independent_of_compatibility": True,
    }


def _state_gap(left: torch.nn.Module, right: torch.nn.Module) -> dict:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if set(left_state) != set(right_state):
        missing = sorted(set(left_state) ^ set(right_state))
        return {"ok": False, "max_abs": None, "missing": missing[:20]}
    worst = 0.0
    for key, value in left_state.items():
        other = right_state[key]
        if tuple(value.shape) != tuple(other.shape):
            return {"ok": False, "max_abs": None, "missing": [key]}
        if value.is_floating_point() or value.is_complex():
            worst = max(worst, float((value.detach().float() - other.detach().float()).abs().max()))
        elif not torch.equal(value, other):
            return {"ok": False, "max_abs": None, "missing": [key]}
    return {"ok": worst == 0.0, "max_abs": worst, "missing": []}


def _buffer_gap(left: torch.nn.Module, right: torch.nn.Module) -> dict:
    named = dict(right.named_buffers())
    worst = 0.0
    worst_name = ""
    mismatch = 0
    for name, value in left.named_buffers():
        other = named.get(name)
        if other is None or tuple(other.shape) != tuple(value.shape):
            mismatch += 1
            continue
        if not value.is_floating_point():
            if not torch.equal(value, other):
                mismatch += 1
            continue
        gap = float((value.detach().float() - other.detach().float()).abs().max())
        if gap >= worst:
            worst = gap
            worst_name = name
    return {"max_abs": worst, "worst": worst_name, "mismatch": mismatch}


def _element_split(left: torch.nn.Module, right: torch.nn.Module) -> dict:
    named = dict(right.named_parameters())
    edges = None
    hist = [0] * (len(EDGES) + 1)
    real_max = 0.0
    real_name = ""
    tiny_max = 0.0
    tiny_name = ""
    none_count = 0
    compared = 0
    for name, param in left.named_parameters():
        grad = param.grad
        other = named[name].grad
        if grad is None or other is None:
            none_count += 1
            continue
        compared += 1
        delta = (grad.detach().float() - other.detach().float()).abs().reshape(-1)
        if edges is None or edges.device != delta.device:
            edges = torch.tensor(EDGES, device=delta.device)
        buckets = torch.bucketize(delta, edges)
        counts = torch.bincount(buckets, minlength=len(hist))
        for index, count in enumerate(counts.tolist()):
            hist[index] += int(count)
        mask = grad.detach().abs().reshape(-1) >= TINY_GRAD
        if bool(mask.any()):
            value = float(delta[mask].max())
            if value >= real_max:
                real_max = value
                real_name = name
        if bool((~mask).any()):
            value = float(delta[~mask].max())
            if value >= tiny_max:
                tiny_max = value
                tiny_name = name
    return {
        "compared": compared,
        "none_grad_parameters": none_count,
        "real_max_abs": real_max,
        "real_name": real_name,
        "tiny_max_abs": tiny_max,
        "tiny_name": tiny_name,
        "histogram_edges": [0.0, *EDGES, "inf"],
        "histogram": hist,
    }


def _weight_split(left: torch.nn.Module, right: torch.nn.Module) -> dict:
    named = dict(right.named_parameters())
    real_max = 0.0
    real_name = ""
    tiny_max = 0.0
    tiny_name = ""
    none_count = 0
    for name, param in left.named_parameters():
        grad = param.grad
        delta = (param.detach() - named[name].detach()).abs().reshape(-1)
        if grad is None:
            none_count += 1
            value = float(delta.max()) if delta.numel() else 0.0
            if value >= tiny_max:
                tiny_max = value
                tiny_name = name
            continue
        mask = grad.detach().abs().reshape(-1) >= TINY_GRAD
        if bool(mask.any()):
            value = float(delta[mask].max())
            if value >= real_max:
                real_max = value
                real_name = name
        if bool((~mask).any()):
            value = float(delta[~mask].max())
            if value >= tiny_max:
                tiny_max = value
                tiny_name = name
    return {
        "real_max_abs": real_max,
        "real_name": real_name,
        "tiny_max_abs": tiny_max,
        "tiny_name": tiny_name,
        "none_grad_parameters": none_count,
    }


def _formula_residual(left, right, pre_l, pre_r, grad_l, grad_r, lr: float, eps: float, decay: float) -> dict:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, param in left.named_parameters():
        if grad_l.get(name) is None or grad_r.get(name) is None:
            continue
        scale = 1.0 - lr * decay
        pred_l = pre_l[name] * scale - lr * grad_l[name] / (grad_l[name].abs() + eps)
        pred_r = pre_r[name] * scale - lr * grad_r[name] / (grad_r[name].abs() + eps)
        actual = param.detach() - named[name].detach()
        value = float((actual - (pred_l - pred_r)).abs().max())
        if value >= worst:
            worst = value
            worst_name = name
    return {"max_abs": worst, "worst": worst_name}


def _clone_pre(module: torch.nn.Module) -> tuple[dict, dict]:
    weights = {}
    grads = {}
    for name, param in module.named_parameters():
        weights[name] = param.detach().clone()
        grads[name] = None if param.grad is None else param.grad.detach().clone()
    return weights, grads


def _make_optim(params, kind: str, lr: float, betas: tuple[float, float], decay: float):
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=1e-8, weight_decay=decay)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
    raise ValueError(kind)


def _flatten_grad(stats: dict, split: dict, weight: dict, wsplit: dict, losses: dict, wave: float, optim: dict, buffer: dict, disc: float) -> dict:
    generator_gap = losses.get("generator", losses.get("total"))
    return {
        "finite": all(value == value and abs(value) != float("inf") for value in losses.values()),
        "loss_gap": losses,
        "loss_generator_gap": None if generator_gap is None else float(generator_gap),
        "generator_grad_max_abs": float(stats["max_abs"]),
        "generator_grad_real_max_abs": float(split["real_max_abs"]),
        "generator_grad_tiny_max_abs": float(split["tiny_max_abs"]),
        "large_cosine_min": float(stats["large_cosine_min"]),
        "worst_grad": stats["worst"]["name"],
        "worst_grad_index": stats["worst"]["index"],
        "grad_histogram": split["histogram"],
        "none_grad_parameters": split["none_grad_parameters"],
        "weight_max_abs": float(weight["max_abs"]),
        "weight_real_max_abs": float(wsplit["real_max_abs"]),
        "weight_tiny_max_abs": float(wsplit["tiny_max_abs"]),
        "weight_real_name": wsplit["real_name"],
        "wave_max_abs": float(wave),
        "optimizer_max_abs": float(optim["max_abs"]),
        "optimizer_worst": optim["worst"],
        "buffer_max_abs": float(buffer["max_abs"]),
        "discriminator_grad_max_abs": float(disc),
    }


def _waves(batch: int, length: int, seed: int, scale: float, *, constant_noise: bool) -> tuple[torch.Tensor, torch.Tensor]:
    data_seed = seed + 10007
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)
    clean = torch.randn(batch, length, device="cuda")
    if constant_noise:
        noisy = clean * scale + 0.05
    else:
        noisy = clean * scale + 0.05 * torch.randn_like(clean)
    return clean.detach(), noisy.detach()


def _load_bank(bundle: dict, slot: int) -> tuple[torch.nn.Module, ...]:
    modules = []
    for key in bundle["modules"]:
        module = bundle["slots"][key][slot]
        _load_state(module, bundle["state"][key])
        modules.append(module)
    return tuple(modules)


def run_semamba(seeds: tuple[int, ...] = SEEDS, *, include_trainer: bool = True) -> None:
    os.environ["AUDIO_DB_SEMAMBA_SCAN"] = "fused"
    from tests.numerics.reference import compare_semamba_step as sem

    device = torch.device("cuda")
    report: dict = {"protocol_flags": apply_flags("official_native"), "environment": environment(device), "seeds": {}}
    _write("semamba_campaign.json", report)
    mel = sem.MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
    plan = _comparison_plan("semamba")
    for seed in seeds:
        print(f"semamba build seed {seed}", flush=True)
        bank, cfg, null_names = _semamba_bank(sem, device, seed)
        clean, noisy = _waves(1, 8000, seed, 0.6, constant_noise=False)
        entry = _init_entry(bank, ("g", "cqt", "mrd"))
        report["seeds"][str(seed)] = entry
        report["null_bias_names"] = null_names
        if not entry["init_ok"]:
            _write("semamba_campaign.json", report)
            print("semamba init mismatch", entry["init"], flush=True)
            continue
        for optim_name, kind, left, right, resync, lr in plan:
            rows = _semamba_compare(
                sem, cfg, mel, bank, clean, noisy, null_names, optim_name, kind, left, right, resync, lr, seed, report
            )
            entry.setdefault(optim_name, {})[kind] = rows
            _write("semamba_campaign.json", report)
        _drop_bank(bank)
    report["decision"] = evaluate_campaign("semamba", report["seeds"])
    report["elapsed_sec"] = None
    _write("semamba_campaign.json", report)
    if include_trainer and seeds == SEEDS:
        _semamba_deterministic(sem, device, report)
        _semamba_trainer(sem, device)
    print("semamba campaign decision", report["decision"]["STRICT_NATIVE_3STEP_ABSOLUTE_THRESHOLD"], flush=True)


def _comparison_plan(model: str) -> list[tuple]:
    matched = SGD_LRS[model]["matched"]
    magnified = SGD_LRS[model]["magnified"]
    adam = ("adamw",)
    rows = []
    for name, lr in (("adamw", None), ("sgd_matched", matched), ("sgd_magnified", magnified)):
        rows.append((name, "oo", "official", "official", False, lr))
        rows.append((name, "aa", "audio", "audio", False, lr))
        rows.append((name, "oa", "official", "audio", False, lr))
        if name in adam:
            rows.append((name, "oa_resync", "official", "audio", True, lr))
            rows.append((name, "oo_resync", "official", "official", True, lr))
    return rows


def _semamba_bank(sem, device, seed: int):
    torch.manual_seed(seed)
    cfg, official_g, audio_g, official_cqt, audio_cqt, official_mrd, audio_mrd = sem._pair(device, 48, 4, 16, 4)
    official_g2 = sem.SEMambapp(cfg).to(device)
    official_g2.load_state_dict(official_g.state_dict(), strict=True)
    audio_g2 = sem.SEMambaPP(sem._flat(cfg, "fused")).to(device)
    audio_g2.load_state_dict(official_g.state_dict(), strict=True)
    official_cqt2 = sem.OfficialCQT().to(device)
    official_cqt2.load_state_dict(official_cqt.state_dict(), strict=True)
    audio_cqt2 = sem.MultiScaleSubbandCQTDiscriminator().to(device)
    audio_cqt2.load_state_dict(official_cqt.state_dict(), strict=True)
    official_mrd2 = sem.OfficialMRD().to(device)
    official_mrd2.load_state_dict(official_mrd.state_dict(), strict=True)
    audio_mrd2 = sem.MultiResolutionDiscriminator().to(device)
    audio_mrd2.load_state_dict(official_mrd.state_dict(), strict=True)
    bank = {
        "official": _bundle(("g", "cqt", "mrd"), (official_g, official_g2), (official_cqt, official_cqt2), (official_mrd, official_mrd2)),
        "audio": _bundle(("g", "cqt", "mrd"), (audio_g, audio_g2), (audio_cqt, audio_cqt2), (audio_mrd, audio_mrd2)),
    }
    return bank, cfg, _null_bias_names(official_g)


def _bundle(keys: tuple[str, ...], *pairs: tuple[torch.nn.Module, torch.nn.Module]) -> dict:
    slots = {key: pair for key, pair in zip(keys, pairs, strict=True)}
    state = {key: _cpu_state(pair[0]) for key, pair in slots.items()}
    return {"modules": keys, "slots": slots, "state": state}


def _init_entry(bank: dict, keys: tuple[str, ...]) -> dict:
    gaps = {key: _state_gap(bank["official"]["slots"][key][0], bank["audio"]["slots"][key][0]) for key in keys}
    clones = {key: _state_gap(bank["official"]["slots"][key][0], bank["official"]["slots"][key][1]) for key in keys}
    ok = all(item["ok"] for item in gaps.values()) and all(item["ok"] for item in clones.values())
    return {"init_ok": ok, "init": {"official_vs_audio": gaps, "official_clone": clones}}


def _semamba_compare(sem, cfg, mel, bank, clean, noisy, null_names, optim_name, kind, left_kind, right_kind, resync, lr, seed, report):
    lr = sem.LR if lr is None else lr
    decay = sem.WEIGHT_DECAY if optim_name == "adamw" else SGD_WEIGHT_DECAY
    optim_kind = "adamw" if optim_name == "adamw" else "sgd"
    left = _load_bank(bank[left_kind], 0)
    right = _load_bank(bank[right_kind], 0 if left_kind != right_kind else 1)
    opt_l = (
        _make_optim(left[0].parameters(), optim_kind, lr, sem.BETAS, sem.WEIGHT_DECAY),
        _make_optim(list(left[1].parameters()) + list(left[2].parameters()), optim_kind, lr, sem.BETAS, sem.WEIGHT_DECAY),
    )
    opt_r = (
        _make_optim(right[0].parameters(), optim_kind, lr, sem.BETAS, sem.WEIGHT_DECAY),
        _make_optim(list(right[1].parameters()) + list(right[2].parameters()), optim_kind, lr, sem.BETAS, sem.WEIGHT_DECAY),
    )
    rows = []
    for step in range(STEPS):
        print(f"semamba seed {seed} {optim_name} {kind} step {step}", flush=True)
        probe = seed == SEEDS[0] and optim_name == "adamw" and kind in {"oa", "oa_resync"} and step == 0
        row = _semamba_step(sem, cfg, mel, clean, noisy, null_names, left_kind, left, opt_l, right_kind, right, opt_r, lr, decay, optim_kind, step, probe)
        row["step"] = step
        rows.append(row)
        if resync:
            for source, dest in zip(left, right, strict=True):
                dest.load_state_dict(source.state_dict())
            _copy_optim(left[0], opt_l[0], right[0], opt_r[0])
            _copy_optim(left[1], opt_l[1], right[1], opt_r[1])
            _copy_optim(left[2], opt_l[1], right[2], opt_r[1])
    return rows


def _semamba_step(sem, cfg, mel, clean, noisy, null_names, kind_l, mods_l, opt_l, kind_r, mods_r, opt_r, lr, decay, optim_kind, step, probe):
    g_l, cqt_l, mrd_l = mods_l
    g_r, cqt_r, mrd_r = mods_r
    for module in (*mods_l, *mods_r):
        module.train()
    for optim in (*opt_l, *opt_r):
        optim.zero_grad(set_to_none=True)
    snap = rng_snapshot()
    retain_l = _Retain()
    retain_r = _Retain()
    if probe:
        for index, layer in enumerate(g_l.mask_decoder.mask_conv):
            retain_l.hook(layer, f"mask_conv.{index}")
            retain_r.hook(g_r.mask_decoder.mask_conv[index], f"mask_conv.{index}")
        retain_l.hook(g_l.dense_encoder.dense_conv_1[0], "dense_conv_1.conv")
        retain_r.hook(g_r.dense_encoder.dense_conv_1[0], "dense_conv_1.conv")
    rng_restore(snap)
    parts_l = _semamba_forward(sem, kind_l, g_l, clean, noisy, cfg)
    rng_restore(snap)
    parts_r = _semamba_forward(sem, kind_r, g_r, clean, noisy, cfg)
    snap = rng_snapshot()
    loss_d_l = _semamba_losses(sem, kind_l, {**parts_l, "audio": parts_l["audio"].detach()}, clean, cqt_l, mrd_l, mel, cfg)["discriminator"]
    rng_restore(snap)
    loss_d_r = _semamba_losses(sem, kind_r, {**parts_r, "audio": parts_r["audio"].detach()}, clean, cqt_r, mrd_r, mel, cfg)["discriminator"]
    loss_d_l.backward()
    loss_d_r.backward()
    generator_leak = _nonzero_grads(g_l) + _nonzero_grads(g_r)
    disc = max(_grad_stats(cqt_l, cqt_r, set())["max_abs"], _grad_stats(mrd_l, mrd_r, set())["max_abs"])
    opt_l[1].step()
    opt_r[1].step()
    opt_l[0].zero_grad(set_to_none=True)
    opt_r[0].zero_grad(set_to_none=True)
    snap = rng_snapshot()
    loss_l = _semamba_losses(sem, kind_l, parts_l, clean, cqt_l, mrd_l, mel, cfg)
    rng_restore(snap)
    loss_r = _semamba_losses(sem, kind_r, parts_r, clean, cqt_r, mrd_r, mel, cfg)
    loss_l["generator"].backward()
    loss_r["generator"].backward()
    pre_l = pre_r = grad_l = grad_r = None
    if step == 0 and optim_kind == "adamw":
        pre_l, grad_l = _clone_pre(g_l)
        pre_r, grad_r = _clone_pre(g_r)
    opt_l[0].step()
    opt_r[0].step()
    stats = _grad_stats(g_l, g_r, set(null_names))
    split = _element_split(g_l, g_r)
    weight = {"max_abs": _max_weight(g_l, g_r), "worst": stats["worst"]["name"]}
    wsplit = _weight_split(g_l, g_r)
    losses = {name: abs(float(loss_l[name].detach()) - float(loss_r[name].detach())) for name in loss_l if torch.is_tensor(loss_l[name]) and loss_l[name].ndim == 0}
    losses["discriminator"] = abs(float(loss_d_l.detach()) - float(loss_d_r.detach()))
    row = _flatten_grad(
        stats,
        split,
        weight,
        wsplit,
        losses,
        _tensor_gap(parts_l["audio"], parts_r["audio"])["max_abs"],
        _optim_gap(g_l, opt_l[0], g_r, opt_r[0]),
        _buffer_gap(g_l, g_r),
        disc,
    )
    row["generator_grad_leak_after_discriminator"] = generator_leak
    row["focused_bias"] = _bias_values(g_l, g_r, ["mask_decoder.mask_conv.1.bias", "dense_encoder.dense_conv_1.0.bias"])
    if pre_l is not None:
        row["formula_residual_max_abs"] = _formula_residual(g_l, g_r, pre_l, pre_r, grad_l, grad_r, lr, 1e-8, decay)["max_abs"]
    if probe:
        row["activation_gaps"] = retain_l.gaps(retain_r)
    retain_l.close()
    retain_r.close()
    return row


def _semamba_forward(sem, kind: str, generator, clean, noisy, cfg):
    if kind == "official":
        return sem._official_parts(generator, clean, noisy, cfg)
    return sem._independent_parts(generator, clean, noisy)


def _semamba_losses(sem, kind, parts, clean, cqt, mrd, mel, cfg):
    if kind == "official":
        return sem._official_losses(parts, clean, cqt, mrd, mel, cfg)
    return sem._independent_losses(parts, clean, cqt, mrd)


def _nonzero_grads(module: torch.nn.Module) -> int:
    count = 0
    for param in module.parameters():
        if param.grad is not None and bool(param.grad.abs().max() > 0):
            count += 1
    return count


def _max_weight(left: torch.nn.Module, right: torch.nn.Module) -> float:
    named = dict(right.named_parameters())
    worst = 0.0
    for name, param in left.named_parameters():
        worst = max(worst, float((param.detach() - named[name].detach()).abs().max()))
    return worst


def _drop_bank(bank: dict) -> None:
    bank.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _semamba_deterministic(sem, device, campaign: dict) -> None:
    payload = {"warnings": [], "strict": None}
    try:
        apply_flags("deterministic_warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _semamba_one_seed(sem, device, SEEDS[0], "semamba_det_warn.json")
            payload["warnings"] = sorted({str(item.message) for item in caught if "deterministic" in str(item.message).lower()})
    except Exception as exc:
        payload["warn_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        apply_flags("official_native")
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "semamba-det-strict"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    payload["strict_returncode"] = child.returncode
    payload["strict_stderr_tail"] = "\n".join(child.stderr.splitlines()[-40:])
    strict_path = OUT / "semamba_det_strict.json"
    if strict_path.is_file():
        payload["strict"] = json.loads(strict_path.read_text(encoding="utf-8"))
    campaign["deterministic"] = payload
    _write("semamba_campaign.json", campaign)


def _semamba_one_seed(sem, device, seed: int, name: str) -> None:
    bank, cfg, null_names = _semamba_bank(sem, device, seed)
    clean, noisy = _waves(1, 8000, seed, 0.6, constant_noise=False)
    mel = sem.MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
    report = {"flags": _flags(), "init": _init_entry(bank, ("g", "cqt", "mrd"))}
    for kind, resync in (("oo", False), ("oa", False), ("oa_resync", True)):
        report[kind] = _semamba_compare(
            sem, cfg, mel, bank, clean, noisy, null_names, "adamw", kind, "official", "official" if kind == "oo" else "audio", resync, None, seed, report
        )
    _write(name, report)
    _drop_bank(bank)


def run_mp(seeds: tuple[int, ...] = SEEDS, *, include_trainer: bool = True) -> None:
    from tests.numerics.reference import compare_mp_senet as cmp

    device = torch.device("cuda")
    started = time.time()
    report: dict = {"protocol_flags": apply_flags("official_native"), "environment": environment(device), "seeds": {}}
    _write("mp_campaign.json", report)
    h = cmp._hyper()
    for seed in seeds:
        print(f"mp build seed {seed}", flush=True)
        bank = _mp_bank(cmp, h, device, seed)
        clean, noisy = _waves(2, 8000, seed, 0.5, constant_noise=True)
        entry = _init_entry(bank, ("g", "d"))
        report["seeds"][str(seed)] = entry
        report["null_bias_names"] = _null_bias_names(bank["official"]["slots"]["g"][0])
        if not entry["init_ok"]:
            _write("mp_campaign.json", report)
            continue
        for optim_name, kind, left, right, resync, lr in _comparison_plan("mp"):
            print(f"mp seed {seed} {optim_name} {kind}", flush=True)
            entry.setdefault(optim_name, {})[kind] = _mp_compare(
                cmp, h, bank, clean, noisy, optim_name, kind, left, right, resync, lr, seed
            )
            _write("mp_campaign.json", report)
        _drop_bank(bank)
    report["decision"] = evaluate_campaign("mp", report["seeds"])
    report["elapsed_sec"] = time.time() - started
    _write("mp_campaign.json", report)
    if include_trainer and seeds == SEEDS:
        _mp_deterministic(cmp, h, device, report)
        _mp_trainer(cmp, h, device)
    print("mp campaign decision", report["decision"]["STRICT_NATIVE_3STEP_ABSOLUTE_THRESHOLD"], flush=True)


def _mp_bank(cmp, h, device, seed: int) -> dict:
    from models.discriminator import MetricDiscriminator as OfficialDiscriminator
    from models.model import MPNet

    audio_g, audio_d, official_g, official_d = cmp._build(h, device, seed=seed)
    official_g2 = MPNet(h, num_tsblocks=h.num_tsconformers).to(device)
    official_d2 = OfficialDiscriminator().to(device)
    official_g2.load_state_dict(official_g.state_dict(), strict=True)
    official_d2.load_state_dict(official_d.state_dict(), strict=True)
    audio_g2 = cmp.build_model("mp_senet", vars(h)).to(device)
    audio_d2 = cmp.MetricDiscriminator().to(device)
    audio_g2.load_state_dict(audio_g.state_dict(), strict=True)
    audio_d2.load_state_dict(audio_d.state_dict(), strict=True)
    return {
        "official": _bundle(("g", "d"), (official_g, official_g2), (official_d, official_d2)),
        "audio": _bundle(("g", "d"), (audio_g, audio_g2), (audio_d, audio_d2)),
    }


def _mp_compare(cmp, h, bank, clean, noisy, optim_name, kind, left_kind, right_kind, resync, lr, seed):
    lr = cmp.LR if lr is None else lr
    decay = cmp.WEIGHT_DECAY if optim_name == "adamw" else SGD_WEIGHT_DECAY
    optim_kind = "adamw" if optim_name == "adamw" else "sgd"
    left = _load_bank(bank[left_kind], 0)
    right = _load_bank(bank[right_kind], 0 if left_kind != right_kind else 1)
    opt_l = (
        _make_optim(left[0].parameters(), optim_kind, lr, cmp.BETAS, cmp.WEIGHT_DECAY),
        _make_optim(left[1].parameters(), optim_kind, lr, cmp.BETAS, cmp.WEIGHT_DECAY),
    )
    opt_r = (
        _make_optim(right[0].parameters(), optim_kind, lr, cmp.BETAS, cmp.WEIGHT_DECAY),
        _make_optim(right[1].parameters(), optim_kind, lr, cmp.BETAS, cmp.WEIGHT_DECAY),
    )
    null_names = set(_null_bias_names(left[0]))
    rows = []
    for step in range(STEPS):
        print(f"mp seed {seed} {optim_name} {kind} step {step}", flush=True)
        row = _mp_step(cmp, h, clean, noisy, null_names, left_kind, left, opt_l, right_kind, right, opt_r, lr, decay, optim_kind, step)
        row["step"] = step
        rows.append(row)
        if resync:
            for source, dest in zip(left, right, strict=True):
                dest.load_state_dict(source.state_dict())
            _copy_optim(left[0], opt_l[0], right[0], opt_r[0])
            _copy_optim(left[1], opt_l[1], right[1], opt_r[1])
    return rows


def _mp_step(cmp, h, clean, noisy, null_names, kind_l, mods_l, opt_l, kind_r, mods_r, opt_r, lr, decay, optim_kind, step):
    g_l, d_l = mods_l
    g_r, d_r = mods_r
    for module in (g_l, d_l, g_r, d_r):
        module.train()
    for optim in (*opt_l, *opt_r):
        optim.zero_grad(set_to_none=True)
    snap = rng_snapshot()
    spec_l = _mp_parts(cmp, h, kind_l, g_l, clean, noisy)
    rng_restore(snap)
    spec_r = _mp_parts(cmp, h, kind_r, g_r, clean, noisy)
    ones = torch.ones(clean.shape[0], device=clean.device)
    snap = rng_snapshot()
    loss_d_l = _mp_discriminator(cmp, kind_l, d_l, spec_l, clean, ones)
    rng_restore(snap)
    loss_d_r = _mp_discriminator(cmp, kind_r, d_r, spec_r, clean, ones)
    loss_d_l.backward()
    loss_d_r.backward()
    leak = _nonzero_grads(g_l) + _nonzero_grads(g_r)
    disc = _grad_stats(d_l, d_r, set())["max_abs"]
    opt_l[1].step()
    opt_r[1].step()
    opt_l[0].zero_grad(set_to_none=True)
    opt_r[0].zero_grad(set_to_none=True)
    snap = rng_snapshot()
    loss_l = _mp_generator(cmp, kind_l, d_l, spec_l, clean, ones)
    rng_restore(snap)
    loss_r = _mp_generator(cmp, kind_r, d_r, spec_r, clean, ones)
    loss_l["total"].backward()
    loss_r["total"].backward()
    pre_l = pre_r = grad_l = grad_r = None
    if step == 0 and optim_kind == "adamw":
        pre_l, grad_l = _clone_pre(g_l)
        pre_r, grad_r = _clone_pre(g_r)
    opt_l[0].step()
    opt_r[0].step()
    stats = _grad_stats(g_l, g_r, null_names)
    split = _element_split(g_l, g_r)
    losses = {name: abs(float(loss_l[name].detach()) - float(loss_r[name].detach())) for name in cmp.LOSS_NAMES}
    losses["discriminator"] = abs(float(loss_d_l.detach()) - float(loss_d_r.detach()))
    losses["generator"] = losses["total"]
    row = _flatten_grad(
        stats,
        split,
        {"max_abs": _max_weight(g_l, g_r)},
        _weight_split(g_l, g_r),
        losses,
        _tensor_gap(spec_l["audio"], spec_r["audio"])["max_abs"],
        _optim_gap(g_l, opt_l[0], g_r, opt_r[0]),
        _buffer_gap(g_l, g_r),
        disc,
    )
    row["generator_grad_leak_after_discriminator"] = leak
    row["focused_bias"] = _bias_values(g_l, g_r, ["dense_encoder.dense_conv_1.0.bias"])
    if pre_l is not None:
        row["formula_residual_max_abs"] = _formula_residual(g_l, g_r, pre_l, pre_r, grad_l, grad_r, lr, 1e-8, decay)["max_abs"]
    return row


def _mp_parts(cmp, h, kind: str, generator, clean, noisy):
    stft = cmp.official_stft if kind == "official" else cmp.mag_pha_stft
    istft = cmp.official_istft if kind == "official" else cmp.mag_pha_istft
    return cmp._spectra(generator, clean, noisy, h, stft, istft)


def _mp_discriminator(cmp, kind: str, discriminator, spectrum, clean, ones):
    real = discriminator(spectrum["clean_mag"], spectrum["clean_mag"])
    fake = discriminator(spectrum["clean_mag"], spectrum["hat_mag"].detach())
    if kind == "official":
        target = cmp._official_pesq(clean, spectrum["audio"])
        return cmp._official_discriminator_loss(real, fake, ones, target)
    target = cmp.metric_targets(clean, spectrum["audio"].detach(), cmp._hyper().sampling_rate)
    return cmp.discriminator_loss(real, fake, ones, target)


def _mp_generator(cmp, kind: str, discriminator, spectrum, clean, ones):
    metric = discriminator(spectrum["clean_mag"], spectrum["hat_mag"])
    if kind == "official":
        return cmp._official_components(spectrum, metric, ones, clean)
    return cmp._components(spectrum, metric, ones, clean)


def _mp_deterministic(cmp, h, device, campaign: dict) -> None:
    payload = {"warnings": []}
    try:
        apply_flags("deterministic_warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank = _mp_bank(cmp, h, device, SEEDS[0])
            clean, noisy = _waves(2, 8000, SEEDS[0], 0.5, constant_noise=True)
            report = {"flags": _flags()}
            for kind, resync in (("aa", False), ("oa_resync", True)):
                left = "audio" if kind == "aa" else "official"
                right = "audio"
                report[kind] = _mp_compare(cmp, h, bank, clean, noisy, "adamw", kind, left, right, resync, None, SEEDS[0])
            _write("mp_det_warn.json", report)
            payload["warnings"] = sorted({str(item.message) for item in caught if "deterministic" in str(item.message).lower()})
            _drop_bank(bank)
    except Exception as exc:
        payload["warn_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        apply_flags("official_native")
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "mp-det-strict"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    payload["strict_returncode"] = child.returncode
    payload["strict_stderr_tail"] = "\n".join(child.stderr.splitlines()[-40:])
    campaign["deterministic"] = payload
    _write("mp_campaign.json", campaign)


class _FixedWaves:
    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], identity: dict) -> None:
        self.pairs = pairs
        self._identity = identity
        self.epoch = 0
        self.update_index = 0
        self.group_size = 1

    def identity(self) -> dict:
        return dict(self._identity)

    def train_epoch(self, epoch: int, start_update: int = 0):
        del epoch, start_update
        return _FixedView(self)

    def validation_epoch(self, epoch: int):
        del epoch
        return iter(())


class _FixedView:
    def __init__(self, module: _FixedWaves) -> None:
        self._module = module
        self.updates = len(module.pairs)
        self.plan_hash = f"final-{self.updates}"

    def batch(self, update_index: int):
        from src.experiments.data import Batch

        noisy, clean = self._module.pairs[update_index]
        return [Batch(inputs=noisy.clone(), targets=clean.clone(), sample_id=f"b{update_index}", realization_id=f"r{update_index}")]

    def close(self) -> None:
        return None


def _cpu_pairs(clean: torch.Tensor, noisy: torch.Tensor, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    for _ in range(count):
        # Distinct micro-batches from the same measurement waveform family.
        noise = torch.randn(clean.shape, generator=generator)
        pairs.append(((noisy + 0.01 * noise).contiguous(), (clean + 0.01 * noise).contiguous()))
    return pairs


def _failure_count(model: str, rows: list[dict]) -> int:
    return sum(len(strict_reasons(model, row)) for row in rows)


def _grad_dict(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {name: None if param.grad is None else param.grad.detach().float().cpu() for name, param in module.named_parameters()}


def _compare_saved_grads(model: str, left: dict, right: dict, left_module, right_module, losses: dict, disc: float) -> dict:
    max_abs = 0.0
    real_max = 0.0
    cosine_min = 1.0
    large_cosine = 1.0
    worst = ""
    for name, grad in left.items():
        other = right.get(name)
        if grad is None or other is None:
            continue
        delta = (grad - other).abs()
        value = float(delta.max())
        if value >= max_abs:
            max_abs = value
            worst = name
        mask = grad.abs().reshape(-1) >= TINY_GRAD
        if bool(mask.any()):
            real_max = max(real_max, float(delta.reshape(-1)[mask].max()))
        if grad.numel() > 1 and float(grad.norm() * other.norm()) > 0:
            cosine = float(torch.nn.functional.cosine_similarity(grad.reshape(-1), other.reshape(-1), dim=0))
            if float(grad.abs().max()) > 1e-5:
                large_cosine = min(large_cosine, cosine)
                cosine_min = min(cosine_min, cosine)
    weight = _weight_split(left_module, right_module)
    stats = {"max_abs": max_abs, "large_cosine_min": large_cosine, "worst": {"name": worst, "index": -1}}
    split = {"real_max_abs": real_max, "tiny_max_abs": 0.0, "histogram": [], "none_grad_parameters": 0}
    return _flatten_grad(
        stats,
        split,
        {"max_abs": _max_weight(left_module, right_module)},
        weight,
        losses,
        0.0,
        {"max_abs": 0.0, "worst": ""},
        {"max_abs": 0.0},
        disc,
    )


def _semamba_trainer(sem, device) -> None:
    from src.experiments.checkpoint import CheckpointManager
    from src.experiments.official.semambapp.models import SEMambaPPCQTDiscriminator, SEMambaPPGenerator, SEMambaPPMRD
    from src.experiments.official.semambapp.trainer import SEMambaPPTrainer
    from src.experiments.runtime import PreparePolicy, ProcessRuntime

    report = {"call_label": "OFFICIAL_COMPONENT_STEP", "official_function": "train", "official_file": str(SEMAMBA_TRAIN), "run_called": False}
    try:
        apply_flags("official_native")
        native = _semamba_trainer_once(sem, device, SEMambaPPTrainer, SEMambaPPGenerator, SEMambaPPMRD, SEMambaPPCQTDiscriminator, PreparePolicy, ProcessRuntime, CheckpointManager, "official_native")
        apply_flags("tight")
        tight = _semamba_trainer_once(sem, device, SEMambaPPTrainer, SEMambaPPGenerator, SEMambaPPMRD, SEMambaPPCQTDiscriminator, PreparePolicy, ProcessRuntime, CheckpointManager, "tight")
        report["native"] = native
        report["tight"] = tight
        report["run_called"] = bool(native.get("run_called"))
        report["status"] = trainer_status(
            run_called=bool(native.get("run_called")),
            logical_steps=int(native.get("logical_steps") or 0),
            scheduler_steps=int(native.get("scheduler_steps") or 0),
            native_failure_count=_failure_count("semamba", native.get("same_state") or []),
            tight_failure_count=_failure_count("semamba", tight.get("same_state") or []),
            graph_retained=bool(native.get("graph_retained")),
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "UNKNOWN"
    finally:
        apply_flags("official_native")
    _write("semamba_trainer.json", report)
    print("semamba trainer", report.get("status"), flush=True)


def _semamba_trainer_once(sem, device, trainer_cls, gen_cls, mrd_cls, cqt_cls, policy_cls, runtime_cls, checkpoint_cls, flag_mode: str) -> dict:
    torch.manual_seed(SEEDS[0])
    cfg, official_g, _audio_g, official_cqt, _audio_cqt, official_mrd, _audio_mrd = sem._pair(device, 48, 4, 16, 4)
    clean, noisy = _waves(1, 8000, SEEDS[0], 0.6, constant_noise=False)
    pairs = _cpu_pairs(clean.cpu(), noisy.cpu(), 4)
    # This process already owns the CUDA context, so the idle check would see itself.
    runtime = runtime_cls(device="cuda", precision="fp32", allow_busy_gpu=True)
    try:
        architecture = dict(hid_feature=48, num_tfmamba=4, d_state=16, d_conv=4, expand=2, norm_epsilon=1e-5, input_channel=2, output_channel=1, n_fft=400, scan_backend="fused", hop=100, win=400, sampling_rate=16000)
        generator = gen_cls(seed=SEEDS[0], **architecture)
        mrd = mrd_cls(seed=SEEDS[0], **architecture)
        cqt = cqt_cls(seed=SEEDS[0], **architecture)
        prepared_g = runtime.prepare_model(generator, policy_cls(trainable=True, distribute="none"))
        prepared_m = runtime.prepare_model(mrd, policy_cls(trainable=True, distribute="none"))
        prepared_c = runtime.prepare_model(cqt, policy_cls(trainable=True, distribute="none"))
        prepared_g.state_module.generator.load_state_dict(official_g.state_dict(), strict=True)
        prepared_m.state_module.discriminator.load_state_dict(official_mrd.state_dict(), strict=True)
        prepared_c.state_module.discriminator.load_state_dict(official_cqt.state_dict(), strict=True)
        data = _FixedWaves(pairs, {"algorithm": "online.semambapp_gsr", "claim_label": "protocol_conformant_substitute_materials", "group_size": 1})
        opt_g = torch.optim.AdamW(prepared_g.parameters, lr=sem.LR, betas=sem.BETAS, eps=1e-8, weight_decay=sem.WEIGHT_DECAY)
        opt_d = torch.optim.AdamW(list(prepared_m.parameters) + list(prepared_c.parameters), lr=sem.LR, betas=sem.BETAS, eps=1e-8, weight_decay=sem.WEIGHT_DECAY)
        opt_g._experiment_component_id = "optim_g"
        opt_d._experiment_component_id = "optim_d"
        sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.99)
        sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.99)
        sched_count = {"n": 0}

        def _count(original):
            def counted(*args, **kwargs):
                sched_count["n"] += 1
                return original(*args, **kwargs)

            return counted

        sched_g.step = _count(sched_g.step)
        sched_d.step = _count(sched_d.step)
        directory = OUT / "checkpoints" / f"semamba-{flag_mode}"
        directory.mkdir(parents=True, exist_ok=True)
        trainer = trainer_cls(
            prepared_g, prepared_m, prepared_c, data, opt_g, opt_d, sched_g, sched_d,
            max_logical_steps=3, accumulation=1, grad_clip=None, validation_interval=0, checkpoint_interval=100,
        )
        off_opt_g = torch.optim.AdamW(official_g.parameters(), lr=sem.LR, betas=sem.BETAS, eps=1e-8, weight_decay=sem.WEIGHT_DECAY)
        off_opt_d = torch.optim.AdamW(list(official_mrd.parameters()) + list(official_cqt.parameters()), lr=sem.LR, betas=sem.BETAS, eps=1e-8, weight_decay=sem.WEIGHT_DECAY)
        mel = sem.MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
        init_gap = _state_gap(prepared_g.state_module.generator, official_g)
        init_state = {
            "g": _cpu_state(official_g),
            "cqt": _cpu_state(official_cqt),
            "mrd": _cpu_state(official_mrd),
        }
        rows = []
        for index in range(STEPS):
            batch = data.train_epoch(0).batch(index)[0]
            wave_clean = batch.targets.to(device)
            wave_noisy = batch.inputs.to(device)
            snap = rng_snapshot()
            official_losses, official_grads = _semamba_component_step(sem, cfg, mel, official_g, official_cqt, official_mrd, off_opt_g, off_opt_d, wave_clean, wave_noisy)
            rng_restore(snap)
            produced = trainer.official_step([batch], runtime)
            trainer_grads = _grad_dict(prepared_g.state_module.generator)
            disc = max(
                _grad_max_from_dicts(_grad_dict(prepared_c.state_module.discriminator), _grad_dict(official_cqt)),
                _grad_max_from_dicts(_grad_dict(prepared_m.state_module.discriminator), _grad_dict(official_mrd)),
            )
            row = _compare_saved_grads(
                "semamba",
                official_grads,
                trainer_grads,
                official_g,
                prepared_g.state_module.generator,
                _shared_gap(official_losses, produced),
                disc,
            )
            row["step"] = index
            row["optimizer_max_abs"] = _optim_gap(official_g, off_opt_g, prepared_g.state_module.generator, opt_g)["max_abs"]
            rows.append(row)
            _align_modules((official_g, official_cqt, official_mrd), (off_opt_g, off_opt_d, off_opt_d), (prepared_g.state_module.generator, prepared_c.state_module.discriminator, prepared_m.state_module.discriminator), (opt_g, opt_d, opt_d))
        same_trace = list(trainer.sync_trace)
        _load_state(official_g, init_state["g"])
        _load_state(official_cqt, init_state["cqt"])
        _load_state(official_mrd, init_state["mrd"])
        off_opt_g.state.clear()
        off_opt_d.state.clear()
        for index in range(STEPS):
            batch = data.train_epoch(0).batch(index)[0]
            _semamba_component_step(sem, cfg, mel, official_g, official_cqt, official_mrd, off_opt_g, off_opt_d, batch.targets.to(device), batch.inputs.to(device))
        _load_state(prepared_g.state_module.generator, init_state["g"])
        _load_state(prepared_c.state_module.discriminator, init_state["cqt"])
        _load_state(prepared_m.state_module.discriminator, init_state["mrd"])
        opt_g.state.clear()
        opt_d.state.clear()
        trainer.sync_trace.clear()
        result = trainer.run(runtime, checkpoint_cls(directory))
        run_trace = list(trainer.sync_trace)
        graph = _retained(same_trace) and _retained(run_trace)
        return {
            "flags": _flags(),
            "init_gap": init_gap,
            "same_state": rows,
            "independent_generator_gap": _state_gap(prepared_g.state_module.generator, official_g),
            "graph_retained": graph,
            "sync_trace": same_trace + run_trace,
            "run_called": True,
            "logical_steps": int(result.logical_step),
            "scheduler_steps": int(sched_count["n"]),
            "step_log": trainer.step_log,
            "grad_clip": trainer.grad_clip,
        }
    finally:
        runtime.shutdown()


def _semamba_component_step(sem, cfg, mel, generator, cqt, mrd, opt_g, opt_d, clean, noisy):
    """One official ``train`` body. Not ``train()`` itself: no loader and no wandb."""

    for module in (generator, cqt, mrd):
        module.train()
    parts = sem._official_parts(generator, clean, noisy, cfg)
    opt_d.zero_grad()
    detached = sem._official_losses({**parts, "audio": parts["audio"].detach()}, clean, cqt, mrd, mel, cfg)
    detached["discriminator"].backward()
    opt_d.step()
    opt_g.zero_grad()
    losses = sem._official_losses(parts, clean, cqt, mrd, mel, cfg)
    losses["generator"].backward()
    grads = _grad_dict(generator)
    floats = {name: float(value.detach()) for name, value in losses.items() if torch.is_tensor(value) and value.ndim == 0}
    floats["discriminator"] = float(detached["discriminator"].detach())
    opt_g.step()
    return floats, grads


def _align_modules(sources, source_opts, dests, dest_opts) -> None:
    for source, dest in zip(sources, dests, strict=True):
        dest.load_state_dict(source.state_dict())
    for source, source_opt, dest, dest_opt in zip(sources, source_opts, dests, dest_opts, strict=True):
        _copy_optim(source, source_opt, dest, dest_opt)


def _shared_gap(left: dict, right: dict) -> dict:
    gap = {}
    for key, value in left.items():
        if key not in right:
            continue
        gap[key] = abs(float(value) - float(right[key]))
    return gap


def _retained(trace: list[str]) -> bool:
    text = "".join(trace)
    return "generator_backward_retained" in trace and "generator_reforward" not in text


def _grad_max_from_dicts(left: dict, right: dict) -> float:
    worst = 0.0
    for name, grad in left.items():
        other = right.get(name)
        if grad is None or other is None:
            continue
        worst = max(worst, float((grad - other).abs().max()))
    return worst


def _mp_trainer(cmp, h, device) -> None:
    from src.experiments.checkpoint import CheckpointManager
    from src.experiments.official.mp_senet.models import MPSENetGANGenerator, MPSENetMetricDiscriminator
    from src.experiments.official.mp_senet.trainer import MPSENetGANTrainer
    from src.experiments.runtime import PreparePolicy, ProcessRuntime

    report = {"call_label": "OFFICIAL_COMPONENT_STEP", "official_function": "train", "official_file": str(MP_TRAIN), "run_called": False}
    try:
        apply_flags("official_native")
        native = _mp_trainer_once(cmp, h, device, MPSENetGANTrainer, MPSENetGANGenerator, MPSENetMetricDiscriminator, PreparePolicy, ProcessRuntime, CheckpointManager, "official_native")
        apply_flags("tight")
        tight = _mp_trainer_once(cmp, h, device, MPSENetGANTrainer, MPSENetGANGenerator, MPSENetMetricDiscriminator, PreparePolicy, ProcessRuntime, CheckpointManager, "tight")
        report["native"] = native
        report["tight"] = tight
        report["run_called"] = bool(native.get("run_called"))
        report["status"] = trainer_status(
            run_called=bool(native.get("run_called")),
            logical_steps=int(native.get("logical_steps") or 0),
            scheduler_steps=int(native.get("scheduler_steps") or 0),
            native_failure_count=_failure_count("mp", native.get("same_state") or []),
            tight_failure_count=_failure_count("mp", tight.get("same_state") or []),
            graph_retained=bool(native.get("graph_retained")),
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "UNKNOWN"
    finally:
        apply_flags("official_native")
    _write("mp_trainer.json", report)
    print("mp trainer", report.get("status"), flush=True)


def _mp_trainer_once(cmp, h, device, trainer_cls, gen_cls, disc_cls, policy_cls, runtime_cls, checkpoint_cls, flag_mode: str) -> dict:
    audio_g, audio_d, official_g, official_d = cmp._build(h, device, seed=SEEDS[0])
    del audio_g, audio_d
    clean, noisy = _waves(2, 8000, SEEDS[0], 0.5, constant_noise=True)
    pairs = _cpu_pairs(clean.cpu(), noisy.cpu(), 4)
    # This process already owns the CUDA context, so the idle check would see itself.
    runtime = runtime_cls(device="cuda", precision="fp32", allow_busy_gpu=True)
    alive = {"value": False}
    try:
        architecture = dict(dense_channel=64, num_tsconformers=4, n_fft=400, hop_size=100, win_size=400, compress_factor=0.3, beta=2.0, sampling_rate=16000)
        generator = gen_cls(seed=SEEDS[0], **architecture)
        discriminator = disc_cls(seed=SEEDS[0], **architecture)
        prepared_g = runtime.prepare_model(generator, policy_cls(trainable=True, distribute="none"))
        prepared_d = runtime.prepare_model(discriminator, policy_cls(trainable=True, distribute="none"))
        prepared_g.state_module.generator.load_state_dict(official_g.state_dict(), strict=True)
        prepared_d.state_module.discriminator.load_state_dict(official_d.state_dict(), strict=True)
        data = _FixedWaves(pairs, {"algorithm": "fixed_pair.mp_senet", "profile": "official_faithful", "selection_uses_test": True, "group_size": 1})
        opt_g = torch.optim.AdamW(prepared_g.parameters, lr=cmp.LR, betas=cmp.BETAS, eps=1e-8, weight_decay=cmp.WEIGHT_DECAY)
        opt_d = torch.optim.AdamW(prepared_d.parameters, lr=cmp.LR, betas=cmp.BETAS, eps=1e-8, weight_decay=cmp.WEIGHT_DECAY)
        opt_g._experiment_component_id = "optim_g"
        opt_d._experiment_component_id = "optim_d"
        sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.99)
        sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.99)
        sched_count = {"n": 0}
        original_g = sched_g.step
        original_d = sched_d.step
        def count_g(*args, **kwargs):
            sched_count["n"] += 1
            return original_g(*args, **kwargs)
        def count_d(*args, **kwargs):
            sched_count["n"] += 1
            return original_d(*args, **kwargs)
        sched_g.step = count_g
        sched_d.step = count_d
        directory = OUT / "checkpoints" / f"mp-{flag_mode}"
        directory.mkdir(parents=True, exist_ok=True)
        trainer = trainer_cls(
            prepared_g, prepared_d, data, opt_g, opt_d, sched_g, sched_d,
            max_logical_steps=3, experiment_claim="official_faithful", accumulation=1,
            validation_interval=0, checkpoint_interval=100, best_checkpoint_start_epoch=40,
        )
        def hook(phase: str, payload=None) -> None:
            if phase == "after_discriminator_step" and isinstance(payload, list) and payload:
                audio = payload[0][1]["audio"]
                alive["value"] = audio.grad_fn is not None
        trainer._on_update_phase = hook
        off_opt_g = torch.optim.AdamW(official_g.parameters(), lr=cmp.LR, betas=cmp.BETAS, eps=1e-8, weight_decay=cmp.WEIGHT_DECAY)
        off_opt_d = torch.optim.AdamW(official_d.parameters(), lr=cmp.LR, betas=cmp.BETAS, eps=1e-8, weight_decay=cmp.WEIGHT_DECAY)
        rows = []
        init_gap = _state_gap(prepared_g.state_module.generator, official_g)
        init_state = {"g": _cpu_state(official_g), "d": _cpu_state(official_d)}
        for index in range(STEPS):
            batch = data.train_epoch(0).batch(index)[0]
            wave_clean = batch.targets.to(device)
            wave_noisy = batch.inputs.to(device)
            snap = rng_snapshot()
            official_losses, official_grads = _mp_component_step(cmp, h, official_g, official_d, off_opt_g, off_opt_d, wave_clean, wave_noisy)
            rng_restore(snap)
            produced = trainer.official_step([batch], runtime)
            row = _compare_saved_grads(
                "mp",
                official_grads,
                _grad_dict(prepared_g.state_module.generator),
                official_g,
                prepared_g.state_module.generator,
                _mp_loss_gap(official_losses, produced),
                _grad_max_from_dicts(_grad_dict(prepared_d.state_module.discriminator), _grad_dict(official_d)),
            )
            row["step"] = index
            row["optimizer_max_abs"] = _optim_gap(official_g, off_opt_g, prepared_g.state_module.generator, opt_g)["max_abs"]
            rows.append(row)
            _align_modules((official_g, official_d), (off_opt_g, off_opt_d), (prepared_g.state_module.generator, prepared_d.state_module.discriminator), (opt_g, opt_d))
        _load_state(official_g, init_state["g"])
        _load_state(official_d, init_state["d"])
        off_opt_g.state.clear()
        off_opt_d.state.clear()
        for index in range(STEPS):
            batch = data.train_epoch(0).batch(index)[0]
            _mp_component_step(cmp, h, official_g, official_d, off_opt_g, off_opt_d, batch.targets.to(device), batch.inputs.to(device))
        _load_state(prepared_g.state_module.generator, init_state["g"])
        _load_state(prepared_d.state_module.discriminator, init_state["d"])
        opt_g.state.clear()
        opt_d.state.clear()
        result = trainer.run(runtime, checkpoint_cls(directory))
        return {
            "flags": _flags(),
            "init_gap": init_gap,
            "same_state": rows,
            "independent_generator_gap": _state_gap(prepared_g.state_module.generator, official_g),
            "graph_retained": bool(alive["value"]),
            "run_called": True,
            "logical_steps": int(result.logical_step),
            "scheduler_steps": int(sched_count["n"]),
            "step_log": trainer.step_log,
        }
    finally:
        runtime.shutdown()


def _mp_component_step(cmp, h, generator, discriminator, opt_g, opt_d, clean, noisy):
    """Official ``train.py`` update body. The surrounding loader is not started."""

    generator.train()
    discriminator.train()
    spec = cmp._spectra(generator, clean, noisy, h, cmp.official_stft, cmp.official_istft)
    ones = torch.ones(clean.shape[0], device=clean.device)
    opt_d.zero_grad()
    loss_d = _mp_discriminator(cmp, "official", discriminator, spec, clean, ones)
    loss_d.backward()
    opt_d.step()
    opt_g.zero_grad()
    losses = _mp_generator(cmp, "official", discriminator, spec, clean, ones)
    losses["total"].backward()
    grads = _grad_dict(generator)
    floats = {name: float(losses[name].detach()) for name in cmp.LOSS_NAMES}
    floats["discriminator"] = float(loss_d.detach())
    opt_g.step()
    return floats, grads


def _mp_loss_gap(official: dict, trainer: dict) -> dict:
    gap = {}
    for name in ("magnitude", "phase", "complex", "consistency", "metric", "time", "discriminator"):
        if name in official and name in trainer:
            gap[name] = abs(float(official[name]) - float(trainer[name]))
    if "total" in official and "generator" in trainer:
        gap["generator"] = abs(float(official["total"]) - float(trainer["generator"]))
    return gap


def _strict_child(model: str) -> None:
    apply_flags("deterministic_strict")
    device = torch.device("cuda")
    try:
        if model == "semamba":
            os.environ["AUDIO_DB_SEMAMBA_SCAN"] = "fused"
            from tests.numerics.reference import compare_semamba_step as sem

            _semamba_one_seed(sem, device, SEEDS[0], "semamba_det_strict.json")
        else:
            from tests.numerics.reference import compare_mp_senet as cmp

            h = cmp._hyper()
            bank = _mp_bank(cmp, h, device, SEEDS[0])
            clean, noisy = _waves(2, 8000, SEEDS[0], 0.5, constant_noise=True)
            report = {"flags": _flags(), "oa": _mp_compare(cmp, h, bank, clean, noisy, "adamw", "oa", "official", "audio", False, None, SEEDS[0])}
            _write("mp_det_strict.json", report)
    except Exception as exc:
        _write(f"{'semamba' if model == 'semamba' else 'mp'}_det_strict.json", {"error": f"{type(exc).__name__}: {exc}", "flags": _flags()})
        raise


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: two_model_final.py semamba|mp|semamba-det-strict|mp-det-strict")
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    if mode in {"semamba", "mp"}:
        written = protocol()
        defaults = written["torch_adamw_defaults"]
        if defaults["weight_decay"] != 0.01 or defaults["eps"] != 1e-8 or defaults["amsgrad"]:
            raise SystemExit(f"AdamW defaults are not the official implicit values: {defaults}")
        _write("protocol.json", written)
    if not torch.cuda.is_available():
        _write(f"{mode}_not_run.json", {"status": "NOT_RUN", "reason": "cuda unavailable"})
        raise SystemExit("cuda unavailable")
    if mode == "semamba":
        started = time.time()
        run_semamba()
        payload = json.loads((OUT / "semamba_campaign.json").read_text(encoding="utf-8"))
        payload["elapsed_sec"] = time.time() - started
        _write("semamba_campaign.json", payload)
    elif mode == "mp":
        run_mp()
    elif mode == "semamba-det-strict":
        _strict_child("semamba")
    elif mode == "mp-det-strict":
        _strict_child("mp")
    elif mode == "semamba-trainer":
        os.environ["AUDIO_DB_SEMAMBA_SCAN"] = "fused"
        from tests.numerics.reference import compare_semamba_step as sem

        _semamba_trainer(sem, torch.device("cuda"))
    elif mode == "mp-trainer":
        from tests.numerics.reference import compare_mp_senet as cmp

        _mp_trainer(cmp, cmp._hyper(), torch.device("cuda"))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
