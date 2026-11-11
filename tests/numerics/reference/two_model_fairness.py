"""Same-input O/O, A/A, and O/A steps for SEMamba++ and MP-SENet.

Diagnostic only. Training code is not modified. Official checkouts are read-only.
Writes under AUDIO_DB_NUMERICS_OUT (default /tmp/audio-db-official-two-models).

    CUBLAS_WORKSPACE_CONFIG=:4096:8 AUDIO_DB_SEMAMBA_SCAN=fused \\
        .venvs/semambapp/bin/python tests/numerics/reference/two_model_fairness.py semamba
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venvs/se/bin/python tests/numerics/reference/two_model_fairness.py mp
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("AUDIO_DB_NUMERICS_OUT", "/tmp/audio-db-official-two-models"))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _git(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "absent"


def environment(device: torch.device) -> dict:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "scan_env": os.environ.get("AUDIO_DB_SEMAMBA_SCAN"),
        "mamba_ssm": _package_version("mamba-ssm"),
        "causal_conv1d": _package_version("causal-conv1d"),
        "triton": _package_version("triton"),
        "audio_db_head": _git(ROOT),
        "mp_senet_official": _git(ROOT / "src" / "se" / "official" / "mp-senet"),
        "semambapp_official": _git(ROOT / "src" / "se" / "official" / "semambapp"),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability(device))
    return info


def _write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    print("wrote", path, flush=True)


def _cpu_state(module: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_state(module: nn.Module, state: dict) -> None:
    device = next(module.parameters()).device
    module.load_state_dict({key: value.to(device) for key, value in state.items()})


def _null_bias_names(module: nn.Module) -> list[str]:
    """Conv biases whose following InstanceNorm cancels them in exact arithmetic."""
    found = []
    conv_types = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
    for name, sequential in module.named_modules():
        if not isinstance(sequential, nn.Sequential):
            continue
        for index, layer in enumerate(sequential):
            if index + 1 >= len(sequential):
                continue
            if isinstance(layer, conv_types) and isinstance(sequential[index + 1], nn.InstanceNorm2d):
                if getattr(layer, "bias", None) is not None:
                    found.append(f"{name}.{index}.bias")
    return found


def _grad_stats(left: nn.Module, right: nn.Module, null_names: set[str]) -> dict:
    named = dict(right.named_parameters())
    worst = {"max_abs": 0.0, "name": "", "index": -1, "left": None, "right": None, "left_max": None}
    worst_real = {"max_abs": 0.0, "name": "", "index": -1, "left": None, "right": None, "left_max": None}
    worst_large = {"max_abs": 0.0, "name": "", "index": -1, "left": None, "right": None, "left_max": None}
    cosine_min = 1.0
    cosine_name = ""
    large_cosine = 1.0
    compared = 0
    for name, param in left.named_parameters():
        other = named[name]
        if param.grad is None or other.grad is None:
            continue
        compared += 1
        delta = (param.grad.detach().float() - other.grad.detach().float()).abs()
        index = int(delta.view(-1).argmax())
        value = float(delta.view(-1)[index])
        scale = float(param.grad.detach().abs().max())
        item = {
            "max_abs": value,
            "name": name,
            "index": index,
            "left": float(param.grad.detach().float().view(-1)[index]),
            "right": float(other.grad.detach().float().view(-1)[index]),
            "left_max": scale,
        }
        if value >= worst["max_abs"]:
            worst = item
        if name not in null_names and value >= worst_real["max_abs"]:
            worst_real = item
        if name not in null_names and scale > 1e-4 and value >= worst_large["max_abs"]:
            worst_large = item
        flat_l = param.grad.detach().float().reshape(-1)
        flat_r = other.grad.detach().float().reshape(-1)
        if flat_l.numel() > 1 and float(flat_l.norm() * flat_r.norm()) > 0:
            cosine = float(torch.nn.functional.cosine_similarity(flat_l, flat_r, dim=0))
            if scale > 1e-5:
                large_cosine = min(large_cosine, cosine)
                if cosine <= cosine_min:
                    cosine_min = cosine
                    cosine_name = name
    return {
        "compared": compared,
        "max_abs": worst["max_abs"],
        "worst": worst,
        "excluding_null_bias": worst_real,
        "excluding_null_bias_and_tiny": worst_large,
        "large_cosine_min": large_cosine,
        "large_cosine_name": cosine_name,
    }


def _weight_stats(left: nn.Module, right: nn.Module, null_names: set[str]) -> dict:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    real = 0.0
    real_name = ""
    for name, param in left.named_parameters():
        value = float((param.detach() - named[name].detach()).abs().max())
        if value >= worst:
            worst = value
            worst_name = name
        if name not in null_names and value >= real:
            real = value
            real_name = name
    return {"max_abs": worst, "worst": worst_name, "excluding_null_bias": real, "excluding_null_bias_name": real_name}


def _bias_values(left: nn.Module, right: nn.Module, names: list[str]) -> dict:
    left_named = dict(left.named_parameters())
    right_named = dict(right.named_parameters())
    report = {}
    for name in names:
        if name not in left_named:
            continue
        grad_l = left_named[name].grad
        grad_r = right_named[name].grad
        if grad_l is None or grad_r is None:
            continue
        report[name] = {
            "left": [float(value) for value in grad_l.detach().float().view(-1)],
            "right": [float(value) for value in grad_r.detach().float().view(-1)],
            "max_abs": float((grad_l.detach() - grad_r.detach()).abs().max()),
        }
    return report


def _optim_gap(left_m: nn.Module, left_o: torch.optim.Optimizer, right_m: nn.Module, right_o: torch.optim.Optimizer) -> dict:
    """Compare Adam moments by parameter name. Module order is not shared."""
    worst = 0.0
    worst_name = ""
    mismatch = 0
    right_named = dict(right_m.named_parameters())
    for name, left_param in left_m.named_parameters():
        state_l = left_o.state.get(left_param, {})
        state_r = right_o.state.get(right_named[name], {})
        if not state_l and not state_r:
            continue
        for key in set(state_l) | set(state_r):
            if key == "step":
                if int(state_l.get(key, -1)) != int(state_r.get(key, -2)):
                    mismatch += 1
                continue
            if key not in state_l or key not in state_r or not torch.is_tensor(state_l[key]):
                mismatch += 1
                continue
            if tuple(state_l[key].shape) != tuple(state_r[key].shape):
                mismatch += 1
                continue
            value = float((state_l[key].detach().float() - state_r[key].detach().float()).abs().max())
            if value > worst:
                worst = value
                worst_name = f"{name}:{key}"
    return {"max_abs": worst, "worst": worst_name, "missing_or_step_mismatch": mismatch}


def _first_adam_gap(left: nn.Module, right: nn.Module, lr: float, eps: float, decay: float) -> dict:
    named = dict(right.named_parameters())
    worst = 0.0
    worst_name = ""
    for name, param in left.named_parameters():
        other = named[name]
        if param.grad is None or other.grad is None:
            continue
        pred_l = param.detach() * (1 - lr * decay) - lr * param.grad.detach() / (param.grad.detach().abs() + eps)
        pred_r = other.detach() * (1 - lr * decay) - lr * other.grad.detach() / (other.grad.detach().abs() + eps)
        value = float((pred_l - pred_r).abs().max())
        if value > worst:
            worst = value
            worst_name = name
    return {"formula_weight_max_abs": worst, "worst": worst_name}


def _copy_optim(source_m: nn.Module, source_o: torch.optim.Optimizer, dest_m: nn.Module, dest_o: torch.optim.Optimizer) -> None:
    dest_named = dict(dest_m.named_parameters())
    for name, source_param in source_m.named_parameters():
        dest_param = dest_named[name]
        source_state = source_o.state.get(source_param)
        if not source_state:
            dest_o.state.pop(dest_param, None)
            continue
        dest_o.state[dest_param] = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in source_state.items()
        }


def _sync(source_modules, source_optims, dest_modules, dest_optims) -> None:
    for source, dest in zip(source_modules, dest_modules, strict=True):
        dest.load_state_dict(source.state_dict())
    for source_m, source_o, dest_m, dest_o in zip(source_modules, source_optims, dest_modules, dest_optims, strict=True):
        _copy_optim(source_m, source_o, dest_m, dest_o)


def null_space_probe(device: torch.device) -> dict:
    """Show that a conv bias before InstanceNorm is a numerical null space."""
    torch.manual_seed(0)
    conv = nn.Conv2d(2, 8, (1, 1)).to(device)
    norm = nn.InstanceNorm2d(8, affine=True).to(device)
    act = nn.PReLU(8).to(device)
    base = torch.randn(2, 2, 16, 32, device=device)

    def bias_grad(value: torch.Tensor) -> torch.Tensor:
        conv.zero_grad(set_to_none=True)
        act(norm(conv(value))).square().mean().backward()
        return conv.bias.grad.detach().float().cpu().clone()

    first = bias_grad(base)
    bumped = bias_grad(torch.nextafter(base, torch.full_like(base, float("inf"))))
    lr = 2e-4
    eps = 1e-8
    amplified = lr * first / (first.abs() + eps) - lr * bumped / (bumped.abs() + eps)
    return {
        "grad_max_abs": float(first.abs().max()),
        "nextafter_input_grad_max_abs": float((first - bumped).abs().max()),
        "sign_flips": int(((first > 0) != (bumped > 0)).sum()),
        "adamw_first_step_weight_gap_at_lr_2e-4": float(amplified.abs().max()),
        "numel": int(first.numel()),
    }


def _tensor_gap(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = (left.detach().float() - right.detach().float()).abs()
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def _loss_map(losses: dict) -> dict[str, float]:
    mapped = {}
    for name, value in losses.items():
        if torch.is_tensor(value) and value.ndim == 0:
            mapped[name] = float(value.detach())
    return mapped


def _term_grads(losses: dict, params: list[tuple[str, torch.Tensor]]) -> dict:
    tensors = [param for _, param in params]
    report = {}
    for name, loss in losses.items():
        if name == "discriminator" or not torch.is_tensor(loss) or loss.ndim != 0:
            continue
        grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
        entry = {}
        for (param_name, _), grad in zip(params, grads, strict=True):
            if grad is None:
                entry[param_name] = None
                continue
            flat = grad.detach().float().view(-1)
            entry[param_name] = {
                "max_abs": float(flat.abs().max()),
                "sum": float(flat.sum()),
                "values": [float(value) for value in flat] if flat.numel() <= 64 else None,
            }
        report[name] = entry
    return report


class _Retain:
    def __init__(self):
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []

    def hook(self, module: nn.Module, name: str) -> None:
        def record(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(value):
                value.retain_grad()
                self.outputs[name] = value

        self.handles.append(module.register_forward_hook(record))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def gaps(self, other: "_Retain") -> dict:
        report = {}
        for name, value in self.outputs.items():
            if name not in other.outputs:
                continue
            report[name] = {"forward": _tensor_gap(value, other.outputs[name])}
            if value.grad is not None and other.outputs[name].grad is not None:
                report[name]["backward"] = _tensor_gap(value.grad, other.outputs[name].grad)
                spatial = tuple(dim for dim in range(value.grad.ndim) if dim != 1)
                if value.grad.ndim == 4:
                    reduced = value.grad.detach().float().sum(dim=spatial)
                    other_reduced = other.outputs[name].grad.detach().float().sum(dim=spatial)
                    magnitude = value.grad.detach().float().abs().sum(dim=spatial).clamp_min(1e-30)
                    report[name]["channel_sum_gap"] = float((reduced - other_reduced).abs().max())
                    report[name]["cancellation_ratio_max"] = float((reduced.abs() / magnitude).max())
        return report


def run_semamba() -> None:
    os.environ["AUDIO_DB_SEMAMBA_SCAN"] = "fused"
    from tests.numerics.reference import compare_semamba_step as sem
    from model.semambapp import SEMambapp

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        _write("semamba_fairness.json", {"status": "NOT_RUN", "reason": "cuda unavailable", "environment": environment(device)})
        return
    started = time.time()
    report: dict = {
        "environment": environment(device),
        "null_space_probe": null_space_probe(device),
        "init_seed": 7,
        "data_seed": 23,
        "note": (
            "Historical O/A weights were not saved. Data uses the trace seed 23 after init. "
            "Init seed 7 makes this process reproducible. All arms share these tensors."
        ),
    }
    _write("semamba_fairness.json", report)
    torch.manual_seed(7)
    cfg, official_g, audio_g, official_cqt, audio_cqt, official_mrd, audio_mrd = sem._pair(device, 48, 4, 16, 4)
    official_g2 = SEMambapp(cfg).to(device)
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
        "official": {
            "g": (official_g, official_g2),
            "cqt": (official_cqt, official_cqt2),
            "mrd": (official_mrd, official_mrd2),
            "state": {
                "g": _cpu_state(official_g),
                "cqt": _cpu_state(official_cqt),
                "mrd": _cpu_state(official_mrd),
            },
        },
        "audio": {
            "g": (audio_g, audio_g2),
            "cqt": (audio_cqt, audio_cqt2),
            "mrd": (audio_mrd, audio_mrd2),
            "state": {
                "g": _cpu_state(audio_g),
                "cqt": _cpu_state(audio_cqt),
                "mrd": _cpu_state(audio_mrd),
            },
        },
    }
    null_names = _null_bias_names(official_g)
    report["null_bias_names"] = null_names
    report["known_bias_present"] = {
        "mask_decoder.mask_conv.1.bias": "mask_decoder.mask_conv.1.bias" in null_names,
        "dense_encoder.dense_conv_1.0.bias": "dense_encoder.dense_conv_1.0.bias" in null_names,
    }
    torch.manual_seed(23)
    clean = torch.randn(1, 8000, device=device)
    noisy = clean * 0.6 + 0.05 * torch.randn_like(clean)
    clean = clean.detach()
    noisy = noisy.detach()
    mel = sem.MultiScaleMelSpectrogramLoss(sampling_rate=16000).to(device)
    interest = [
        "mask_decoder.mask_conv.1.bias",
        "mask_decoder.mask_conv.1.weight",
        "dense_encoder.dense_conv_1.0.bias",
        "dense_encoder.dense_conv_1.0.weight",
    ]

    def forward(kind: str, generator, waveform_clean, waveform_noisy):
        if kind == "official":
            return sem._official_parts(generator, waveform_clean, waveform_noisy, cfg)
        return sem._independent_parts(generator, waveform_clean, waveform_noisy)

    def losses_of(kind: str, parts, cqt, mrd):
        if kind == "official":
            return sem._official_losses(parts, clean, cqt, mrd, mel, cfg)
        return sem._independent_losses(parts, clean, cqt, mrd)

    def prepare(kind: str, slot: int):
        bundle = bank[kind]
        generator = bundle["g"][slot]
        cqt = bundle["cqt"][slot]
        mrd = bundle["mrd"][slot]
        _load_state(generator, bundle["state"]["g"])
        _load_state(cqt, bundle["state"]["cqt"])
        _load_state(mrd, bundle["state"]["mrd"])
        opt_g = torch.optim.AdamW(generator.parameters(), lr=sem.LR, betas=sem.BETAS, weight_decay=sem.WEIGHT_DECAY)
        opt_d = torch.optim.AdamW(
            list(cqt.parameters()) + list(mrd.parameters()),
            lr=sem.LR,
            betas=sem.BETAS,
            weight_decay=sem.WEIGHT_DECAY,
        )
        return generator, cqt, mrd, opt_g, opt_d

    def one_step(kind_l, mods_l, opt_l, kind_r, mods_r, opt_r, *, terms: bool, activations: bool):
        (g_l, cqt_l, mrd_l), (opt_g_l, opt_d_l) = mods_l, opt_l
        (g_r, cqt_r, mrd_r), (opt_g_r, opt_d_r) = mods_r, opt_r
        for module in (g_l, cqt_l, mrd_l, g_r, cqt_r, mrd_r):
            module.train()
        for optim in (opt_g_l, opt_d_l, opt_g_r, opt_d_r):
            optim.zero_grad(set_to_none=True)
        parts_l = forward(kind_l, g_l, clean, noisy)
        parts_r = forward(kind_r, g_r, clean, noisy)
        detached_l = losses_of(kind_l, {**parts_l, "audio": parts_l["audio"].detach()}, cqt_l, mrd_l)["discriminator"]
        detached_r = losses_of(kind_r, {**parts_r, "audio": parts_r["audio"].detach()}, cqt_r, mrd_r)["discriminator"]
        detached_l.backward()
        detached_r.backward()
        discriminator_grad = max(
            _grad_stats(cqt_l, cqt_r, set())["max_abs"],
            _grad_stats(mrd_l, mrd_r, set())["max_abs"],
        )
        opt_d_l.step()
        opt_d_r.step()
        opt_g_l.zero_grad(set_to_none=True)
        opt_g_r.zero_grad(set_to_none=True)
        retain_l = _Retain()
        retain_r = _Retain()
        if activations:
            for index, layer in enumerate(g_l.mask_decoder.mask_conv):
                retain_l.hook(layer, f"mask_conv.{index}")
            for index, layer in enumerate(g_r.mask_decoder.mask_conv):
                retain_r.hook(layer, f"mask_conv.{index}")
            retain_l.hook(g_l.dense_encoder.dense_conv_1[0], "dense_conv_1.conv")
            retain_r.hook(g_r.dense_encoder.dense_conv_1[0], "dense_conv_1.conv")
        parts_l = forward(kind_l, g_l, clean, noisy)
        parts_r = forward(kind_r, g_r, clean, noisy)
        loss_l = losses_of(kind_l, parts_l, cqt_l, mrd_l)
        loss_r = losses_of(kind_r, parts_r, cqt_r, mrd_r)
        term_report = None
        if terms:
            wanted = [name for name in interest if name.endswith("bias")]
            params_l = [(name, dict(g_l.named_parameters())[name]) for name in wanted]
            params_r = [(name, dict(g_r.named_parameters())[name]) for name in wanted]
            term_report = {"left": _term_grads(loss_l, params_l), "right": _term_grads(loss_r, params_r)}
            for tensor in list(retain_l.outputs.values()) + list(retain_r.outputs.values()):
                tensor.grad = None
        loss_l["generator"].backward()
        loss_r["generator"].backward()
        activation_gap = retain_l.gaps(retain_r) if activations else None
        retain_l.close()
        retain_r.close()
        left_losses = _loss_map(loss_l)
        right_losses = _loss_map(loss_r)
        left_losses["discriminator"] = float(detached_l.detach())
        right_losses["discriminator"] = float(detached_r.detach())
        row = {
            "loss_gap": {name: abs(left_losses[name] - right_losses[name]) for name in left_losses if name in right_losses},
            "left_loss": left_losses,
            "forward": {
                "wave": _tensor_gap(parts_l["audio"], parts_r["audio"]),
                "mag": _tensor_gap(parts_l["mag"], parts_r["mag"]),
                "phase": _tensor_gap(parts_l["phase"], parts_r["phase"]),
                "complex": _tensor_gap(parts_l["complex"], parts_r["complex"]),
            },
            "generator_grad": _grad_stats(g_l, g_r, set(null_names)),
            "focus_bias_grads": _bias_values(
                g_l,
                g_r,
                ["mask_decoder.mask_conv.1.bias", "dense_encoder.dense_conv_1.0.bias"],
            ),
            "discriminator_grad_max_abs": discriminator_grad,
        }
        fresh = len(opt_g_l.state) == 0
        if fresh:
            row["formula"] = _first_adam_gap(g_l, g_r, sem.LR, opt_g_l.defaults["eps"], sem.WEIGHT_DECAY)
        if term_report is not None:
            row["loss_term_grads"] = term_report
        if activation_gap is not None:
            row["activation_gaps"] = activation_gap
        opt_g_l.step()
        opt_g_r.step()
        row["weight"] = _weight_stats(g_l, g_r, set(null_names))
        row["optimizer"] = _optim_gap(g_l, opt_g_l, g_r, opt_g_r)
        if fresh:
            row["weight_minus_formula"] = abs(row["weight"]["max_abs"] - row["formula"]["formula_weight_max_abs"])
        return row

    def run_pair(kind_l: str, kind_r: str, steps: int, *, resync: bool, terms: bool, key: str) -> None:
        g_l, cqt_l, mrd_l, opt_g_l, opt_d_l = prepare(kind_l, 0)
        g_r, cqt_r, mrd_r, opt_g_r, opt_d_r = prepare(kind_r, 1)
        report[key] = []
        for index in range(steps):
            print(f"semamba {kind_l}/{kind_r} resync={resync} step {index}", flush=True)
            row = one_step(
                kind_l,
                (g_l, cqt_l, mrd_l),
                (opt_g_l, opt_d_l),
                kind_r,
                (g_r, cqt_r, mrd_r),
                (opt_g_r, opt_d_r),
                terms=terms and index == 0,
                activations=terms and index == 0,
            )
            row["step"] = index
            report[key].append(row)
            _write("semamba_fairness.json", report)
            if resync:
                for source, dest in zip((g_l, cqt_l, mrd_l), (g_r, cqt_r, mrd_r), strict=True):
                    dest.load_state_dict(source.state_dict())
                _copy_optim(g_l, opt_g_l, g_r, opt_g_r)
                _copy_optim(cqt_l, opt_d_l, cqt_r, opt_d_r)
                _copy_optim(mrd_l, opt_d_l, mrd_r, opt_d_r)

    run_pair("official", "audio", 3, resync=False, terms=True, key="oa_trace")
    run_pair("official", "audio", 3, resync=True, terms=False, key="oa_resync")
    run_pair("official", "official", 3, resync=False, terms=False, key="oo")
    run_pair("audio", "audio", 3, resync=False, terms=False, key="aa")
    report["elapsed_sec"] = time.time() - started
    _write("semamba_fairness.json", report)


def _compress(spectrum: torch.Tensor, compress_factor: float):
    real = torch.view_as_real(spectrum)
    magnitude = torch.sqrt(real.pow(2).sum(-1) + 1e-9)
    phase = torch.atan2(real[..., 1] + 1e-10, real[..., 0] + 1e-5)
    magnitude = magnitude.pow(compress_factor)
    compressed = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    return magnitude, phase, compressed


def _native_stft(waveform, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    window = torch.hann_window(win_size, dtype=waveform.dtype).to(waveform.device)
    spectrum = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    return _compress(spectrum, compress_factor)


def _index_stft(waveform, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    from src.experiments.calc.stft import _reflect_pad

    window = torch.hann_window(win_size, dtype=waveform.dtype).to(waveform.device)
    framed = _reflect_pad(waveform, n_fft // 2) if center else waveform
    spectrum = torch.stft(
        framed,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=False,
        normalized=False,
        return_complex=True,
    )
    return _compress(spectrum, compress_factor)


def _plain_istft(magnitude, phase, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    magnitude = magnitude.pow(1.0 / compress_factor)
    spectrum = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    window = torch.hann_window(win_size, dtype=spectrum.real.dtype).to(spectrum.device)
    return torch.istft(spectrum, n_fft, hop_length=hop_size, win_length=win_size, window=window, center=center)


def _stft_backward_gap(left_fn, right_fn, waveform: torch.Tensor) -> dict:
    def once(fn):
        wave = waveform.detach().clone().requires_grad_(True)
        mag, _phase, _compressed = fn(wave, 400, 100, 400, 0.3)
        mag.sum().backward()
        return float(mag.detach().sum()), wave.grad.detach().clone()

    left_loss, left_grad = once(left_fn)
    right_loss, right_grad = once(right_fn)
    again_loss, again_grad = once(left_fn)
    return {
        "cross_loss_gap": abs(left_loss - right_loss),
        "cross_grad_max_abs": float((left_grad - right_grad).abs().max()),
        "self_loss_gap": abs(left_loss - again_loss),
        "self_grad_max_abs": float((left_grad - again_grad).abs().max()),
    }


def run_mp(profile: str = "full") -> None:
    out_name = "mp_fairness.json" if profile == "full" else "mp_followup.json"
    from tests.numerics.reference import compare_mp_senet as cmp
    from models.discriminator import MetricDiscriminator as OfficialDiscriminator
    from models.model import MPNet

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        _write(out_name, {"status": "NOT_RUN", "reason": "cuda unavailable", "environment": environment(device)})
        return
    started = time.time()
    h = cmp._hyper()
    original = {
        "official_stft": cmp.official_stft,
        "official_istft": cmp.official_istft,
        "mag_pha_stft": cmp.mag_pha_stft,
        "mag_pha_istft": cmp.mag_pha_istft,
    }
    report: dict = {"environment": environment(device), "null_space_probe": null_space_probe(device), "data_seed": 11, "init_seed": 7}
    torch.manual_seed(11)
    clean = torch.randn(2, 8000, device=device)
    noisy = clean * 0.5 + 0.05
    clean = clean.detach()
    noisy = noisy.detach()
    report["stft_probe_on_training_waveform"] = {
        "official_versus_audiodb": _stft_backward_gap(original["official_stft"], original["mag_pha_stft"], clean),
        "official_versus_index": _stft_backward_gap(original["official_stft"], _index_stft, clean),
        "native_explicit_versus_index": _stft_backward_gap(_native_stft, _index_stft, clean),
    }
    _write(out_name, report)
    audio_g, audio_d, official_g, official_d = cmp._build(h, device)
    official_g2 = MPNet(h, num_tsblocks=h.num_tsconformers).to(device)
    official_d2 = OfficialDiscriminator().to(device)
    official_g2.load_state_dict(official_g.state_dict(), strict=True)
    official_d2.load_state_dict(official_d.state_dict(), strict=True)
    audio_g2 = cmp.build_model("mp_senet", vars(h)).to(device)
    audio_d2 = cmp.MetricDiscriminator().to(device)
    audio_g2.load_state_dict(audio_g.state_dict(), strict=True)
    audio_d2.load_state_dict(audio_d.state_dict(), strict=True)
    bank = {
        "official": {
            "g": (official_g, official_g2),
            "d": (official_d, official_d2),
            "state_g": _cpu_state(official_g),
            "state_d": _cpu_state(official_d),
        },
        "audio": {
            "g": (audio_g, audio_g2),
            "d": (audio_d, audio_d2),
            "state_g": _cpu_state(audio_g),
            "state_d": _cpu_state(audio_d),
        },
    }
    null_names = _null_bias_names(official_g)
    report["null_bias_names"] = null_names
    report["known_bias_present"] = {"dense_encoder.dense_conv_1.0.bias": "dense_encoder.dense_conv_1.0.bias" in null_names}

    def install(mode: str) -> None:
        if mode == "det_force_native":
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(False)
        if mode == "native":
            cmp.official_stft = original["official_stft"]
            cmp.official_istft = original["official_istft"]
            cmp.mag_pha_stft = original["mag_pha_stft"]
            cmp.mag_pha_istft = original["mag_pha_istft"]
        elif mode == "both_official_stft":
            cmp.official_stft = original["official_stft"]
            cmp.official_istft = original["official_istft"]
            cmp.mag_pha_stft = original["official_stft"]
            cmp.mag_pha_istft = original["official_istft"]
        elif mode == "both_index_stft":
            cmp.official_stft = _index_stft
            cmp.mag_pha_stft = _index_stft
            cmp.official_istft = _plain_istft
            cmp.mag_pha_istft = _plain_istft
        elif mode == "det_force_native":
            cmp.official_stft = _native_stft
            cmp.mag_pha_stft = _native_stft
            cmp.official_istft = _plain_istft
            cmp.mag_pha_istft = _plain_istft
        else:
            raise ValueError(mode)

    def parts(kind: str, generator):
        stft = cmp.official_stft if kind == "official" else cmp.mag_pha_stft
        istft = cmp.official_istft if kind == "official" else cmp.mag_pha_istft
        snap = cmp.rng_snapshot()
        value = cmp._spectra(generator, clean, noisy, h, stft, istft)
        cmp.rng_restore(snap)
        return value

    def discriminator_objective(kind: str, discriminator, spectrum):
        ones = torch.ones(clean.shape[0], device=clean.device)
        snap = cmp.rng_snapshot()
        real = discriminator(spectrum["clean_mag"], spectrum["clean_mag"])
        fake = discriminator(spectrum["clean_mag"], spectrum["hat_mag"].detach())
        cmp.rng_restore(snap)
        if kind == "official":
            target = cmp._official_pesq(clean, spectrum["audio"])
            loss = cmp._official_discriminator_loss(real, fake, ones, target)
        else:
            target = cmp.metric_targets(clean, spectrum["audio"].detach(), h.sampling_rate)
            loss = cmp.discriminator_loss(real, fake, ones, target)
        return loss, ones

    def generator_objective(kind: str, discriminator, spectrum, ones):
        snap = cmp.rng_snapshot()
        metric = discriminator(spectrum["clean_mag"], spectrum["hat_mag"])
        cmp.rng_restore(snap)
        if kind == "official":
            loss = cmp._official_components(spectrum, metric, ones, clean)
        else:
            loss = cmp._components(spectrum, metric, ones, clean)
        return loss

    def prepare(kind: str, slot: int):
        bundle = bank[kind]
        generator = bundle["g"][slot]
        discriminator = bundle["d"][slot]
        _load_state(generator, bundle["state_g"])
        _load_state(discriminator, bundle["state_d"])
        opt_g = torch.optim.AdamW(generator.parameters(), lr=cmp.LR, betas=cmp.BETAS, weight_decay=cmp.WEIGHT_DECAY)
        opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cmp.LR, betas=cmp.BETAS, weight_decay=cmp.WEIGHT_DECAY)
        return generator, discriminator, opt_g, opt_d

    def run_pair(kind_l: str, kind_r: str, steps: int, *, resync: bool, bucket: dict, key: str) -> None:
        # Same kind uses the two clones. A cross pair uses slot 0 of each bank.
        g_l, d_l, opt_g_l, opt_d_l = prepare(kind_l, 0)
        g_r, d_r, opt_g_r, opt_d_r = prepare(kind_r, 1 if kind_l == kind_r else 0)
        bucket[key] = []
        for index in range(steps):
            print(f"mp {kind_l}/{kind_r} resync={resync} step {index + 1}", flush=True)
            for module in (g_l, d_l, g_r, d_r):
                module.train()
            for optim in (opt_g_l, opt_d_l, opt_g_r, opt_d_r):
                optim.zero_grad(set_to_none=True)
            # One generator forward feeds both updates, matching compare_mp_senet.gan_step.
            spec_l = parts(kind_l, g_l)
            spec_r = parts(kind_r, g_r)
            spec_l["audio"].retain_grad()
            spec_r["audio"].retain_grad()
            loss_d_l, ones_l = discriminator_objective(kind_l, d_l, spec_l)
            loss_d_r, ones_r = discriminator_objective(kind_r, d_r, spec_r)
            loss_d_l.backward()
            loss_d_r.backward()
            disc_grad = _grad_stats(d_l, d_r, set())
            opt_d_l.step()
            opt_d_r.step()
            opt_g_l.zero_grad(set_to_none=True)
            opt_g_r.zero_grad(set_to_none=True)
            fresh = len(opt_g_l.state) == 0
            loss_g_l = generator_objective(kind_l, d_l, spec_l, ones_l)
            loss_g_r = generator_objective(kind_r, d_r, spec_r, ones_r)
            loss_g_l["total"].backward()
            loss_g_r["total"].backward()
            row = {
                "step": index + 1,
                "loss_gap": {name: abs(float(loss_g_l[name].detach()) - float(loss_g_r[name].detach())) for name in cmp.LOSS_NAMES},
                "left_loss": {name: float(loss_g_l[name].detach()) for name in cmp.LOSS_NAMES},
                "discriminator_loss_gap": abs(float(loss_d_l.detach()) - float(loss_d_r.detach())),
                "forward": {
                    "wave": _tensor_gap(spec_l["audio"], spec_r["audio"]),
                    "stft": _tensor_gap(spec_l["noisy_mag"], spec_r["noisy_mag"]),
                    "magnitude": _tensor_gap(spec_l["magnitude"], spec_r["magnitude"]),
                    "phase": _tensor_gap(spec_l["phase"], spec_r["phase"]),
                },
                "waveform_grad": _tensor_gap(spec_l["audio"].grad, spec_r["audio"].grad),
                "generator_grad": _grad_stats(g_l, g_r, set(null_names)),
                "discriminator_grad": disc_grad,
                "focused_bias": _bias_values(g_l, g_r, ["dense_encoder.dense_conv_1.0.bias"]),
            }
            if fresh:
                row["formula"] = _first_adam_gap(g_l, g_r, cmp.LR, opt_g_l.defaults["eps"], cmp.WEIGHT_DECAY)
            opt_g_l.step()
            opt_g_r.step()
            row["weight"] = _weight_stats(g_l, g_r, set(null_names))
            row["optimizer"] = _optim_gap(g_l, opt_g_l, g_r, opt_g_r)
            row["discriminator_weight"] = _weight_stats(d_l, d_r, set())
            if fresh:
                row["weight_minus_formula"] = abs(row["weight"]["max_abs"] - row["formula"]["formula_weight_max_abs"])
            bucket[key].append(row)
            _write(out_name, report)
            if resync:
                _sync((g_l, d_l), (opt_g_l, opt_d_l), (g_r, d_r), (opt_g_r, opt_d_r))

    if profile == "followup":
        # Name-based optimizer sync. The first full run's resync copied Adam
        # moments by parameter index and is not used.
        report["modes"] = {}
        install("native")
        entry = {"deterministic_algorithms": False, "note": "resync copies Adam state by parameter name"}
        report["modes"]["native"] = entry
        run_pair("official", "audio", 3, resync=True, bucket=entry, key="oa_resync")
        install("det_force_native")
        entry = {"deterministic_algorithms": True}
        report["modes"]["det_force_native"] = entry
        run_pair("audio", "audio", 3, resync=False, bucket=entry, key="aa")
        run_pair("official", "audio", 3, resync=True, bucket=entry, key="oa_resync")
        report["elapsed_sec"] = time.time() - started
        _write(out_name, report)
        return
    modes = ["native", "both_official_stft", "both_index_stft", "det_force_native"]
    report["modes"] = {}
    for mode in modes:
        print("mp mode", mode, flush=True)
        install(mode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = {"deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())}
            report["modes"][mode] = entry
            run_pair("official", "official", 3, resync=False, bucket=entry, key="oo")
            run_pair("official", "audio", 3, resync=False, bucket=entry, key="oa")
            run_pair("official", "audio", 3, resync=True, bucket=entry, key="oa_resync")
            if mode == "native":
                run_pair("audio", "audio", 3, resync=False, bucket=entry, key="aa")
            entry["deterministic_warnings"] = sorted(
                {str(item.message) for item in caught if "deterministic" in str(item.message).lower()}
            )
        report["modes"][mode] = entry
        report["environment"]["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        _write(out_name, report)
    report["elapsed_sec"] = time.time() - started
    _write(out_name, report)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"semamba", "mp", "mp-followup"}:
        raise SystemExit("usage: two_model_fairness.py semamba|mp|mp-followup")
    if sys.argv[1] == "semamba":
        run_semamba()
    elif sys.argv[1] == "mp-followup":
        run_mp("followup")
    else:
        run_mp()


if __name__ == "__main__":
    main()
