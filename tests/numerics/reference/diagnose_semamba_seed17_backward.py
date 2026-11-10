"""Locate the first SEMamba++ backward mismatch on the saved seed-17 probe.

The probe input and initial weights are the official-width fused run
``semamba-epoch-seed-17``. Each run reloads those weights and repeats one
generator update backward without stepping the optimizer. selective_scan CUDA
calls are recorded in forward and backward order.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.numerics.four_process.compare import seed_probe  # noqa: E402
from tests.numerics.four_process.real_models import (  # noqa: E402
    _semamba_generator,
    _semamba_parts,
)


SOURCE = ROOT / "data/results/training/followup-20261102-voicebank/semamba-epoch-seed-17"
OUTPUT = ROOT / "data/results/training/seed17-backward-20261103-named"


def _pair(a, b):
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "mismatch": "shape"}
    diff = (a.double() - b.double()).abs()
    denom = a.double().reshape(-1).norm().clamp_min(1e-30)
    flat_a = a.double().reshape(-1)
    flat_b = b.double().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0)
    return {
        "max_abs": float(diff.max()),
        "rel_l2": float(diff.reshape(-1).norm() / denom),
        "cosine": float(cosine),
        "numel": int(a.numel()),
        "different": int((a != b).sum()),
    }


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    return value


class ScanTrace:
    def __init__(self):
        self.fwd = []
        self.bwd = []
        self.stacks = []
        self._installed = False
        self._orig_fwd = None
        self._orig_bwd = None

    def install(self):
        import selective_scan_cuda

        self._orig_fwd = selective_scan_cuda.fwd
        self._orig_bwd = selective_scan_cuda.bwd

        def fwd(*args, **kwargs):
            if len(self.stacks) < 2:
                frames = []
                for frame in traceback.extract_stack()[:-1]:
                    if "selective_scan" in frame.filename or "audio-db" in frame.filename or "mamba_ssm" in frame.filename:
                        frames.append(f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}")
                self.stacks.append(frames[-8:])
            out = self._orig_fwd(*args, **kwargs)
            self.fwd.append({
                "inputs": {
                    "u": _cpu(args[0]),
                    "delta": _cpu(args[1]),
                    "A": _cpu(args[2]),
                    "B": _cpu(args[3]),
                    "C": _cpu(args[4]),
                    "D": _cpu(args[5]),
                    "z": _cpu(args[6]),
                },
                "out": _cpu(out[0]),
                "shape": list(args[0].shape),
                "dtype": str(args[0].dtype).replace("torch.", ""),
            })
            return out

        def bwd(*args, **kwargs):
            out = self._orig_bwd(*args, **kwargs)
            names = ("du", "ddelta", "dA", "dB", "dC", "dD", "ddelta_bias", "dz", "out_z")
            saved = {
                "dout": _cpu(args[8]),
                "u": _cpu(args[0]),
                "delta": _cpu(args[1]),
                "A": _cpu(args[2]),
                "B": _cpu(args[3]),
                "C": _cpu(args[4]),
                "D": _cpu(args[5]),
                "z": _cpu(args[6]),
                "delta_bias": _cpu(args[7]),
            }
            grads = {}
            for name, value in zip(names, out):
                if torch.is_tensor(value):
                    grads[name] = _cpu(value)
            self.bwd.append({"inputs": saved, "grads": grads, "shape": list(args[0].shape)})
            return out

        selective_scan_cuda.fwd = fwd
        selective_scan_cuda.bwd = bwd
        self._installed = True

    def remove(self):
        if not self._installed:
            return
        import selective_scan_cuda

        selective_scan_cuda.fwd = self._orig_fwd
        selective_scan_cuda.bwd = self._orig_bwd
        self._installed = False

    def clear(self):
        self.fwd.clear()
        self.bwd.clear()
        self.stacks.clear()


def _first_mismatch(left, right, section):
    found = []
    for index, (a, b) in enumerate(zip(left, right)):
        tensors = a["inputs"] if section == "forward-input" else a.get("grads", a["inputs"])
        other = b["inputs"] if section == "forward-input" else b.get("grads", b["inputs"])
        keys = ("dout", "u", "delta", "A", "B", "C", "D", "z") if section != "backward-grad" else (
            "du", "ddelta", "dA", "dB", "dC", "dD", "ddelta_bias", "dz", "out_z"
        )
        if section == "forward-output":
            stat = _pair(a["out"], b["out"])
            if stat.get("different", 1):
                found.append({"call": index, "tensor": "out", **stat})
                break
            continue
        for key in keys:
            if key not in tensors or key not in other:
                continue
            stat = _pair(tensors[key], other[key])
            if stat.get("different", 1):
                found.append({"call": index, "tensor": key, **stat})
                return found
    return found


def _named_grads(module):
    grads = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            grads[name] = parameter.grad.detach().cpu()
    return grads


def _named_report(left, right):
    shared = sorted(set(left) & set(right))
    worst = None
    rels = []
    for name in shared:
        stat = _pair(left[name], right[name])
        rels.append(stat["rel_l2"])
        if worst is None or stat["max_abs"] > worst["max_abs"]:
            worst = {"name": name, **stat}
    return {
        "shared": len(shared),
        "only_left": sorted(set(left) - set(right))[:8],
        "only_right": sorted(set(right) - set(left))[:8],
        "worst": worst,
        "median_rel_l2": float(torch.tensor(rels).median()) if rels else None,
        "max_rel_l2": max(rels) if rels else None,
    }


def _call_rows(left, right):
    rows = []
    for index, (a, b) in enumerate(zip(left, right)):
        row = {"call": index, "shape": a.get("shape")}
        for key, store in (("dout", "inputs"), ("du", "grads"), ("dB", "grads"), ("dC", "grads")):
            if key not in a[store] or key not in b[store]:
                continue
            stat = _pair(a[store][key], b[store][key])
            row[key] = {"max_abs": stat["max_abs"], "rel_l2": stat["rel_l2"], "different": stat["different"]}
        rows.append(row)
    return rows


def _run(implementation, payload, cfg, trace):
    from tests.numerics.four_process.real_models import SEMambaPP

    adapter = SEMambaPP()
    adapter.configure(cfg)
    seed_probe(cfg["seed"], True)
    device = torch.device("cuda:0")
    modules = adapter.build(implementation, device)
    for name, module in modules.items():
        module.load_state_dict(payload["initial"][name], strict=True)
    seed_probe(cfg["seed"], True)
    for module in modules.values():
        module.train()
    clean = payload["clean"].to(device)
    noisy = payload["noisy"].to(device)
    trace.clear()
    parts = _semamba_parts(implementation, modules["generator"], clean, noisy)
    losses = _semamba_generator(
        implementation, modules["discriminator_band"], modules["discriminator_mrd"], clean, parts, adapter._mel
    )
    audio = parts["audio"]
    audio.retain_grad()
    losses["generator"].backward()
    torch.cuda.synchronize()
    summary = {
        "implementation": implementation,
        "forward_calls": len(trace.fwd),
        "backward_calls": len(trace.bwd),
        "shapes": [call["shape"] for call in trace.fwd],
        "loss": float(losses["generator"].detach()),
        "audio_grad": audio.grad.detach().cpu(),
        "audio": audio.detach().cpu(),
        "named_grad": _named_grads(modules["generator"]),
        "stack": list(trace.stacks[:1]),
        "fwd": list(trace.fwd),
        "bwd": list(trace.bwd),
    }
    del modules
    torch.cuda.empty_cache()
    return summary


def _scan_args(tensors):
    bias = tensors["delta_bias"]
    return (
        tensors["u"], tensors["delta"], tensors["A"], tensors["B"], tensors["C"], tensors["D"], tensors["z"],
        None if bias is None else bias, True,
    )


def _replay(sample, repeats):
    import selective_scan_cuda

    moved = {key: value.cuda() if torch.is_tensor(value) else value for key, value in sample["inputs"].items()}
    dout = moved["dout"]
    runs = []
    for _ in range(repeats):
        out, intermediates, *rest = selective_scan_cuda.fwd(*_scan_args(moved))
        grads = selective_scan_cuda.bwd(
            moved["u"], moved["delta"], moved["A"], moved["B"], moved["C"], moved["D"], moved["z"],
            None if moved["delta_bias"] is None else moved["delta_bias"],
            dout, intermediates, out, None, True, False,
        )
        torch.cuda.synchronize()
        names = ("du", "ddelta", "dA", "dB", "dC", "dD", "ddelta_bias", "dz", "out_z")
        runs.append({name: value.detach().cpu() for name, value in zip(names, grads) if torch.is_tensor(value)})
    report = {}
    for name in runs[0]:
        report[name] = [_pair(runs[0][name], run[name]) for run in runs[1:]]
    report["shape"] = list(sample["inputs"]["u"].shape)
    report["dtype"] = str(sample["inputs"]["u"].dtype).replace("torch.", "")
    return report


def _variant(tensors, length=None, batch=None):
    sequence = tensors["u"].shape[-1]
    out = {}
    for key, value in tensors.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        current = value
        if batch is not None and current.dim() >= 3 and current.shape[0] > batch:
            current = current[:batch].contiguous()
        if length is not None and current.dim() >= 1 and current.shape[-1] == sequence and sequence > length:
            current = current[..., :length].contiguous()
        out[key] = current
    return out


def _shorten(sample):
    """Repeat the captured activation at a shorter length and a single batch row."""
    import selective_scan_cuda

    variants = []
    full = {key: value.cuda() if torch.is_tensor(value) else value for key, value in sample["inputs"].items()}
    specs = {
        "full": full,
        "length_16": _variant(full, length=16),
        "batch_1": _variant(full, batch=1),
    }
    for name, tensors in specs.items():
        dout = torch.randn_like(tensors["u"])
        runs = []
        for _ in range(3):
            out, intermediates, *rest = selective_scan_cuda.fwd(*_scan_args(tensors))
            bias = tensors["delta_bias"]
            grads = selective_scan_cuda.bwd(
                tensors["u"], tensors["delta"], tensors["A"], tensors["B"], tensors["C"], tensors["D"], tensors["z"],
                None if bias is None else bias, dout, intermediates, out, None, True, False,
            )
            torch.cuda.synchronize()
            runs.append({"dB": grads[3].detach().cpu(), "dC": grads[4].detach().cpu()})
        variants.append({
            "name": name,
            "shape": list(tensors["u"].shape),
            "dB_repeat": _pair(runs[0]["dB"], runs[1]["dB"]),
            "dC_repeat": _pair(runs[0]["dC"], runs[1]["dC"]),
        })
    return variants


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload = torch.load(SOURCE / "input.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((SOURCE / "config.json").read_text())
    trace = ScanTrace()
    trace.install()
    try:
        official_a = _run("official", payload, cfg, trace)
        official_b = _run("official", payload, cfg, trace)
        independent = _run("independent", payload, cfg, trace)
        stacks = trace.stacks
    finally:
        trace.remove()

    def bundle(left, right):
        return {
            "forward_input": _first_mismatch(left["fwd"], right["fwd"], "forward-input"),
            "forward_output": _first_mismatch(left["fwd"], right["fwd"], "forward-output"),
            "backward_input": _first_mismatch(left["bwd"], right["bwd"], "forward-input"),
            "backward_grad": _first_mismatch(left["bwd"], right["bwd"], "backward-grad"),
            "audio": _pair(left["audio"], right["audio"]),
            "audio_grad": _pair(left["audio_grad"], right["audio_grad"]),
            "named_grad": _named_report(left["named_grad"], right["named_grad"]),
            "scan_calls": _call_rows(left["bwd"], right["bwd"]),
            "loss_abs": abs(left["loss"] - right["loss"]),
            "forward_calls": [left["forward_calls"], right["forward_calls"]],
            "backward_calls": [left["backward_calls"], right["backward_calls"]],
        }

    first_bwd = official_a["bwd"][0]
    replay_sample = {"inputs": first_bwd["inputs"]}
    # The saved backward input names u/delta/A/B/C are the scan tensors.
    report = {
        "source": str(SOURCE),
        "seed": cfg["seed"],
        "official_width": True,
        "scan_backend": "fused",
        "hypothesis": "selective_scan_cuda.bwd is the first seed-17 backward mismatch",
        "call_stacks": {
            "official_a": official_a["stack"],
            "official_b": official_b["stack"],
            "independent": independent["stack"],
        },
        "shapes": official_a["shapes"],
        "official_a_vs_b": bundle(official_a, official_b),
        "official_a_vs_independent": bundle(official_a, independent),
        "replay_first_backward_call": _replay(replay_sample, 4),
        "shape_variants_dC": _shorten(replay_sample),
    }
    # Drop bulk tensors before writing. The report keeps only the first mismatch.
    destination = OUTPUT / "scan-diff.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "official_a_vs_b": report["official_a_vs_b"],
        "official_a_vs_independent": report["official_a_vs_independent"],
        "replay_dB": report["replay_first_backward_call"].get("dB"),
        "replay_dC": report["replay_first_backward_call"].get("dC"),
        "variants": report["shape_variants_dC"],
        "stacks": report["call_stacks"],
    }, indent=2))


if __name__ == "__main__":
    main()
