"""Same-checkpoint SEMamba++ backward, official repeats versus independent.

The criterion below is the decision rule for this script. It is fixed in
source before a run and is copied into the result JSON unchanged.

Same weights, same input, and the same RNG snapshot are restored before every
repeat. Optimizer state does not carry across repeats.

bias_rel_l2 is the relative L2 between the mean official gradient and the
mean independent gradient. self_median is the larger of the two
within-implementation median pairwise relative L2 values. cross_median is the
median official-versus-independent pairwise relative L2.

The backward matches kernel nondeterminism only when bias_rel_l2 <= self_median
and cross_median <= self_median. A larger cross median is an implementation
gap, not a license to widen this rule.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.experiments.calc.rng import capture_rng, restore_rng  # noqa: E402
from tests.numerics.four_process.compare import seed_probe  # noqa: E402
from tests.numerics.four_process.real_models import SEMambaPP  # noqa: E402

CRITERION = (
    "bias_rel_l2 <= self_median and cross_median <= self_median, "
    "with self_median the larger within-implementation median pairwise relative L2"
)
EPS = 1e-8


def _rel(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    delta = (a.double() - b.double()).reshape(-1)
    denom = a.double().reshape(-1).norm().clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(a.double().reshape(-1), b.double().reshape(-1), dim=0)
    signs = (a.reshape(-1) * b.reshape(-1)) < 0
    small = a.reshape(-1).abs() < EPS
    return {
        "max_abs": float((a - b).abs().max()),
        "rel_l2": float(delta.norm() / denom),
        "cosine": float(cosine),
        "sign_mismatches": int(signs.sum()),
        "below_eps": int(small.sum()),
        "below_eps_sign_mismatches": int((signs & small).sum()),
        "norm": float(a.double().reshape(-1).norm()),
    }


def _pack(grads: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.cat([grads[key].float().reshape(-1) for key in keys])


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _pairs(rows: list[torch.Tensor]) -> list[dict[str, float]]:
    return [_rel(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]


def _snapshot(modules):
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }


def _restore_modules(modules, state):
    for name, module in modules.items():
        module.load_state_dict(state[name], strict=True)


def _group(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "generator" and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


def _summarize(packs: dict[str, list[torch.Tensor]]) -> dict:
    official = packs["official"]
    independent = packs["independent"]
    self_official = _pairs(official)
    self_independent = _pairs(independent)
    cross = [_rel(left, right) for left in official for right in independent]
    mean_official = torch.stack([row.double() for row in official]).mean(0)
    mean_independent = torch.stack([row.double() for row in independent]).mean(0)
    bias = _rel(mean_official, mean_independent)
    self_median = max(
        _median([row["rel_l2"] for row in self_official]),
        _median([row["rel_l2"] for row in self_independent]),
    )
    cross_median = _median([row["rel_l2"] for row in cross])
    return {
        "official_pairs": self_official,
        "independent_pairs": self_independent,
        "cross_pairs": cross,
        "bias": bias,
        "self_median": self_median,
        "cross_median": cross_median,
        "passes_criterion": bool(bias["rel_l2"] <= self_median and cross_median <= self_median),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modules", type=Path, help="checkpoint whose module weights replace input.pt initial weights")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.repeats < 2:
        raise SystemExit("repeats must be at least 2")
    started = time.perf_counter()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    module_source = "input.pt initial"
    if args.modules is not None:
        checkpoint = torch.load(args.modules, map_location="cpu", weights_only=False)
        payload["initial"] = checkpoint["modules"]
        module_source = f"{args.modules} step {checkpoint.get('data_position')}"
    adapter = SEMambaPP()
    adapter.configure({"scan_backend": "fused", "scheduler_step": "epoch", "official_width": True})
    clean = payload["clean"].to(device)
    noisy = payload["noisy"].to(device)
    if clean.dim() == 1:
        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
    stored = {"official": [], "independent": []}
    grouped: dict[str, dict[str, list[torch.Tensor]]] = {}
    losses = {"official": [], "independent": []}
    forwards = {"official": [], "independent": []}
    keys = None
    for implementation in ("official", "independent"):
        seed_probe(17, False)
        modules = adapter.build(implementation, device)
        for name, module in modules.items():
            module.load_state_dict(payload["initial"][name], strict=True)
        optimizers = adapter.optimizers(modules, adapter.lr)
        schedulers = adapter.schedulers(optimizers)
        weights = _snapshot(modules)
        opt_state = {name: opt.state_dict() for name, opt in optimizers.items()}
        sched_state = {name: sched.state_dict() for name, sched in schedulers.items()}
        seed_probe(17, True)
        rng = capture_rng(cuda=True)
        for _ in range(args.repeats):
            _restore_modules(modules, weights)
            for name, opt in optimizers.items():
                opt.load_state_dict(opt_state[name])
            for name, sched in schedulers.items():
                sched.load_state_dict(sched_state[name])
            restore_rng(rng, cuda=True)
            step = adapter.train_step(implementation, modules, optimizers, schedulers, clean, noisy, 0)
            names = sorted(step.gradients)
            if keys is None:
                keys = names
            elif names != keys:
                raise RuntimeError("gradient names changed between repeats")
            stored[implementation].append(_pack(step.gradients, keys))
            losses[implementation].append({name: float(value) for name, value in step.losses.items()})
            forwards[implementation].append(step.forward.detach().float().cpu())
            pieces: dict[str, list[torch.Tensor]] = {}
            for key in keys:
                pieces.setdefault(_group(key), []).append(step.gradients[key].float().reshape(-1))
            for group, tensors in pieces.items():
                grouped.setdefault(group, {"official": [], "independent": []})
                grouped[group][implementation].append(torch.cat(tensors))
        del modules, optimizers, schedulers
        torch.cuda.empty_cache()
    report = {
        "criterion": CRITERION,
        "input": str(args.input),
        "modules": module_source,
        "repeats": args.repeats,
        "seed": 17,
        "scan_backend": "fused",
        "official_width": True,
        "scheduler_step": "epoch",
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "all_parameters": _summarize(stored),
        "groups": {name: _summarize(packs) for name, packs in sorted(grouped.items())},
        "losses": losses,
        "forward_pairs": {
            "official": _pairs(forwards["official"]),
            "independent": _pairs(forwards["independent"]),
            "cross": [_rel(left, right) for left in forwards["official"] for right in forwards["independent"]],
        },
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passes": report["all_parameters"]["passes_criterion"],
        "bias_rel_l2": report["all_parameters"]["bias"]["rel_l2"],
        "self_median": report["all_parameters"]["self_median"],
        "cross_median": report["all_parameters"]["cross_median"],
        "seconds": report["seconds"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
