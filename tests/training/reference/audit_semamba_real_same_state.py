"""Real-audio SEMamba++ same-state step and a four-update branch.

The decision text below is fixed before the run. Do not edit it after seeing
the output. A failed median comparison is recorded and is not, by itself, a
failure or a pass.

States are the official_a weights after 0, 4, and 20 AdamW updates of
final-accept-semamba-seed17-retry-20261103. Moments are replayed with
AdamW.step on the saved gradients. A state is usable only when that replay
matches the saved parameter tensors.

Each same-state repeat restores parameters, optimizer moments, the epoch
scheduler, and one shared RNG snapshot. The input is the next prepared
utterance when one exists (indices 0 and 4). After 20 updates the prepared
set is exhausted, so the probe repeats utterance 19 and labels that fact.

bias_rel_l2 is the relative L2 between the mean official gradient and the
mean independent gradient. self_median is the larger within-implementation
median pairwise relative L2. A systematic gap is bias_rel_l2 > self_median,
or a forward / loss-component mean absolute difference above the
within-implementation maximum. cross_median <= self_median is reported only.

Selective-scan forward inputs must match exactly. Scan backward is explained
by kernel rerun when every backward-output max abs between official and
independent is at most the official self-rerun max abs. Parameter backward
is explained when there is no systematic gap and the scan check passes.

The branch runs four further updates from the verified 4-update state on
two official and two independent copies. It is accumulation evidence, not a
20-step distribution test.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.core.rng import capture_rng, restore_rng  # noqa: E402
from tests.training.four_process.compare import seed_probe  # noqa: E402
from tests.training.four_process.real_models import SEMambaPP  # noqa: E402
from tests.training.four_process.shared_fork import remap_optimizer  # noqa: E402

CRITERION = (
    "systematic gap if bias_rel_l2 > self_median or a forward/loss mean absolute "
    "difference exceeds the within-implementation maximum; "
    "scan backward is kernel rerun only when forward inputs match exactly and "
    "each backward-output max abs is <= the official self-rerun max abs; "
    "cross_median <= self_median is recorded and decides neither GO nor NO-GO"
)
SOURCE = ROOT / "data/results/training/final-accept-semamba-seed17-retry-20261103"
REPEATS = 4


def _rel(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    delta = (a.double() - b.double()).reshape(-1)
    denom = a.double().reshape(-1).norm().clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(
        a.double().reshape(-1), b.double().reshape(-1), dim=0
    )
    return {
        "max_abs": float((a - b).abs().max()) if a.numel() else 0.0,
        "rel_l2": float(delta.norm() / denom) if a.numel() else 0.0,
        "cosine": float(cosine) if a.numel() else 1.0,
        "norm": float(a.double().reshape(-1).norm()) if a.numel() else 0.0,
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _pack(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    keys = sorted(grads)
    if not keys:
        return torch.zeros(1)
    return torch.cat([grads[key].float().reshape(-1) for key in keys])


def _params(modules) -> dict[str, torch.Tensor]:
    found = {}
    for prefix, module in modules.items():
        for name, value in module.named_parameters():
            found[f"{prefix}.{name}"] = value
    return found


def _param_max_abs(modules, saved) -> float:
    worst = 0.0
    for prefix, module in modules.items():
        current = module.state_dict()
        for name, value in module.named_parameters():
            worst = max(worst, float((current[name].detach().cpu() - saved[prefix][name]).abs().max()))
    return worst


def _assign(modules, grads: dict[str, torch.Tensor]) -> None:
    for prefix, module in modules.items():
        for name, value in module.named_parameters():
            grad = grads.get(f"{prefix}.{name}")
            value.grad = None if grad is None else grad.to(device=value.device, dtype=value.dtype)


def _clone_state(modules, optimizers, schedulers):
    return {
        "modules": {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in modules.items()
        },
        "optimizers": {name: copy.deepcopy(opt.state_dict()) for name, opt in optimizers.items()},
        "schedulers": {name: copy.deepcopy(sched.state_dict()) for name, sched in schedulers.items()},
    }


def _restore(modules, optimizers, schedulers, state, rng) -> None:
    for name, module in modules.items():
        module.load_state_dict(state["modules"][name], strict=True)
    for name, opt in optimizers.items():
        opt.load_state_dict(copy.deepcopy(state["optimizers"][name]))
    for name, sched in schedulers.items():
        sched.load_state_dict(copy.deepcopy(state["schedulers"][name]))
    restore_rng(rng, cuda=True)


def _remap_states(official_modules, independent_modules, optimizers):
    groups = (
        ("generator", ("generator",)),
        ("discriminator", ("discriminator_band", "discriminator_mrd")),
    )
    mapped = {}
    for opt_name, prefixes in groups:
        def names(models):
            return [
                prefix + "." + name
                for prefix in prefixes
                for name, _ in models[prefix].named_parameters()
            ]
        state = copy.deepcopy(optimizers[opt_name])
        remap_optimizer(state, names(official_modules), names(independent_modules))
        mapped[opt_name] = state
    return mapped


def _summarize(packs: dict[str, list[torch.Tensor]], forwards, losses):
    official = packs["official"]
    independent = packs["independent"]

    def pairs(rows):
        return [_rel(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]

    self_official = pairs(official)
    self_independent = pairs(independent)
    cross = [_rel(left, right) for left in official for right in independent]
    mean_official = torch.stack([row.double() for row in official]).mean(0)
    mean_independent = torch.stack([row.double() for row in independent]).mean(0)
    bias = _rel(mean_official, mean_independent)
    mean_forward_official = torch.stack([row.double() for row in forwards["official"]]).mean(0)
    mean_forward_independent = torch.stack([row.double() for row in forwards["independent"]]).mean(0)
    forward_bias = float((mean_forward_official - mean_forward_independent).abs().max())
    self_median = max(
        _median([row["rel_l2"] for row in self_official]),
        _median([row["rel_l2"] for row in self_independent]),
    )
    cross_median = _median([row["rel_l2"] for row in cross])
    loss_report = {}
    for key in losses["official"][0]:
        off = [float(row[key]) for row in losses["official"]]
        ind = [float(row[key]) for row in losses["independent"]]
        self_max = max(
            max(abs(a - b) for i, a in enumerate(off) for b in off[i + 1 :]),
            max(abs(a - b) for i, a in enumerate(ind) for b in ind[i + 1 :]),
        )
        bias_abs = abs(sum(off) / len(off) - sum(ind) / len(ind))
        loss_report[key] = {
            "self_max_abs": self_max,
            "bias_abs": bias_abs,
            "systematic": bool(bias_abs > self_max),
        }
    forward_pairs = {
        "self_official_max_abs": max(row["max_abs"] for row in pairs(forwards["official"])),
        "self_independent_max_abs": max(row["max_abs"] for row in pairs(forwards["independent"])),
        "cross": [_rel(left, right) for left in forwards["official"] for right in forwards["independent"]],
    }
    self_forward_max = max(
        forward_pairs["self_official_max_abs"], forward_pairs["self_independent_max_abs"]
    )
    systematic = bool(
        bias["rel_l2"] > self_median
        or forward_bias > self_forward_max
        or any(item["systematic"] for item in loss_report.values())
    )
    return {
        "bias": bias,
        "self_median": self_median,
        "cross_median": cross_median,
        "cross_over_self": None if self_median == 0 else cross_median / self_median,
        "median_boolean": bool(cross_median <= self_median),
        "systematic_gap": systematic,
        "official_pairs": self_official,
        "independent_pairs": self_independent,
        "cross_pairs": cross,
        "losses": loss_report,
        "forward_bias_abs": forward_bias,
        "forward_self_max_abs": self_forward_max,
        "forward_cross_max_abs": max(row["max_abs"] for row in forward_pairs["cross"]),
    }


def _replay(official, optimizers, prepared, checkpoint):
    _load = prepared["initial"]
    for name, module in official.items():
        module.load_state_dict(_load[name], strict=True)
    for opt in optimizers.values():
        opt.state.clear()
    # Rebuild moments from a fresh AdamW so replay does not keep a previous step count.
    fresh = SEMambaPP()
    fresh.configure({"scan_backend": "fused", "scheduler_step": "epoch", "official_width": True})
    optimizers.clear()
    optimizers.update(fresh.optimizers(official, fresh.lr))
    states = {0: _clone_state(official, optimizers, {})}
    worst = 0.0
    per_step = []
    for index, snapshot in enumerate(checkpoint["snapshots"], start=1):
        _assign(official, snapshot["grad"])
        optimizers["discriminator"].step()
        optimizers["generator"].step()
        gap = _param_max_abs(official, snapshot["weight"])
        worst = max(worst, gap)
        per_step.append(gap)
        if index in (4, 20):
            states[index] = _clone_state(official, optimizers, {})
    return states, {"max_abs": worst, "per_step_max_abs": per_step}


def _step_metrics(step):
    losses = {
        key: float(value.detach())
        for key, value in step.losses.items()
        if torch.is_tensor(value) and value.ndim == 0
    }
    return step.forward.detach().float().cpu(), losses, {
        key: value.detach().float().cpu() for key, value in step.gradients.items()
    }


def _scan_probe(adapter, implementations, state_for, clean, noisy, rng):
    import selective_scan_cuda

    saved = {"fwd": [], "bwd": []}
    current = {"label": None, "fwd": [], "bwd": []}

    def fwd(*args, **kwargs):
        out = original_fwd(*args, **kwargs)
        slot = current["fwd"]
        index = len(slot)
        record = {}
        if current["label"] == "official-0":
            saved["fwd"].append([value.detach().clone() if torch.is_tensor(value) else None for value in args[:7]])
            record = {"max_abs": [0.0] * 7}
        else:
            record = {
                "max_abs": [
                    float((value - ref).abs().max()) if torch.is_tensor(value) and ref is not None else None
                    for value, ref in zip(args[:7], saved["fwd"][index])
                ]
            }
        slot.append(record)
        return out

    def bwd(*args, **kwargs):
        out = original_bwd(*args, **kwargs)
        index = len(current["bwd"])
        tensors = [value.detach().clone() if torch.is_tensor(value) else None for value in out]
        record = {}
        if current["label"] == "official-0":
            saved["bwd"].append(tensors)
            record = {"max_abs": [0.0 if value is not None else None for value in tensors]}
        else:
            record = {
                "max_abs": [
                    float((value - ref).abs().max()) if torch.is_tensor(value) and ref is not None else None
                    for value, ref in zip(tensors, saved["bwd"][index])
                ]
            }
        current["bwd"].append(record)
        return out

    original_fwd = selective_scan_cuda.fwd
    original_bwd = selective_scan_cuda.bwd
    selective_scan_cuda.fwd = fwd
    selective_scan_cuda.bwd = bwd
    runs = []
    try:
        for label, implementation in implementations:
            current["label"] = label
            current["fwd"], current["bwd"] = [], []
            modules = adapter.build(implementation, clean.device)
            optimizers = adapter.optimizers(modules, adapter.lr)
            schedulers = adapter.schedulers(optimizers)
            chosen = state_for(implementation, modules, optimizers, schedulers)
            _restore(modules, optimizers, schedulers, chosen, rng)
            step = adapter.train_step(implementation, modules, optimizers, schedulers, clean, noisy, 0)
            runs.append({
                "label": label,
                "implementation": implementation,
                "loss": float(step.losses["generator"]),
                "forward_calls": len(current["fwd"]),
                "backward_calls": len(current["bwd"]),
                "forward": current["fwd"],
                "backward": current["bwd"],
            })
            del modules
            torch.cuda.empty_cache()
    finally:
        selective_scan_cuda.fwd = original_fwd
        selective_scan_cuda.bwd = original_bwd
    self_run = next(item for item in runs if item["label"] == "official-1")
    cross_run = next(item for item in runs if item["label"] == "independent")
    def worst(run):
        values = [value for row in run["backward"] for value in row["max_abs"] if value is not None]
        inputs = [value for row in run["forward"] for value in row["max_abs"] if value is not None]
        return {
            "forward_input_max_abs": max(inputs) if inputs else None,
            "backward_output_max_abs": max(values) if values else None,
        }
    self_stats = worst(self_run)
    cross_stats = worst(cross_run)
    explained = bool(
        cross_stats["forward_input_max_abs"] == 0.0
        and self_stats["backward_output_max_abs"] is not None
        and cross_stats["backward_output_max_abs"] <= self_stats["backward_output_max_abs"]
    )
    return {"runs": [
        {key: value for key, value in run.items() if key not in ("forward", "backward")}
        | {"forward_input_max_abs": worst(run)["forward_input_max_abs"],
           "backward_output_max_abs": worst(run)["backward_output_max_abs"]}
        for run in runs
    ], "official_self": self_stats, "cross": cross_stats, "explained_by_scan_rerun": explained}


def _branch(adapter, official_modules, base_state, batches, rng, restore_mel):
    """Four updates from the shared step-4 state. Two copies of each implementation."""
    packs = {name: [] for name in ("official_a", "official_b", "independent_a", "independent_b")}
    losses = {name: [] for name in packs}
    forwards = {name: [] for name in packs}
    for name in packs:
        implementation = name.split("_")[0]
        if implementation == "official":
            restore_mel()
            modules = official_modules
        else:
            modules = adapter.build(implementation, batches[0][0].device)
        optimizers = adapter.optimizers(modules, adapter.lr)
        schedulers = adapter.schedulers(optimizers)
        state = base_state
        if implementation != "official":
            mapped = _remap_states(official_modules, modules, base_state["optimizers"])
            state = {
                "modules": base_state["modules"],
                "optimizers": mapped,
                "schedulers": base_state["schedulers"],
            }
        _restore(modules, optimizers, schedulers, state, rng)
        for clean, noisy in batches:
            step = adapter.train_step(implementation, modules, optimizers, schedulers, clean, noisy, 0)
            forward, loss, grads = _step_metrics(step)
            packs[name].append(_pack(grads))
            losses[name].append(loss)
            forwards[name].append(forward)
        del modules
        torch.cuda.empty_cache()
    report = []
    for index in range(len(batches)):
        grouped = {
            "official": [packs["official_a"][index], packs["official_b"][index]],
            "independent": [packs["independent_a"][index], packs["independent_b"][index]],
        }
        # One pair per group. The median of a single pair is that pair.
        summary = _summarize(
            grouped,
            {
                "official": [forwards["official_a"][index], forwards["official_b"][index]],
                "independent": [forwards["independent_a"][index], forwards["independent_b"][index]],
            },
            {
                "official": [losses["official_a"][index], losses["official_b"][index]],
                "independent": [losses["independent_a"][index], losses["independent_b"][index]],
            },
        )
        report.append({"update": index + 1, "summary": summary})
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    started = time.perf_counter()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prepared = torch.load(SOURCE / "input.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(SOURCE / "official_a" / "checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["data_position"] != 20 or len(checkpoint["snapshots"]) != 20:
        raise SystemExit("official_a checkpoint is not the 20-update state")
    adapter = SEMambaPP()
    cfg = {"scan_backend": "fused", "scheduler_step": "epoch", "official_width": True}
    adapter.configure(cfg)
    official = adapter.build("official", device)
    official_mel = adapter._mel

    def restore_mel():
        adapter._mel = official_mel
    optimizers = adapter.optimizers(official, adapter.lr)
    schedulers = adapter.schedulers(optimizers)
    replay_states, replay = _replay(official, optimizers, prepared, checkpoint)
    # Schedulers were not part of the replay. Epoch mode never steps them.
    for state in replay_states.values():
        state["schedulers"] = {name: copy.deepcopy(sched.state_dict()) for name, sched in schedulers.items()}
    checkpoint_scheduler = {
        name: checkpoint["schedulers"][name]["last_epoch"] for name in checkpoint["schedulers"]
    }
    del checkpoint
    print(json.dumps({"replay_max_abs": replay["max_abs"], "scheduler": checkpoint_scheduler}), flush=True)
    seed_probe(17, True)
    rng = capture_rng(cuda=True)

    def waveform(index):
        clean = prepared["clean"][index].to(device)
        noisy = prepared["noisy"][index].to(device)
        if clean.dim() == 1:
            clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
        return clean, noisy

    independent = adapter.build("independent", device)
    results = []
    scan = []
    for step, utterance in ((0, 0), (4, 4), (20, 19)):
        clean, noisy = waveform(utterance)
        base = replay_states[step]
        mapped = _remap_states(official, independent, base["optimizers"])
        packs = {"official": [], "independent": []}
        forwards = {"official": [], "independent": []}
        losses = {"official": [], "independent": []}
        holders = {
            "official": (official, adapter.optimizers(official, adapter.lr), schedulers, base),
            "independent": (
                independent,
                adapter.optimizers(independent, adapter.lr),
                adapter.schedulers(adapter.optimizers(independent, adapter.lr)),
                {
                    "modules": base["modules"],
                    "optimizers": mapped,
                    "schedulers": base["schedulers"],
                },
            ),
        }
        # Rebuild schedulers against the optimizers that will actually step.
        for implementation in ("official", "independent"):
            modules, _old, _sched, state = holders[implementation]
            opts = adapter.optimizers(modules, adapter.lr)
            sch = adapter.schedulers(opts)
            holders[implementation] = (modules, opts, sch, state)
            for _ in range(REPEATS):
                if implementation == "official":
                    restore_mel()
                _restore(modules, opts, sch, state, rng)
                trained = adapter.train_step(implementation, modules, opts, sch, clean, noisy, 0)
                forward, loss, grads = _step_metrics(trained)
                packs[implementation].append(_pack(grads))
                forwards[implementation].append(forward)
                losses[implementation].append(loss)
        summary = _summarize(packs, forwards, losses)

        def state_for(implementation, modules, opts, sch):
            del modules, opts, sch
            if implementation == "official":
                return base
            return {
                "modules": base["modules"],
                "optimizers": mapped,
                "schedulers": base["schedulers"],
            }

        scan_report = _scan_probe(
            adapter,
            (("official-0", "official"), ("official-1", "official"), ("independent", "independent")),
            state_for,
            clean,
            noisy,
            rng,
        )
        explained = (not summary["systematic_gap"]) and scan_report["explained_by_scan_rerun"]
        item = {
            "updates": step,
            "utterance_index": utterance,
            "utterance_note": "prepared utterance" if step < 20 else "step 20 has no next utterance; repeated index 19",
            "case_id": prepared["items"][utterance]["case_id"],
            "crop_start": prepared["items"][utterance]["crop_start"],
            "summary": summary,
            "scan": scan_report,
            "explained": explained,
        }
        results.append(item)
        scan.append(scan_report["explained_by_scan_rerun"])
        print(json.dumps({
            "updates": step,
            "bias_rel_l2": summary["bias"]["rel_l2"],
            "self_median": summary["self_median"],
            "cross_median": summary["cross_median"],
            "systematic_gap": summary["systematic_gap"],
            "scan_explained": scan_report["explained_by_scan_rerun"],
            "explained": explained,
        }), flush=True)
    batches = [waveform(index) for index in range(4, 8)]
    branch = _branch(adapter, official, replay_states[4], batches, rng, restore_mel)
    for row in branch:
        summary = row["summary"]
        print(json.dumps({
            "branch_update": row["update"],
            "bias_rel_l2": summary["bias"]["rel_l2"],
            "self_median": summary["self_median"],
            "cross_median": summary["cross_median"],
            "systematic_gap": summary["systematic_gap"],
        }), flush=True)
    payload = {
        "criterion": CRITERION,
        "source": str(SOURCE.relative_to(ROOT)),
        "repeats": REPEATS,
        "replay": replay,
        "checkpoint_scheduler_last_epoch": checkpoint_scheduler,
        "same_state": results,
        "branch_from_update_4": branch,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} replay_max_abs={replay['max_abs']}")


if __name__ == "__main__":
    main()
