"""Same-state CUDA repeats for MP-SENet and DeepFilterNet3, plus native STFT noise.

Fixed before the run. Four official and four independent repeats restore the
same initial weights and one RNG snapshot. Optimizer moments stay at zero.
A systematic gap is bias_rel_l2 > self_median, or a forward mean-absolute
difference above the within-implementation maximum. The median boolean is
recorded and is not the decision.

The STFT probe repeats torch.stft backward on one waveform. It measures
native reflection-padding noise. It is not a second model implementation.
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

from src.training.core.rng import capture_rng, restore_rng  # noqa: E402
from tests.training.four_process.compare import seed_probe  # noqa: E402

REPEATS = 4


def _rel(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    delta = (a.double() - b.double()).reshape(-1)
    denom = a.double().reshape(-1).norm().clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(a.double().reshape(-1), b.double().reshape(-1), dim=0)
    return {
        "max_abs": float((a - b).abs().max()),
        "rel_l2": float(delta.norm() / denom),
        "cosine": float(cosine),
        "norm": float(a.double().reshape(-1).norm()),
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _pack(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([grads[key].float().reshape(-1) for key in sorted(grads)])


def _summarize(packs, forwards):
    def pairs(rows):
        return [_rel(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]

    self_pairs = pairs(packs["official"]) + []
    official_pairs = pairs(packs["official"])
    independent_pairs = pairs(packs["independent"])
    cross = [_rel(a, b) for a in packs["official"] for b in packs["independent"]]
    bias = _rel(
        torch.stack([row.double() for row in packs["official"]]).mean(0),
        torch.stack([row.double() for row in packs["independent"]]).mean(0),
    )
    self_median = max(
        _median([row["rel_l2"] for row in official_pairs]),
        _median([row["rel_l2"] for row in independent_pairs]),
    )
    cross_median = _median([row["rel_l2"] for row in cross])
    forward_pairs_official = pairs(forwards["official"])
    forward_pairs_independent = pairs(forwards["independent"])
    self_forward = max(
        max(row["max_abs"] for row in forward_pairs_official),
        max(row["max_abs"] for row in forward_pairs_independent),
    )
    forward_bias = float((
        torch.stack([row.double() for row in forwards["official"]]).mean(0)
        - torch.stack([row.double() for row in forwards["independent"]]).mean(0)
    ).abs().max())
    return {
        "bias": bias,
        "self_median": self_median,
        "cross_median": cross_median,
        "median_boolean": bool(cross_median <= self_median),
        "forward_bias_abs": forward_bias,
        "forward_self_max_abs": self_forward,
        "systematic_gap": bool(bias["rel_l2"] > self_median or forward_bias > self_forward),
        "official_rel_l2": [row["rel_l2"] for row in official_pairs],
        "independent_rel_l2": [row["rel_l2"] for row in independent_pairs],
        "cross_rel_l2": [row["rel_l2"] for row in cross],
        "unused": len(self_pairs),
    }


def _restore(modules, state, rng):
    for name, module in modules.items():
        module.load_state_dict(state[name], strict=True)
    restore_rng(rng, cuda=True)


def _run(adapter, prepared, device):
    initial = prepared["initial"]
    clean = prepared["clean"][0].to(device)
    noisy = prepared["noisy"][0].to(device)
    if clean.dim() == 1:
        clean, noisy = clean.unsqueeze(0), noisy.unsqueeze(0)
    packs = {"official": [], "independent": []}
    forwards = {"official": [], "independent": []}
    losses = {"official": [], "independent": []}
    for implementation in ("official", "independent"):
        modules = adapter.build(implementation, device)
        for name, module in modules.items():
            module.load_state_dict(initial[name], strict=True)
        optimizers = adapter.optimizers(modules, 0)
        schedulers = adapter.schedulers(optimizers)
        seed_probe(17, True)
        rng = capture_rng(cuda=True)
        saved = {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                 for name, module in modules.items()}
        for _ in range(REPEATS):
            _restore(modules, saved, rng)
            for opt in optimizers.values():
                opt.state.clear()
            step = adapter.train_step(implementation, modules, optimizers, schedulers, clean, noisy, 0)
            packs[implementation].append(_pack(step.gradients))
            forwards[implementation].append(step.forward.detach().float().cpu())
            losses[implementation].append(float(step.losses["generator"].detach()))
        del modules
        torch.cuda.empty_cache()
    summary = _summarize(packs, forwards)
    summary["generator_loss"] = losses
    return summary


def _stft_noise(waveform: torch.Tensor) -> dict:
    window = torch.hann_window(400, device=waveform.device)
    grads = []
    base = waveform.detach()
    for _ in range(REPEATS):
        wave = base.detach().requires_grad_(True)
        spectrum = torch.stft(
            wave, 400, 100, 400, window, center=True, pad_mode="reflect",
            normalized=False, return_complex=True,
        )
        spectrum.abs().sum().backward()
        grads.append(wave.grad.detach().float().cpu().reshape(-1))
    pairs = [_rel(grads[i], grads[j]) for i in range(len(grads)) for j in range(i + 1, len(grads))]
    return {
        "pair_rel_l2": [row["rel_l2"] for row in pairs],
        "pair_max_abs": [row["max_abs"] for row in pairs],
        "median_rel_l2": _median([row["rel_l2"] for row in pairs]),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from tests.training.four_process.adapters import MPSENet
    from tests.training.four_process.real_models import DeepFilterNet3

    mp_in = torch.load(
        ROOT / "data/results/training/final-accept-mp-fixed-gpu-seed17-20261103/input.pt",
        map_location="cpu", weights_only=False,
    )
    mp = MPSENet()
    mp.configure({"official_width": True, "scheduler_step": "epoch"})
    mp_summary = _run(mp, mp_in, device)
    noisy = mp_in["noisy"][0].to(device)
    if noisy.dim() == 1:
        noisy = noisy.unsqueeze(0)
    stft = _stft_noise(noisy)
    print(json.dumps({"mp_systematic": mp_summary["systematic_gap"], "mp_bias": mp_summary["bias"]["rel_l2"],
                      "mp_self": mp_summary["self_median"], "stft_median": stft["median_rel_l2"]}), flush=True)

    os.environ["DF_PROBE_OFFICIAL_LIBDF"] = "1"
    df_in = torch.load(
        ROOT / "data/results/training/final-accept-deepfilternet3-gpu-resume-source-20261103/input.pt",
        map_location="cpu", weights_only=False,
    )
    df = DeepFilterNet3()
    df.configure({"official_width": True, "probe": {
        "frontend": "official libdf vs independent DFStream",
    }})
    df_summary = _run(df, df_in, device)
    print(json.dumps({"df_systematic": df_summary["systematic_gap"], "df_bias": df_summary["bias"]["rel_l2"],
                      "df_self": df_summary["self_median"], "df_cross": df_summary["cross_median"]}), flush=True)
    payload = {
        "criterion": "systematic gap if bias_rel_l2 > self_median or forward bias exceeds self max abs",
        "repeats": REPEATS,
        "mp_senet": mp_summary,
        "mp_native_stft_backward": stft,
        "deepfilternet3": df_summary,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
