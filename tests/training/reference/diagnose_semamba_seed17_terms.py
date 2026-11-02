"""Split the seed-17 SEMamba++ waveform gradient by loss term.

Forward of the saved official-width probe matches. This script asks which
loss term first disagrees in backward, and which parameter has the largest
relative gradient difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.training.four_process.compare import seed_probe  # noqa: E402
from tests.training.four_process.real_models import (  # noqa: E402
    SEMambaPP,
    _semamba_generator,
    _semamba_parts,
)

SOURCE = ROOT / "data/results/training/followup-20261102-voicebank/semamba-epoch-seed-17"
OUTPUT = ROOT / "data/results/training/seed17-backward-20261103-terms"


def _pair(a, b):
    diff = (a.double() - b.double()).abs()
    denom = a.double().reshape(-1).norm().clamp_min(1e-30)
    return {
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "rel_l2": float(diff.reshape(-1).norm() / denom) if diff.numel() else 0.0,
        "cosine": float(torch.nn.functional.cosine_similarity(a.double().reshape(-1), b.double().reshape(-1), dim=0))
        if a.numel() else 1.0,
        "different": int((a != b).sum()) if a.shape == b.shape else -1,
        "left_norm": float(a.double().reshape(-1).norm()),
    }


def _run(implementation, payload, cfg):
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
    parts = _semamba_parts(implementation, modules["generator"], clean, noisy)
    losses = _semamba_generator(
        implementation, modules["discriminator_band"], modules["discriminator_mrd"], clean, parts, adapter._mel
    )
    audio = parts["audio"]
    terms = {}
    for name, value in losses.items():
        grad = torch.autograd.grad(value, audio, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            terms[name] = None
            continue
        terms[name] = grad.detach().cpu()
    modules["generator"].zero_grad(set_to_none=True)
    losses["generator"].backward()
    named = {
        name: parameter.grad.detach().cpu()
        for name, parameter in modules["generator"].named_parameters()
        if parameter.grad is not None
    }
    result = {"terms": terms, "named": named, "loss": {key: float(value.detach()) for key, value in losses.items()}}
    del modules
    torch.cuda.empty_cache()
    return result


def _worst(left, right):
    worst_abs = None
    worst_rel = None
    rels = []
    for name in sorted(set(left) & set(right)):
        stat = _pair(left[name].reshape(-1), right[name].reshape(-1))
        rels.append(stat["rel_l2"])
        row = {"name": name, **stat}
        if worst_abs is None or stat["max_abs"] > worst_abs["max_abs"]:
            worst_abs = row
        if worst_rel is None or stat["rel_l2"] > worst_rel["rel_l2"]:
            worst_rel = row
    return {
        "worst_abs": worst_abs,
        "worst_rel": worst_rel,
        "median_rel_l2": float(torch.tensor(rels).median()) if rels else None,
    }


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = torch.load(SOURCE / "input.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((SOURCE / "config.json").read_text())
    official_a = _run("official", payload, cfg)
    official_b = _run("official", payload, cfg)
    independent = _run("independent", payload, cfg)

    def terms(left, right):
        return {
            name: None if left["terms"][name] is None or right["terms"].get(name) is None
            else _pair(left["terms"][name], right["terms"][name])
            for name in left["terms"]
            if name in right["terms"]
        }

    report = {
        "seed": cfg["seed"],
        "loss_official_a": official_a["loss"],
        "loss_independent": independent["loss"],
        "official_a_vs_b_terms": terms(official_a, official_b),
        "official_vs_independent_terms": terms(official_a, independent),
        "official_a_vs_b_parameters": _worst(official_a["named"], official_b["named"]),
        "official_vs_independent_parameters": _worst(official_a["named"], independent["named"]),
    }
    (OUTPUT / "terms.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
