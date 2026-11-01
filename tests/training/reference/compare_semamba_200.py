"""Compare four SEMamba++ 200-step runs and score STOI on the saved validation audio.

    .venvs/lab/bin/python tests/training/reference/compare_semamba_200.py \
        data/results/training/semamba-200step-20261101
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from src.evaluation.metrics.stoi import score_stoi

PAIRS = (("official-a", "official-b"), ("independent-a", "independent-b"), ("official-a", "independent-a"), ("official-b", "independent-b"))
DETAIL = (0, 1, 2, 5, 10, 20, 50, 100, 200)


def main() -> None:
    root = Path(sys.argv[1])
    runs = {name: root / name for name in ("official-a", "official-b", "independent-a", "independent-b")}
    missing = [name for name, path in runs.items() if not (path / "loss.jsonl").is_file()]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    losses = {name: _losses(path) for name, path in runs.items()}
    report = {"steps": {}, "validation": {}, "first_divergence": None}
    for step in sorted(losses["official-a"]):
        report["steps"][step] = {
            f"{left}_vs_{right}": _loss_gap(losses[left][step], losses[right][step])
            for left, right in PAIRS
            if step in losses[left] and step in losses[right]
        }
    for step in DETAIL:
        report["validation"][step] = {}
        for left, right in PAIRS:
            report["validation"][step][f"{left}_vs_{right}"] = _val_gap(runs[left], runs[right], step)
    report["first_divergence"] = _first(losses)
    report["gradients"] = _grads(runs["official-a"], runs["independent-a"])
    (root / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"first_divergence": report["first_divergence"], "gradients": report["gradients"]}, indent=2))


def _losses(path: Path) -> dict[int, dict]:
    rows = {}
    for line in (path / "loss.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[int(row["step"])] = row
    return rows


def _loss_gap(left: dict, right: dict) -> dict:
    keys = ("generator", "discriminator", "adversarial", "matching", "mel", "magnitude", "phase", "complex", "consistency", "generator_grad_norm", "discriminator_grad_norm")
    return {key: None if left.get(key) is None or right.get(key) is None else abs(float(left[key]) - float(right[key])) for key in keys}


def _val_gap(left: Path, right: Path, step: int) -> dict:
    left_path = left / f"val_{step:08d}.pt"
    right_path = right / f"val_{step:08d}.pt"
    left_json_path = left / f"val_{step:08d}.json"
    right_json_path = right / f"val_{step:08d}.json"
    if not left_path.is_file() or not right_path.is_file():
        return {"missing": True}
    left_rows = torch.load(left_path, map_location="cpu", weights_only=False)["rows"]
    right_rows = torch.load(right_path, map_location="cpu", weights_only=False)["rows"]
    left_json = json.loads(left_json_path.read_text(encoding="utf-8"))
    right_json = json.loads(right_json_path.read_text(encoding="utf-8"))
    cosine = []
    stoi_gap = []
    stoi_left = []
    for item, other in zip(left_rows, right_rows):
        length = min(item["enhanced"].numel(), other["enhanced"].numel())
        cosine.append(_cosine(item["enhanced"][:length], other["enhanced"][:length]))
        scored_left = score_stoi(item["clean"], item["enhanced"], 16000)
        scored_right = score_stoi(other["clean"], other["enhanced"], 16000)
        if scored_left.get("value") is not None:
            stoi_left.append(scored_left["value"])
        if scored_left.get("value") is not None and scored_right.get("value") is not None:
            stoi_gap.append(abs(scored_left["value"] - scored_right["value"]))
    sisdr = [
        abs(a["enhanced"]["value"] - b["enhanced"]["value"])
        for a, b in zip(left_json, right_json)
        if a["enhanced"].get("value") is not None and b["enhanced"].get("value") is not None
    ]
    enhanced = [row["enhanced"]["value"] for row in left_json if row["enhanced"].get("value") is not None]
    noisy = [row["noisy"]["value"] for row in left_json if row["noisy"].get("value") is not None]
    return {
        "waveform_cosine_min": min(cosine) if cosine else None,
        "si_sdr_abs_max": max(sisdr) if sisdr else None,
        "stoi_abs_max": max(stoi_gap) if stoi_gap else None,
        "left_si_sdr_mean": sum(enhanced) / len(enhanced) if enhanced else None,
        "left_noisy_si_sdr_mean": sum(noisy) / len(noisy) if noisy else None,
        "left_stoi_mean": sum(stoi_left) / len(stoi_left) if stoi_left else None,
    }


def _first(losses: dict) -> dict | None:
    for step in sorted(losses["official-a"]):
        gap = abs(losses["official-a"][step]["generator"] - losses["independent-a"][step]["generator"])
        rerun = abs(losses["official-a"][step]["generator"] - losses["official-b"][step]["generator"])
        if gap > 1e-4:
            return {"step": step, "generator_abs": gap, "official_rerun_abs": rerun}
    return None


def _grads(official: Path, independent: Path) -> dict:
    path_o = official / "grads_00000001.pt"
    path_i = independent / "grads_00000001.pt"
    if not path_o.is_file() or not path_i.is_file():
        return {"missing": True}
    left = torch.load(path_o, map_location="cpu", weights_only=False)
    right = torch.load(path_i, map_location="cpu", weights_only=False)
    return {
        "audio": _pair(left.get("audio"), right["audio"][0] if isinstance(right.get("audio"), list) else right.get("audio")),
        "generator": _named(left.get("generator_grad"), right.get("generator_grad")),
    }


def _pair(left, right) -> dict | None:
    if left is None or right is None:
        return None
    delta = (left.float().reshape(-1) - right.float().reshape(-1)).abs()
    return {"max_abs": float(delta.max()), "cosine": _cosine(left, right)}


def _named(left, right) -> dict | None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    worst = 0.0
    cosine_min = 1.0
    large = 1.0
    compared = 0
    for key, value in left.items():
        other = _match(right, key)
        if other is None or other.shape != value.shape:
            continue
        compared += 1
        delta = float((value - other).abs().max())
        worst = max(worst, delta)
        cosine = _cosine(value, other)
        cosine_min = min(cosine_min, cosine)
        if float(value.abs().max()) > 1e-5:
            large = min(large, cosine)
    return {"compared": compared, "max_abs": worst, "cosine_min": cosine_min, "large_cosine_min": large}


def _match(grads: dict, key: str):
    if key in grads:
        return grads[key]
    suffix = key.split(".", 1)[-1]
    found = [value for name, value in grads.items() if name.split(".", 1)[-1] == suffix]
    if len(found) == 1:
        return found[0]
    return None


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    denom = left.norm() * right.norm()
    if float(denom) == 0:
        return 0.0
    return float(torch.dot(left, right) / denom)


if __name__ == "__main__":
    main()
