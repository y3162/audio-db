"""Score saved 64-utterance enhancements with SI-SDR, STOI, PESQ, and cosine.

PESQ uses the library gate: a downsample-boundary jump is ``na``, not a
corrected score. Existing JSON files are not overwritten.

    .venvs/lab/bin/python tests/training/reference/summarize_valid64.py \\
        data/results/training/semamba-valid64-20261101
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics.pesq import score_pesq
from src.evaluation.metrics.sisdr import score_si_sdr
from src.evaluation.metrics.stoi import score_stoi
from src.training.data.dataset import load_recipe_dataset

DB = ROOT / "data/db/semamba-voicebank-train-20261101.duckdb"


def main() -> None:
    out = Path(sys.argv[1])
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    meta = {row["case_id"]: row for row in selection["utterances"]}
    valid = load_recipe_dataset(DB, "voicebank_28_valid", role="validation", seed=1234, sample_rate=16000, segment_size=None, training=False)
    positions = {example.case_id: index for index, example in enumerate(valid.examples)}
    waves = {}
    for case_id in meta:
        batch = valid.collate([positions[case_id]])
        waves[case_id] = (batch["clean"][0], batch["noisy"][0])
    written = []
    stored = {}
    for path in sorted(out.glob("*.pt")):
        destination = path.with_suffix(".json")
        if destination.exists():
            written.append(destination)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            stored[path.stem] = {row["case_id"]: enhanced for row, enhanced in zip(payload["rows"], payload["enhanced"], strict=True)}
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = []
        for record, enhanced in zip(payload["rows"], payload["enhanced"], strict=True):
            clean, noisy = waves[record["case_id"]]
            info = meta[record["case_id"]]
            length = min(clean.numel(), noisy.numel(), enhanced.numel())
            rows.append(
                {
                    "case_id": record["case_id"],
                    "filename": info["filename"],
                    "speaker": info["speaker"],
                    "noise": info["noise"],
                    "snr_db": info["snr_db"],
                    "samples": int(clean.numel()),
                    "enhanced_samples": int(enhanced.numel()),
                    "common": int(length),
                    "noisy_sisdr": score_si_sdr(clean, noisy),
                    "sisdr": score_si_sdr(clean, enhanced),
                    "stoi": score_stoi(clean, enhanced, 16000),
                    "pesq": score_pesq(clean, enhanced, 16000),
                    "cosine": _cosine(clean[:length], enhanced[:length]),
                }
            )
        body = {"source": path.name, "side": payload["side"], "state": payload["state"], "rows": rows, "summary": _summary(rows)}
        destination.write_text(json.dumps(body), encoding="utf-8")
        written.append(destination)
        stored[path.stem] = {row["case_id"]: enhanced for row, enhanced in zip(payload["rows"], payload["enhanced"], strict=True)}
        print(path.name, body["summary"]["sisdr_mean"], body["summary"]["pesq_na"], flush=True)
    groups = []
    for path in written:
        body = json.loads(path.read_text(encoding="utf-8"))
        groups.append({"file": path.name, **body["summary"], "side": body["side"], "state": body["state"]})
    combined = out / f"summary-{len(groups):02d}.json"
    if combined.exists():
        raise SystemExit(f"refusing to overwrite {combined}")
    pairs = _pairs(out, stored)
    combined.write_text(json.dumps({"groups": groups, "pairs": pairs}, indent=2), encoding="utf-8")
    print("wrote", combined, flush=True)


def _pairs(out: Path, stored: dict[str, dict[str, torch.Tensor]]) -> list[dict]:
    rows = {}
    for path in out.glob("*.json"):
        if path.name.startswith("summary"):
            continue
        body = json.loads(path.read_text(encoding="utf-8"))
        if "rows" not in body:
            continue
        rows[path.stem] = {row["case_id"]: row for row in body["rows"]}
    names = (
        "official-a-step{step}",
        "official-b-step{step}",
        "independent-a-step{step}",
        "independent-b-step{step}",
    )
    report = []
    for step in ("00000000", "00000050", "00000100", "00000200"):
        keys = [name.format(step=step) for name in names]
        if any(key not in rows or key not in stored for key in keys):
            continue
        report.append(
            {
                "step": int(step),
                "official_rerun": _pair(rows[keys[0]], rows[keys[1]], stored[keys[0]], stored[keys[1]]),
                "independent_rerun": _pair(rows[keys[2]], rows[keys[3]], stored[keys[2]], stored[keys[3]]),
                "official_vs_independent": _pair(rows[keys[0]], rows[keys[2]], stored[keys[0]], stored[keys[2]]),
            }
        )
    return report


def _pair(left_rows, right_rows, left_waves, right_waves) -> dict:
    sisdr = []
    stoi = []
    cosine = []
    for case_id, left in left_rows.items():
        right = right_rows[case_id]
        if left["sisdr"].get("status") == "ok" and right["sisdr"].get("status") == "ok":
            sisdr.append(abs(float(left["sisdr"]["value"]) - float(right["sisdr"]["value"])))
        if left["stoi"].get("status") == "ok" and right["stoi"].get("status") == "ok":
            stoi.append(abs(float(left["stoi"]["value"]) - float(right["stoi"]["value"])))
        a, b = left_waves[case_id], right_waves[case_id]
        cosine.append(_cosine(a[: min(a.numel(), b.numel())], b[: min(a.numel(), b.numel())]))
    return {
        "sisdr_abs_mean": None if not sisdr else sum(sisdr) / len(sisdr),
        "sisdr_abs_max": None if not sisdr else max(sisdr),
        "stoi_abs_mean": None if not stoi else sum(stoi) / len(stoi),
        "stoi_abs_max": None if not stoi else max(stoi),
        "cosine_min": min(cosine),
        "cosine_mean": sum(cosine) / len(cosine),
    }


def _summary(rows: list[dict]) -> dict:
    def values(name: str) -> list[float]:
        return [float(row[name]["value"]) for row in rows if row[name].get("status") == "ok" and row[name].get("value") is not None]

    sisdr = values("sisdr")
    noisy = values("noisy_sisdr")
    stoi = values("stoi")
    pesq = values("pesq")
    paired = [
        float(row["sisdr"]["value"]) - float(row["noisy_sisdr"]["value"])
        for row in rows
        if row["sisdr"].get("status") == "ok" and row["noisy_sisdr"].get("status") == "ok"
    ]
    return {
        "count": len(rows),
        "sisdr_mean": None if not sisdr else sum(sisdr) / len(sisdr),
        "noisy_sisdr_mean": None if not noisy else sum(noisy) / len(noisy),
        "sisdr_improvement_mean": None if not paired else sum(paired) / len(paired),
        "stoi_mean": None if not stoi else sum(stoi) / len(stoi),
        "pesq_mean": None if not pesq else sum(pesq) / len(pesq),
        "pesq_count": len(pesq),
        "pesq_na": sum(row["pesq"].get("status") == "na" for row in rows),
        "cosine_min": min(row["cosine"] for row in rows),
        "cosine_mean": sum(row["cosine"] for row in rows) / len(rows),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denom = float(left.norm() * right.norm())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


if __name__ == "__main__":
    main()
