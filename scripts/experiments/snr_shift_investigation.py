"""Length distribution, pad_short plans, and full SNR-shift recipe audit.

Writes under data/results/investigations/snr-shift-20261114/. Does not write
wav files and does not modify recipes, databases, or snapshots.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RECIPE = Path("/net/spring/work/seki/database/recipes/voicebank-snr-shift-v1/recipes.jsonl")
EXPECTED_SHA = "9502da7b24b81bed04781c1e66726945dfcb3024045111df8bbb5839e0d01c14"
OUT = ROOT / "data/results/investigations/snr-shift-20261114"
MULTIPLIER = 10 ** (15 / 20)
RATE16 = 16000


def _quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _length_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[int]] = defaultdict(list)
    speakers: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        length = int(row["noise_end_exclusive_48k"]) - int(row["noise_start_48k"])
        speaker = str(row["utterance_id"]).split("_", 1)[0]
        split = "valid" if row["split"] == "train" and speaker in {"p286", "p287"} else row["split"]
        by_split[split].append(length)
        speakers[split][speaker].append(length)

    def pack(lengths: list[int], thresholds: tuple[int, ...]) -> dict[str, Any]:
        report: dict[str, Any] = {
            "count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": _quantile(lengths, 0.5),
            "quantiles": {name: _quantile(lengths, q) for name, q in (("p05", 0.05), ("p25", 0.25), ("p75", 0.75), ("p95", 0.95))},
        }
        for seconds in thresholds:
            samples = seconds * 48000
            under = sum(length < samples for length in lengths)
            exact = sum(length == samples for length in lengths)
            over = sum(length > samples for length in lengths)
            report[f"under_{seconds}s"] = under
            report[f"exact_{seconds}s"] = exact
            report[f"over_{seconds}s"] = over
            report[f"under_{seconds}s_fraction"] = under / len(lengths)
        return report

    speaker_bias = {}
    for split, grouped in speakers.items():
        rows_out = []
        for speaker, lengths in sorted(grouped.items()):
            short3 = sum(length < 3 * 48000 for length in lengths)
            short4 = sum(length < 4 * 48000 for length in lengths)
            rows_out.append({
                "speaker": speaker,
                "count": len(lengths),
                "under_3s": short3,
                "under_4s": short4,
                "under_3s_fraction": short3 / len(lengths),
                "under_4s_fraction": short4 / len(lengths),
            })
        speaker_bias[split] = rows_out
    return {"sample_rate": 48000, "splits": {name: pack(values, (3, 4)) for name, values in by_split.items()}, "speakers": speaker_bias}


def _plan_stats(name: str, samples: list[tuple[str, int]], *, epochs: list[int], seed: int, num_samples: int | None, chunk: int, shuffle: bool, pad_short: bool) -> list[dict[str, Any]]:
    from src.experiments.data.render.chunking import plan_chunk_epoch

    lengths = {case_id: length for case_id, length in samples}
    reports = []
    for epoch in epochs:
        plan = plan_chunk_epoch(
            samples,
            espnet_epoch=epoch,
            seed=seed,
            world_size=1,
            num_samples=num_samples,
            chunk_length=chunk,
            global_batch=4,
            num_cache_chunks=1024,
            shuffle=shuffle,
            shift_ratio=0.5,
            pad_short=pad_short,
        )
        chunks = [chunk_row for batch in plan.rank_batches[0] for chunk_row in batch]
        selected = plan.selected[0]
        used = {item.case_id for item in chunks}
        padded = [item for item in chunks if lengths[item.case_id] < chunk]
        ratios = [lengths[item.case_id] / chunk for item in padded]
        order = hashlib.sha256(json.dumps(
            [[item.case_id, item.start, item.stop] for item in chunks], separators=(",", ":")
        ).encode()).hexdigest()
        reports.append({
            "model": name,
            "espnet_epoch": epoch,
            "pad_short": pad_short,
            "shuffle": shuffle,
            "num_samples": num_samples,
            "selected_count": len(selected),
            "selected_unique": len(set(selected)),
            "unique_chunked": len(used),
            "chunks": len(chunks),
            "excluded_selected_unique": len(set(selected) - used),
            "padded_chunks": len(padded),
            "padded_real_fraction_mean": None if not ratios else sum(ratios) / len(ratios),
            "optimizer_updates": plan.updates,
            "order_sha256": order,
            "plan_hash": plan.plan_hash,
            "discarded_tail_chunks": plan.discarded_tail_chunks[0],
        })
    return reports


def _snapshot_samples(selector: str, slot: str) -> list[tuple[str, int]]:
    from src.experiments.data.fixed_pair import SnapshotSource
    from src.experiments.data.sampling.planner import chunk_case_length

    source = SnapshotSource(selector, ROOT / "data/db/snapshots-snr-shift-v1-perrank", "train")
    source.open()
    try:
        return [(case_id, chunk_case_length(source.store, case_id)) for case_id in source.case_ids(slot)]
    finally:
        source.close()


def _audit_chunk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import soundfile as sf
    import soxr

    from src.experiments.data.prepare.voicebank_shift_recipe import shift_case_id

    root = ROOT
    cache: dict[str, tuple[str, np.ndarray]] = {}
    results = []
    for row in rows:
        item: dict[str, Any] = {"utterance_id": row.get("utterance_id"), "ok": False}
        try:
            clean_rel = row["clean_path"]
            noise_rel = row["noise_path"]
            clean_path = root / clean_rel
            noise_path = root / noise_rel
            if noise_rel not in cache:
                digest = hashlib.sha256(noise_path.read_bytes()).hexdigest()
                cache[noise_rel] = (digest, np.empty(0))
            noise_sha = cache[noise_rel][0]
            clean_sha = hashlib.sha256(clean_path.read_bytes()).hexdigest()
            clean, rate = sf.read(clean_path, dtype="float64", always_2d=False)
            start, end = int(row["noise_start_48k"]), int(row["noise_end_exclusive_48k"])
            with sf.SoundFile(noise_path) as handle:
                handle.seek(start)
                noise = handle.read(end - start, dtype="float64")
            g = float(row["clean_gain_g"])
            beta = float(row["noise_gain_before_peak_beta"])
            shifted = float(row["shifted_noise_gain"])
            reference = g * clean
            mixture = reference + shifted * noise
            finite = bool(np.isfinite(reference).all() and np.isfinite(mixture).all())
            ref32 = np.ascontiguousarray(reference, dtype=np.float32)
            mix32 = np.ascontiguousarray(mixture, dtype=np.float32)
            ref16 = np.asarray(soxr.resample(ref32, 48000, 16000, quality="HQ"), dtype=np.float32)
            mix16 = np.asarray(soxr.resample(mix32, 48000, 16000, quality="HQ"), dtype=np.float32)
            clipped = bool(np.any(np.abs(mix32) > 1) or np.any(np.abs(mix16) > 1))
            # The training path does not clip. This records whether a clip would have changed samples.
            clip_delta = float(np.max(np.abs(mix16 - np.clip(mix16, -1, 1)))) if mix16.size else 0.0
            pcm = np.clip(np.round(mix16 * 32767), -32768, 32767) / 32767
            noise_comp = mixture - reference
            rms_s = float(np.sqrt(np.mean(reference ** 2)))
            rms_n = float(np.sqrt(np.mean(noise_comp ** 2)))
            snr48 = None if rms_s == 0 or rms_n == 0 else 20 * math.log10(rms_s / rms_n)
            comp16 = mix16.astype(np.float64) - ref16.astype(np.float64)
            rms_s16 = float(np.sqrt(np.mean(ref16.astype(np.float64) ** 2)))
            rms_n16 = float(np.sqrt(np.mean(comp16 ** 2)))
            snr16 = None if rms_s16 == 0 or rms_n16 == 0 else 20 * math.log10(rms_s16 / rms_n16)
            predicted = g * beta * MULTIPLIER
            item.update({
                "ok": finite and rate == 48000 and noise.shape == clean.shape and clean_sha == row["clean_sha256"] and noise_sha == row["noise_sha256"],
                "case_id": shift_case_id(row),
                "split": row["split"],
                "speaker": str(row["utterance_id"]).split("_", 1)[0],
                "length_48": int(clean.shape[0]),
                "span": end - start,
                "length_16": int(mix16.shape[0]),
                "expected_16": (int(clean.shape[0]) + 1) // 3,
                "finite": finite,
                "clean_sha_ok": clean_sha == row["clean_sha256"],
                "noise_sha_ok": noise_sha == row["noise_sha256"],
                "shape_ok": bool(noise.shape == clean.shape and rate == 48000),
                "g": g,
                "official_log_snr_db": float(row["official_log_snr_db"]),
                "target_log_snr_db": float(row["target_log_snr_db"]),
                "target_minus_official": float(row["target_log_snr_db"]) - float(row["official_log_snr_db"]),
                "gain_prediction_abs": abs(predicted - shifted),
                "component_shift_db": None if g * beta == 0 else 20 * math.log10(shifted / (g * beta)),
                "rms_snr_48": snr48,
                "rms_snr_16": snr16,
                "peak_48": float(np.max(np.abs(mixture))),
                "peak_16": float(np.max(np.abs(mix16))),
                "clip_would_change_16": clip_delta > 0,
                "pcm16_roundtrip_max_abs": float(np.max(np.abs(mix16 - pcm))),
                "peak_steps": int(row["peak_steps"]),
                "energy_48": float(np.mean(mixture ** 2)),
            })
            item["ok"] = bool(item["ok"] and item["length_16"] == item["expected_16"] and item["length_48"] == item["span"])
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    return results


def _hist(values: list[float], edges: list[float]) -> dict[str, int]:
    counts = Counter()
    for value in values:
        placed = False
        for edge in edges:
            if value < edge:
                counts[f"<{edge:g}"] += 1
                placed = True
                break
        if not placed:
            counts[f">={edges[-1]:g}"] += 1
    return dict(counts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = RECIPE.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    length_report = _length_report(rows)
    grid_train = _snapshot_samples("voicebank.snr_shift.tf_gridnet@1", "train.cases")
    grid_valid = _snapshot_samples("voicebank.snr_shift.tf_gridnet@1", "valid.cases")
    loco_train = _snapshot_samples("voicebank.snr_shift.tf_locoformer@1", "train.cases")
    loco_valid = _snapshot_samples("voicebank.snr_shift.tf_locoformer@1", "valid.cases")
    epochs = [1, 2, 3, 4]
    plans = []
    plans.extend(_plan_stats("tf_gridnet", grid_train, epochs=epochs, seed=0, num_samples=10000, chunk=48000, shuffle=True, pad_short=True))
    plans.extend(_plan_stats("tf_gridnet", grid_train, epochs=epochs, seed=0, num_samples=10000, chunk=48000, shuffle=True, pad_short=False))
    plans.extend(_plan_stats("tf_gridnet_valid", grid_valid, epochs=epochs, seed=0, num_samples=None, chunk=48000, shuffle=False, pad_short=False))
    plans.extend(_plan_stats("tf_locoformer", loco_train, epochs=epochs, seed=0, num_samples=4000, chunk=64000, shuffle=True, pad_short=True))
    plans.extend(_plan_stats("tf_locoformer", loco_train, epochs=epochs, seed=0, num_samples=4000, chunk=64000, shuffle=True, pad_short=False))
    plans.extend(_plan_stats("tf_locoformer_valid", loco_valid, epochs=epochs, seed=0, num_samples=None, chunk=64000, shuffle=False, pad_short=False))
    (OUT / "lengths.json").write_text(json.dumps(length_report, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "plans.json").write_text(json.dumps(plans, indent=2, sort_keys=True), encoding="utf-8")
    print(f"plans {len(plans)} sha {actual == EXPECTED_SHA} rows {len(rows)}", flush=True)

    workers = min(16, os.cpu_count() or 1)
    chunks = [rows[index::workers] for index in range(workers)]
    audited: list[dict[str, Any]] = []
    with Pool(workers) as pool:
        for part in pool.imap_unordered(_audit_chunk, chunks):
            audited.extend(part)
            print(f"audited {len(audited)}", flush=True)
    ok = [row for row in audited if row.get("ok")]
    bad = [row for row in audited if not row.get("ok")]
    def column(name: str) -> list[float]:
        return [float(row[name]) for row in ok if row.get(name) is not None]

    summary = {
        "recipe_sha256": actual,
        "recipe_sha_ok": actual == EXPECTED_SHA,
        "rows": len(rows),
        "ok": len(ok),
        "failed": len(bad),
        "gain_not_one": sum(row.get("g") != 1 for row in ok),
        "peak48_over_1": sum(float(row["peak_48"]) > 1 for row in ok),
        "peak16_over_1": sum(float(row["peak_16"]) > 1 for row in ok),
        "clip_would_change_16": sum(bool(row["clip_would_change_16"]) for row in ok),
        "pcm16_roundtrip_max": max(column("pcm16_roundtrip_max_abs"), default=None),
        "max_gain_prediction_abs": max(column("gain_prediction_abs"), default=None),
        "max_abs_target_error": max((abs(float(row["target_minus_official"]) + 15) for row in ok), default=None),
        "max_abs_component_shift_error": max((abs(float(row["component_shift_db"]) - 15) for row in ok if row.get("component_shift_db") is not None), default=None),
        "rms_snr_48": _hist(column("rms_snr_48"), [-20, -10, -5, 0, 5, 10, 15, 20, 30]),
        "official_snr": _hist(column("official_log_snr_db"), [0, 2.5, 7.5, 12.5, 17.5, 20]),
        "length_mismatch": sum(row.get("length_48") != row.get("span") for row in audited),
        "nonfinite": sum(not row.get("finite", False) for row in audited),
        "hash_mismatch": sum(not row.get("clean_sha_ok", False) or not row.get("noise_sha_ok", False) for row in audited),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "audit_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (OUT / "audit_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in bad:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
