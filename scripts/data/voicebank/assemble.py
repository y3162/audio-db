"""Assemble a recipe directory from the investigation archive. Refuses an existing output."""
import argparse
import collections
import csv
import hashlib
import json
import os
import pathlib
import platform
import sys

YAML = pathlib.Path("configs/data/voicebank_snr_shift.yaml")
ARTIFICIAL = {
    "babble": pathlib.Path("data/corpora/Babble/babble.wav"),
    "ssn": pathlib.Path("data/corpora/SSN/ssn.wav"),
}


def yaml_scalar(text, key):
    prefix = key + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"missing {key}")


def corpus_demand(mix_source):
    name = pathlib.Path(mix_source).name
    suffix = "_ch01.wav"
    if not name.endswith(suffix):
        raise SystemExit(f"unmapped source {mix_source}")
    return pathlib.Path("data/corpora/DEMAND48") / name[: -len(suffix)] / "ch01.wav"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = pathlib.Path(args.dataset_root)
    dest = pathlib.Path(args.output)
    if dest.exists():
        raise SystemExit(f"refuse existing {dest}")
    resolved = dest.resolve()
    for relative in (
        "recipes/voicebank-snr-shift-v1",
        "recipes/voicebank-snr-shift-v2",
        "investigations/voicebank-20261112",
    ):
        protected = (dataset / relative).resolve()
        if resolved == protected or protected in resolved.parents:
            raise SystemExit(f"refuse write under {protected}")
    investigation = dataset / yaml_scalar(YAML.read_text(encoding="utf-8"), "investigation_rel")
    index = json.loads((investigation / "INDEX.json").read_text(encoding="utf-8"))
    utterances = investigation / index["utterances_csv"]
    unresolved = investigation / index["unresolved_analysis"]
    artificial_path = investigation / index["artificial_search"]
    rows = list(csv.DictReader(utterances.open(encoding="utf-8")))
    artificial = {record["id"]: record for line in artificial_path.open(encoding="utf-8") for record in [json.loads(line)]}
    old = {
        (record["split"], record["utterance_id"]): record
        for line in unresolved.open(encoding="utf-8")
        for record in [json.loads(line)]
    }
    hashes = {kind: file_sha(path) for kind, path in ARTIFICIAL.items()}
    out = []
    reviewed = []
    for record in rows:
        utterance_id = record["obs_utterance_id"]
        kind = record["obs_noise_type"]
        is_artificial = kind in hashes
        found = artificial.get(utterance_id, {})
        old_status = record["position_status"]
        new_status = old_status
        delta = None
        if not is_artificial and old_status == "unresolved":
            previous = old[(record["obs_split"], utterance_id)]
            delta = float(record["obs_v2_snr_difference_db"]) + float(record["p56_new_p56_db"]) - float(record["mix_p56_db"])
            ratio = float(record["obs_v2_second_first_ratio"])
            passed = (
                abs(delta) < 0.05
                and ratio >= 50
                and float(record["p56_new_rmse"]) < 2e-5
                and int(record["obs_v2_start_48k"]) == int(record["mix_start_sample"])
            )
            new_status = "verified" if passed else "unresolved"
            reviewed.append(
                dict(
                    id=utterance_id,
                    old_status=old_status,
                    new_status=new_status,
                    source=str(corpus_demand(record["mix_source"])),
                    source_sha256=record["mix_source_sha256"],
                    channel=1,
                    start=record["inferred_start_sample_48k"],
                    end_exclusive=record["inferred_end_sample_48k"],
                    official_snr_db=record["obs_original_snr_db"],
                    old_p56_snr_difference_db=record["obs_v2_snr_difference_db"],
                    latest_p56_fitted_snr_difference_db=delta,
                    rmse=record["p56_new_rmse"],
                    competitor_ratio=ratio,
                    competitors=previous.get("two_gain", {}).get("ranked_candidates", []),
                    local_competitor=previous.get("near_competitor"),
                    reason="旧二係数候補・位置一致・競合比と新P.56入力のSNR差を検証" if passed else "基準不足",
                )
            )
        if not is_artificial:
            source = corpus_demand(record["mix_source"])
            source_sha = file_sha(source)
            if source_sha != record["mix_source_sha256"]:
                raise SystemExit(f"source sha mismatch {source}")
            start = record["inferred_start_sample_48k"]
            end = record["inferred_end_sample_48k"]
            gain = record["estimated_common_clean_gain"]
            noise_gain = record["estimated_noise_gain_after_peak"]
            beta = record["estimated_noise_gain_before_peak"]
            rmse = record["p56_new_rmse"]
            clean_sha = record["mix_clean_sha256"]
            steps = record["p56_new_steps"]
            raw = record["p56_raw_vctk_path"]
            raw_sha = record["p56_raw_vctk_sha256"]
            offset = record["p56_raw_alignment_offset"]
            second = record["obs_candidate_second_ncc16"]
            evidence = "prior full-corpus validation"
            status = "ready" if new_status == "verified" else "needs_review"
            reason = "位置と最新共通モデルの検証済み候補" if status == "ready" else "位置候補の競合またはSNR検証が不十分"
        else:
            source = ARTIFICIAL[kind] if found.get("status") == "ok" else ""
            source_sha = hashes[kind] if source else ""
            start = found.get("start")
            end = found.get("end")
            gain = found.get("g")
            beta = found.get("beta")
            noise_gain = gain * beta if gain is not None and beta is not None else None
            rmse = found.get("rmse")
            clean_sha = found.get("clean_sha256")
            steps = found.get("steps")
            raw = found.get("raw_path")
            raw_sha = found.get("raw_sha256")
            offset = found.get("raw_alignment_offset")
            second = found.get("second_ncc16")
            evidence = "16k global single-gain search, 48k local refine, P56 reconstruction and two-gain fit"
            if found.get("status") != "ok":
                status = "blocked"
                reason = "位置探索または入力読み込み失敗: " + str(found.get("error", "not processed"))
                new_status = "unknown"
            elif (
                rmse < 2e-5
                and found["max_abs"] < 1 / 32768 * 1.5
                and found["fit_rmse"] < 2e-5
                and found["ncc16"] > 0.98
                and found["second_ncc16"] < 0.5
                and found["fit_beta"] > 0
            ):
                status = "ready"
                reason = "公式音声と元素材の位置探索・共通モデル・競合候補に整合"
                new_status = "verified"
            else:
                status = "needs_review"
                reason = "素材の位置候補または共通モデルの再構成が基準未達"
                new_status = "unresolved"
        shifted = float(noise_gain) * 10 ** 0.75 if noise_gain not in ("", None) else None
        out.append(
            dict(
                utterance_id=utterance_id,
                split=record["obs_split"],
                noise_type=kind,
                clean_path=record["obs_clean_asset"],
                clean_sha256=clean_sha,
                noisy_path=record["obs_noisy_asset"],
                noise_path=str(source),
                noise_sha256=source_sha,
                noise_channel_1based=1 if source else None,
                noise_start_48k=start,
                noise_end_exclusive_48k=end,
                official_log_snr_db=float(record["obs_original_snr_db"]),
                target_log_snr_db=float(record["obs_original_snr_db"]) - 15,
                clean_gain_g=gain,
                noise_gain_before_peak_beta=beta,
                noise_gain_after_peak=noise_gain,
                shifted_noise_gain=shifted,
                peak_steps=steps,
                reconstruction_rmse=rmse,
                position_old=old_status,
                position_new=new_status,
                latest_p56_fitted_snr_difference_db=delta,
                second_ncc16=second,
                source_vctk=raw,
                source_vctk_sha256=raw_sha,
                vctk_alignment_offset=offset,
                evidence=evidence,
                status=status,
                reason=reason,
                artificial_fit_alpha=found.get("fit_alpha"),
                artificial_fit_beta=found.get("fit_beta"),
                artificial_fit_rmse=found.get("fit_rmse"),
                artificial_max_abs_error=found.get("max_abs"),
            )
        )
    if len(out) != 12396 or len({(row["split"], row["utterance_id"]) for row in out}) != 12396 or len(reviewed) != 16:
        raise SystemExit(f"unexpected assembly {len(out)} reviewed={len(reviewed)}")
    dest.mkdir(parents=True)
    for filename, data in (
        ("recipes.jsonl", out),
        ("demand16_review.jsonl", reviewed),
        ("artificial_results.jsonl", [row for row in out if row["noise_type"] in hashes]),
    ):
        with (dest / filename).open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    counts = collections.Counter(row["status"] for row in out)
    (dest / "counts.json").write_text(
        json.dumps(
            {
                "total": len(out),
                "statuses": counts,
                "by_noise_status": dict(collections.Counter(row["noise_type"] + "/" + row["status"] for row in out)),
                "demand16": dict(collections.Counter(row["new_status"] for row in reviewed)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "algorithm": "vctk-aligned-p56-peak-shift-v1",
        "formula": "shifted = g * clean + shifted_noise_gain * noise[start:end]; reference = g * clean",
        "snr_shift_db": -15,
        "noise_multiplier": 10 ** 0.75,
        "sample_rate_hz": 48000,
        "output": "IEEE float WAV, no clipping or renormalization",
        "paths": "relative to repository root",
        "end": "exclusive",
        "noise_channel": "1-based",
        "provenance": "g, beta, position are inferred, not official code; target_log_snr is a label relative to log, actual definition is component SNR difference",
        "missing": "null or empty means unknown; not zero",
        "residual": "noisy-clean=(g-1)*clean+g*beta*noise+quantization error; not independent source",
    }
    (dest / "common.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    inputs = [utterances, unresolved, artificial_path, *ARTIFICIAL.values(), pathlib.Path(__file__)]
    (dest / "manifest.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "files_sha256": {str(path): file_sha(path) for path in inputs},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(counts, collections.Counter(row["new_status"] for row in reviewed))


if __name__ == "__main__":
    main()
