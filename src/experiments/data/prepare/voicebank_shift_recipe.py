"""Normalize the recorded SNR-shift parameters without rounding the gains."""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest


def canonical_shift_recipe(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a content-addressable recipe; the identifier is not part of it.

    Gains are decimal strings so the shared six-decimal JSON normalization
    does not change the recorded binary64 values.
    """
    if row.get("status") != "ready":
        raise ReproductionError("SNR-shift recipe is not ready")

    def source(path_key: str, hash_key: str) -> dict[str, str]:
        value, sha = row.get(path_key), row.get(hash_key)
        if not isinstance(value, str) or not value or "\\" in value:
            raise ReproductionError(f"invalid {path_key}")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not value.startswith("data/corpora/"):
            raise ReproductionError(f"invalid {path_key}")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ReproductionError(f"invalid {hash_key}")
        return {"path": value, "sha256": sha}

    def gain(key: str) -> str:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ReproductionError(f"invalid {key}")
        try:
            number = float(value)
        except ValueError as exc:
            raise ReproductionError(f"invalid {key}") from exc
        if not math.isfinite(number):
            raise ReproductionError(f"invalid {key}")
        return repr(number)

    def index(key: str) -> int:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ReproductionError(f"invalid {key}")
        try:
            return int(value)
        except ValueError as exc:
            raise ReproductionError(f"invalid {key}") from exc

    start, end = index("noise_start_48k"), index("noise_end_exclusive_48k")
    if start < 0 or end <= start:
        raise ReproductionError("invalid noise span")
    return {
        "algorithm": "vctk-aligned-p56-peak-shift-v1",
        "sample_rate": 48000,
        "sources": {"clean": source("clean_path", "clean_sha256"),
                    "noise": source("noise_path", "noise_sha256")},
        "noise_start": start,
        "noise_end_exclusive": end,
        "clean_gain": gain("clean_gain_g"),
        "shifted_noise_gain": gain("shifted_noise_gain"),
        "reference": "scaled_clean",
    }


def shift_case_id(row: Mapping[str, Any]) -> str:
    return digest(canonical_shift_recipe(row))


def snapshot_realization(staged_case: Mapping[str, Any]) -> dict[str, Any]:
    """Represent one staged recipe as a 16 kHz snapshot realization document.

    This constructs the document only. The algorithm is not yet registered in
    the reproduction graph or connected to a training data module.
    """
    from src.experiments.data.prepare.resample import soxr_hq_frame_count

    recipe = staged_case.get("recipe")
    if not isinstance(recipe, dict) or staged_case.get("case_id") != digest(recipe):
        raise ReproductionError("staged SNR-shift case does not match its recipe")
    if recipe.get("algorithm") != "vctk-aligned-p56-peak-shift-v1" or recipe.get("sample_rate") != 48000:
        raise ReproductionError("staged SNR-shift algorithm is invalid")
    try:
        clean = recipe["sources"]["clean"]
        noise = recipe["sources"]["noise"]
        native_length = recipe["noise_end_exclusive"] - recipe["noise_start"]
        if native_length < 1:
            raise ValueError("empty noise span")
        sources = {}
        for name, item in (("clean", clean), ("noise", noise)):
            path = item["path"]
            if (not isinstance(path, str) or not path.startswith("data/corpora/")
                    or ".." in PurePosixPath(path).parts or "\\" in path):
                raise ValueError("invalid source path")
            sha = item["sha256"]
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("invalid source hash")
            sources[name] = {"asset": "corpora/" + path.removeprefix("data/corpora/"),
                             "sha256": sha}
        return {
            "realization_version": 1,
            "algorithm": "fixed_pair.voicebank_snr_shift",
            "algorithm_version": 1,
            "sample_rate": 16000,
            "sources": sources,
            "draws": {"source_recipe_case_id": staged_case["case_id"],
                      "noise_start_48k": recipe["noise_start"],
                      "noise_end_exclusive_48k": recipe["noise_end_exclusive"],
                      "clean_gain": recipe["clean_gain"],
                      "shifted_noise_gain": recipe["shifted_noise_gain"]},
            "steps": ["vctk-aligned-p56-peak-shift-v1", "soxr_hq_float32_48000_to_16000"],
            "output": {"length": soxr_hq_frame_count(native_length), "teacher": "dry_clean"},
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ReproductionError("invalid staged SNR-shift realization") from exc


def build_staged_index(source: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one split and recipe index for all four models without writing files."""
    from src.experiments.data.prepare.voicebank_shift_audit import HOLDOUT, audit_recipes

    content = source.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ReproductionError("SNR-shift source recipe SHA-256 mismatch")
    try:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReproductionError("SNR-shift recipe JSONL is invalid") from exc
    audit = audit_recipes(rows)
    expected = {"train": 10700, "valid": 872, "test": 824}
    if {name: row["utterances"] for name, row in audit["splits"].items()} != expected:
        raise ReproductionError("SNR-shift split counts do not match the declared split")
    if any(c["shared_utterance_ids"] or c["shared_speakers"] for c in audit["comparisons"].values()):
        raise ReproductionError("SNR-shift splits share an utterance or speaker")
    index = []
    for row in rows:
        utterance = row["utterance_id"]
        speaker = utterance.split("_", 1)[0]
        split = "valid" if row["split"] == "train" and speaker in HOLDOUT else row["split"]
        recipe = canonical_shift_recipe(row)
        index.append({"case_id": digest(recipe), "split": split, "utterance_id": utterance,
                      "speaker_id": speaker, "recipe": recipe})
    if len({r["case_id"] for r in index}) != len(index):
        raise ReproductionError("SNR-shift source contains duplicate normalized recipes")
    metadata = {
        "format": "voicebank-snr-shift-staging-v1",
        "source_sha256": actual,
        "case_count": len(index),
        "splits": expected,
        "models": ["mp_senet", "semambapp", "tf_locoformer", "tf_gridnet"],
        "selection_split": "valid",
        "selection_uses_test": False,
        "noise_overlap_audit": audit,
        "note": ("speaker and utterance splits are disjoint; validation noise overlaps training noise"
                 if audit["validation_noise_independent"] is False else
                 "speaker and utterance splits are disjoint; see noise overlap audit"),
    }
    return metadata, index


def write_staged_index(source: Path, expected_sha256: str, destination: Path) -> None:
    """Write a new persistent staging directory and refuse all existing paths."""
    metadata, index = build_staged_index(source, expected_sha256)
    destination = Path(destination)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ReproductionError("SNR-shift staging destination already exists") from exc
    with (destination / "cases.jsonl").open("x", encoding="utf-8") as handle:
        for row in index:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    metadata["cases_sha256"] = hashlib.sha256((destination / "cases.jsonl").read_bytes()).hexdigest()
    # The metadata is written last; its absence marks an incomplete staging.
    with (destination / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def read_staged_index(destination: Path, source: Path, expected_source_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a complete staging index, checking its content and split identity."""
    from collections import Counter

    from src.experiments.data.prepare.voicebank_shift_audit import HOLDOUT, audit_recipes

    destination = Path(destination)
    try:
        metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        content = (destination / "cases.jsonl").read_bytes()
        cases = [json.loads(line) for line in content.splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReproductionError("SNR-shift staging is incomplete or invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("format") != "voicebank-snr-shift-staging-v1":
        raise ReproductionError("SNR-shift staging format is invalid")
    if metadata.get("source_sha256") != expected_source_sha256:
        raise ReproductionError("SNR-shift staging source hash does not match")
    if metadata.get("cases_sha256") != hashlib.sha256(content).hexdigest():
        raise ReproductionError("SNR-shift staging case hash does not match")
    if metadata.get("selection_split") != "valid" or metadata.get("selection_uses_test") is not False:
        raise ReproductionError("SNR-shift staging selection must use validation only")
    if metadata.get("models") != ["mp_senet", "semambapp", "tf_locoformer", "tf_gridnet"]:
        raise ReproductionError("SNR-shift staging model list does not match")
    rows = []
    seen = set()
    for item in cases:
        try:
            recipe = item["recipe"]
            utterance, speaker, split = item["utterance_id"], item["speaker_id"], item["split"]
            if not isinstance(utterance, str) or speaker != utterance.split("_", 1)[0]:
                raise ValueError("invalid speaker")
            if split not in ("train", "valid", "test") or (split == "valid") != (speaker in HOLDOUT):
                raise ValueError("invalid split")
            if item["case_id"] != digest(recipe) or item["case_id"] in seen:
                raise ValueError("invalid case id")
            seen.add(item["case_id"])
            source_split = "train" if split == "valid" else split
            rows.append({"utterance_id": utterance, "split": source_split, "status": "ready",
                         "noise_sha256": recipe["sources"]["noise"]["sha256"],
                         "noise_start_48k": recipe["noise_start"],
                         "noise_end_exclusive_48k": recipe["noise_end_exclusive"]})
        except (KeyError, TypeError, ValueError) as exc:
            raise ReproductionError("SNR-shift staging case is invalid") from exc
    counts = dict(Counter(case["split"] for case in cases))
    if counts != {"train": 10700, "valid": 872, "test": 824} or metadata.get("case_count") != len(cases):
        raise ReproductionError("SNR-shift staging case counts do not match")
    if metadata.get("splits") != counts or metadata.get("noise_overlap_audit") != audit_recipes(rows):
        raise ReproductionError("SNR-shift staging audit does not match its cases")
    expected_metadata, expected_cases = build_staged_index(source, expected_source_sha256)
    if cases != expected_cases or any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ReproductionError("SNR-shift staging does not match the canonical source")
    return metadata, cases


def main() -> int:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Stage or verify the canonical VoiceBank SNR-shift recipes")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/data/voicebank_snr_shift.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.dataset_root / config["recipes_rel"]
    expected = config["sha256"]["recipes.jsonl"]
    if args.verify:
        metadata, _ = read_staged_index(args.staging, source, expected)
        print(f"verified cases={metadata['case_count']} sha256={metadata['cases_sha256']}")
    else:
        write_staged_index(source, expected, args.staging)
        print(f"staged {args.staging}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
