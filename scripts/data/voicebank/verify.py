"""Check the VoiceBank SNR-shift v1 recipes. Does not write audio."""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

YAML = Path("configs/data/voicebank_snr_shift.yaml")
NOISE_TYPES = (
    "babble",
    "ssn",
    "bus",
    "cafe",
    "cafeteria",
    "car",
    "kitchen",
    "living",
    "meeting",
    "metro",
    "office",
    "psquare",
    "restaurant",
    "station",
    "traffic",
)


def yaml_scalar(text, key):
    prefix = key + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"missing {key} in {YAML}")


def dataset_root(argv_root):
    value = argv_root or os.environ.get("AUDIO_DB_DATASET_ROOT")
    if not value:
        raise SystemExit("specify --dataset-root or AUDIO_DB_DATASET_ROOT")
    return Path(value)


def load_depends(root, repo):
    depends_rel = yaml_scalar(YAML.read_text(encoding="utf-8"), "depends_rel")
    depends = json.loads((root / depends_rel).read_text(encoding="utf-8"))
    opened = {}
    for key, item in depends.items():
        if not isinstance(item, dict) or "path" not in item:
            continue
        path = root / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise SystemExit(f"{key} sha256 mismatch: {digest}")
        opened[key] = path
    recipes = [json.loads(line) for line in opened["recipes"].read_text(encoding="utf-8").splitlines()]
    common = json.loads(opened["common"].read_text(encoding="utf-8"))
    if common.get("algorithm") != "vctk-aligned-p56-peak-shift-v1" or common.get("snr_shift_db") != -15:
        raise SystemExit("common.json does not match vctk-aligned-p56-peak-shift-v1 at -15 dB")
    for row in recipes:
        for field in ("clean_path", "noisy_path", "noise_path", "source_vctk"):
            candidate = Path(row[field])
            if candidate.is_absolute() or ".." in candidate.parts or not (repo / candidate).is_file():
                raise SystemExit(f"unresolved {field} {row[field]}")
    return recipes


def read_noise(row, repo, clean_length):
    with sf.SoundFile(repo / row["noise_path"]) as handle:
        if handle.samplerate != 48000 or handle.channels != 1:
            raise SystemExit(f"noise format {row['noise_path']}")
        start = int(row["noise_start_48k"])
        end = int(row["noise_end_exclusive_48k"])
        if not 0 <= start < end <= handle.frames or end - start != clean_length:
            raise SystemExit(f"noise span {row['utterance_id']}")
        handle.seek(start)
        noise = handle.read(end - start)
    if len(noise) != clean_length or getattr(noise, "ndim", 1) != 1:
        raise SystemExit(f"noise shape {row['utterance_id']}")
    return noise


def component_delta(clean, noise, row):
    gain = float(row["clean_gain_g"])
    before = float(row["noise_gain_after_peak"])
    after = float(row["shifted_noise_gain"])

    def snr(signal, interference):
        return 10 * np.log10(np.dot(signal, signal) / np.dot(interference, interference))

    return gain, before, after, float(snr(gain * clean, after * noise) - snr(gain * clean, before * noise))


def select_rows(rows):
    if len(rows) != 12396 or len({(row["split"], row["utterance_id"]) for row in rows}) != len(rows):
        raise SystemExit("recipe count or identity check failed")
    ready = [row for row in rows if row["status"] == "ready"]
    selected = random.Random(42).sample(ready, min(40, len(ready)))
    for kind in ("babble", "ssn"):
        selected += [row for row in ready if row["noise_type"] == kind][:2]
    first = {}
    for row in rows:
        first.setdefault(row["noise_type"], row)
    missing = [kind for kind in NOISE_TYPES if kind not in first]
    if missing:
        raise SystemExit(f"missing noise types {missing}")
    return selected, [first[kind] for kind in NOISE_TYPES]


def evaluate(row, repo):
    clean, sample_rate = sf.read(repo / row["clean_path"])
    noisy, _sample_rate = sf.read(repo / row["noisy_path"])
    if sample_rate != 48000:
        raise SystemExit(f"sample rate {row['utterance_id']}")
    noise = read_noise(row, repo, len(clean))
    gain, before, after, delta = component_delta(clean, noise, row)
    base = gain * clean + before * noise
    shifted = gain * clean + after * noise
    rmse = float(np.sqrt(np.mean((base - noisy) ** 2)))
    if not (rmse < 2e-5 and abs(delta + 15) < 1e-9):
        raise SystemExit(f"check failed {row['utterance_id']} rmse={rmse} delta={delta}")
    return {
        "id": row["utterance_id"],
        "noise": row["noise_type"],
        "rmse": rmse,
        "component_snr_delta_db": delta,
        "shifted_peak": float(np.max(np.abs(shifted))),
        "reference": "g * clean",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = dataset_root(args.dataset_root)
    repo = Path(args.repo_root)
    rows = load_depends(root, repo)
    selected, by_noise = select_rows(rows)
    results = [evaluate(row, repo) for row in selected]
    peaks = [evaluate(row, repo) for row in by_noise]
    text = json.dumps(
        {
            "method": "v1; seed 42 random 40 ready plus first 2 babble and ssn; first row of each of 15 noise types; no output audio written",
            "results": results,
            "noise_types": [{key: item[key] for key in ("id", "noise", "shifted_peak")} for item in peaks],
        },
        indent=2,
    ) + "\n"
    if args.output:
        dest = Path(args.output)
        if dest.exists():
            raise SystemExit(f"refuse existing {dest}")
        v1 = (root / "recipes" / "voicebank-snr-shift-v1").resolve()
        resolved = dest.resolve()
        if resolved == v1 or v1 in resolved.parents:
            raise SystemExit("refuse write into v1")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    print(
        "verified",
        len(results),
        "noise_types",
        len(peaks),
        "max_rmse",
        max(item["rmse"] for item in results),
        "max_peak",
        max(item["shifted_peak"] for item in results + peaks),
    )


if __name__ == "__main__":
    sys.exit(main())
