"""Compare g != 1 residuals with the stored identity. Refuses an existing output."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

YAML = Path("configs/data/voicebank_snr_shift.yaml")


def yaml_scalar(text, key):
    prefix = key + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"missing {key} in {YAML}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root_value = args.dataset_root or os.environ.get("AUDIO_DB_DATASET_ROOT")
    if not root_value:
        raise SystemExit("specify --dataset-root or AUDIO_DB_DATASET_ROOT")
    root = Path(root_value)
    repo = Path(args.repo_root)
    dest = Path(args.output)
    if dest.exists():
        raise SystemExit(f"refuse existing {dest}")
    v1 = (root / "recipes" / "voicebank-snr-shift-v1").resolve()
    if dest.resolve() == v1 or v1 in dest.resolve().parents:
        raise SystemExit("refuse write into v1")
    depends = json.loads((root / yaml_scalar(YAML.read_text(encoding="utf-8"), "depends_rel")).read_text(encoding="utf-8"))
    recipes_item = depends["recipes"]
    recipes_path = root / recipes_item["path"]
    digest = hashlib.sha256(recipes_path.read_bytes()).hexdigest()
    if digest != recipes_item["sha256"]:
        raise SystemExit(f"recipes sha256 mismatch: {digest}")
    rows = [json.loads(line) for line in recipes_path.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if float(row["clean_gain_g"]) != 1]
    results = []
    for row in selected:
        clean, _sample_rate = sf.read(repo / row["clean_path"])
        noisy, _sample_rate = sf.read(repo / row["noisy_path"])
        with sf.SoundFile(repo / row["noise_path"]) as handle:
            handle.seek(int(row["noise_start_48k"]))
            noise = handle.read(len(clean))
        gain = float(row["clean_gain_g"])
        beta = float(row["noise_gain_after_peak"])
        residual = noisy - clean
        pure = beta * noise
        predicted = (gain - 1) * clean + pure
        results.append(
            {
                "id": row["utterance_id"],
                "g": gain,
                "naive_residual_vs_noise_rmse": float(np.sqrt(np.mean((residual - pure) ** 2))),
                "identity_residual_rmse": float(np.sqrt(np.mean((residual - predicted) ** 2))),
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"count": len(results), "cases": results}, indent=2) + "\n", encoding="utf-8")
    print(
        "count",
        len(results),
        "max naive",
        max(item["naive_residual_vs_noise_rmse"] for item in results),
        "max identity",
        max(item["identity_residual_rmse"] for item in results),
    )


if __name__ == "__main__":
    sys.exit(main())
