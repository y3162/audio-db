"""Fixed 64-utterance slice of voicebank_28_valid.

Cells are (speaker, noise, SNR) for validation speakers p286 and p287.
There are 10 noises and 4 SNRs, so 80 cells. The two alphabetically last
noises, station and traffic, are left out, leaving 8 x 4 x 2 = 64. Within a
cell the utterance is chosen by SHA-256(seed, speaker, noise, SNR), not by a
score. Test speakers and the train split are checked for overlap.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.data.dataset import load_recipe_dataset

DB = ROOT / "data/db/semamba-voicebank-train-20261101.duckdb"
LOG = ROOT / "data/corpora/VoiceBank-DEMAND/log_trainset_28spk.txt"
SEED = 1234
VALID_SPEAKERS = ("p286", "p287")
TEST_SPEAKERS = {"p232", "p257"}
DROPPED_NOISES = ("station", "traffic")


def main() -> None:
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    conditions = {}
    for line in LOG.read_text(encoding="utf-8").splitlines():
        stem, noise, snr = line.split()
        conditions[stem] = {"noise": noise, "snr_db": int(snr)}
    valid = load_recipe_dataset(DB, "voicebank_28_valid", role="validation", seed=SEED, sample_rate=16000, segment_size=None, training=False)
    train = load_recipe_dataset(DB, "voicebank_28_train", role="train", seed=SEED, sample_rate=16000, segment_size=None, training=False)
    train_ids = {example.case_id for example in train.examples}
    cells: dict[tuple, list[dict]] = {}
    speakers = set()
    for example in valid.examples:
        stem = Path(example.recipe["sources"]["clean"]["path"]).stem
        speaker = stem.split("_")[0]
        speakers.add(speaker)
        if example.case_id in train_ids:
            raise SystemExit(f"train overlap {example.case_id}")
        info = conditions[stem]
        cells.setdefault((speaker, info["noise"], info["snr_db"]), []).append(
            {"case_id": example.case_id, "filename": stem, "speaker": speaker, **info}
        )
    if speakers != set(VALID_SPEAKERS):
        raise SystemExit(f"unexpected validation speakers {sorted(speakers)}")
    if speakers & TEST_SPEAKERS:
        raise SystemExit("validation includes a test speaker")
    kept = []
    for key in sorted(cells):
        speaker, noise, snr = key
        if noise in DROPPED_NOISES:
            continue
        rows = sorted(cells[key], key=lambda row: row["filename"])
        digest = hashlib.sha256(f"{SEED}:{speaker}:{noise}:{snr}".encode()).digest()
        picked = rows[int.from_bytes(digest[:8], "big") % len(rows)]
        picked["cell_size"] = len(rows)
        kept.append(picked)
    if len(kept) != 64:
        raise SystemExit(f"expected 64 utterances, got {len(kept)}")
    payload = {
        "seed": SEED,
        "database": str(DB),
        "validation_set": "voicebank_28_valid",
        "validation_count": len(valid),
        "train_count": len(train),
        "train_overlap": 0,
        "validation_speakers": list(VALID_SPEAKERS),
        "test_speakers_excluded": sorted(TEST_SPEAKERS),
        "dropped_noises": list(DROPPED_NOISES),
        "drop_rule": "alphabetically last noise names, fixed before scoring",
        "within_cell": "SHA-256 of seed, speaker, noise, and SNR",
        "utterances": kept,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out, len(kept), flush=True)


if __name__ == "__main__":
    main()
