"""Regenerate babble/ssn position search into an empty directory."""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import concurrent.futures

import numpy as np
import soundfile as sf
from scipy.signal import correlate, fftconvolve, resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asl_P56_fast import asl_P56

YAML = pathlib.Path("configs/data/voicebank_snr_shift.yaml")
CORPUS_NOISE = {
    "babble": pathlib.Path("data/corpora/Babble/babble.wav"),
    "ssn": pathlib.Path("data/corpora/SSN/ssn.wav"),
}
noise = {}
n16 = {}
sq = {}


def yaml_scalar(text, key):
    prefix = key + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"missing {key}")


def run(record):
    try:
        kind = record["obs_noise_type"]
        clean, sample_rate = sf.read(record["obs_clean_asset"])
        noisy, _sample_rate = sf.read(record["obs_noisy_asset"])
        length = len(clean)
        residual16 = resample_poly(noisy - clean, 1, 3)
        reference = n16[kind]
        energy = sq[kind][len(residual16) :] - sq[kind][: -len(residual16)]
        correlation = fftconvolve(reference, residual16[::-1], mode="valid")
        score = correlation * correlation / np.maximum(energy, 1e-30)
        index = int(np.argmax(score))
        score[max(0, index - len(residual16)) : index + len(residual16) + 1] = 0
        second = float(np.sqrt(max(score) / np.dot(residual16, residual16)))
        low = max(0, index * 3 - 8)
        high = min(len(noise[kind]) - length, index * 3 + 8)
        window = noise[kind][low : high + length]
        fine = fftconvolve(window, (noisy - clean)[::-1], mode="valid")
        window_energy = np.r_[0.0, np.cumsum(window * window)]
        window_energy = window_energy[length:] - window_energy[:-length]
        fine_index = int(np.argmax(fine * fine / np.maximum(window_energy, 1e-30)))
        start = low + fine_index
        segment = noise[kind][start : start + length]
        raw_path = pathlib.Path("data/corpora/VCTK/wav48") / record["obs_utterance_id"].split("_")[0] / (record["obs_utterance_id"] + ".wav")
        raw, _sample_rate = sf.read(raw_path)
        offset = int(np.argmax(abs(correlate(raw, clean, mode="valid", method="fft"))))
        raw = raw[offset : offset + length]
        raw = raw * np.max(abs(clean)) / np.max(abs(raw))
        level, _, _ = asl_P56(raw, sample_rate, 16)
        beta = np.sqrt(level / np.mean(segment * segment) / 10 ** (float(record["obs_original_snr_db"]) / 10))
        mixture = clean + beta * segment
        gain = 1.0
        steps = 0
        for factor in (1.1, 1.2, 1.3, 1.4):
            if max(abs(mixture)) * gain <= 1:
                break
            gain /= factor
            steps += 1
        diff = gain * mixture - noisy
        fit = np.linalg.lstsq(np.column_stack((clean, segment)), noisy, rcond=None)[0]
        fit_error = noisy - fit[0] * clean - fit[1] * segment
        return dict(
            id=record["obs_utterance_id"],
            kind=kind,
            start=start,
            end=start + length,
            ncc16=float(np.sqrt(max(correlation[index] * correlation[index] / max(energy[index], 1e-30), 0)) / np.linalg.norm(residual16)),
            second_ncc16=second,
            beta=float(beta),
            g=gain,
            steps=steps,
            rmse=float(np.sqrt(np.mean(diff * diff))),
            max_abs=float(np.max(abs(diff))),
            fit_alpha=float(fit[0]),
            fit_beta=float(fit[1]),
            fit_rmse=float(np.sqrt(np.mean(fit_error * fit_error))),
            clean_sha256=hashlib.sha256(pathlib.Path(record["obs_clean_asset"]).read_bytes()).hexdigest(),
            raw_path=str(raw_path),
            raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            raw_alignment_offset=offset,
            p56_ms=float(level),
            status="ok",
        )
    except Exception as exc:
        return dict(id=record["obs_utterance_id"], kind=record["obs_noise_type"], status="error", error=repr(exc))


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
    with (investigation / index["utterances_csv"]).open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["obs_noise_type"] in CORPUS_NOISE]
    global noise, n16, sq
    noise = {kind: sf.read(path)[0] for kind, path in CORPUS_NOISE.items()}
    n16 = {kind: resample_poly(values, 1, 3) for kind, values in noise.items()}
    sq = {kind: np.r_[0.0, np.cumsum(values * values)] for kind, values in n16.items()}
    dest.mkdir(parents=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool, (dest / "artificial_search.jsonl").open("w", encoding="utf-8") as handle:
        for number, value in enumerate(pool.map(run, rows, chunksize=1)):
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            if number % 100 == 0:
                print(number, value, flush=True)


if __name__ == "__main__":
    main()
