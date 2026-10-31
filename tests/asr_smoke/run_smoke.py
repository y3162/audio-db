"""Real-model smoke checks. Not part of the fast unit-test run.

Usage:
    .venvs/asr/bin/python -m tests.asr_smoke.run_smoke parakeet whisper wav2vec2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr import load_recognizer
from src.evaluation.audio.io import load_waveform
from src.evaluation.datasets import EvaluationDB

CHECKS = {
    "parakeet": ("parakeet", "nvidia/parakeet-tdt-0.6b-v2", "cuda:1"),
    "parakeet-ctc": ("parakeet", "nvidia/parakeet-ctc-0.6b", "cuda:1"),
    "parakeet-rnnt": ("parakeet", "nvidia/parakeet-rnnt-0.6b", "cuda:1"),
    "whisper": ("whisper", "tiny", "cuda:1"),
    "wav2vec2": ("wav2vec2", "facebook/wav2vec2-base-960h", "cuda:1"),
    "hubert": ("hubert", "facebook/hubert-large-ls960-ft", "cuda:1"),
    "wavlm": ("wavlm", "patrickvonplaten/wavlm-libri-clean-100h-base-plus", "cuda:1"),
}


def main(argv: list[str]) -> int:
    names = argv or ["parakeet", "whisper", "wav2vec2"]
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        print("unknown checks:", ", ".join(unknown), file=sys.stderr)
        return 2
    database = EvaluationDB(ROOT / "data/db/resource.duckdb", project_root=ROOT)
    sample = database.load_dataset("libri_demand_snr0")[0]
    waveform, rate = load_waveform(sample.clean_path, 2.0)
    print(f"audio samples={waveform.numel()} rate={rate} reference={sample.transcript}")
    failures = 0
    for name in names:
        family, model_id, device = CHECKS[name]
        print(f"--- {name} {model_id} device={device}")
        started = time.perf_counter()
        try:
            recognizer = load_recognizer(family, model_id=model_id, device=device, language="en")
            result = recognizer.transcribe(waveform, rate)
        except Exception as exc:
            failures += 1
            print(f"FAIL {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        if not isinstance(result.text, str):
            failures += 1
            print(f"FAIL text type {type(result.text).__name__}")
            continue
        print(
            f"OK decode={result.decode} model_id={result.model_id} "
            f"seconds={elapsed:.1f} text={result.text!r}"
        )
        del recognizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
