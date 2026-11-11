"""Parallel draws for the DeepFilterNet3 distribution window.

Each index has its own PCG64 seed. Workers do not share a random stream.
The histogram is the sum of those independent draws.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from src.experiments.data.protocols.online_dfn3 import (
    NUM_SAMPLES,
    SAMPLE_RATE,
    DFN3OnlineSpec,
    render_mixture,
)

_STATE: dict[str, object] = {}
_THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class _Clip:
    def __init__(self, data) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.sample_rate = SAMPLE_RATE
        self.frames = int(self.data.shape[0])

    def read(self, start: int, stop: int) -> np.ndarray:
        return self.data[int(start) : int(stop)]


def slices(count: int, workers: int) -> list[tuple[int, int]]:
    """Half-open index ranges. Every index in ``0 .. count-1`` appears once."""
    count = int(count)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count == 0:
        return []
    width = max(1, min(int(workers), count))
    size = (count + width - 1) // width
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < count:
        stop = min(count, start + size)
        bounds.append((start, stop))
        start = stop
    return bounds


def _init_worker(seed: int) -> None:
    source = np.random.default_rng(int(seed))
    _STATE["spec"] = DFN3OnlineSpec.shipped()
    _STATE["speech"] = [_Clip(source.normal(0.0, 0.1, NUM_SAMPLES + 8000)) for _ in range(4)]
    _STATE["noise"] = [_Clip(source.normal(0.0, 0.1, NUM_SAMPLES + 8000)) for _ in range(4)]
    _STATE["rir"] = [_Clip(source.normal(0.0, 0.01, 512))]


def _render_slice(bounds: tuple[int, int]) -> dict[str, object]:
    start, stop = bounds
    spec = _STATE["spec"]
    speech = _STATE["speech"]
    noise = _STATE["noise"]
    rir = _STATE["rir"]
    snr: dict[int, int] = {}
    gain: dict[int, int] = {}
    noises: dict[int, int] = {}
    reverb = 0
    synthetic = 0
    noise_total = 0
    snr_error_max: float | None = None
    snr_error_count = 0
    failures: list[str] = []
    for index in range(int(start), int(stop)):
        mixture = render_mixture(
            spec,
            speech,
            noise,
            rir,
            speech_index=index % 4,
            split="train",
            rng=np.random.Generator(np.random.PCG64(int(_STATE["seed_base"]) + index)),
        )
        snr[int(mixture.snr_db)] = snr.get(int(mixture.snr_db), 0) + 1
        gain[int(mixture.gain_db)] = gain.get(int(mixture.gain_db), 0) + 1
        noises[int(mixture.n_noises)] = noises.get(int(mixture.n_noises), 0) + 1
        reverb += int(mixture.reverb_speech)
        synthetic += int(mixture.synthetic_noises)
        noise_total += int(mixture.n_noises)
        if mixture.noisy.shape != (NUM_SAMPLES,) or not np.isfinite(mixture.noisy).all():
            failures.append(f"{index}: noisy waveform is not a finite {NUM_SAMPLES}-sample buffer")
        elif float(np.max(np.abs(mixture.noisy))) > 1.0 + 1e-4:
            failures.append(f"{index}: noisy peak exceeds 1")
        if mixture.noise_only:
            if mixture.measured_snr_db is not None or not np.all(mixture.clean == 0):
                failures.append(f"{index}: noise-only draw kept a target or a measured SNR")
        elif mixture.measured_snr_db is None:
            failures.append(f"{index}: measured SNR is missing")
        else:
            error = abs(float(mixture.measured_snr_db) - float(mixture.snr_db))
            snr_error_max = error if snr_error_max is None else max(snr_error_max, error)
            snr_error_count += 1
    return {
        "snr": snr,
        "gain": gain,
        "noises": noises,
        "reverb": reverb,
        "synthetic": synthetic,
        "noise_total": noise_total,
        "snr_error_max": snr_error_max,
        "snr_error_count": snr_error_count,
        "failures": failures,
    }


def _init_and_seed(seed: int) -> None:
    _init_worker(seed)
    _STATE["seed_base"] = int(seed)


def render_distribution(count: int, seed: int, *, workers: int) -> dict[str, object]:
    """Render ``count`` mixtures and return histogram totals."""
    bounds = slices(count, workers)
    previous = {key: os.environ.get(key) for key in _THREAD_KEYS}
    for key in _THREAD_KEYS:
        os.environ[key] = "1"
    try:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, len(bounds)),
            mp_context=context,
            initializer=_init_and_seed,
            initargs=(int(seed),),
        ) as pool:
            parts = list(pool.map(_render_slice, bounds))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return _merge(parts)


def _merge(parts: list[dict[str, object]]) -> dict[str, object]:
    spec = DFN3OnlineSpec.shipped()
    snr = {int(value): 0 for value in spec.snrs}
    gain = {int(value): 0 for value in spec.gains}
    noises = {value: 0 for value in range(2, 6)}
    reverb = 0
    synthetic = 0
    noise_total = 0
    snr_error_max: float | None = None
    snr_error_count = 0
    failures: list[str] = []
    for part in parts:
        for key, value in part["snr"].items():
            snr[int(key)] += int(value)
        for key, value in part["gain"].items():
            gain[int(key)] += int(value)
        for key, value in part["noises"].items():
            noises[int(key)] += int(value)
        reverb += int(part["reverb"])
        synthetic += int(part["synthetic"])
        noise_total += int(part["noise_total"])
        snr_error_count += int(part["snr_error_count"])
        if part["snr_error_max"] is not None:
            error = float(part["snr_error_max"])
            snr_error_max = error if snr_error_max is None else max(snr_error_max, error)
        failures.extend(part["failures"])
    return {
        "snr": snr,
        "gain": gain,
        "noises": noises,
        "reverb": reverb,
        "synthetic": synthetic,
        "noise_total": noise_total,
        "snr_error_max": snr_error_max,
        "snr_error_count": snr_error_count,
        "failures": failures,
    }
