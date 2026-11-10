"""DNS-Challenge Interspeech 2020 mixtures for TF-Locoformer.

The mix follows ``audiolib.snr_mixer`` and ``noisyspeech_synthesizer_multiprocessing.py``
at DNS-Challenge ``70f19285`` (``interspeech2020/master``). Clean and noise are
peak-normalized, scaled to -25 dBFS, mixed at an integer SNR, then scaled to an
integer target level. A peak above 0.99 scales all three waveforms down.
Short pieces are joined with 0.2 s of silence. Output is PCM_16.

The official script seeds ``random`` and ``numpy`` once, then hands a shared
file counter to a process pool. Which file a worker reads depends on
scheduling, and forked workers can share one NumPy stream. This builder does
not replay that stream. Each ``gen_index`` draws crops, SNR, and level from
``Generator(PCG64(mix_seed('dns', seed, gen_index)))``, and file order is one
sorted-then-shuffled list. Stored spans are a pure function of that seed and
the catalog lengths. Sample-dependent clipping and activity retries are not
folded into the plan: official retries are not bit-reproducible. Rendering
uses the stored spans, then the same ``snr_mixer`` and PCM_16 rounding.
"""

from __future__ import annotations

import contextvars
import io
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import build_id_for, digest, validate_realization
from src.experiments.data.sampling.seeds import mix_seed

GENERATOR = "dns2020_synth"
GENERATOR_VERSION = 1
ALGORITHM = "pregenerated.dns_tflocoformer"
SAMPLE_RATE = 16000
AUDIO_SECONDS = 30.0
SILENCE_SECONDS = 0.2
SNR_LOWER = -5
SNR_UPPER = 15
LEVEL_LOWER = -35
LEVEL_UPPER = -15
CLEAN_ACTIVITY = 0.6
NOISE_ACTIVITY = 0.0
CLIP_THRESHOLD = 0.99
MAX_PIECE_TRIES = 50
MAX_CASE_ATTEMPTS = 50
RETRY_NONE = "length_plan_no_sample_retry"
RETRY_ACCEPT = "deterministic_case_stream_v1"
RENDERER_V1 = "dns_pcm16_v1"
RENDERER_ACCEPT = "dns_pcm16_accept_v2"
RESAMPLE_ID = "librosa_1.0.0_soxr_hq"
MAX_NAME = 100
EPS = float(np.finfo(float).eps)
DNS_COMMIT = "70f19285c36cca4df2338f9248775ddc50980c6b"
ESPNET_COMMIT = "90eed8e53498e7af682bc6ff39d9067ae440d6a4"
UTTID_RE = re.compile(r".*reader_([0-9]*)_.*_([0-9]*)\.wav$", re.IGNORECASE)
READER_RE = re.compile(r"reader_([0-9]+)", re.IGNORECASE)
_PROFILE: contextvars.ContextVar["DnsProfile | None"] = contextvars.ContextVar("dns_accept_profile", default=None)


class DnsProfile:
    """Per-process timers for one acceptance plan. Absent unless a caller opts in."""

    __slots__ = ("io", "decode", "resample", "activity", "clip")

    def __init__(self) -> None:
        self.io = 0.0
        self.decode = 0.0
        self.resample = 0.0
        self.activity = 0.0
        self.clip = 0.0

    def add(self, bucket: str, seconds: float) -> None:
        setattr(self, bucket, getattr(self, bucket) + float(seconds))

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__slots__}


class profile_acceptance:
    """Time decoder, resample, activity, and clipping work inside this block."""

    def __enter__(self) -> DnsProfile:
        self.profile = DnsProfile()
        self.token = _PROFILE.set(self.profile)
        return self.profile

    def __exit__(self, exc_type, exc, traceback) -> None:
        _PROFILE.reset(self.token)


def _profile_add(bucket: str, started: float) -> None:
    profile = _PROFILE.get()
    if profile is not None:
        profile.add(bucket, time.perf_counter() - started)


def python_shuffle(names: Sequence[str], seed: int) -> list[str]:
    """Sort, then shuffle with ``random.Random``. Official ``glob`` order is not stable."""
    ordered = sorted(names)
    random.Random(int(seed)).shuffle(ordered)
    return ordered


def shuffle_pools(clean: Sequence[str], noise: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    """One ``random.Random`` shuffles clean, then noise. Official uses one ``random.seed`` too."""
    rng = random.Random(int(seed))
    clean_names = sorted(clean)
    noise_names = sorted(noise)
    rng.shuffle(clean_names)
    rng.shuffle(noise_names)
    return clean_names, noise_names


def is_clipped(audio: np.ndarray, clipping_threshold: float = CLIP_THRESHOLD) -> bool:
    """Official ``any(abs(audio) > threshold)``. ``np.any`` is the same predicate."""
    started = time.perf_counter()
    values = np.asarray(audio)
    if values.size == 0:
        result = False
    else:
        result = bool(np.any(np.abs(values) > clipping_threshold))
    _profile_add("clip", started)
    return result


def activitydetector(audio: np.ndarray, fs: int = SAMPLE_RATE, energy_thresh: float = 0.13, target_level: float = -25) -> float:
    """Official ``activitydetector``. The signal is scaled to ``target_level`` first."""
    started = time.perf_counter()
    scaled = normalize(np.asarray(audio, dtype=np.float64), target_level)
    window = int(fs * 50 / 1000)
    if window < 1:
        raise ReproductionError("activity window is empty")
    if len(scaled) == 0:
        raise ReproductionError("activity detector received an empty signal")
    count = (len(scaled) + window - 1) // window
    energies = np.empty(count, dtype=np.float64)
    for index in range(count):
        frame = scaled[index * window:(index + 1) * window]
        # Left-to-right float64 sum. It matches official ``sum(frame ** 2)``.
        energies[index] = np.add.reduce(frame * frame)
    frame_rms = 20 * np.log10(energies + EPS)
    probability = 1.0 / (1.0 + np.exp(-(-1.0 + 0.2 * frame_rms)))
    previous = 0.0
    active = 0
    for value in probability:
        current = float(value)
        if current > previous:
            smoothed = current * 0.8 + previous * (1 - 0.8)
        else:
            smoothed = current * 0.05 + previous * (1 - 0.05)
        if smoothed > energy_thresh:
            active += 1
        previous = current
    result = active / count
    _profile_add("activity", started)
    return result


def normalize(audio: np.ndarray, target_level: float = -25) -> np.ndarray:
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    return audio * scalar


def snr_mixer(
    clean: np.ndarray,
    noise: np.ndarray,
    snr: int,
    target_level: int,
    *,
    clipping_threshold: float = CLIP_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Official ``snr_mixer`` with the target level already drawn.

    The level is an integer in ``[target_level_lower, target_level_upper)``.
    Official draws it with ``np.random.randint``, whose upper bound is excluded.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    clean = clean / (float(np.max(np.abs(clean))) + EPS)
    clean = normalize(clean, -25)
    rmsclean = (clean ** 2).mean() ** 0.5
    noise = noise / (float(np.max(np.abs(noise))) + EPS)
    noise = normalize(noise, -25)
    rmsnoise = (noise ** 2).mean() ** 0.5
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (target_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    applied = int(target_level)
    if is_clipped(noisyspeech, clipping_threshold):
        peak = float(np.max(np.abs(noisyspeech))) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / peak
        clean = clean / peak
        noisenewlevel = noisenewlevel / peak
        applied = int(20 * np.log10(scalarnoisy / peak * (rmsnoisy + EPS)))
    return clean, noisenewlevel, noisyspeech, applied


def measured_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    clean_rms = float(np.sqrt(np.mean(np.asarray(clean, dtype=np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean(np.asarray(noise, dtype=np.float64) ** 2)))
    if clean_rms == 0.0 or noise_rms == 0.0:
        raise ReproductionError("SNR is undefined for a silent waveform")
    return 20.0 * float(np.log10(clean_rms / noise_rms))


def pcm16_roundtrip(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """``soundfile.write`` default PCM_16, read back. Nothing is written to disk."""
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float64), sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    read, rate = sf.read(buffer, dtype="float64")
    if int(rate) != sample_rate:
        raise ReproductionError("PCM_16 round trip changed the sampling rate")
    return np.asarray(read, dtype=np.float64)


def official_uttid(clean_pieces: Sequence[dict[str, Any]], noise_pieces: Sequence[dict[str, Any]], snr: int, file_id: int) -> str:
    """``dns_data_prep.sh`` uttid sed on the noisy basename this generator would have written."""
    clean = "-".join(Path(str(piece["name"])).stem for piece in clean_pieces if "name" in piece)[:MAX_NAME]
    noise = "-".join(Path(str(piece["name"])).stem for piece in noise_pieces if "name" in piece)[:MAX_NAME]
    basename = f"{clean}_{noise}_snr{int(snr)}_fileid_{int(file_id)}.wav"
    match = UTTID_RE.fullmatch(basename)
    if match is None:
        raise ReproductionError(f"noisy name {basename} has no reader_/fileid_ uttid")
    return f"reader_{match.group(1)}_fileid_{match.group(2)}"


def reader_id(name: str) -> str | None:
    match = READER_RE.search(Path(name).name)
    if match is None:
        return None
    return match.group(1)


def build_definition(
    *,
    seed: int,
    num_files: int,
    clean_count: int,
    noise_count: int,
    order_sha256: str,
    inputs: dict[str, str],
    shard_size: int = 1000,
    audio_seconds: float = AUDIO_SECONDS,
    total_hours: float | None = None,
    excluded_non_16k_noise: int = 0,
    acceptance: str | None = None,
) -> dict[str, Any]:
    if num_files < 2:
        raise ReproductionError("DNS plan needs at least two mixtures so train and valid are both non-empty")
    params = {
        "sample_rate": SAMPLE_RATE,
        "audio_seconds": float(audio_seconds),
        "silence_seconds": SILENCE_SECONDS,
        "num_files": int(num_files),
        "total_hours": None if total_hours is None else float(total_hours),
        "snr_lower": SNR_LOWER,
        "snr_upper": SNR_UPPER,
        "target_level_lower": LEVEL_LOWER,
        "target_level_upper": LEVEL_UPPER,
        "clean_activity_threshold": CLEAN_ACTIVITY,
        "noise_activity_threshold": NOISE_ACTIVITY,
        "is_test_set": False,
        "clean_count": int(clean_count),
        "noise_count": int(noise_count),
        "excluded_non_16k_noise": int(excluded_non_16k_noise),
        "order_sha256": order_sha256,
        "file_selection": "per_case_cycle",
    }
    if acceptance is not None:
        if acceptance != RETRY_ACCEPT:
            raise ReproductionError(f"unsupported DNS acceptance policy: {acceptance}")
        params["retry_policy"] = RETRY_ACCEPT
        params["renderer"] = RENDERER_ACCEPT
        params["resample"] = RESAMPLE_ID
        params["max_piece_tries"] = MAX_PIECE_TRIES
        params["max_case_attempts"] = MAX_CASE_ATTEMPTS
    build_id = build_id_for(
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        params=params,
        seed=int(seed),
        inputs=dict(inputs),
    )
    return {
        "build_id": build_id,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "shard_size": int(shard_size),
        "params": params,
        "inputs": {str(key): str(value) for key, value in sorted(inputs.items())},
        "deviations": [
            {
                "id": "seeded_per_case",
                "class": "algorithm_change",
                "text": (
                    "Official DNS generation seeds random and numpy once and advances a "
                    "shared multiprocessing counter that starts at 0 and increments before "
                    "the modulo. File assignment depends on the process schedule, and clip "
                    "or activity retries consume that counter. This plan does not replay "
                    "that stream. Names are sorted, shuffled once with random.Random(seed), "
                    "and each gen_index walks that list from gen_index % count. Crops, SNR, "
                    "and level come from Generator(PCG64(mix_seed('dns', seed, gen_index))). "
                    + (
                        "Sample-dependent retries are not applied."
                        if acceptance is None
                        else (
                            "Each case then applies the official activity and clipping predicates "
                            "on one private stream. A shared multiprocessing counter is not replayed."
                        )
                    )
                ),
            }
        ],
    }


def num_files_for_hours(total_hours: float, audio_seconds: float = AUDIO_SECONDS) -> int:
    """``int(total_hours * 3600 / audio_length)`` from the official synthesizer."""
    if total_hours <= 0 or audio_seconds <= 0:
        raise ReproductionError("total_hours and audio_seconds must be positive")
    count = int(total_hours * 3600 / audio_seconds)
    if count < 2:
        raise ReproductionError("the requested duration does not produce two mixtures")
    return count


def resample_to_16k(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Full-file resample used by official ``build_audio`` before the crop.

    DNS-Challenge ``70f19285`` calls ``librosa.resample(audio, fs_input, 16000)``.
    That positional form does not run on librosa 1.0.0. The librosa release used
    for the published mixtures is not recorded. This function calls librosa
    1.0.0 with ``res_type='soxr_hq'``, which is that release's default. 16 kHz
    input is returned unchanged and is not passed through the resampler.
    """
    wave = np.asarray(audio, dtype=np.float64)
    rate = int(source_rate)
    if rate == SAMPLE_RATE:
        return wave
    if rate < 1 or wave.size < 1:
        raise ReproductionError(f"cannot resample {wave.size} samples at {rate} Hz")
    import librosa

    started = time.perf_counter()
    converted = librosa.resample(wave, orig_sr=rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    _profile_add("resample", started)
    out = np.asarray(converted, dtype=np.float64).reshape(-1)
    if out.size < 1 or not np.isfinite(out).all():
        raise ReproductionError(f"resample from {rate} Hz produced no finite 16 kHz samples")
    return out


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    """``audioread`` without its optional RMS normalize: soundfile, then channel mean."""
    started = time.perf_counter()
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    _profile_add("decode", started)
    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    return mono, int(rate)


def load_16k(path: Path, cache: dict[str, tuple[np.ndarray, int]] | None = None) -> tuple[np.ndarray, int]:
    """Native mono audio converted to 16 kHz. ``cache`` is discarded with the caller."""
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    mono, rate = read_mono(path)
    converted = mono if rate == SAMPLE_RATE else resample_to_16k(mono, rate)
    loaded = (converted, rate)
    if cache is not None:
        cache[key] = loaded
    return loaded


def signal_length(path: Path, cache: dict[str, tuple[np.ndarray, int]] | None = None) -> tuple[int, int]:
    """``(source_rate, length at 16 kHz)`` without reading a 16 kHz file body."""
    key = str(path)
    if cache is not None and key in cache:
        audio, rate = cache[key]
        return int(rate), int(audio.shape[0])
    started = time.perf_counter()
    info = sf.info(path)
    _profile_add("io", started)
    rate = int(info.samplerate)
    if rate == SAMPLE_RATE:
        return rate, int(info.frames)
    audio, source_rate = load_16k(path, cache)
    return int(source_rate), int(audio.shape[0])


def read_signal_span(
    path: Path,
    start: int,
    length: int,
    cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[np.ndarray, int]:
    """Crop after the official full-file resample. 16 kHz files are read as a span."""
    if start < 0 or length < 0:
        raise ReproductionError(f"{path} span {start}:{start + length} is empty")
    key = str(path)
    if cache is not None and key in cache:
        audio, rate = cache[key]
        crop = audio[start:start + length]
        if crop.shape[0] != length:
            raise ReproductionError(f"{path} span {start}:{start + length} is outside the 16 kHz signal")
        return crop, int(rate)
    started = time.perf_counter()
    info = sf.info(path)
    _profile_add("io", started)
    rate = int(info.samplerate)
    if rate == SAMPLE_RATE:
        if start + length > int(info.frames):
            raise ReproductionError(f"{path} span {start}:{start + length} is outside {info.frames} frames")
        started = time.perf_counter()
        audio, got = sf.read(path, start=start, stop=start + length, dtype="float64", always_2d=True)
        _profile_add("decode", started)
        if int(got) != SAMPLE_RATE:
            raise ReproductionError(f"{path} is {got} Hz; the plan expected {SAMPLE_RATE}")
        mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
        if mono.shape[0] != length:
            raise ReproductionError(f"{path} span {start}:{start + length} returned {mono.shape[0]} samples")
        return mono, rate
    audio, source_rate = load_16k(path, cache)
    crop = audio[start:start + length]
    if crop.shape[0] != length:
        raise ReproductionError(f"{path} span {start}:{start + length} is outside the resampled signal")
    return crop, int(source_rate)


def plan_accepted_realizations(
    definition: dict[str, Any],
    clean_names: Sequence[str],
    clean_paths: dict[str, Path],
    noise_names: Sequence[str],
    noise_paths: dict[str, Path],
    *,
    cache: dict[str, tuple[np.ndarray, int]] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accepted realizations for every ``gen_index``. Audio is read. Unaccepted candidates are not stored."""
    count = int(definition["params"]["num_files"])
    return plan_accepted_span(
        definition,
        clean_names,
        clean_paths,
        noise_names,
        noise_paths,
        start=0,
        stop=count,
        cache=cache,
        workers=workers,
    )


def plan_accepted_span(
    definition: dict[str, Any],
    clean_names: Sequence[str],
    clean_paths: dict[str, Path],
    noise_names: Sequence[str],
    noise_paths: dict[str, Path],
    *,
    start: int,
    stop: int,
    cache: dict[str, tuple[np.ndarray, int]] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accepted realizations for ``gen_index`` in ``[start, stop)``.

    ``workers`` above 1 plans disjoint ranges in separate processes. Each case has
    its own generator, so the documents do not depend on the worker count or on
    which range contains the case. The source cache passed as ``None`` is created
    inside the case and discarded when that case returns. Mixture waveforms are
    not stored.
    """
    params = definition["params"]
    if params.get("retry_policy") != RETRY_ACCEPT:
        raise ReproductionError("accepted DNS plans require deterministic_case_stream_v1")
    total = int(params["num_files"])
    start = int(start)
    stop = int(stop)
    if start < 0 or stop > total or start >= stop:
        raise ReproductionError(f"DNS acceptance span {start}:{stop} is outside 0:{total}")
    if digest({"clean": list(clean_names), "noise": list(noise_names)}) != params["order_sha256"]:
        raise ReproductionError("file order does not match the build order_sha256")
    worker_count = int(workers)
    if worker_count < 1:
        raise ReproductionError("DNS acceptance workers must be positive")
    width = stop - start
    if worker_count == 1 or width < 2:
        pairs = [
            _accept_one(definition, gen_index, clean_names, clean_paths, noise_names, noise_paths, cache)
            for gen_index in range(start, stop)
        ]
    else:
        pairs = _accept_parallel(
            definition, start, stop, clean_names, clean_paths, noise_names, noise_paths, worker_count
        )
    documents = []
    totals = {
        "cases": 0,
        "activity_rejects": 0,
        "post_clip_rejects": 0,
        "short_rejects": 0,
        "clean_clip_skips": 0,
        "noise_clip_skips": 0,
        "outer_attempts": [],
        "case_stats": [],
        "resampled_noise_pieces": 0,
    }
    for document, stats in pairs:
        documents.append(document)
        totals["case_stats"].append({key: int(value) for key, value in stats.items()})
        totals["cases"] += 1
        for key in ("activity_rejects", "post_clip_rejects", "short_rejects", "clean_clip_skips", "noise_clip_skips"):
            totals[key] += int(stats[key])
        totals["outer_attempts"].append(int(stats["retries"]))
        totals["resampled_noise_pieces"] += int(stats["resampled_noise_pieces"])
    totals["failed_cases"] = 0
    return documents, totals


def _accept_one(
    definition: dict[str, Any],
    gen_index: int,
    clean_names: Sequence[str],
    clean_paths: dict[str, Path],
    noise_names: Sequence[str],
    noise_paths: dict[str, Path],
    cache: dict[str, tuple[np.ndarray, int]] | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    document, stats = plan_accepted_case(
        definition,
        gen_index,
        clean_names,
        clean_paths,
        noise_names,
        noise_paths,
        cache=cache,
    )
    validate_realization(_stored(document))
    if int(definition["params"]["num_files"]) >= 20000 and (gen_index + 1) % 20000 == 0:
        print(f"accepted\t{gen_index + 1}\t{definition['params']['num_files']}", flush=True)
    return document, stats


def _accept_parallel(
    definition: dict[str, Any],
    start: int,
    stop: int,
    clean_names: Sequence[str],
    clean_paths: dict[str, Path],
    noise_names: Sequence[str],
    noise_paths: dict[str, Path],
    workers: int,
) -> list[tuple[dict[str, Any], dict[str, int]]]:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    width = max(1, (int(stop) - int(start) + workers - 1) // workers)
    ranges = [(cursor, min(int(stop), cursor + width)) for cursor in range(int(start), int(stop), width)]
    payload = {
        "definition": definition,
        "clean_names": list(clean_names),
        "clean_paths": {name: str(path) for name, path in clean_paths.items()},
        "noise_names": list(noise_names),
        "noise_paths": {name: str(path) for name, path in noise_paths.items()},
    }
    # spawn keeps the result independent of the parent process state.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(ranges), mp_context=context) as pool:
        chunks = list(pool.map(_accept_range, [(payload, start, stop) for start, stop in ranges]))
    ordered: list[tuple[dict[str, Any], dict[str, int]]] = []
    for chunk in chunks:
        ordered.extend(chunk)
    if [int(item[0]["draws"]["gen_index"]) for item in ordered] != list(range(int(start), int(stop))):
        raise ReproductionError("parallel DNS acceptance did not return gen_index order")
    return ordered


def _accept_range(job: tuple[dict[str, Any], int, int]) -> list[tuple[dict[str, Any], dict[str, int]]]:
    payload, start, stop = job
    clean_paths = {name: Path(path) for name, path in payload["clean_paths"].items()}
    noise_paths = {name: Path(path) for name, path in payload["noise_paths"].items()}
    return [
        _accept_one(
            payload["definition"],
            gen_index,
            payload["clean_names"],
            clean_paths,
            payload["noise_names"],
            noise_paths,
            None,
        )
        for gen_index in range(int(start), int(stop))
    ]


def plan_accepted_case(
    definition: dict[str, Any],
    gen_index: int,
    clean_names: Sequence[str],
    clean_paths: dict[str, Path],
    noise_names: Sequence[str],
    noise_paths: dict[str, Path],
    *,
    cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """One case. The stream continues across rejects and does not depend on other cases."""
    params = definition["params"]
    target = int(round(float(params["audio_seconds"]) * SAMPLE_RATE))
    silence = int(round(float(params["silence_seconds"]) * SAMPLE_RATE))
    if target < 2 or silence < 0:
        raise ReproductionError("DNS target length must be positive")
    if not clean_names or not noise_names:
        raise ReproductionError("accepted DNS plan needs clean and noise files")
    if cache is None:
        cache = {}
    rng = np.random.Generator(np.random.PCG64(mix_seed("dns", int(definition["seed"]), gen_index)))
    clean_cursor = int(gen_index) % len(clean_names)
    noise_cursor = int(gen_index) % len(noise_names)
    stats = {
        "retries": 0,
        "activity_rejects": 0,
        "post_clip_rejects": 0,
        "short_rejects": 0,
        "clean_clip_skips": 0,
        "noise_clip_skips": 0,
        "resampled_noise_pieces": 0,
    }
    limit = int(params.get("max_case_attempts") or MAX_CASE_ATTEMPTS)
    clean_threshold = float(params.get("clean_activity_threshold", CLEAN_ACTIVITY))
    noise_threshold = float(params.get("noise_activity_threshold", NOISE_ACTIVITY))
    for _attempt in range(limit):
        clean, clean_cursor, clean_skips, clean_ok = _build_signal(
            clean_names,
            clean_paths,
            clean_cursor,
            target,
            silence,
            rng,
            is_clean=True,
            is_test_set=bool(params.get("is_test_set", False)),
            cache=cache,
        )
        stats["clean_clip_skips"] += clean_skips
        if not clean_ok:
            stats["short_rejects"] += 1
            stats["retries"] += 1
            continue
        clean_wave = _pieces_to_wave(clean, clean_paths, cache=cache)
        if clean_threshold == 0.0:
            activity = None
        else:
            activity = float(activitydetector(clean_wave))
            if not activity > clean_threshold:
                stats["activity_rejects"] += 1
                stats["retries"] += 1
                continue
        noise, noise_cursor, noise_skips, noise_ok = _build_signal(
            noise_names,
            noise_paths,
            noise_cursor,
            int(clean_wave.shape[0]),
            silence,
            rng,
            is_clean=False,
            is_test_set=False,
            cache=cache,
        )
        stats["noise_clip_skips"] += noise_skips
        if not noise_ok:
            stats["short_rejects"] += 1
            stats["retries"] += 1
            continue
        if noise_threshold != 0.0:
            noise_wave = _pieces_to_wave(noise, noise_paths, cache=cache)
            if not float(activitydetector(noise_wave)) > noise_threshold:
                stats["activity_rejects"] += 1
                stats["retries"] += 1
                continue
        snr = int(rng.integers(int(params["snr_lower"]), int(params["snr_upper"])))
        level = int(rng.integers(int(params["target_level_lower"]), int(params["target_level_upper"])))
        noise_wave = _pieces_to_wave(noise, noise_paths, cache=cache)
        mixed_clean, mixed_noise, noisy, _applied = snr_mixer(clean_wave, noise_wave, snr, level)
        if is_clipped(mixed_clean) or is_clipped(mixed_noise) or is_clipped(noisy):
            stats["post_clip_rejects"] += 1
            stats["retries"] += 1
            continue
        stats["resampled_noise_pieces"] = sum(
            1 for piece in noise if int(piece.get("source_rate", SAMPLE_RATE)) != SAMPLE_RATE
        )
        uttid = official_uttid(clean, noise, snr, gen_index)
        document = {
            "realization_version": 1,
            "algorithm": ALGORITHM,
            "algorithm_version": 1,
            "sample_rate": SAMPLE_RATE,
            "sources": {"speech": clean, "noise": noise},
            "draws": {
                "gen_index": gen_index,
                "build_id": definition["build_id"],
                "snr_db": snr,
                "target_level_dbfs": level,
                "utt_id": uttid,
                "retries": int(stats["retries"]),
                "retry_policy": RETRY_ACCEPT,
                "renderer": RENDERER_ACCEPT,
                "resample": RESAMPLE_ID,
                "clean_activity": None if activity is None else activity,
                "clean_clip_skips": int(stats["clean_clip_skips"]),
                "noise_clip_skips": int(stats["noise_clip_skips"]),
                "activity_rejects": int(stats["activity_rejects"]),
                "post_clip_rejects": int(stats["post_clip_rejects"]),
                "short_rejects": int(stats["short_rejects"]),
            },
            "steps": [
                "resample_16k",
                "concat_silence",
                "activity_accept",
                "peak_normalize",
                "rms_-25dbfs",
                "snr_scale",
                "target_level",
                "clip_guard",
                "post_mix_accept",
                "pcm16",
            ],
            "output": {"length": target, "teacher": "dry_clean"},
        }
        if int(clean_wave.shape[0]) != target:
            raise ReproductionError(f"gen_index {gen_index} accepted clean length {clean_wave.shape[0]}, plan says {target}")
        return document, stats
    raise ReproductionError(
        f"gen_index {gen_index} had no accepted mixture after {limit} attempts: {stats}"
    )


def plan_realizations(
    definition: dict[str, Any],
    clean_names: Sequence[str],
    clean_frames: dict[str, int],
    noise_names: Sequence[str],
    noise_frames: dict[str, int],
) -> list[dict[str, Any]]:
    """One realization per ``gen_index``. Audio is not read."""
    params = definition["params"]
    count = int(params["num_files"])
    if digest({"clean": list(clean_names), "noise": list(noise_names)}) != params["order_sha256"]:
        raise ReproductionError("file order does not match the build order_sha256")
    documents = []
    for gen_index in range(count):
        document = plan_case(
            definition,
            gen_index,
            clean_names,
            clean_frames,
            noise_names,
            noise_frames,
        )
        validate_realization(_stored(document))
        documents.append(document)
        if count >= 20000 and (gen_index + 1) % 20000 == 0:
            print(f"planned\t{gen_index + 1}\t{count}", flush=True)
    return documents


def plan_case(
    definition: dict[str, Any],
    gen_index: int,
    clean_names: Sequence[str],
    clean_frames: dict[str, int],
    noise_names: Sequence[str],
    noise_frames: dict[str, int],
) -> dict[str, Any]:
    params = definition["params"]
    target = int(round(float(params["audio_seconds"]) * SAMPLE_RATE))
    silence = int(round(float(params["silence_seconds"]) * SAMPLE_RATE))
    if target < 2 or silence < 0:
        raise ReproductionError("DNS target length must be positive")
    rng = np.random.Generator(np.random.PCG64(mix_seed("dns", int(definition["seed"]), gen_index)))
    clean = _assemble(clean_names, clean_frames, gen_index % len(clean_names), target, silence, rng)
    noise = _assemble(noise_names, noise_frames, gen_index % len(noise_names), target, silence, rng)
    snr = int(rng.integers(int(params["snr_lower"]), int(params["snr_upper"])))
    level = int(rng.integers(int(params["target_level_lower"]), int(params["target_level_upper"])))
    uttid = official_uttid(clean, noise, snr, gen_index)
    return {
        "realization_version": 1,
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "sample_rate": SAMPLE_RATE,
        "sources": {"speech": clean, "noise": noise},
        "draws": {
            "gen_index": gen_index,
            "build_id": definition["build_id"],
            "snr_db": snr,
            "target_level_dbfs": level,
            "utt_id": uttid,
            "retries": 0,
            "retry_policy": "length_plan_no_sample_retry",
        },
        "steps": [
            "concat_silence",
            "peak_normalize",
            "rms_-25dbfs",
            "snr_scale",
            "target_level",
            "clip_guard",
            "pcm16",
        ],
        "output": {"length": target, "teacher": "dry_clean"},
    }


def render_realization(document: dict[str, Any], directory_for: dict[str, Path | dict[str, Path]]) -> dict[str, np.ndarray]:
    """Mix one stored plan. The RNG is not drawn again.

    ``length_plan_no_sample_retry`` reads stored 16 kHz spans. An accepted plan
    may resample a non-16 kHz source before the stored crop. Neither path
    searches again, and neither writes a waveform.
    """
    policy = str(document["draws"].get("retry_policy") or RETRY_NONE)
    if policy == RETRY_NONE:
        allow_resample = False
    elif policy == RETRY_ACCEPT:
        allow_resample = True
    else:
        raise ReproductionError(f"unknown DNS retry policy: {policy}")
    clean = _render_pieces(document["sources"]["speech"], directory_for["speech"], allow_resample=allow_resample)
    noise = _render_pieces(document["sources"]["noise"], directory_for["noise"], allow_resample=allow_resample)
    length = int(document["output"]["length"])
    if clean.shape[0] != length or noise.shape[0] != length:
        raise ReproductionError(
            f"gen_index {document['draws']['gen_index']} rendered {clean.shape[0]}/{noise.shape[0]}, plan says {length}"
        )
    mixed_clean, mixed_noise, noisy, _applied = snr_mixer(
        clean,
        noise,
        int(document["draws"]["snr_db"]),
        int(document["draws"]["target_level_dbfs"]),
    )
    out = {
        "clean": pcm16_roundtrip(mixed_clean),
        "noise": pcm16_roundtrip(mixed_noise),
        "noisy": pcm16_roundtrip(noisy),
    }
    for name, wave in out.items():
        if wave.shape[0] != length or not np.isfinite(wave).all():
            raise ReproductionError(f"gen_index {document['draws']['gen_index']} {name} PCM_16 is not the planned length")
    return out


def duration_kept(frames: int) -> bool:
    """ESPnet ``utt2num_samples`` keeps ``min < length < max`` with 0.1 s and 31 s."""
    return int(0.1 * SAMPLE_RATE) < int(frames) < 31 * SAMPLE_RATE


def split_train_valid(documents: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Sort uttids and give the first ``count * 9 // 10`` to train. The tail is validation."""
    ordered = sorted(documents, key=lambda item: str(item["draws"]["utt_id"]))
    seen: set[str] = set()
    for item in ordered:
        uttid = str(item["draws"]["utt_id"])
        if uttid in seen:
            raise ReproductionError(f"duplicate DNS uttid {uttid}")
        seen.add(uttid)
    train_count = len(ordered) * 9 // 10
    raw_train = ordered[:train_count]
    raw_valid = ordered[train_count:]
    train = [item for item in raw_train if duration_kept(int(item["output"]["length"]))]
    valid = [item for item in raw_valid if duration_kept(int(item["output"]["length"]))]
    if not train or not valid:
        raise ReproductionError("duration filter removed every train or valid mixture")
    return train, valid, {
        "files": len(ordered),
        "train_before_duration": len(raw_train),
        "valid_before_duration": len(raw_valid),
        "train_kept": len(train),
        "valid_kept": len(valid),
        "train_dropped": len(raw_train) - len(train),
        "valid_dropped": len(raw_valid) - len(valid),
    }


def order_sha256(clean_names: Sequence[str], noise_names: Sequence[str]) -> str:
    return digest({"clean": list(clean_names), "noise": list(noise_names)})


def _stored(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: document[key]
        for key in (
            "realization_version",
            "algorithm",
            "algorithm_version",
            "sample_rate",
            "sources",
            "draws",
            "steps",
            "output",
        )
    }


def _assemble(
    names: Sequence[str],
    frames: dict[str, int],
    start: int,
    target: int,
    silence: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if not names:
        raise ReproductionError("cannot assemble an empty file list")
    remaining = target
    pieces: list[dict[str, Any]] = []
    cursor = int(start)
    steps = 0
    count = len(names)
    while remaining > 0:
        if steps > count:
            raise ReproductionError("source pool cannot fill one mixture")
        name = names[cursor % count]
        length = int(frames[name])
        if length < 1:
            raise ReproductionError(f"{name} has no frames")
        if length > remaining:
            offset = int(rng.integers(0, length - remaining))
            use = remaining
        else:
            offset = 0
            use = length
        pieces.append({"name": name, "start": offset, "length": use})
        remaining -= use
        if remaining > 0:
            gap = min(remaining, silence)
            if gap:
                pieces.append({"silence": gap})
                remaining -= gap
        cursor += 1
        steps += 1
    return pieces


def _build_signal(
    names: Sequence[str],
    paths: dict[str, Path],
    cursor: int,
    target: int,
    silence: int,
    rng: np.random.Generator,
    *,
    is_clean: bool,
    is_test_set: bool,
    cache: dict[str, tuple[np.ndarray, int]] | None,
) -> tuple[list[dict[str, Any]], int, int, bool]:
    """Official ``build_audio`` on one case-local cursor.

    A clipped crop consumes one of ``MAX_PIECE_TRIES`` and the random offset,
    then the next file is used. The already concatenated prefix stays. The
    cursor advances for every read, including rejects.
    """
    if not names:
        raise ReproductionError("cannot assemble an empty file list")
    remaining = int(target)
    pieces: list[dict[str, Any]] = []
    tries_left = MAX_PIECE_TRIES
    skips = 0
    count = len(names)
    position = int(cursor)
    while remaining > 0 and tries_left > 0:
        name = names[position % count]
        position += 1
        path = paths[name]
        source_rate, available = signal_length(path, cache)
        if available > remaining and (not is_clean or not is_test_set):
            offset = int(rng.integers(0, available - remaining))
            use = remaining
        else:
            offset = 0
            use = available
        if use < 1:
            skips += 1
            tries_left -= 1
            continue
        crop, _rate = read_signal_span(path, offset, use, cache)
        if is_clipped(crop):
            skips += 1
            tries_left -= 1
            continue
        pieces.append({"name": name, "start": offset, "length": use, "source_rate": source_rate})
        remaining -= use
        if remaining > 0:
            gap = min(remaining, silence)
            if gap:
                pieces.append({"silence": gap})
                remaining -= gap
    # A test-set clean file longer than the target is kept whole, so remaining
    # can be negative. Training plans set is_test_set false and stay exact.
    return pieces, position, skips, remaining <= 0


def _pieces_to_wave(
    pieces: Sequence[dict[str, Any]],
    paths: dict[str, Path],
    *,
    cache: dict[str, tuple[np.ndarray, int]] | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for piece in pieces:
        if "silence" in piece:
            chunks.append(np.zeros(int(piece["silence"]), dtype=np.float64))
            continue
        start = int(piece["start"])
        length = int(piece["length"])
        crop, _rate = read_signal_span(paths[str(piece["name"])], start, length, cache)
        chunks.append(crop)
    if not chunks:
        raise ReproductionError("realization has no samples")
    return np.concatenate(chunks)


def _render_pieces(
    pieces: Sequence[dict[str, Any]],
    root: Path | dict[str, Path],
    *,
    allow_resample: bool = False,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for piece in pieces:
        if "silence" in piece:
            chunks.append(np.zeros(int(piece["silence"]), dtype=np.float64))
            continue
        name = str(piece["name"])
        path = root[name] if isinstance(root, dict) else Path(root) / name
        start = int(piece["start"])
        length = int(piece["length"])
        source_rate = piece.get("source_rate")
        if source_rate is not None and int(source_rate) != SAMPLE_RATE:
            if not allow_resample:
                raise ReproductionError(f"{path} is {source_rate} Hz; this plan does not resample")
            mono, rate = read_mono(path)
            if int(rate) != int(source_rate):
                raise ReproductionError(f"{path} is {rate} Hz; the plan stored {source_rate} Hz")
            converted = resample_to_16k(mono, rate)
            mono = converted[start:start + length]
        else:
            audio, rate = sf.read(path, start=start, stop=start + length, dtype="float64", always_2d=True)
            if int(rate) != SAMPLE_RATE:
                raise ReproductionError(f"{path} is {rate} Hz; the plan only stores 16 kHz spans")
            mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
        if mono.shape[0] != length:
            raise ReproductionError(f"{path} span {start}:{start + length} returned {mono.shape[0]} samples")
        chunks.append(np.asarray(mono, dtype=np.float64))
    if not chunks:
        raise ReproductionError("realization has no samples")
    return np.concatenate(chunks)


def measure_realization(document: dict[str, Any], directory_for: dict[str, Path | dict[str, Path]]) -> dict[str, Any]:
    """Render one stored plan and compare it with the official acceptance checks.

    Official generation would discard a clipped source piece and would rebuild
    a clean signal whose activity is not above 0.6, or a mix whose clean, noise,
    or noisy peak still exceeds 0.99. This measurement does not change the plan.
    """
    speech = directory_for["speech"]
    noise_root = directory_for["noise"]
    policy = str(document["draws"].get("retry_policy") or RETRY_NONE)
    allow_resample = policy == RETRY_ACCEPT
    source_clipped = _clipped_pieces(document["sources"]["speech"], speech) + _clipped_pieces(
        document["sources"]["noise"], noise_root
    )
    clean = _render_pieces(document["sources"]["speech"], speech, allow_resample=allow_resample)
    noise = _render_pieces(document["sources"]["noise"], noise_root, allow_resample=allow_resample)
    activity = float(activitydetector(clean))
    mixed_clean, mixed_noise, noisy, _applied = snr_mixer(
        clean,
        noise,
        int(document["draws"]["snr_db"]),
        int(document["draws"]["target_level_dbfs"]),
    )
    post_clipped = int(is_clipped(mixed_clean) or is_clipped(mixed_noise) or is_clipped(noisy))
    rendered = render_realization(document, directory_for)
    again = render_realization(document, directory_for)
    if any(not np.array_equal(rendered[name], again[name]) for name in ("clean", "noise", "noisy")):
        raise ReproductionError(f"gen_index {document['draws']['gen_index']} PCM_16 is not repeatable")
    snr = measured_snr_db(rendered["clean"], rendered["noise"])
    return {
        "gen_index": int(document["draws"]["gen_index"]),
        "clean_activity": activity,
        "clean_activity_rejected": bool(activity <= CLEAN_ACTIVITY),
        "source_clipped_pieces": int(source_clipped),
        "post_mix_clipped": post_clipped,
        "snr_db": int(document["draws"]["snr_db"]),
        "snr_error_db": float(snr - int(document["draws"]["snr_db"])),
        "length": int(rendered["noisy"].shape[0]),
    }


def _clipped_pieces(pieces: Sequence[dict[str, Any]], root: Path | dict[str, Path]) -> int:
    found = 0
    for piece in pieces:
        if "name" not in piece:
            continue
        name = str(piece["name"])
        path = root[name] if isinstance(root, dict) else Path(root) / name
        start = int(piece["start"])
        length = int(piece["length"])
        source_rate = piece.get("source_rate")
        if source_rate is not None and int(source_rate) != SAMPLE_RATE:
            mono, rate = read_mono(path)
            crop = resample_to_16k(mono, rate)[start:start + length]
        else:
            audio, _rate = sf.read(path, start=start, stop=start + length, dtype="float64", always_2d=True)
            crop = np.asarray(audio, dtype=np.float64).mean(axis=1)
        if is_clipped(crop):
            found += 1
    return found
