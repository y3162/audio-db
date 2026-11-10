"""DeepFilterNet3 online mixer.

The draw order follows ``TdDataset::get_sample`` at
``d375b2d8309e0935d165700c91da9de862a99c31``. The random stream is NumPy
PCG64, not Rust ``StdRng``. Distributions follow the shipped ``config.ini``
and the code defaults that ini does not override.

A waveform is a function of the protocol, the snapshot, the training seed,
the split, the epoch (train only), the sample index, and the audio bytes.
Rank and worker count are not inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import DFN3_ONLINE_V1
from src.experiments.data.sampling.seeds import mix_seed

SAMPLE_RATE = 48000
NUM_SAMPLES = 144000
RENDERER_VERSION = "dfn3_online_v1"
_REPLACE_LIMIT = 64


@dataclass(frozen=True)
class DFN3OnlineSpec:
    sample_rate: int = SAMPLE_RATE
    num_samples: int = NUM_SAMPLES
    snrs: tuple[int, ...] = tuple(DFN3_ONLINE_V1["snrs"])
    gains: tuple[int, ...] = tuple(DFN3_ONLINE_V1["gains"])
    noise_count_low: int = 2
    noise_count_high_exclusive: int = 6
    p_synthetic_noise: float = 0.05
    synthetic_decay: tuple[float, float] = (-2.0, 2.0)
    synthetic_gains: tuple[int, ...] = (-24, -12, -6, 0)
    p_reverb: float = 0.1
    drr: float = 0.3
    rt60: float = 0.5
    offset_late_ms: int = 20
    p_decay_floor: float = 0.5
    rir_resample: tuple[float, float] = (0.8, 1.2)
    rir_max_samples: int = 1_440_000
    p_remove_dc: float = 0.25
    p_lfilt: float = 0.25
    lfilt_ab: tuple[float, float] = (-0.375, 0.375)
    p_resample: float = 0.1
    resample_ratio: tuple[float, float] = (0.9, 1.1)
    resample_step_hz: int = 500
    p_noise_clip: float = 0.1
    clip_c: tuple[float, float] = (0.01, 0.5)

    @classmethod
    def shipped(cls) -> DFN3OnlineSpec:
        return cls()


@dataclass(frozen=True)
class OnlineMixture:
    noisy: np.ndarray
    clean: np.ndarray
    sample_rate: int
    snr_db: int
    gain_db: int
    n_noises: int
    synthetic_noises: int
    reverb_speech: bool
    reverb_noise: bool
    noise_only: bool
    measured_snr_db: float | None


def sample_seed(
    protocol_id: str,
    snapshot_hash: str,
    seed: int,
    split: str,
    epoch: int,
    sample_index: int,
) -> int:
    """Train includes epoch. Valid and test do not, matching the official split rule."""
    epoch_key = int(epoch) if split == "train" else 0
    return mix_seed(
        "dfn3-online",
        protocol_id,
        snapshot_hash,
        int(seed),
        split,
        epoch_key,
        int(sample_index),
    )


def speech_key_positions(
    count: int,
    factor: float,
    *,
    protocol_id: str,
    snapshot_hash: str,
    seed: int,
    split: str,
    epoch: int,
) -> list[int]:
    """Integer factors shuffle once. A non-integer factor reshuffles every epoch."""
    if count < 1:
        raise ReproductionError("online speech list is empty")
    length = max(int(round(float(count) * float(factor))), 1)
    fractional = abs(float(factor) - round(float(factor))) > 1e-9
    key_epoch = int(epoch) if fractional else 0
    rng = np.random.Generator(
        np.random.PCG64(
            mix_seed(
                "dfn3-speech-keys",
                protocol_id,
                snapshot_hash,
                int(seed),
                split,
                key_epoch,
            )
        )
    )
    order = rng.permutation(count).tolist()
    positions: list[int] = []
    while len(positions) < length:
        positions.extend(order)
    return positions[:length]


def render_mixture(
    spec: DFN3OnlineSpec,
    speech: list,
    noise: list,
    rir: list,
    *,
    speech_index: int,
    split: str,
    rng: np.random.Generator,
) -> OnlineMixture:
    """One 3-second mixture. ``speech[speech_index]`` is the first utterance."""
    if split not in ("train", "valid", "test"):
        raise ReproductionError(f"unknown split {split}")
    if not noise:
        raise ReproductionError("online mixing needs noise clips")
    if spec.p_reverb > 0 and not rir:
        raise ReproductionError("p_reverb is positive but no RIR clips were provided")
    snr = int(_choose(rng, spec.snrs))
    gain = int(_choose(rng, spec.gains))
    noise_only = snr <= -100
    if noise_only:
        dry = np.zeros(spec.num_samples, dtype=np.float64)
        max_freq = spec.sample_rate // 2
    else:
        dry, max_freq = _load_speech(spec, speech, speech_index, rng)
    n_noises = int(rng.integers(spec.noise_count_low, spec.noise_count_high_exclusive))
    pieces = []
    gains = []
    synthetic = 0
    for _ in range(n_noises):
        piece, piece_gain, was_synthetic = _load_noise(spec, noise, split, rng)
        pieces.append(piece)
        gains.append(piece_gain)
        synthetic += int(was_synthetic)
    mixed_noise = _combine_noises(pieces, gains, spec.num_samples, rng)
    target = dry
    distorted = dry.copy()
    reverb_speech = False
    reverb_noise = False
    if rir:
        target, distorted, mixed_noise, reverb_speech, reverb_noise = _maybe_reverb(
            spec, dry, mixed_noise, rir, rng
        )
    if max_freq < spec.sample_rate // 2:
        mixed_noise = _low_pass(mixed_noise, max_freq, spec.sample_rate)[: spec.num_samples]
    clean, scaled_noise, mixture = mix_signals(target, distorted, mixed_noise, snr, gain)
    measured = None
    if not noise_only:
        measured = _snr_db(clean, scaled_noise)
    return OnlineMixture(
        noisy=_f32(mixture),
        clean=_f32(clean),
        sample_rate=spec.sample_rate,
        snr_db=snr,
        gain_db=gain,
        n_noises=n_noises,
        synthetic_noises=synthetic,
        reverb_speech=reverb_speech,
        reverb_noise=reverb_noise,
        noise_only=noise_only,
        measured_snr_db=measured,
    )


def mix_signals(
    target: np.ndarray,
    distorted: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    gain_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official ``mix_audio_signal`` plus the clip guard.

    SNR uses the target energy, including silence samples. The mixture adds
    the distorted speech, which is the fully reverberant speech when reverb
    is applied to speech.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    distorted = np.asarray(distorted, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise, dtype=np.float64).reshape(-1)
    if target.shape != distorted.shape or target.shape != noise.shape:
        raise ReproductionError("mix inputs must share one length")
    gain = 10.0 ** (float(gain_db) / 20.0)
    clean = target * gain
    clean_mix = distorted * gain
    noise = noise * _mix_factor(clean, noise, float(snr_db))
    mixture = clean_mix + noise
    peak = max(_peak(clean), _peak(noise), _peak(mixture))
    if peak - 1.0 > 1e-10:
        scale = 1.0 / (peak + 1e-10)
        clean = clean * scale
        noise = noise * scale
        mixture = mixture * scale
    return clean, noise, mixture


def synthesize_noise(
    rng: np.random.Generator,
    decay: float,
    num_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Colored noise from ``gen_noise``. The FFT length is the sample rate."""
    rate = int(sample_rate)
    white = rng.normal(0.0, 1.0, size=rate).astype(np.float64)
    if float(decay) != 0.0:
        spec = np.fft.rfft(white)
        bins = spec.shape[0]
        mask = np.linspace(1.0, np.sqrt(bins), bins, dtype=np.float64) ** float(decay)
        white = np.fft.irfft(spec / mask, n=rate)
    peak = max(float(np.max(np.abs(white))), 1.0)
    white *= float(rng.uniform(0.01, 0.95)) / peak
    repeats = int(np.ceil(num_samples / rate))
    tiled = np.tile(white, repeats)[:num_samples]
    return tiled.astype(np.float64)


def reverb_pair(
    speech: np.ndarray,
    noise: np.ndarray,
    rir: np.ndarray,
    sample_rate: int,
    *,
    drr: float,
    rt60: float,
    offset_samples: int,
    apply_speech: bool,
    apply_noise: bool,
    decay_rt60: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One shared RIR. Returns target, fully reverberant speech, and noise.

    ``decay_rt60`` is the optional early random decay. ``rt60`` is the later
    target decay (shipped 0.5). The target is ``drr * dry + (1 - drr) * little``.
    """
    dry = np.asarray(speech, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise, dtype=np.float64).reshape(-1).copy()
    impulse = np.asarray(rir, dtype=np.float64).reshape(-1)
    if dry.shape != noise.shape:
        raise ReproductionError("reverb speech and noise lengths differ")
    length = int(dry.shape[0])
    if impulse.size < 1:
        raise ReproductionError("RIR is empty")
    max_idx = int(np.argmax(np.abs(impulse)))
    if decay_rt60 is not None:
        impulse = _suppress_late(impulse, sample_rate, max_idx, float(decay_rt60))
    impulse = _trim_rir(impulse, max_idx)
    impulse = _l2(impulse)
    reverberant = dry.copy()
    target = dry.copy()
    if apply_speech:
        speech_rms = _rms(dry)
        reverberant = _convolve_prefix(dry, impulse, length)
        late = max_idx + int(offset_samples)
        little_rir = _l2(_suppress_late(impulse, sample_rate, late, float(rt60)))
        little = _convolve_prefix(dry, little_rir, length)
        target = float(drr) * dry + (1.0 - float(drr)) * little
        target = target * (speech_rms / (_rms(target) + 1e-10))
    if apply_noise:
        noise = _convolve_prefix(noise, impulse, length)
    return target, reverberant, noise


def _load_speech(spec: DFN3OnlineSpec, speech: list, speech_index: int, rng: np.random.Generator):
    if speech_index < 0 or speech_index >= len(speech):
        raise ReproductionError("speech index is outside the key list")
    cursor = speech_index
    pieces: list[np.ndarray] = []
    max_freq = spec.sample_rate // 2
    replacements = 0
    while True:
        if replacements > _REPLACE_LIMIT:
            raise ReproductionError("speech replacement limit exceeded")
        want = int(spec.num_samples * 1.1) - sum(int(piece.size) for piece in pieces)
        if want < 1:
            want = 1
        clip = speech[cursor]
        piece = _read_window(clip, want, spec.sample_rate, rng)
        max_freq = min(max_freq, int(clip.sample_rate) // 2)
        if _rms(piece) < 1e-10:
            cursor = int(rng.integers(0, len(speech)))
            replacements += 1
            continue
        piece = _speech_aug(spec, piece, rng)
        if _rms(piece) < 1e-10:
            cursor = int(rng.integers(0, len(speech)))
            replacements += 1
            continue
        pieces.append(piece)
        if sum(int(item.size) for item in pieces) >= spec.num_samples:
            break
        cursor = int(rng.integers(0, len(speech)))
    joined = np.concatenate(pieces)
    if joined.size > spec.num_samples:
        start = int(rng.integers(0, joined.size - spec.num_samples))
        joined = joined[start : start + spec.num_samples]
    if joined.size != spec.num_samples:
        raise ReproductionError("speech window was not filled by concatenation")
    return joined.astype(np.float64), max_freq


def _load_noise(spec: DFN3OnlineSpec, noise: list, split: str, rng: np.random.Generator):
    probability = spec.p_synthetic_noise if split == "train" else 0.0
    if _gate(rng, probability):
        decay = float(rng.uniform(spec.synthetic_decay[0], spec.synthetic_decay[1]))
        piece = synthesize_noise(rng, decay, spec.num_samples, spec.sample_rate)
        return piece, int(_choose(rng, spec.synthetic_gains)), True
    replacements = 0
    while True:
        if replacements > _REPLACE_LIMIT:
            raise ReproductionError("noise replacement limit exceeded")
        clip = noise[int(rng.integers(0, len(noise)))]
        piece = _read_window(clip, spec.num_samples, spec.sample_rate, rng)
        replacements += 1
        if piece.size < 100 or _peak(piece) < 1e-10:
            continue
        piece = _noise_aug(spec, piece, split, rng)
        if piece.size > spec.num_samples:
            piece = piece[: spec.num_samples]
        return piece.astype(np.float64), int(_choose(rng, spec.gains)), False


def _combine_noises(pieces: list[np.ndarray], gains: list[int], length: int, rng: np.random.Generator):
    """Sum gain-scaled noises. Divide by the channel count, which is 1.

    A piece that is not strictly longer than ``length`` is repeated until it
    is, then cropped. That includes a piece that already had the right length.
    """
    acc = np.zeros(length, dtype=np.float64)
    for piece, gain in zip(pieces, gains, strict=True):
        wave = np.asarray(piece, dtype=np.float64).reshape(-1)
        if wave.size < 1:
            raise ReproductionError("a noise piece is empty")
        while wave.size <= length:
            wave = np.concatenate([wave, wave])
        extra = int(wave.size - length)
        start = int(rng.integers(0, extra))
        wave = wave[start : start + length]
        acc += wave * (10.0 ** (float(gain) / 20.0))
    return acc


def _maybe_reverb(spec: DFN3OnlineSpec, speech, noise, rir, rng: np.random.Generator):
    apply_speech = _gate(rng, spec.p_reverb)
    apply_noise = _gate(rng, spec.p_reverb)
    if not apply_speech and not apply_noise:
        return speech, speech.copy(), noise, False, False
    clip = rir[int(rng.integers(0, len(rir)))]
    impulse = _read_rir(clip, spec, rng)
    decay = None
    if _gate(rng, max(spec.p_reverb, spec.p_decay_floor)):
        decay = float(rng.uniform(0.2, 1.0))
    offset = spec.offset_late_ms * spec.sample_rate // 1000
    target, reverberant, noise_out = reverb_pair(
        speech,
        noise,
        impulse,
        spec.sample_rate,
        drr=spec.drr,
        rt60=spec.rt60,
        offset_samples=offset,
        apply_speech=apply_speech,
        apply_noise=apply_noise,
        decay_rt60=decay,
    )
    distorted = reverberant if apply_speech else speech.copy()
    return target, distorted, noise_out, apply_speech, apply_noise


def _read_rir(clip, spec: DFN3OnlineSpec, rng: np.random.Generator) -> np.ndarray:
    frames = int(clip.frames)
    start = 0
    stop = frames
    if frames > spec.rir_max_samples:
        start = int(rng.integers(0, frames - spec.rir_max_samples))
        stop = start + spec.rir_max_samples
    impulse = np.asarray(clip.read(start, stop), dtype=np.float64).reshape(-1)
    if int(clip.sample_rate) != spec.sample_rate:
        impulse = _resample(impulse, int(clip.sample_rate), spec.sample_rate)
    if _gate(rng, spec.p_reverb):
        new_rate = _rounded_rate(
            float(rng.uniform(spec.rir_resample[0], spec.rir_resample[1])) * spec.sample_rate,
            spec.resample_step_hz,
        )
        if new_rate != spec.sample_rate and new_rate > 0:
            impulse = _resample(impulse, spec.sample_rate, new_rate)
    return impulse


def _speech_aug(spec: DFN3OnlineSpec, wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if _gate(rng, spec.p_remove_dc):
        wave = wave - float(np.mean(wave))
    wave = _maybe_lfilt(spec, wave, spec.p_lfilt, rng)
    wave = _maybe_resample(spec, wave, spec.p_resample, spec.sample_rate, rng)
    return wave


def _noise_aug(spec: DFN3OnlineSpec, wave: np.ndarray, split: str, rng: np.random.Generator) -> np.ndarray:
    wave = _maybe_lfilt(spec, wave, spec.p_lfilt, rng)
    wave = _maybe_resample(spec, wave, spec.p_resample, spec.sample_rate, rng)
    if split == "train" and _gate(rng, spec.p_noise_clip):
        peak = _peak(wave)
        low, high = spec.clip_c
        limit = float(rng.uniform(low * peak, high * peak))
        wave = np.clip(wave, -limit, limit)
    return wave


def _maybe_lfilt(spec: DFN3OnlineSpec, wave: np.ndarray, probability: float, rng: np.random.Generator):
    if not _gate(rng, probability):
        return wave
    low, high = spec.lfilt_ab
    a = [float(rng.uniform(low, high)), float(rng.uniform(low, high))]
    b = [float(rng.uniform(low, high)), float(rng.uniform(low, high))]
    return _biquad(wave, b, a)


def _maybe_resample(spec, wave, probability, sample_rate, rng):
    if not _gate(rng, probability):
        return wave
    low, high = spec.resample_ratio
    new_rate = _rounded_rate(float(rng.uniform(low, high)) * sample_rate, spec.resample_step_hz)
    if new_rate == sample_rate or new_rate < 1:
        return wave
    return _resample(wave, sample_rate, new_rate)


def _read_window(clip, want_target: int, target_rate: int, rng: np.random.Generator) -> np.ndarray:
    native = int(clip.sample_rate)
    frames = int(clip.frames)
    want_native = max(int(want_target) * native // target_rate, 1)
    take = min(frames, want_native)
    start = 0
    if frames > take:
        start = int(rng.integers(0, frames - take))
    piece = np.asarray(clip.read(start, start + take), dtype=np.float64).reshape(-1)
    if native != target_rate:
        piece = _resample(piece, native, target_rate)
        if piece.size > want_target:
            piece = piece[:want_target]
    return piece


def _biquad(wave: np.ndarray, b: list[float], a: list[float]) -> np.ndarray:
    """``biquad_norm_inplace`` with a0 = b0 = 1. ``a`` and ``b`` are the two taps."""
    out = np.asarray(wave, dtype=np.float64).copy()
    mem0 = 0.0
    mem1 = 0.0
    b1, b2 = b
    a1, a2 = a
    for index, value in enumerate(out):
        y = float(value) + mem0
        mem0 = mem1 + (b1 * float(value) - a1 * y)
        mem1 = b2 * float(value) - a2 * y
        out[index] = y
    return out


def _resample(wave: np.ndarray, sample_rate: int, new_rate: int) -> np.ndarray:
    if sample_rate == new_rate:
        return np.asarray(wave, dtype=np.float64)
    out_len = max(int(np.ceil(wave.size * float(new_rate) / float(sample_rate))), 1)
    import scipy.signal

    return scipy.signal.resample(np.asarray(wave, dtype=np.float64), out_len).astype(np.float64)


def _low_pass(wave: np.ndarray, cutoff: int, sample_rate: int) -> np.ndarray:
    mid = int(cutoff) * 2
    if mid < 2 or mid == sample_rate:
        return wave
    original = int(wave.size)
    changed = _resample(_resample(wave, sample_rate, mid), mid, sample_rate)
    if changed.size >= original:
        return changed[:original]
    return np.pad(changed, (0, original - changed.size))


def _suppress_late(rir: np.ndarray, sample_rate: int, offset: int, rt60: float) -> np.ndarray:
    impulse = np.asarray(rir, dtype=np.float64).reshape(-1).copy()
    if offset >= impulse.size:
        return impulse
    dt = 1.0 / float(sample_rate)
    level = 10.0 ** (-60.0 / 20.0)
    tau = -float(rt60) / np.log10(level)
    steps = np.arange(impulse.size - offset, dtype=np.float64)
    impulse[offset:] *= 10.0 ** (-steps * dt / tau)
    return impulse


def _trim_rir(rir: np.ndarray, ref_idx: int) -> np.ndarray:
    impulse = np.asarray(rir, dtype=np.float64).reshape(-1)
    ref = float(impulse[ref_idx])
    if ref <= 0.0:
        return impulse
    floor = 10.0 ** ((-80.0 + np.log10(ref) * 20.0) / 20.0)
    end = impulse.size
    for back, value in enumerate(impulse[::-1]):
        if abs(float(value)) < floor:
            end = impulse.size - back
        else:
            break
    if end < 1:
        end = 1
    return impulse[:end]


def _l2(wave: np.ndarray) -> np.ndarray:
    energy = float(np.sqrt(np.sum(np.asarray(wave, dtype=np.float64) ** 2)))
    return np.asarray(wave, dtype=np.float64) / energy


def _convolve_prefix(wave: np.ndarray, impulse: np.ndarray, length: int) -> np.ndarray:
    import scipy.signal

    full = scipy.signal.fftconvolve(wave, impulse, mode="full")
    if full.size < length:
        full = np.pad(full, (0, length - full.size))
    return full[:length]


def _mix_factor(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> float:
    clean_energy = float(np.sum(np.asarray(clean, dtype=np.float64) ** 2)) + 1e-10
    noise_energy = float(np.sum(np.asarray(noise, dtype=np.float64) ** 2)) + 1e-10
    snr = 10.0 ** (float(snr_db) / 10.0)
    return float(1.0 / np.sqrt((noise_energy / clean_energy) * snr + 1e-10))


def _snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    clean_energy = float(np.sum(clean ** 2)) + 1e-10
    noise_energy = float(np.sum(noise ** 2)) + 1e-10
    return float(10.0 * np.log10(clean_energy / noise_energy))


def _gate(rng: np.random.Generator, probability: float) -> bool:
    """Official ``p == 0`` draws nothing. ``0 < p < 1`` applies when ``u <= p``."""
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return float(rng.random()) <= float(probability)


def _choose(rng: np.random.Generator, values):
    return values[int(rng.integers(0, len(values)))]


def _rounded_rate(value: float, step: int) -> int:
    return int(round(float(value) / float(step)) * step)


def _rms(wave: np.ndarray) -> float:
    data = np.asarray(wave, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data ** 2)))


def _peak(wave: np.ndarray) -> float:
    data = np.asarray(wave, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 0.0
    return float(np.max(np.abs(data)))


def _f32(wave: np.ndarray) -> np.ndarray:
    out = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not np.isfinite(out).all():
        raise ReproductionError("online mixture is not finite")
    return out
