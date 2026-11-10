"""SEMamba++ online degradation.

The order and formulas follow ``simulate_degradation.py`` and
``simulate_degradation_utils.py`` at
``cd369c89a8106072c077b468417b765b227afd74``, with HEAD ``config.yaml``.
This is not the DeepFilterNet3 mixer and not ``config_orig.yaml``.

One example is a function of the protocol, the snapshot, the training seed,
the split, the epoch (train only), and the speech index. Rank and worker id
are not inputs. The random stream is one NumPy PCG64. It is not CPython
``random`` plus a fresh ``default_rng(None)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import SEMAMBA_GSR_V1
from src.experiments.data.sampling.seeds import mix_seed

SAMPLE_RATE = 16000
SEGMENT_SIZE = 48000
RENDERER_VERSION = "semamba_gsr_v1"
PEAK_EPS = 1e-9
SNR_POWER_FLOOR = 1e-10
VAD_FRAME = 1024
VAD_SHIFT = 512
VAD_THRESHOLD = 0.01
LOWPASS_RIPPLE = 1
LOWPASS_STOP = 40
_ORDER = {"reverb": 1, "noise": 2, "clipping": 3, "bandwidth": 5}
_TYPES = ("noise", "reverb", "bandwidth", "clipping")


@dataclass(frozen=True)
class SemambaDraw:
    """The second configuration draw. The first draw is discarded."""

    snr_db: int
    bandwidth_sr: int
    bandwidth_type: str
    lowpass_type: str
    lowpass_order: int
    clipping_min: float
    clipping_max: float
    packet_duration: int
    packet_loss_rate: float
    selected: tuple[str, ...]


@dataclass(frozen=True)
class SemambaExample:
    clean: np.ndarray
    degraded: np.ndarray
    sample_rate: int
    crop_start: int
    padded: bool
    snr_db: int
    vad_snr_db: float | None
    selected: tuple[str, ...]
    discarded_selected: tuple[str, ...]
    noise_index: int
    rir_index: int
    full_clean_peak: float
    full_degraded_peak: float
    bandwidth_length_padded: bool


def sample_seed(
    protocol_id: str,
    snapshot_hash: str,
    seed: int,
    split: str,
    epoch: int,
    sample_index: int,
) -> int:
    """Train includes epoch. Valid and test do not."""
    epoch_key = int(epoch) if split == "train" else 0
    return mix_seed(
        "semamba-gsr",
        protocol_id,
        snapshot_hash,
        int(seed),
        split,
        epoch_key,
        int(sample_index),
    )


def draw_once(rng: np.random.Generator) -> SemambaDraw:
    """One pass of ``random_select_and_order`` for HEAD ``config.yaml``."""
    cfg = SEMAMBA_GSR_V1
    snr = int(rng.choice(cfg["snr"]))
    bandwidth_sr = int(rng.choice(cfg["bandwidth_sr"]))
    bandwidth_type = str(rng.choice(cfg["bandwidth_type"]))
    lowpass_type = str(rng.choice(cfg["lowpass_type"]))
    lowpass_order = int(rng.choice(cfg["lowpass_order"]))
    clipping_min = float(rng.uniform(cfg["clipping_min"][0], cfg["clipping_min"][1]))
    clipping_max = float(rng.uniform(cfg["clipping_max"][0], cfg["clipping_max"][1]))
    packet_duration = int(rng.choice(cfg["packet_duration"]))
    packet_loss_rate = float(rng.uniform(cfg["packet_loss_rate"][0], cfg["packet_loss_rate"][1]))
    probs = {
        "noise": float(cfg["noise_prob"]),
        "reverb": float(cfg["reverb_prob"]),
        "bandwidth": float(cfg["bandwidth_prob"]),
        "clipping": float(cfg["clipping_prob"]),
    }
    selected = [name for name in _TYPES if float(rng.random()) < probs[name]]
    if not selected:
        # HEAD config.yaml sets noise_prob to 1, so this branch is unreachable.
        # The official fallback passes unnormalized probabilities to np.random.choice.
        raise ReproductionError(
            "no degradation was selected; HEAD config.yaml always includes noise"
        )
    selected = tuple(sorted(selected, key=lambda name: _ORDER[name]))
    if "packet_loss" in selected:
        raise ReproductionError("packet_loss is not a selected degradation")
    return SemambaDraw(
        snr_db=snr,
        bandwidth_sr=bandwidth_sr,
        bandwidth_type=bandwidth_type,
        lowpass_type=lowpass_type,
        lowpass_order=lowpass_order,
        clipping_min=clipping_min,
        clipping_max=clipping_max,
        packet_duration=packet_duration,
        packet_loss_rate=packet_loss_rate,
        selected=selected,
    )


def draw_used(rng: np.random.Generator) -> tuple[SemambaDraw, SemambaDraw]:
    """Draw twice. Official ``__getitem__`` discards the first result."""
    discarded = draw_once(rng)
    used = draw_once(rng)
    return used, discarded


def framing(x: np.ndarray, frame_length: int = 512, frame_shift: int = 256) -> np.ndarray:
    """Official ``framing`` with ``centered=False`` and ``padded=True``."""
    if x.size == 0:
        raise ReproductionError("framing received an empty signal")
    if frame_length < 1 or frame_length > x.shape[-1] or frame_shift <= 0:
        raise ReproductionError("framing lengths are outside the official checks")
    nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
    if nadd:
        pad = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad, mode="constant", constant_values=0)
    if frame_length == 1 and frame_length == frame_shift:
        return x[..., None]
    shape = x.shape[:-1] + ((x.shape[-1] - frame_length) // frame_shift + 1, frame_length)
    strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def detect_non_silence(
    x: np.ndarray,
    threshold: float = VAD_THRESHOLD,
    frame_length: int = VAD_FRAME,
    frame_shift: int = VAD_SHIFT,
    window: str = "boxcar",
) -> np.ndarray:
    """Power VAD from the pinned ``detect_non_silence``."""
    signal = np.asarray(x)
    if signal.shape[-1] < frame_length:
        return np.full(signal.shape, fill_value=True, dtype=bool)
    if signal.dtype.kind == "i":
        signal = signal.astype(np.float64)
    framed = framing(signal, frame_length, frame_shift)
    framed = framed * scipy.signal.get_window(window, frame_length).astype(framed.dtype)
    power = (framed**2).mean(axis=-1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(signal.shape, fill_value=True, dtype=bool)
    detect_frames = power / mean_power > threshold
    detects = np.broadcast_to(detect_frames[..., None], detect_frames.shape + (frame_shift,))
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    return np.pad(
        detects,
        [(0, 0)] * (signal.ndim - 1) + [(0, signal.shape[-1] - detects.shape[-1])],
        mode="edge",
    )


def vad_power(x: np.ndarray) -> float:
    mask = detect_non_silence(x)
    picked = np.asarray(x)[mask]
    if picked.size == 0:
        return float("nan")
    return float((picked.astype(np.float64) ** 2).mean())


def mix_noise(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """VAD-scaled additive noise. Returns noisy, scaled noise, and the VAD SNR."""
    speech = _row(speech)
    noise = _row(noise)
    len_speech = speech.shape[-1]
    len_noise = noise.shape[-1]
    if len_noise < len_speech:
        offset = int(rng.integers(0, len_speech - len_noise))
        noise = np.pad(
            noise,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = int(rng.integers(0, len_noise - len_speech))
        noise = noise[:, offset : offset + len_speech]
    power_speech = vad_power(speech)
    power_noise = vad_power(noise)
    scale = 10 ** (-float(snr_db) / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, SNR_POWER_FLOOR))
    scaled = scale * noise
    noisy = speech + scaled
    measured = None
    if np.isfinite(power_speech) and power_speech > 0 and np.isfinite(scale):
        scaled_power = float((scale**2) * max(power_noise, 0.0))
        if scaled_power > 0:
            measured = float(10.0 * np.log10(power_speech / scaled_power))
    return noisy, scaled, measured


def add_reverberation(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Full convolution, then the input length. No RIR gain or delay fix."""
    speech = _row(speech)
    rir = _row(rir)
    reverberant = scipy.signal.convolve(speech, rir, mode="full")
    return reverberant[:, : speech.shape[1]]


def clipping(speech: np.ndarray, min_quantile: float, max_quantile: float) -> np.ndarray:
    """Per-channel quantile clip. This is not ``np.clip(x, -1, 1)``."""
    speech = _row(speech)
    quantiles = np.quantile(speech, np.asarray([min_quantile, max_quantile]), axis=-1, keepdims=False)
    low, high = quantiles[0], quantiles[1]
    return np.stack([np.clip(speech[i], low[i], high[i]) for i in range(speech.shape[0])], axis=0)


def bandwidth_limitation(
    speech: np.ndarray,
    fs: int,
    fs_new: int,
    res_type: str,
    lowpass_type: str,
    order: int,
) -> tuple[np.ndarray, bool]:
    """Low-pass, downsample, upsample, then the original length.

    A shorter resampler output is zero-padded. The official slice would make
    the later length assert fail. ``padded`` reports that guard.
    """
    speech = _row(speech)
    if fs == fs_new:
        return speech, False
    if fs < fs_new:
        raise ReproductionError(f"bandwidth target rate {fs_new} exceeds {fs}")
    nyq_orig = fs / 2
    nyq_new = fs_new / 2
    norm_cutoff = nyq_new / nyq_orig
    if lowpass_type == "butter":
        coefficients = scipy.signal.butter(order, norm_cutoff, btype="low", analog=False)
    elif lowpass_type == "cheby1":
        coefficients = scipy.signal.cheby1(order, LOWPASS_RIPPLE, norm_cutoff, btype="low")
    elif lowpass_type == "cheby2":
        coefficients = scipy.signal.cheby2(order, LOWPASS_RIPPLE, norm_cutoff, btype="low")
    elif lowpass_type == "bessel":
        coefficients = scipy.signal.bessel(order, norm_cutoff, btype="low", norm="phase")
    elif lowpass_type == "ellip":
        coefficients = scipy.signal.ellip(order, LOWPASS_RIPPLE, LOWPASS_STOP, norm_cutoff, btype="low")
    else:
        raise ReproductionError(f"unknown lowpass type {lowpass_type}")
    b, a = coefficients
    filtered = scipy.signal.filtfilt(b, a, speech)
    import librosa

    down = librosa.resample(filtered, orig_sr=fs, target_sr=fs_new, res_type=res_type)
    up = librosa.resample(down, orig_sr=fs_new, target_sr=fs, res_type=res_type)
    target = speech.shape[1]
    padded = up.shape[-1] < target
    if padded:
        up = np.pad(up, [(0, 0), (0, target - up.shape[-1])], mode="constant")
    return up[:, :target], padded


def apply_degradations(
    speech: np.ndarray,
    noise: np.ndarray,
    rir: np.ndarray,
    draw: SemambaDraw,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float | None, bool]:
    """Apply the selected degradations in the fixed order. Teacher stays dry."""
    clean = _row(speech)
    degraded = clean
    measured = None
    bandwidth_padded = False
    for name in draw.selected:
        if name == "reverb":
            degraded = add_reverberation(degraded, rir)
        elif name == "noise":
            degraded, _, measured = mix_noise(degraded, noise, draw.snr_db, rng)
        elif name == "clipping":
            degraded = clipping(degraded, draw.clipping_min, draw.clipping_max)
        elif name == "bandwidth":
            degraded, bandwidth_padded = bandwidth_limitation(
                degraded,
                SAMPLE_RATE,
                draw.bandwidth_sr,
                draw.bandwidth_type,
                draw.lowpass_type,
                draw.lowpass_order,
            )
        else:
            raise ReproductionError(f"unexpected degradation {name}")
        if degraded.shape[-1] != clean.shape[-1]:
            raise ReproductionError(f"{name} changed the waveform length")
    return clean, degraded, measured, bandwidth_padded


def independent_peak(audio: np.ndarray, eps: float = PEAK_EPS) -> tuple[np.ndarray, float]:
    """``x / (max|x| + eps)`` in float32, matching ``peak_normalize``."""
    values = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    scaled = values / (np.float32(peak) + np.float32(eps))
    return scaled, peak


def peak_and_crop(
    clean: np.ndarray,
    degraded: np.ndarray,
    rng: np.random.Generator,
    segment: int = SEGMENT_SIZE,
) -> tuple[np.ndarray, np.ndarray, int, bool, float, float]:
    """Normalize each full waveform, then take the same crop or tail pad."""
    if clean.shape != degraded.shape:
        raise ReproductionError("clean and degraded lengths differ before the peak norm")
    clean_f, clean_peak = independent_peak(np.reshape(clean, -1))
    degraded_f, degraded_peak = independent_peak(np.reshape(degraded, -1))
    length = int(clean_f.shape[-1])
    if length >= segment:
        start = int(rng.integers(0, length - segment + 1))
        return (
            clean_f[start : start + segment],
            degraded_f[start : start + segment],
            start,
            False,
            clean_peak,
            degraded_peak,
        )
    pad = segment - length
    return (
        np.pad(clean_f, (0, pad)),
        np.pad(degraded_f, (0, pad)),
        0,
        True,
        clean_peak,
        degraded_peak,
    )


def render_example(
    speech,
    noise,
    rir,
    *,
    speech_index: int,
    split: str,
    rng: np.random.Generator,
) -> SemambaExample:
    """One 16 kHz, 48,000-sample pair. ``speech[speech_index]`` is the teacher source."""
    if split not in ("train", "valid", "test"):
        raise ReproductionError(f"unknown split {split}")
    if not speech or not noise or not rir:
        raise ReproductionError("SEMamba++ online rendering needs speech, noise, and RIR")
    if speech_index < 0 or speech_index >= len(speech):
        raise ReproductionError(f"speech index {speech_index} is outside the list")
    noise_index = int(rng.integers(0, len(noise)))
    rir_index = int(rng.integers(0, len(rir)))
    used, discarded = draw_used(rng)
    clean, degraded, measured, bandwidth_padded = apply_degradations(
        _read(speech[speech_index]),
        _read(noise[noise_index]),
        _read(rir[rir_index]),
        used,
        rng,
    )
    cropped_clean, cropped_degraded, start, padded, clean_peak, degraded_peak = peak_and_crop(
        clean, degraded, rng, SEGMENT_SIZE
    )
    if cropped_clean.shape != (SEGMENT_SIZE,) or cropped_degraded.shape != cropped_clean.shape:
        raise ReproductionError("SEMamba++ crop did not return 48000 aligned samples")
    return SemambaExample(
        clean=np.ascontiguousarray(cropped_clean, dtype=np.float32),
        degraded=np.ascontiguousarray(cropped_degraded, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        crop_start=start,
        padded=padded,
        snr_db=used.snr_db,
        vad_snr_db=measured,
        selected=used.selected,
        discarded_selected=discarded.selected,
        noise_index=noise_index,
        rir_index=rir_index,
        full_clean_peak=clean_peak,
        full_degraded_peak=degraded_peak,
        bandwidth_length_padded=bandwidth_padded,
    )


def _row(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    if audio.ndim != 2 or audio.shape[0] != 1 or audio.shape[1] < 1:
        raise ReproductionError(f"expected a mono waveform, got shape {audio.shape}")
    return np.ascontiguousarray(audio)


def _read(clip) -> np.ndarray:
    audio = clip.read() if hasattr(clip, "read") else clip
    return _row(audio)
