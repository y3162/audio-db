"""On-demand VoiceBank+DEMAND reconstruction.

The realization stores one shared recipe per utterance: VCTK slice, noise
slice, peak-normalization scale, and the gains fitted by the 2026-11-12
inverse analysis. ``snr_shift_db`` comes from the dataset protocol. Rendering
reads VCTK and the noise file only. Official clean and noisy files are not
inputs.

``render_components`` returns float64 and does not clip or quantize.
``official_pcm16`` is the separate comparison against the published PCM.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import VOICEBANK_SNR_MIX_V1

ALGORITHM = "pregenerated.voicebank_snr_shift"
SAMPLE_RATE = 48000
STEPS = (
    "slice_vctk",
    "scale_to_stored_peak",
    "slice_noise",
    "apply_stored_gains",
    "scale_noise_for_snr_shift",
    "add",
)
SPEECH_KEYS = {"asset", "sha256", "channel", "start", "end"}
NOISE_KEYS = {"asset", "sha256", "channel", "start", "end", "kind", "name"}
NOISE_KINDS = {"demand", "babble", "ssn"}
DRAW_KEYS = {
    "gen_index",
    "build_id",
    "utterance_id",
    "split",
    "speaker_id",
    "official_snr_db",
    "clean_scale",
    "clean_gain_g",
    "noise_gain_before_peak_beta",
    "noise_gain_after_peak",
    "peak_steps",
    "target_peak_pcm",
    "silence",
}
SILENCE_KEYS = {"mode", "source_frames", "leading_removed", "trailing_removed"}
SPLITS = {"train", "valid", "test"}
PEAK_FACTORS = (1.1, 1.2, 1.3, 1.4)


def decimal_string(value: float) -> str:
    """Shortest round-trip decimal that ``float`` accepts unchanged."""

    number = float(value)
    if not math.isfinite(number):
        raise ReproductionError("decimal value is not finite")
    text = format(number, ".17g")
    if float(text) != number:
        raise ReproductionError(f"decimal value {value!r} does not round-trip")
    return text


def parse_decimal(value: Any, where: str) -> float:
    if not isinstance(value, str) or not value:
        raise ReproductionError(f"{where} must be a decimal string")
    try:
        number = float(value)
    except ValueError as exc:
        raise ReproductionError(f"{where} is not a decimal string") from exc
    if not math.isfinite(number) or decimal_string(number) != value:
        raise ReproductionError(f"{where} is not a canonical decimal string")
    return number


def noise_scale(snr_shift_db: float) -> float:
    """Multiply the stored noise gain. Negative shift makes the noise louder.

    ``snr_shift_db`` is target SNR minus the published SNR label.
    """

    shift = float(snr_shift_db)
    if not math.isfinite(shift):
        raise ReproductionError("snr_shift_db is not finite")
    return float(10 ** (-shift / 20.0))


def peak_product(steps: int) -> float:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= len(PEAK_FACTORS):
        raise ReproductionError("peak_steps must be an integer from 0 to 4")
    product = 1.0
    for index in range(steps):
        product *= PEAK_FACTORS[index]
    return product


def pcm16_round_nearest_even(samples: np.ndarray) -> np.ndarray:
    """Saturating round-to-nearest-even into PCM16. Ties follow ``numpy.rint``."""

    scaled = np.asarray(samples, dtype=np.float64) * np.float64(32768.0)
    if not np.isfinite(scaled).all():
        raise ReproductionError("PCM quantization received a non-finite sample")
    rounded = np.rint(scaled)
    return np.clip(rounded, -32768, 32767).astype(np.int16)


def validate_realization_document(document: dict[str, Any]) -> None:
    """Reject a VoiceBank realization that is not the shared v1 recipe."""

    if document.get("sample_rate") != SAMPLE_RATE:
        raise ReproductionError("voicebank realization sample_rate must be 48000")
    if tuple(document.get("steps") or ()) != STEPS:
        raise ReproductionError("voicebank realization steps are not the shipped order")
    sources = document.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"speech", "noise"}:
        raise ReproductionError("voicebank realization sources must be speech and noise")
    speech = _source(sources["speech"], "sources.speech", SPEECH_KEYS)
    noise = _source(sources["noise"], "sources.noise", NOISE_KEYS)
    if noise.get("kind") not in NOISE_KINDS:
        raise ReproductionError("voicebank noise kind is unknown")
    if not isinstance(noise.get("name"), str) or not noise["name"]:
        raise ReproductionError("voicebank noise name must be a string")
    draws = document.get("draws")
    if not isinstance(draws, dict) or set(draws) != DRAW_KEYS:
        extra = sorted(set(draws or {}) - DRAW_KEYS) if isinstance(draws, dict) else ["draws"]
        missing = sorted(DRAW_KEYS - set(draws or {})) if isinstance(draws, dict) else []
        raise ReproductionError(
            f"voicebank draws fields are wrong (missing {missing}, extra {extra})"
        )
    length = int(document["output"]["length"])
    if speech["end"] - speech["start"] != length or noise["end"] - noise["start"] != length:
        raise ReproductionError("voicebank speech and noise slices must match output.length")
    silence = draws["silence"]
    if not isinstance(silence, dict) or set(silence) != SILENCE_KEYS:
        raise ReproductionError("voicebank silence fields are wrong")
    if silence.get("mode") != "contiguous_slice":
        raise ReproductionError("voicebank silence mode must be contiguous_slice")
    source_frames = _non_negative(silence.get("source_frames"), "silence.source_frames")
    leading = _non_negative(silence.get("leading_removed"), "silence.leading_removed")
    trailing = _non_negative(silence.get("trailing_removed"), "silence.trailing_removed")
    if leading != speech["start"] or trailing != source_frames - speech["end"]:
        raise ReproductionError("voicebank silence removal does not match the speech slice")
    if source_frames < speech["end"]:
        raise ReproductionError("voicebank speech slice ends past the source")
    if draws.get("split") not in SPLITS:
        raise ReproductionError("voicebank split is unknown")
    speaker = draws.get("speaker_id")
    utterance = draws.get("utterance_id")
    if not isinstance(speaker, str) or not speaker:
        raise ReproductionError("voicebank speaker_id must be a string")
    if not isinstance(utterance, str) or not utterance.startswith(speaker + "_"):
        raise ReproductionError("voicebank utterance_id must start with the speaker id")
    _non_negative(draws.get("gen_index"), "draws.gen_index")
    build_id = draws.get("build_id")
    if not isinstance(build_id, str) or len(build_id) != 64:
        raise ReproductionError("voicebank build_id must be 64 hex characters")
    snr = draws.get("official_snr_db")
    if isinstance(snr, bool) or not isinstance(snr, (int, float)) or not math.isfinite(float(snr)):
        raise ReproductionError("official_snr_db must be a finite number")
    peak = draws.get("target_peak_pcm")
    if isinstance(peak, bool) or not isinstance(peak, int) or not 1 <= peak <= 32767:
        raise ReproductionError("target_peak_pcm must be an integer from 1 to 32767")
    if peak != int(VOICEBANK_SNR_MIX_V1["published_clean_peak_pcm"]):
        raise ReproductionError("target_peak_pcm is not the published clean peak")
    steps = draws.get("peak_steps")
    peak_product(steps if isinstance(steps, int) else -1)
    scale = parse_decimal(draws.get("clean_scale"), "clean_scale")
    gain = parse_decimal(draws.get("clean_gain_g"), "clean_gain_g")
    beta = parse_decimal(draws.get("noise_gain_before_peak_beta"), "noise_gain_before_peak_beta")
    after = parse_decimal(draws.get("noise_gain_after_peak"), "noise_gain_after_peak")
    if scale <= 0 or gain <= 0 or beta <= 0 or after <= 0:
        raise ReproductionError("voicebank scale and gains must be positive")
    expected_g = 1.0 / peak_product(int(steps))
    if abs(gain - expected_g) > 1e-9 * max(1.0, expected_g):
        raise ReproductionError("clean_gain_g does not match the stored peak steps")
    if abs(after - gain * beta) > 1e-8 * max(1.0, abs(after)):
        raise ReproductionError("noise_gain_after_peak does not match g * beta")


def check_voicebank_assets(case: dict[str, Any], assets: dict[str, dict[str, Any]]) -> None:
    """The stored hashes and channel counts have to match the asset rows."""

    document = case["doc_json"]
    for name in ("speech", "noise"):
        source = document["sources"][name]
        asset = assets.get(source["asset"])
        if asset is None:
            raise ReproductionError(f"voicebank case {case['case_id']} is missing {source['asset']}")
        if asset.get("sha256") != source["sha256"]:
            raise ReproductionError(f"voicebank case {case['case_id']} {name} sha256 does not match the asset")
        if int(asset["sample_rate"]) != SAMPLE_RATE:
            raise ReproductionError(f"voicebank asset {asset['asset_id']} is not 48 kHz")
        if int(source["channel"]) >= int(asset["channels"]):
            raise ReproductionError(f"voicebank asset {asset['asset_id']} has no channel {source['channel']}")
        if int(source["end"]) > int(asset["frames"]):
            raise ReproductionError(f"voicebank asset {asset['asset_id']} is shorter than the slice")
    draws = document["draws"]
    if str(draws["build_id"]) != str(case.get("build_id")):
        raise ReproductionError(f"voicebank case {case['case_id']} build_id does not match the row")
    if str(case.get("speaker_id") or "") != draws["speaker_id"]:
        raise ReproductionError(f"voicebank case {case['case_id']} speaker does not match the recipe")
    if str(case.get("utterance_key") or "") != draws["utterance_id"]:
        raise ReproductionError(f"voicebank case {case['case_id']} utterance does not match the recipe")


def render_components(
    document: dict[str, Any],
    speech: np.ndarray,
    noise: np.ndarray,
    *,
    snr_shift_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Float64 clean component and mixture. No clip and no PCM rounding."""

    validate_realization_document(document)
    length = int(document["output"]["length"])
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if speech.shape != (length,) or noise.shape != (length,):
        raise ReproductionError(
            f"voicebank slices are {speech.shape} and {noise.shape}; the recipe length is {length}"
        )
    if not np.isfinite(speech).all() or not np.isfinite(noise).all():
        raise ReproductionError("voicebank source slice is non-finite")
    draws = document["draws"]
    clean = speech * parse_decimal(draws["clean_scale"], "clean_scale")
    clean = clean * parse_decimal(draws["clean_gain_g"], "clean_gain_g")
    noisy = clean + parse_decimal(draws["noise_gain_after_peak"], "noise_gain_after_peak") * noise_scale(snr_shift_db) * noise
    if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ReproductionError("voicebank mixture is non-finite")
    return clean, noisy


def official_pcm16(document: dict[str, Any], speech: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCM candidate for the published clean and noisy files.

    Clean is the peak-normalized VCTK slice. The common gain ``g`` is applied
    to the mixture, which is where the published noisy file carries it.
    """

    clean, noisy = render_components(document, speech, noise, snr_shift_db=0)
    gain = parse_decimal(document["draws"]["clean_gain_g"], "clean_gain_g")
    speech_before_gain = clean / gain
    return pcm16_round_nearest_even(speech_before_gain), pcm16_round_nearest_even(noisy)


def p56_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Active-speech mean square over the noise segment RMS, in dB.

    The level is ``10 log10`` of the P.56 mean square. It is not dB SPL.
    A non-positive level or noise power is NaN, not 0 dB. The published
    label matches this ratio on the peak-normalized speech and the
    pre-peak noise. Method B thresholds are absolute, so the common gain
    ``g`` changes the measured value even though it scales both signals.
    """

    level, _activity, _threshold = active_speech_level(np.asarray(clean, dtype=np.float64), SAMPLE_RATE, 16)
    power = float(np.mean(np.asarray(noise, dtype=np.float64) ** 2))
    if level <= 0 or power <= 0 or not math.isfinite(level) or not math.isfinite(power):
        return float("nan")
    return float(10.0 * math.log10(level) - 10.0 * math.log10(power))


def component_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Full-span power ratio. This is not the published P.56 definition."""

    speech = float(np.dot(clean, clean))
    disturbance = float(np.dot(noise, noise))
    if speech <= 0 or disturbance <= 0 or not math.isfinite(speech) or not math.isfinite(disturbance):
        return float("nan")
    return float(10.0 * math.log10(speech / disturbance))


def read_span(path: Path, *, channel: int, start: int, end: int, sha256: str | None = None) -> np.ndarray:
    """Read one half-open mono span as float64. PCM16 files scale by 32768."""

    if "VoiceBank-DEMAND" in path.parts:
        raise ReproductionError("voicebank reconstruction must not read a published clean or noisy pair")
    if sha256 is not None:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ReproductionError(f"sha256 mismatch for {path}")
    import soundfile as sf

    if not path.is_file():
        raise ReproductionError(f"audio file is missing: {path}")
    info = sf.info(str(path))
    if int(info.samplerate) != SAMPLE_RATE:
        raise ReproductionError(f"{path} is {info.samplerate} Hz; voicebank reconstruction stays at 48000")
    if int(channel) < 0 or int(channel) >= int(info.channels):
        raise ReproductionError(f"{path} has no channel {channel}")
    if int(end) > int(info.frames):
        raise ReproductionError(f"{path} ends at {info.frames}, before sample {end}")
    audio, rate = sf.read(str(path), start=int(start), stop=int(end), dtype="float64", always_2d=True)
    if int(rate) != SAMPLE_RATE or int(audio.shape[0]) != int(end) - int(start):
        raise ReproductionError(f"{path} slice {start}:{end} was not read in full")
    span = np.ascontiguousarray(audio[:, int(channel)], dtype=np.float64)
    if not np.isfinite(span).all():
        raise ReproductionError(f"{path} slice is non-finite")
    return span


def render_case(store: Any, case_id: str, *, snr_shift_db: float, verify_on_read: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Render one snapshot case from its VCTK and noise assets."""

    document = store.docs.get(case_id)
    if not isinstance(document, dict):
        raise ReproductionError(f"case {case_id} has no voicebank realization")
    if document.get("algorithm") != ALGORITHM:
        raise ReproductionError(f"case {case_id} is not a voicebank snr-shift realization")
    speech_source = document["sources"]["speech"]
    noise_source = document["sources"]["noise"]
    speech = read_span(
        store.path(speech_source["asset"]),
        channel=int(speech_source["channel"]),
        start=int(speech_source["start"]),
        end=int(speech_source["end"]),
        sha256=speech_source["sha256"] if verify_on_read else None,
    )
    noise = read_span(
        store.path(noise_source["asset"]),
        channel=int(noise_source["channel"]),
        start=int(noise_source["start"]),
        end=int(noise_source["end"]),
        sha256=noise_source["sha256"] if verify_on_read else None,
    )
    return render_components(document, speech, noise, snr_shift_db=snr_shift_db)


def clean_scale_for(samples: np.ndarray) -> str:
    """Peak-normalize a VCTK slice to PCM 16384, stored as a decimal string."""

    peak = float(np.max(np.abs(np.asarray(samples, dtype=np.float64)))) if np.asarray(samples).size else 0.0
    if peak <= 0 or not math.isfinite(peak):
        raise ReproductionError("clean slice peak is not positive")
    return decimal_string((16384.0 / 32768.0) / peak)


def pcm_delta_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Signed PCM16 difference. Length mismatch is reported and not padded."""

    left = np.asarray(actual)
    right = np.asarray(reference)
    if left.shape != right.shape:
        return {
            "aligned": False,
            "actual_samples": int(left.size),
            "reference_samples": int(right.size),
        }
    delta = left.astype(np.int64) - right.astype(np.int64)
    if delta.size == 0:
        return {"aligned": True, "samples": 0, "exact": 0, "max_abs_lsb": 0, "min_lsb": 0, "max_lsb": 0, "sum_abs_lsb": 0, "sum_sq_lsb": 0, "histogram": {}}
    values, counts = np.unique(delta, return_counts=True)
    return {
        "aligned": True,
        "samples": int(delta.size),
        "exact": int(np.count_nonzero(delta == 0)),
        "max_abs_lsb": int(np.max(np.abs(delta))),
        "min_lsb": int(np.min(delta)),
        "max_lsb": int(np.max(delta)),
        "sum_abs_lsb": int(np.sum(np.abs(delta))),
        "sum_sq_lsb": int(delta @ delta),
        "histogram": {str(int(value)): int(count) for value, count in zip(values, counts)},
    }


def dataset_shift(dataset: dict[str, Any]) -> float:
    rule = dataset.get("rule_json")
    if dataset.get("algorithm") != ALGORITHM or not isinstance(rule, dict):
        raise ReproductionError("dataset is not voicebank snr shift")
    shift = rule.get("snr_shift_db")
    if isinstance(shift, bool) or not isinstance(shift, (int, float)):
        raise ReproductionError("dataset snr_shift_db is missing")
    return float(shift)


def active_speech_level(samples: np.ndarray, sample_rate: int, bits: int) -> tuple[float, float, float]:
    """ITU-T P.56 method B as used by the 2026-11-12 inverse analysis.

    Returns active mean square, activity factor, and threshold. The publisher's
    binary was not recovered, so this remains a measured candidate.
    """

    if bits != 16:
        raise ReproductionError("voicebank P.56 uses 16-bit thresholds")
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if signal.size < 2:
        return 0.0, 0.0, 0.0
    smoothing = 0.03
    hangover = 0.2
    margin = 15.9
    thresholds = bits - 1
    hang_samples = int(math.ceil(sample_rate * hangover))
    pole = math.exp(-1.0 / (sample_rate * smoothing))
    levels = np.ldexp(np.ones(thresholds, dtype=np.float64), np.arange(-15, thresholds - 15))
    envelope = _envelope(np.abs(signal), pole)
    counts = _count_activity(envelope, levels, hang_samples).astype(np.float64)
    # The reference implementation returns an undefined threshold when the first
    # bin never moves. Zero here means "no active level", not 0 dB.
    if counts[0] == -1:
        return 0.0, 0.0, 0.0
    counts = counts + 2.0
    energy = float(signal @ signal)
    eps = float(np.finfo(float).eps)
    active = np.empty(thresholds, dtype=np.float64)
    threshold_db = np.empty(thresholds, dtype=np.float64)
    active[0] = 10.0 * math.log10(energy / counts[0] + eps)
    threshold_db[0] = 20.0 * math.log10(float(levels[0]) + eps)
    if active[0] - threshold_db[0] < margin:
        return 0.0, 0.0, 0.0
    for index in range(1, thresholds):
        active[index] = 10.0 * math.log10(energy / (counts[index] + eps) + eps)
        threshold_db[index] = 20.0 * math.log10(float(levels[index]) + eps)
    for index in range(1, thresholds):
        if counts[index] == 0:
            continue
        if active[index] - threshold_db[index] <= margin:
            level_db, cross = _interpolate(
                float(active[index]),
                float(active[index - 1]),
                float(threshold_db[index]),
                float(threshold_db[index - 1]),
                margin,
                0.5,
            )
            mean_square = 10.0 ** (level_db / 10.0)
            activity = (energy / signal.size) / mean_square if mean_square > 0 else 0.0
            return float(mean_square), float(activity), float(10.0 ** (cross / 20.0))
    return 0.0, 0.0, 0.0


def _source(value: Any, where: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != allowed:
        raise ReproductionError(f"{where} fields are wrong")
    asset = value.get("asset")
    if not isinstance(asset, str) or not asset.startswith("corpora/") or ".." in asset:
        raise ReproductionError(f"{where}.asset must be a corpora asset id")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ReproductionError(f"{where}.sha256 must be 64 lowercase hex characters")
    channel = value.get("channel")
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
        raise ReproductionError(f"{where}.channel must be a non-negative integer")
    start = _non_negative(value.get("start"), f"{where}.start")
    end = _non_negative(value.get("end"), f"{where}.end")
    if end <= start:
        raise ReproductionError(f"{where} end must be after start")
    return value


def _non_negative(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReproductionError(f"{where} must be a non-negative integer")
    return value


def _envelope(magnitude: np.ndarray, pole: float) -> np.ndarray:
    import scipy.signal

    coefficient = 1.0 - pole
    first = scipy.signal.lfilter([coefficient, 0.0], [1.0, -pole], magnitude)
    return scipy.signal.lfilter([coefficient, 0.0], [1.0, -pole], first)


_COMPILED_COUNT: Any = None


def _count_activity(envelope: np.ndarray, levels: np.ndarray, hang_samples: int) -> np.ndarray:
    global _COMPILED_COUNT
    envelope = np.ascontiguousarray(envelope, dtype=np.float64)
    levels = np.ascontiguousarray(levels, dtype=np.float64)
    if _COMPILED_COUNT is False:
        return _count_activity_python(envelope, levels, int(hang_samples))
    if _COMPILED_COUNT is None:
        try:
            import numba

            _COMPILED_COUNT = numba.njit(_count_activity_python)
        except Exception:
            _COMPILED_COUNT = False
            return _count_activity_python(envelope, levels, int(hang_samples))
    try:
        return np.asarray(_COMPILED_COUNT(envelope, levels, int(hang_samples)))
    except Exception:
        _COMPILED_COUNT = False
        return _count_activity_python(envelope, levels, int(hang_samples))


def _count_activity_python(envelope: np.ndarray, levels: np.ndarray, hang_samples: int) -> np.ndarray:
    counts = np.full(levels.shape[0], -1, dtype=np.int64)
    hang = np.full(levels.shape[0], hang_samples, dtype=np.int64)
    for sample in envelope:
        for index in range(levels.shape[0]):
            if sample >= levels[index]:
                counts[index] += 1
                hang[index] = 0
            elif hang[index] < hang_samples:
                counts[index] += 1
                hang[index] += 1
            else:
                break
    return counts


def _interpolate(upper_level: float, lower_level: float, upper_threshold: float, lower_threshold: float, margin: float, tolerance: float) -> tuple[float, float]:
    if abs(upper_level - upper_threshold - margin) < tolerance:
        return upper_level, upper_threshold
    if abs(lower_level - lower_threshold - margin) < tolerance:
        return lower_level, lower_threshold
    level = (upper_level + lower_level) / 2.0
    threshold = (upper_threshold + lower_threshold) / 2.0
    iteration = 1
    while True:
        gap = level - threshold - margin
        if abs(gap) <= tolerance:
            return level, threshold
        iteration += 1
        if iteration > 20:
            tolerance *= 1.1
        if gap > tolerance:
            level = (upper_level + level) / 2.0
            threshold = (upper_threshold + threshold) / 2.0
        elif gap < -tolerance:
            level = (level + lower_level) / 2.0
            threshold = (threshold + lower_threshold) / 2.0
