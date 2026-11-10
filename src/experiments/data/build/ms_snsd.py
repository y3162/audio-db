"""MS-SNSD mixtures for the ESPnet public SE recipe.

The mix follows ``audiolib.snr_mixer`` and ``ms_snsd_noisyspeech_synthesizer.py``
at ESPnet ``0d136c5``. ``audioread(norm=True)`` scales each whole source file
to -25 dBFS before concatenation or the noise prefix crop. ``snr_mixer`` then
scales the concatenated waveforms to -25 dBFS again. Nominal SNR uses ``sqrt``,
so the measured SNR is half the nominal value. Clips shorter than or equal to
10 s are concatenated with 0.2 s of silence, and a longer clean clip is kept
whole. Noise is cut from the start to the clean length. There is no random offset.

The official script draws from the process-global ``np.random`` and never
seeds it. This builder draws each clip from
``Generator(PCG64(mix_seed('gen', seed, split, clip_index)))`` and sorts the
file list before one seeded shuffle. That seed is a recorded deviation
(D-07), not a claim that an official unseeded run can be reproduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import build_id_for, digest, validate_realization
from src.experiments.data.sampling.seeds import mix_seed

GENERATOR = "ms_snsd_synth"
GENERATOR_VERSION = 1
ALGORITHM = "pregenerated.ms_snsd_tfgridnet"
SAMPLE_RATE = 16000
NOMINAL_SNRS = (0.0, 10.0, 20.0, 30.0, 40.0)
SILENCE_SECONDS = 0.2
AUDIO_SECONDS = 10.0
TARGET_DBFS = -25.0
# ESPnet enh.sh keeps utt2num_samples with min < length < max.
# ms_snsd/enh1/run.sh sets max_wav_duration=31 and the template default min is 0.1 s.
DURATION_MIN_EXCLUSIVE = int(0.1 * SAMPLE_RATE)
DURATION_MAX_EXCLUSIVE = 31 * SAMPLE_RATE
OFFICIAL_COMMIT = "0d136c5456d403cb2c0a4c13e76e8136e32e9b7a"


def nominal_snrs(lower: float = 0.0, upper: float = 40.0, levels: int = 5) -> tuple[float, ...]:
    if levels < 1:
        raise ReproductionError("SNR levels must be positive")
    return tuple(float(value) for value in np.linspace(lower, upper, levels))


def scale_to_dbfs(waveform: np.ndarray, level_db: float = TARGET_DBFS) -> np.ndarray:
    """Match ``audioread(..., norm=True)`` and the first half of ``snr_mixer``.

    The official divisor is the RMS with no epsilon. A silent file is an error
    here; the official script would divide by zero.
    """
    audio = np.asarray(waveform, dtype=np.float64)
    rms = (audio ** 2).mean() ** 0.5
    if not np.isfinite(rms) or rms == 0.0:
        raise ReproductionError("cannot scale a silent waveform to dBFS")
    return audio * (10 ** (level_db / 20.0) / rms)


def snr_mixer(clean: np.ndarray, noise: np.ndarray, snr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official ``snr_mixer``. The returned noise is the scaled noise, not the input."""
    clean_scaled = scale_to_dbfs(clean)
    noise_scaled = scale_to_dbfs(noise)
    clean_rms = float(np.sqrt(np.mean(clean_scaled * clean_scaled)))
    noise_rms = float(np.sqrt(np.mean(noise_scaled * noise_scaled)))
    noisescalar = np.sqrt(clean_rms / (10 ** (float(snr) / 20.0)) / noise_rms)
    scaled_noise = noise_scaled * noisescalar
    return clean_scaled, scaled_noise, clean_scaled + scaled_noise


def measured_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    clean_rms = float(np.sqrt(np.mean(np.asarray(clean, dtype=np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean(np.asarray(noise, dtype=np.float64) ** 2)))
    if clean_rms == 0.0 or noise_rms == 0.0:
        raise ReproductionError("SNR is undefined for a silent waveform")
    return 20.0 * float(np.log10(clean_rms / noise_rms))


def list_wavs(directory: Path) -> list[Path]:
    """Flat ``*.wav`` list, sorted by file name. Official ``glob`` order is not stable."""
    if not directory.is_dir():
        raise ReproductionError(f"audio directory is missing: {directory}")
    paths = sorted(path for path in directory.glob("*.wav") if path.is_file())
    if not paths:
        raise ReproductionError(f"no wav files in {directory}")
    return paths


def ordered_paths(paths: Sequence[Path], *, seed: int, split: str) -> list[str]:
    """One seeded shuffle of the sorted relative names. The same seed repeats it."""
    names = [path.name for path in paths]
    generator = np.random.Generator(np.random.PCG64(mix_seed("list", seed, split, "ms_snsd")))
    generator.shuffle(names)
    return names


def plan_clip(
    clean_names: Sequence[str],
    noise_names: Sequence[str],
    clean_frames: Sequence[int],
    noise_frames: Sequence[int],
    *,
    seed: int,
    split: str,
    clip_index: int,
    audio_length: int,
    silence: int,
) -> dict[str, Any]:
    """Choose source pieces for one clip. This does not read or mix audio."""
    if clip_index < 0 or audio_length < 2 or silence < 0:
        raise ReproductionError("clip plan sizes must be non-negative, and the window must be positive")
    if len(clean_names) != len(clean_frames) or len(noise_names) != len(noise_frames):
        raise ReproductionError("names and frame counts differ")
    rng = np.random.Generator(np.random.PCG64(mix_seed("gen", seed, split, clip_index)))
    clean_pieces, clean_total = _assemble(clean_names, clean_frames, rng, audio_length, silence, stop_at=None)
    noise_pieces, _noise_total = _assemble(
        noise_names, noise_frames, rng, audio_length, silence, stop_at=clean_total
    )
    return {
        "clip_index": clip_index,
        "clean": clean_pieces,
        "noise": noise_pieces,
        "length": clean_total,
        "silence": silence,
    }


def build_definition(
    *,
    seed: int,
    split: str,
    clean_names: Sequence[str],
    noise_names: Sequence[str],
    total_hours: float,
    max_clips: int | None,
    shard_size: int = 5,
    inputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    snrs = nominal_snrs()
    if shard_size % len(snrs) != 0:
        raise ReproductionError("shard_size must be a multiple of the five nominal SNRs")
    params = {
        "split": split,
        "sample_rate": SAMPLE_RATE,
        "audio_seconds": AUDIO_SECONDS,
        "silence_seconds": SILENCE_SECONDS,
        "total_hours": float(total_hours),
        "max_clips": max_clips,
        "snrs": list(snrs),
        "target_dbfs": TARGET_DBFS,
        "clean_order": list(clean_names),
        "noise_order": list(noise_names),
        "shard_size": int(shard_size),
    }
    if inputs is None:
        # Planning tests that do not register a build keep this digest.
        # Registered builds pass material-set ids and use build_id_for.
        build_id = digest(
            {
                "generator": GENERATOR,
                "generator_version": GENERATOR_VERSION,
                "seed": seed,
                "params": params,
            }
        )
    else:
        if not isinstance(inputs, dict) or not inputs:
            raise ReproductionError("build inputs must be a non-empty object")
        build_id = build_id_for(
            generator=GENERATOR,
            generator_version=GENERATOR_VERSION,
            params=params,
            seed=int(seed),
            inputs=dict(inputs),
        )
    body = {
        "build_id": build_id,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "shard_size": shard_size,
        "params": params,
        "deviations": [
            {
                "id": "seeded_draws",
                "class": "algorithm_change",
                "text": (
                    "Official MS-SNSD generation uses unseeded np.random. "
                    "Each clip uses mix_seed('gen', seed, split, clip_index), and the "
                    "file list is sorted and shuffled once with mix_seed('list', ...). "
                    "An official unseeded corpus cannot be reproduced file-for-file."
                ),
            }
        ],
    }
    if inputs is not None:
        body["inputs"] = {str(key): str(value) for key, value in sorted(inputs.items())}
    return body


def plan_realizations(
    definition: dict[str, Any],
    clean_frames: dict[str, int],
    noise_frames: dict[str, int],
) -> list[dict[str, Any]]:
    """Realizations until ``total_hours`` or ``max_clips``. Audio is not mixed."""
    params = definition["params"]
    snrs = tuple(float(value) for value in params["snrs"])
    audio_length = int(params["audio_seconds"] * SAMPLE_RATE)
    silence = int(params["silence_seconds"] * SAMPLE_RATE)
    clean_names = list(params["clean_order"])
    noise_names = list(params["noise_order"])
    limit_samples = int(float(params["total_hours"]) * 3600 * SAMPLE_RATE)
    max_clips = params["max_clips"]
    produced = 0
    clip_index = 0
    documents: list[dict[str, Any]] = []
    while produced < limit_samples and (max_clips is None or clip_index < int(max_clips)):
        clip = plan_clip(
            clean_names,
            noise_names,
            [clean_frames[name] for name in clean_names],
            [noise_frames[name] for name in noise_names],
            seed=int(definition["seed"]),
            split=str(params["split"]),
            clip_index=clip_index,
            audio_length=audio_length,
            silence=silence,
        )
        for snr_index, snr in enumerate(snrs):
            gen_index = clip_index * len(snrs) + snr_index
            document = _realization(definition, clip, gen_index, snr)
            stored = {key: value for key, value in document.items() if key in {
                "realization_version", "algorithm", "algorithm_version", "sample_rate",
                "sources", "draws", "steps", "output",
            }}
            validate_realization(stored)
            documents.append(document)
            produced += int(clip["length"])
        clip_index += 1
    if not documents:
        raise ReproductionError("MS-SNSD plan produced no realizations")
    return documents


def render_realization(
    document: dict[str, Any],
    directory_for: dict[str, Path | dict[str, Path]],
) -> dict[str, np.ndarray]:
    """Mix one planned realization. Draws are read from the document, not from a new RNG."""
    clean = _render_pieces(document["sources"]["speech"], directory_for["speech"])
    noise = _render_pieces(document["sources"]["noise"], directory_for["noise"])
    clean = clean[: int(document["output"]["length"])]
    noise = noise[: clean.shape[0]]
    mixed_clean, mixed_noise, noisy = snr_mixer(clean, noise, float(document["draws"]["nominal_snr_db"]))
    if mixed_clean.shape != noisy.shape or mixed_noise.shape != noisy.shape:
        raise ReproductionError("mixed waveforms differ in length")
    return {"clean": mixed_clean, "noise": mixed_noise, "noisy": noisy}


def _assemble(names, frames, rng, audio_length: int, silence: int, stop_at: int | None) -> tuple[list[dict[str, Any]], int]:
    count = len(names)
    if count < 1:
        raise ReproductionError("cannot assemble an empty file list")
    index = int(rng.integers(0, count))
    pieces = [_piece(names[index], frames[index])]
    total = int(frames[index])
    target = audio_length if stop_at is None else stop_at
    if stop_at is None and total > target:
        return pieces, total
    if stop_at is not None and total >= target:
        return _truncate(pieces, target)
    while total <= target:
        index += 1
        if index >= count - 1:
            index = int(rng.integers(0, count))
        pieces.append({"silence": silence})
        pieces.append(_piece(names[index], frames[index]))
        total += silence + int(frames[index])
    if stop_at is not None:
        return _truncate(pieces, stop_at)
    return pieces, total


def _piece(name: str, frames: int) -> dict[str, Any]:
    return {"name": name, "start": 0, "length": int(frames)}


def _truncate(pieces: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    total = 0
    for piece in pieces:
        if "silence" in piece:
            room = target - total
            if room <= 0:
                break
            silence = min(int(piece["silence"]), room)
            if silence:
                kept.append({"silence": silence})
                total += silence
            continue
        room = target - total
        if room <= 0:
            break
        length = min(int(piece["length"]), room)
        kept.append({"name": piece["name"], "start": int(piece["start"]), "length": length})
        total += length
    return kept, total


def official_utt_id(clip_index: int, snr: float) -> str:
    """Basename used by ``ms_snsd_data_prep.sh``.

    The official counter starts at 1. ``str(SNR[i])`` for the linspace values
    is ``0.0``, ``10.0``, ``20.0``, ``30.0``, ``40.0``.
    """
    if clip_index < 0:
        raise ReproductionError("clip_index must be non-negative")
    number = clip_index + 1
    return f"noisy{number}_SNRdb_{str(float(snr))}_clnsp{number}"


def _realization(definition: dict[str, Any], clip: dict[str, Any], gen_index: int, snr: float) -> dict[str, Any]:
    return {
        "realization_version": 1,
        "algorithm": ALGORITHM,
        "algorithm_version": 1,
        "sample_rate": SAMPLE_RATE,
        "sources": {"speech": clip["clean"], "noise": clip["noise"]},
        "draws": {
            "gen_index": gen_index,
            "clip_index": clip["clip_index"],
            "nominal_snr_db": snr,
            "utt_id": official_utt_id(int(clip["clip_index"]), float(snr)),
            "build_id": definition["build_id"],
        },
        "steps": ["dbfs_m25", "concat_silence", "snr_mixer_sqrt", "pcm16"],
        "output": {"length": clip["length"], "teacher": "dry_clean"},
    }


def _render_pieces(pieces: Sequence[dict[str, Any]], directory: Path | dict[str, Path]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for piece in pieces:
        if "silence" in piece:
            chunks.append(np.zeros(int(piece["silence"]), dtype=np.float64))
            continue
        name = str(piece["name"])
        if isinstance(directory, dict):
            path = directory.get(name)
            if path is None:
                raise ReproductionError(f"realization source {name} is not in the material set")
        else:
            path = directory / name
        audio, rate = sf.read(path, dtype="float64", always_2d=False)
        if int(rate) != SAMPLE_RATE:
            raise ReproductionError(f"{path} is {rate} Hz; MS-SNSD generation does not resample")
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # audioread(norm=True) scales the whole file, then the synthesizer crops.
        audio = scale_to_dbfs(audio)
        start = int(piece["start"])
        stop = start + int(piece["length"])
        if stop > audio.shape[0]:
            raise ReproductionError(f"{path} is shorter than the plan")
        chunks.append(audio[start:stop])
    if not chunks:
        raise ReproductionError("realization has no audio pieces")
    return np.concatenate(chunks)


def read_frames(directory: Path, names: Sequence[str]) -> dict[str, int]:
    frames: dict[str, int] = {}
    for name in names:
        info = sf.info(directory / name)
        if int(info.samplerate) != SAMPLE_RATE or int(info.channels) < 1:
            raise ReproductionError(f"{name} is not {SAMPLE_RATE} Hz audio")
        frames[name] = int(info.frames)
    return frames


def train_valid_counts(count: int) -> tuple[int, int]:
    """``ms_snsd_data_prep.sh`` uses bash ``count * 8 / 10``."""
    if count < 2:
        raise ReproductionError("the 80/20 split needs at least two noisy files")
    train_count = count * 8 // 10
    valid_count = count - train_count
    if train_count < 1 or valid_count < 1:
        raise ReproductionError(f"80/20 split of {count} files leaves an empty side")
    return train_count, valid_count


def duration_kept(frames: int) -> bool:
    """ESPnet dump keeps ``min < utt2num_samples < max`` at 16 kHz."""
    return DURATION_MIN_EXCLUSIVE < int(frames) < DURATION_MAX_EXCLUSIVE


def sort_like_kaldi(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """``sort -k1,1`` / ``find | sort`` order of the official noisy basename."""
    return sorted(documents, key=lambda item: str(item["draws"]["utt_id"]))


def _clip_id(utt_id: str) -> str:
    marker = "_clnsp"
    index = utt_id.rfind(marker)
    if index < 0:
        raise ReproductionError(f"utt id has no clnsp marker: {utt_id}")
    return utt_id[index:]


def _clip_atomic_cut(ordered: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Shell cut, moved to a clip boundary when it would split one clip.

    ``find | sort`` then ``count * 8 / 10`` can land inside one clip's five SNR
    names. The five files share one clean waveform, so leaving them on both
    sides makes train and valid content hashes overlap. The nearer clip edge
    is used, and the distance is the number of files moved.
    """
    count = len(ordered)
    cut, _valid = train_valid_counts(count)
    if cut <= 0 or cut >= count:
        return cut, 0
    left = _clip_id(str(ordered[cut - 1]["draws"]["utt_id"]))
    right = _clip_id(str(ordered[cut]["draws"]["utt_id"]))
    if left != right:
        return cut, 0
    start = cut - 1
    while start > 0 and _clip_id(str(ordered[start - 1]["draws"]["utt_id"])) == left:
        start -= 1
    end = cut
    while end + 1 < count and _clip_id(str(ordered[end + 1]["draws"]["utt_id"])) == left:
        end += 1
    if abs(start - cut) <= abs((end + 1) - cut):
        return start, cut - start
    return end + 1, (end + 1) - cut


def split_train_valid(
    documents: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """80/20 on sorted utt ids, then drop lengths outside the ESPnet dump window."""
    ordered = sort_like_kaldi(documents)
    train_count, moved = _clip_atomic_cut(ordered)
    raw_train = ordered[:train_count]
    raw_valid = ordered[train_count:]
    train = [item for item in raw_train if duration_kept(int(item["output"]["length"]))]
    valid = [item for item in raw_valid if duration_kept(int(item["output"]["length"]))]
    if not train or not valid:
        raise ReproductionError("duration filter removed every train or valid utterance")
    report = {
        "files": len(ordered),
        "train_before_duration": len(raw_train),
        "valid_before_duration": len(raw_valid),
        "train_kept": len(train),
        "valid_kept": len(valid),
        "train_dropped": len(raw_train) - len(train),
        "valid_dropped": len(raw_valid) - len(valid),
    }
    if moved:
        report["shell_cut"] = len(ordered) * 8 // 10
        report["files_moved_to_keep_clip"] = moved
    return train, valid, report


def filter_test(documents: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ordered = sort_like_kaldi(documents)
    kept = [item for item in ordered if duration_kept(int(item["output"]["length"]))]
    if not kept:
        raise ReproductionError("duration filter removed every test utterance")
    return kept, {"files": len(ordered), "kept": len(kept), "dropped": len(ordered) - len(kept)}


def shape_lines(rows: Sequence[tuple[str, int]]) -> str:
    """``utt2num_samples`` from ``format_wav_scp.py``: ``utt_id`` and ``len(wave)``.

    A mono soundfile array is 1-D, so ESPnet's comma-joined ``array.shape`` is the
    same integer. The chunk sampler reads that first integer as the length.
    """
    lines = []
    for utt_id, frames in rows:
        if not utt_id or int(frames) < 1:
            raise ReproductionError("shape rows need a utt id and a positive length")
        lines.append(f"{utt_id} {int(frames)}")
    return "\n".join(lines) + "\n"


def _pcm16_roundtrip(wave: np.ndarray) -> np.ndarray:
    """Match libsndfile's float PCM_16 read-back.

    Positive peaks stop at 32767/32768. Codes are ``floor(clipped * 32768)``
    and the file reads back as ``code / 32768``.
    """
    limit = 32767.0 / 32768.0
    codes = np.floor(np.clip(np.asarray(wave, dtype=np.float64), -1.0, limit) * 32768.0)
    return codes / 32768.0


