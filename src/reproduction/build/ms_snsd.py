"""MS-SNSD mixtures for the ESPnet public SE recipe.

The mix follows ``audiolib.snr_mixer`` and ``ms_snsd_noisyspeech_synthesizer.py``
at ESPnet ``0d136c5``: each source is scaled to -25 dBFS, nominal SNR uses
``sqrt`` so the measured SNR is half the nominal value, clips shorter than
10 s are concatenated with 0.2 s of silence, and a longer clip is kept whole.
Noise is cut from the start to the clean length. There is no random offset.

The official script draws from the process-global ``np.random`` and never
seeds it. This builder draws each clip from
``Generator(PCG64(mix_seed('gen', seed, split, clip_index)))`` and sorts the
file list before one seeded shuffle. That seed is a recorded deviation
(D-07), not a claim that an official unseeded run can be reproduced.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from src.data.recipe import sha256_file
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import build_id_for, digest, validate_realization
from src.reproduction.pipeline.seeds import mix_seed

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
    """Match ``audioread(..., norm=True)`` and the first half of ``snr_mixer``."""
    audio = np.asarray(waveform, dtype=np.float64)
    rms = float(np.sqrt(np.mean(audio * audio)))
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
    directory_for: dict[str, Path],
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


def write_build(
    definition: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    audio_root: Path,
    staging_root: Path,
    clean_dir: Path,
    noise_dir: Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Write wavs and a shard manifest. A finished shard is not rewritten."""
    build_id = str(definition["build_id"])
    audio_dir = audio_root / GENERATOR / build_id[:16]
    stage_dir = staging_root / "builds" / build_id
    _refuse_conflicting_definition(stage_dir, definition)
    estimate = _estimate_bytes(documents)
    _require_space(audio_dir if audio_dir.exists() else audio_root, estimate)
    stage_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(stage_dir / "build.json", json.dumps(definition, sort_keys=True, indent=2) + "\n")
    _atomic_text(stage_dir / "plan.jsonl", "".join(json.dumps(item, sort_keys=True) + "\n" for item in documents))
    shard_size = int(definition["shard_size"])
    shard_ids = sorted({int(item["draws"]["gen_index"]) // shard_size for item in documents})
    written = []
    for shard_id in shard_ids:
        manifest_path = stage_dir / "shards" / f"shard-{shard_id:04d}.json"
        if resume and manifest_path.is_file():
            written.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        _clear_incomplete_shard(audio_dir, documents, shard_id, shard_size)
        rows = []
        for document in documents:
            gen_index = int(document["draws"]["gen_index"])
            if gen_index // shard_size != shard_id:
                continue
            waves = render_realization(document, {"speech": clean_dir, "noise": noise_dir})
            for stream, wave in waves.items():
                rel = _audio_rel(stream, gen_index, shard_size)
                path = audio_dir / rel
                _write_wav(path, wave)
                rows.append(
                    {
                        "gen_index": gen_index,
                        "stream": stream,
                        "rel_path": rel,
                        "sha256": sha256_file(path),
                        "frames": int(wave.shape[0]),
                        "sample_rate": SAMPLE_RATE,
                        "size_bytes": path.stat().st_size,
                    }
                )
        manifest = {"build_id": build_id, "shard": shard_id, "files": rows}
        _atomic_text(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        written.append(manifest)
    verified = {
        "build_id": build_id,
        "shards": [item["shard"] for item in written],
        "files": sum(len(item["files"]) for item in written),
        "ready": True,
    }
    _atomic_text(stage_dir / "verified.json", json.dumps(verified, sort_keys=True, indent=2) + "\n")
    return {"audio_dir": audio_dir, "staging_dir": stage_dir, "verified": verified}


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


def _render_pieces(pieces: Sequence[dict[str, Any]], directory: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for piece in pieces:
        if "silence" in piece:
            chunks.append(np.zeros(int(piece["silence"]), dtype=np.float64))
            continue
        path = directory / str(piece["name"])
        audio, rate = sf.read(path, dtype="float64", always_2d=False)
        if int(rate) != SAMPLE_RATE:
            raise ReproductionError(f"{path} is {rate} Hz; MS-SNSD generation does not resample")
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        start = int(piece["start"])
        stop = start + int(piece["length"])
        if stop > audio.shape[0]:
            raise ReproductionError(f"{path} is shorter than the plan")
        chunks.append(audio[start:stop])
    if not chunks:
        raise ReproductionError("realization has no audio pieces")
    return np.concatenate(chunks)


def _audio_rel(stream: str, gen_index: int, shard_size: int) -> str:
    """Path under ``generator/build_id[:16]/``. The directory index is ``shard_size``."""
    if shard_size < 1:
        raise ReproductionError("shard_size must be positive")
    shard = gen_index // shard_size
    return f"{stream}/s{shard:04d}/fid{gen_index:07d}.wav"


def _write_wav(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    sf.write(
        temporary,
        np.asarray(waveform, dtype=np.float64),
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _refuse_conflicting_definition(stage_dir: Path, definition: dict[str, Any]) -> None:
    path = stage_dir / "build.json"
    if not path.is_file():
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != definition:
        raise ReproductionError(f"build {definition['build_id']} already has a different definition")


def _estimate_bytes(documents: list[dict[str, Any]]) -> int:
    frames = sum(int(item["output"]["length"]) for item in documents)
    return frames * 2 * 3


def _require_space(path: Path, estimate: int) -> None:
    target = path if path.exists() else path.parent
    free = shutil.disk_usage(target).free
    if free < int(estimate * 1.2):
        raise ReproductionError(f"not enough free space for {estimate} bytes under {target}")


def _clear_incomplete_shard(audio_dir: Path, documents: list[dict[str, Any]], shard_id: int, shard_size: int) -> None:
    for document in documents:
        gen_index = int(document["draws"]["gen_index"])
        if gen_index // shard_size != shard_id:
            continue
        for stream in ("clean", "noise", "noisy"):
            path = audio_dir / _audio_rel(stream, gen_index, shard_size)
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".part").unlink(missing_ok=True)


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


def verify_build(
    definition: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    audio_dir: Path,
    staging_dir: Path,
    clean_dir: Path,
    noise_dir: Path,
) -> dict[str, Any]:
    """Check completed shards. A hash mismatch is an error and is not rewritten."""
    by_index = {int(item["draws"]["gen_index"]): item for item in documents}
    shard_size = int(definition["shard_size"])
    shard_ids = sorted({int(item["draws"]["gen_index"]) // shard_size for item in documents})
    checked = 0
    clipped = 0
    peak = 0.0
    snr_errors = []
    for shard_id in shard_ids:
        manifest_path = staging_dir / "shards" / f"shard-{shard_id:04d}.json"
        if not manifest_path.is_file():
            raise ReproductionError(f"shard {shard_id} has no manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("build_id") != definition["build_id"] or int(manifest.get("shard", -1)) != shard_id:
            raise ReproductionError(f"shard {shard_id} manifest does not match the build")
        grouped: dict[int, dict[str, dict[str, Any]]] = {}
        for row in manifest["files"]:
            grouped.setdefault(int(row["gen_index"]), {})[str(row["stream"])] = row
        for gen_index, streams in sorted(grouped.items()):
            document = by_index.get(gen_index)
            if document is None:
                raise ReproductionError(f"manifest gen_index {gen_index} is not in the plan")
            if set(streams) != {"clean", "noise", "noisy"}:
                raise ReproductionError(f"gen_index {gen_index} is missing a stream")
            rendered = render_realization(document, {"speech": clean_dir, "noise": noise_dir})
            nominal = float(document["draws"]["nominal_snr_db"])
            measured = measured_snr_db(rendered["clean"], rendered["noise"])
            if abs(measured - nominal / 2.0) > 1e-4:
                snr_errors.append((gen_index, measured, nominal / 2.0))
            clean_level = _level_dbfs(rendered["clean"])
            if abs(clean_level - TARGET_DBFS) > 1e-4:
                raise ReproductionError(f"gen_index {gen_index} clean level is {clean_level} dBFS")
            if rendered["clean"].shape != rendered["noisy"].shape or rendered["noise"].shape != rendered["noisy"].shape:
                raise ReproductionError(f"gen_index {gen_index} streams are not time-aligned")
            residual = rendered["noisy"] - rendered["clean"] - rendered["noise"]
            if float(np.max(np.abs(residual))) > 1e-8:
                raise ReproductionError(f"gen_index {gen_index} is not clean + noise")
            for stream, wave in rendered.items():
                row = streams[stream]
                path = audio_dir / str(row["rel_path"])
                _check_written(path, row, wave, gen_index, stream)
                overflow = int(np.count_nonzero(np.abs(wave) >= 1.0))
                clipped += overflow
                peak = max(peak, float(np.max(np.abs(wave))))
                checked += 1
    if snr_errors:
        raise ReproductionError(f"measured SNR does not match half the nominal SNR: {snr_errors[:3]}")
    report = {
        "build_id": definition["build_id"],
        "status": "PASS",
        "shard_ids": [f"{shard_id:04d}" for shard_id in shard_ids],
        "files": checked,
        "clipped_samples": clipped,
        "peak_abs": peak,
        "snr": "half_nominal",
        "level_dbfs": TARGET_DBFS,
        "pcm16": True,
    }
    _atomic_text(staging_dir / "verify.json", json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _level_dbfs(wave: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(wave, dtype=np.float64) ** 2)))
    if rms <= 0.0:
        raise ReproductionError("level is undefined for a silent waveform")
    return 20.0 * float(np.log10(rms))


def _pcm16_roundtrip(wave: np.ndarray) -> np.ndarray:
    """Match libsndfile's float PCM_16 read-back.

    Positive peaks stop at 32767/32768. Codes are ``floor(clipped * 32768)``
    and the file reads back as ``code / 32768``.
    """
    limit = 32767.0 / 32768.0
    codes = np.floor(np.clip(np.asarray(wave, dtype=np.float64), -1.0, limit) * 32768.0)
    return codes / 32768.0


def _check_written(path: Path, row: dict[str, Any], wave: np.ndarray, gen_index: int, stream: str) -> None:
    if path.name.endswith(".part") or not path.is_file():
        raise ReproductionError(f"{stream} gen_index {gen_index} is missing or still a partial file")
    info = sf.info(path)
    if int(info.samplerate) != SAMPLE_RATE or int(info.channels) != 1 or info.subtype != "PCM_16":
        raise ReproductionError(f"{path} is not 16 kHz mono PCM_16")
    if int(info.frames) != int(wave.shape[0]) or int(info.frames) != int(row["frames"]):
        raise ReproductionError(f"{path} frame count does not match the plan")
    if path.stat().st_size != int(row["size_bytes"]):
        raise ReproductionError(f"{path} size does not match the manifest")
    digest_value = sha256_file(path)
    if digest_value != row["sha256"]:
        raise ReproductionError(f"{path} sha256 does not match the manifest")
    audio, rate = sf.read(path, dtype="float64", always_2d=False)
    if int(rate) != SAMPLE_RATE or audio.shape != wave.shape:
        raise ReproductionError(f"{path} decoded shape does not match the plan")
    expected = _pcm16_roundtrip(wave)
    # Float rounding can move one sample by a single PCM code.
    if float(np.max(np.abs(audio - expected))) > (1.0 / 32768.0) + 1e-9:
        raise ReproductionError(f"{path} is not the PCM_16 rendering of the planned mix")
