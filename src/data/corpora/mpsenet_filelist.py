"""MP-SENet's VoiceBank-DEMAND file list.

The official loader keeps the utterance id before ``|`` and reads
``<clean_dir>/<id>.wav`` and ``<noisy_dir>/<id>.wav``. The path after ``|``
is stored as metadata and is not used as the audio location. The list has no
transcript and no SNR.
"""

from __future__ import annotations

from pathlib import Path

from src.data.audio import read_wav_stream_info
from src.evaluation.sample import AudioSample


def load_mpsenet_filelist(
    path: str | Path,
    *,
    clean_dir: str | Path,
    noisy_dir: str | Path,
    split: str = "test",
) -> list[AudioSample]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"MP-SENet file list not found: {file_path}")
    clean_root = Path(clean_dir)
    noisy_root = Path(noisy_dir)
    rows = _rows(file_path)
    missing: list[str] = []
    samples: list[AudioSample] = []
    for utterance_id, listed in rows:
        clean = clean_root / f"{utterance_id}.wav"
        noisy = noisy_root / f"{utterance_id}.wav"
        if not clean.is_file() or not noisy.is_file():
            missing.append(utterance_id)
            continue
        clean_rate, _clean_frames, clean_channels = read_wav_stream_info(clean)
        noisy_rate, _noisy_frames, noisy_channels = read_wav_stream_info(noisy)
        if clean_channels != 1 or noisy_channels != 1:
            raise ValueError(f"{utterance_id} is not mono")
        if clean_rate != noisy_rate:
            raise ValueError(f"{utterance_id} clean and noisy sample rates differ")
        speaker = utterance_id.split("_", 1)[0]
        transcript, transcript_note = _paired_transcript(clean_root, utterance_id)
        samples.append(
            AudioSample(
                utterance_id=f"voicebank:{split}:{utterance_id}",
                dataset="VoiceBank-DEMAND",
                split=split,
                sample_rate=clean_rate,
                speaker_id=speaker,
                transcript=transcript,
                clean_path=clean.as_posix(),
                noisy_path=noisy.as_posix(),
                metadata={
                    "source": "mp_senet_filelist",
                    "file_list": file_path.as_posix(),
                    "listed_path": listed,
                    "utterance": utterance_id,
                    "transcript_note": transcript_note,
                },
            )
        )
    if missing:
        raise ValueError(
            f"VoiceBank-DEMAND list is incomplete: {len(missing)} of {len(rows)} pairs are missing, "
            f"for example {missing[0]}. No rows were registered."
        )
    return samples


def _paired_transcript(clean_root: Path, utterance_id: str) -> tuple[str | None, str]:
    """Read the official text file. The MP-SENet list itself has no transcript."""
    folder = {
        "clean_testset_wav": "testset_txt",
        "clean_trainset_28spk_wav": "trainset_28spk_txt",
        "clean_trainset_56spk_wav": "trainset_56spk_txt",
    }.get(clean_root.name)
    if folder is None:
        return None, f"no text directory mapped from {clean_root.name}"
    text_path = clean_root.parent / folder / f"{utterance_id}.txt"
    if not text_path.is_file():
        return None, f"missing {text_path.name}"
    text = text_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None, "empty text file"
    return text, "official text file"


def _rows(path: Path) -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        utterance_id, separator, listed = line.partition("|")
        utterance_id = utterance_id.strip()
        if not utterance_id or "/" in utterance_id or "\\" in utterance_id:
            raise ValueError(f"{path}:{lineno} has no utterance id")
        if utterance_id in seen:
            raise ValueError(f"{path}:{lineno} repeats {utterance_id}")
        seen.add(utterance_id)
        rows.append((utterance_id, listed.strip() if separator else None))
    if not rows:
        raise ValueError(f"MP-SENet file list is empty: {path}")
    return rows
