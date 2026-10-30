import os
from collections.abc import Iterator
from pathlib import Path

from src.data.audio import read_audio_stream_info
from src.data.constants import CORPORA_ROOT
from src.data.corpora.parallel_scan import iter_in_threads
from src.data.corpora.paths import common_path_fields, is_audio_file
from src.data.models.audio_file import AudioFileRow

CHIME3_AUDIO_ROOT = Path("data") / "audio" / "16kHz"


def iter_rows() -> Iterator[AudioFileRow]:
    root = CORPORA_ROOT / "CHiME3" / CHIME3_AUDIO_ROOT
    condition_dirs: list[tuple[str, Path]] = []
    with os.scandir(root) as track_types:
        for track_entry in track_types:
            if not track_entry.is_dir():
                continue
            track_type = track_entry.name
            with os.scandir(track_entry.path) as conditions:
                for condition in conditions:
                    if condition.is_dir():
                        condition_dirs.append((track_type, Path(condition.path)))
    return iter_in_threads(condition_dirs, _rows_for_condition_dir)


def _rows_for_condition_dir(args: tuple[str, Path]) -> list[AudioFileRow]:
    track_type, condition_dir = args
    condition_label = condition_dir.name
    eval_split, noise_type, reverberation = _split_condition_label(condition_label)
    rows: list[AudioFileRow] = []
    with os.scandir(condition_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not is_audio_file(entry.name):
                continue
            audio_path = Path(entry.path)
            chime_speaker_label, wsj_utterance_id, mic_channel = _parse_chime_stem(audio_path.stem)
            path, corpus, subset, rel_dir, file_name, extension = common_path_fields(
                audio_path, "CHiME3", track_type
            )
            sample_rate, frames, channels = read_audio_stream_info(audio_path)
            rows.append(
                AudioFileRow(
                    path=path,
                    corpus=corpus,
                    subset=subset,
                    relative_dir=rel_dir,
                    file_name=file_name,
                    extension=extension,
                    sample_rate=sample_rate,
                    frames=frames,
                    channels=channels,
                    track_type=track_type,
                    condition_label=condition_label,
                    eval_split=eval_split,
                    noise_type=noise_type,
                    reverberation=reverberation,
                    chime_speaker_label=chime_speaker_label,
                    wsj_utterance_id=wsj_utterance_id,
                    mic_channel=mic_channel,
                )
            )
    return rows


def _split_condition_label(condition_label: str) -> tuple[str | None, str | None, str | None]:
    parts = condition_label.split("_")
    if len(parts) < 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def _parse_chime_stem(stem: str) -> tuple[str | None, str | None, str | None]:
    mic_channel = None
    if ".CH" in stem:
        stem, _, mic_channel = stem.partition(".CH")
        mic_channel = f"CH{mic_channel}" if mic_channel else None
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None, mic_channel
    if parts[0] == "BGD":
        return "BGD", "_".join(parts[1:-1]) if len(parts) > 2 else parts[1], mic_channel
    chime_speaker_label = parts[0]
    wsj_utterance_id = parts[1] if len(parts) > 1 else None
    return chime_speaker_label, wsj_utterance_id, mic_channel
