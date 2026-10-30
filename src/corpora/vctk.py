import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.audio import read_audio_stream_info
from src.constants import CORPORA_ROOT
from src.corpora.parallel_scan import iter_in_threads
from src.corpora.paths import common_path_fields, is_audio_file, rel_corpus_path
from src.corpora.transcripts import read_transcript_file
from src.models.audio_file import AudioFileRow


@dataclass(frozen=True, slots=True)
class VctkSpeakerMeta:
    age: int
    gender: str
    accent: str
    region: str


def iter_rows() -> Iterator[AudioFileRow]:
    root = CORPORA_ROOT / "VCTK"
    speakers = _load_speaker_info(root / "speaker-info.txt")
    speaker_dirs: list[Path] = []
    with os.scandir(root / "wav48") as entries:
        for entry in entries:
            if entry.is_dir():
                speaker_dirs.append(Path(entry.path))
    return iter_in_threads(speaker_dirs, lambda speaker_dir: _rows_for_speaker_dir(speaker_dir, speakers))


def _load_speaker_info(path: Path) -> dict[str, VctkSpeakerMeta]:
    speakers: dict[str, VctkSpeakerMeta] = {}
    with open(path, encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.split()
            if len(parts) < 6:
                continue
            speaker_id, age, gender = parts[0], int(parts[1]), parts[2]
            accent = parts[3]
            region = " ".join(parts[4:])
            speakers[speaker_id] = VctkSpeakerMeta(
                age=age,
                gender=gender,
                accent=accent,
                region=region,
            )
    return speakers


def _rows_for_speaker_dir(
    speaker_dir: Path,
    speakers: dict[str, VctkSpeakerMeta],
) -> list[AudioFileRow]:
    speaker_id = speaker_dir.name
    speaker = speakers.get(speaker_id)
    rows: list[AudioFileRow] = []
    prefix = f"{speaker_id}_"
    with os.scandir(speaker_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not is_audio_file(entry.name):
                continue
            audio_path = Path(entry.path)
            stem = audio_path.stem
            if not stem.startswith(prefix):
                continue
            utterance_index = stem[len(prefix) :]
            transcript_file = speaker_dir.parent.parent / "txt" / speaker_id / f"{stem}.txt"
            transcript_txt_path = rel_corpus_path(transcript_file)
            transcript = read_transcript_file(transcript_file) if transcript_file.is_file() else None
            path, corpus, subset, rel_dir, file_name, extension = common_path_fields(
                audio_path, "VCTK", "wav48"
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
                    speaker_id=speaker_id,
                    utterance_id=utterance_index,
                    utterance_index=utterance_index,
                    speaker_name=None,
                    speaker_gender=speaker.gender if speaker else None,
                    speaker_age=speaker.age if speaker else None,
                    speaker_accent=speaker.accent if speaker else None,
                    speaker_region=speaker.region if speaker else None,
                    transcript_txt_path=transcript_txt_path,
                    transcript=transcript,
                )
            )
    return rows
