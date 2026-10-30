from dataclasses import dataclass
from typing import Any


AUDIO_FILE_COLUMNS: tuple[str, ...] = (
    "path",
    "corpus",
    "subset",
    "relative_dir",
    "file_name",
    "extension",
    "sample_rate",
    "frames",
    "channels",
    "speaker_id",
    "chapter_id",
    "utterance_id",
    "segment_start_id",
    "segment_end_id",
    "speaker_name",
    "speaker_gender",
    "chapter_title",
    "book_title",
    "chapter_duration_min",
    "transcript_path",
    "utterance_index",
    "speaker_age",
    "speaker_accent",
    "speaker_region",
    "transcript_txt_path",
    "transcript",
    "track_type",
    "condition_label",
    "eval_split",
    "noise_type",
    "reverberation",
    "chime_speaker_label",
    "wsj_utterance_id",
    "mic_channel",
    "scene_code",
    "scene_class",
    "channel_label",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioFileRow:
    path: str
    corpus: str
    subset: str | None = None
    relative_dir: str | None = None
    file_name: str = ""
    extension: str | None = None
    sample_rate: int
    frames: int
    channels: int
    speaker_id: str | None = None
    chapter_id: str | None = None
    utterance_id: str | None = None
    segment_start_id: str | None = None
    segment_end_id: str | None = None
    speaker_name: str | None = None
    speaker_gender: str | None = None
    chapter_title: str | None = None
    book_title: str | None = None
    chapter_duration_min: float | None = None
    transcript_path: str | None = None
    utterance_index: str | None = None
    speaker_age: int | None = None
    speaker_accent: str | None = None
    speaker_region: str | None = None
    transcript_txt_path: str | None = None
    transcript: str | None = None
    track_type: str | None = None
    condition_label: str | None = None
    eval_split: str | None = None
    noise_type: str | None = None
    reverberation: str | None = None
    chime_speaker_label: str | None = None
    wsj_utterance_id: str | None = None
    mic_channel: str | None = None
    scene_code: str | None = None
    scene_class: str | None = None
    channel_label: str | None = None

    def values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in AUDIO_FILE_COLUMNS)
