import os
from collections.abc import Iterator
from pathlib import Path

from src.constants import CORPORA_ROOT
from src.corpora.libri_pipe_metadata import ChapterMeta, SpeakerMeta, load_chapters, load_speakers
from src.corpora.parallel_scan import iter_in_threads, iter_libri_chapter_dirs
from src.corpora.paths import common_path_fields, is_audio_file, rel_corpus_path
from src.corpora.transcripts import load_libritts_chapter_transcripts
from src.models.audio_file import AudioFileRow


def iter_rows() -> Iterator[AudioFileRow]:
    root = CORPORA_ROOT / "LibriTTS"
    speakers = load_speakers(root / "SPEAKERS.txt")
    chapters = load_chapters(root / "CHAPTERS.txt")
    return iter_in_threads(
        iter_libri_chapter_dirs(root),
        lambda chapter_dir: _rows_for_chapter(chapter_dir, speakers, chapters),
    )


def _rows_for_chapter(
    chapter_dir: Path,
    speakers: dict[str, SpeakerMeta],
    chapters: dict[str, ChapterMeta],
) -> list[AudioFileRow]:
    speaker_dir = chapter_dir.parent
    subset_name = speaker_dir.parent.name
    speaker_id = speaker_dir.name
    chapter_id = chapter_dir.name
    speaker = speakers.get(speaker_id)
    chapter = chapters.get(chapter_id)
    transcript_file = chapter_dir / f"{speaker_id}_{chapter_id}.trans.tsv"
    transcript_path = rel_corpus_path(transcript_file)
    transcripts = load_libritts_chapter_transcripts(transcript_file) if transcript_file.is_file() else {}
    prefix = f"{speaker_id}_{chapter_id}_"

    rows: list[AudioFileRow] = []
    with os.scandir(chapter_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not is_audio_file(entry.name):
                continue
            audio_path = Path(entry.path)
            stem = audio_path.stem
            if not stem.startswith(prefix):
                continue
            tail = stem[len(prefix) :]
            segment_start_id, _, segment_end_id = tail.partition("_")
            if not segment_end_id:
                continue
            path, corpus, subset, rel_dir, file_name, extension = common_path_fields(
                audio_path, "LibriTTS", subset_name
            )
            rows.append(
                AudioFileRow(
                    path=path,
                    corpus=corpus,
                    subset=subset,
                    relative_dir=rel_dir,
                    file_name=file_name,
                    extension=extension,
                    speaker_id=speaker_id,
                    chapter_id=chapter_id,
                    utterance_id=tail.replace("_", "-"),
                    segment_start_id=segment_start_id,
                    segment_end_id=segment_end_id,
                    speaker_name=speaker.name if speaker else None,
                    speaker_gender=speaker.gender if speaker else None,
                    chapter_title=chapter.title if chapter else None,
                    book_title=chapter.book_title if chapter else None,
                    chapter_duration_min=chapter.duration_min if chapter else None,
                    transcript_path=transcript_path,
                    transcript=transcripts.get(stem),
                )
            )
    return rows
