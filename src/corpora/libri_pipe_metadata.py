from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SpeakerMeta:
    gender: str
    name: str


@dataclass(frozen=True, slots=True)
class ChapterMeta:
    duration_min: float
    title: str
    book_title: str


def load_pipe_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            rows.append([part.strip() for part in line.split("|")])
    return rows


def load_speakers(path: Path) -> dict[str, SpeakerMeta]:
    speakers: dict[str, SpeakerMeta] = {}
    for parts in load_pipe_rows(path):
        if len(parts) < 5:
            continue
        speakers[parts[0]] = SpeakerMeta(gender=parts[1], name=parts[4])
    return speakers


def load_chapters(path: Path) -> dict[str, ChapterMeta]:
    chapters: dict[str, ChapterMeta] = {}
    for parts in load_pipe_rows(path):
        if len(parts) < 8:
            continue
        chapters[parts[0]] = ChapterMeta(
            duration_min=float(parts[2]),
            title=parts[6],
            book_title=parts[7],
        )
    return chapters
