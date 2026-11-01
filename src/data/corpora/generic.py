"""Header scan for corpora that do not have a dedicated parser.

Paths are sorted before reading so the set of rows does not depend on
directory order. Unknown speaker, transcript, and split fields stay empty.
Unreadable files are counted and omitted rather than filled with placeholder rates.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from src.data.audio import read_audio_stream_info
from src.data.constants import CORPORA_ROOT
from src.data.corpora.parallel_scan import iter_in_threads
from src.data.corpora.paths import common_path_fields, is_audio_file
from src.data.models.audio_file import AudioFileRow

SKIP_DIRS = {"_archives", "__MACOSX", ".git"}
FAILURE_LOG = Path("data/staging/ingest/unreadable.jsonl")
_failures: list[dict[str, str]] = []
_failure_lock = threading.Lock()


def iter_generic(corpus: str) -> Iterator[AudioFileRow]:
    root = CORPORA_ROOT / corpus
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = _list_audio(root)
    chunk = 256
    groups = [paths[index : index + chunk] for index in range(0, len(paths), chunk)]
    return iter_in_threads(groups, lambda group: _rows(corpus, root, group))


def flush_failures() -> int:
    if not _failures:
        return 0
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILURE_LOG.open("a", encoding="utf-8") as handle:
        for item in _failures:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    count = len(_failures)
    _failures.clear()
    return count


def _list_audio(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS)
        for name in filenames:
            if is_audio_file(name):
                found.append(Path(dirpath) / name)
    found.sort()
    return found


def _rows(corpus: str, root: Path, paths: list[Path]) -> list[AudioFileRow]:
    rows: list[AudioFileRow] = []
    for audio_path in paths:
        try:
            sample_rate, frames, channels = read_audio_stream_info(audio_path)
        except (OSError, ValueError) as exc:
            with _failure_lock:
                _failures.append({"path": audio_path.as_posix(), "error": str(exc)})
            continue
        relative = audio_path.relative_to(root)
        subset = relative.parts[0] if len(relative.parts) > 1 else None
        path, corpus_name, subset_name, rel_dir, file_name, extension = common_path_fields(
            audio_path, corpus, subset
        )
        speaker_id = None
        utterance_id = None
        transcript = None
        if corpus == "VoiceBank-DEMAND":
            speaker_id, utterance_id, transcript = _voicebank(audio_path, root)
        rows.append(
            AudioFileRow(
                path=path,
                corpus=corpus_name,
                subset=subset_name,
                relative_dir=rel_dir,
                file_name=file_name,
                extension=extension,
                sample_rate=sample_rate,
                frames=frames,
                channels=channels,
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                transcript=transcript,
            )
        )
    return rows


def _voicebank(audio_path: Path, root: Path) -> tuple[str | None, str | None, str | None]:
    stem = audio_path.stem
    speaker = stem.split("_", 1)[0] if "_" in stem else None
    text_dir = {
        "clean_testset_wav": "testset_txt",
        "noisy_testset_wav": "testset_txt",
        "clean_trainset_28spk_wav": "trainset_28spk_txt",
        "noisy_trainset_28spk_wav": "trainset_28spk_txt",
        "clean_trainset_56spk_wav": "trainset_56spk_txt",
        "noisy_trainset_56spk_wav": "trainset_56spk_txt",
    }.get(audio_path.parent.name)
    transcript = None
    if text_dir is not None:
        text_path = root / text_dir / f"{stem}.txt"
        if text_path.is_file():
            transcript = text_path.read_text(encoding="utf-8", errors="replace").strip() or None
    return speaker, stem, transcript
