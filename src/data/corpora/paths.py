from pathlib import Path

from src.data.constants import AUDIO_EXTENSIONS, CORPORA_ROOT


def rel_corpus_path(path: Path) -> str:
    return path.relative_to(CORPORA_ROOT).as_posix()


def common_path_fields(
    path: Path, corpus: str, subset: str | None
) -> tuple[str, str, str | None, str | None, str, str | None]:
    rel = rel_corpus_path(path)
    rel_dir = path.parent.relative_to(CORPORA_ROOT).as_posix()
    extension = path.suffix[1:] if path.suffix else None
    return rel, corpus, subset, rel_dir, path.name, extension


def is_audio_file(name: str) -> bool:
    return Path(name).suffix.lower() in AUDIO_EXTENSIONS
