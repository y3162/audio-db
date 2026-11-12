import os
from pathlib import Path

from src.data.constants import AUDIO_EXTENSIONS, CORPORA_ROOT

_CORPUS_PREFIX = f"{CORPORA_ROOT.as_posix().rstrip('/')}/"


def _posix(path: Path) -> str:
    text = os.fspath(path)
    if "\\" in text:
        return text.replace("\\", "/")
    return text


def _extension(name: str) -> str | None:
    dot = name.rfind(".")
    if dot <= 0:
        return None
    return name[dot + 1 :] or None


def rel_corpus_path(path: Path) -> str:
    text = _posix(path)
    if text.startswith(_CORPUS_PREFIX):
        return text[len(_CORPUS_PREFIX) :]
    return path.relative_to(CORPORA_ROOT).as_posix()


def subset_under(path: Path, root: Path) -> str | None:
    """First directory under ``root``, or None when the file sits directly in it."""
    text = _posix(path)
    root_text = _posix(root).rstrip("/") + "/"
    if text.startswith(root_text):
        rest = text[len(root_text) :]
        slash = rest.find("/")
        if slash == -1:
            return None
        return rest[:slash]
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else None


def common_path_fields(
    path: Path, corpus: str, subset: str | None
) -> tuple[str, str, str | None, str | None, str, str | None]:
    rel = rel_corpus_path(path)
    slash = rel.rfind("/")
    if slash == -1:
        rel_dir = path.parent.relative_to(CORPORA_ROOT).as_posix()
        name = path.name
    else:
        rel_dir = rel[:slash]
        name = rel[slash + 1 :]
    return rel, corpus, subset, rel_dir, name, _extension(name)


def is_audio_file(name: str) -> bool:
    return Path(name).suffix.lower() in AUDIO_EXTENSIONS
