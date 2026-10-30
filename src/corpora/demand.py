import os
from collections.abc import Iterator
from pathlib import Path

from src.audio import read_audio_stream_info
from src.constants import CORPORA_ROOT, SCENE_CLASS_BY_PREFIX
from src.corpora.parallel_scan import iter_in_threads
from src.corpora.paths import common_path_fields, is_audio_file
from src.models.audio_file import AudioFileRow


def iter_rows() -> Iterator[AudioFileRow]:
    root = CORPORA_ROOT / "DEMAND"
    scene_dirs: list[Path] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir():
                scene_dirs.append(Path(entry.path))
    return iter_in_threads(scene_dirs, _rows_for_scene_dir)


def _rows_for_scene_dir(scene_dir: Path) -> list[AudioFileRow]:
    scene_code = scene_dir.name
    scene_class = SCENE_CLASS_BY_PREFIX.get(scene_code[0])
    rows: list[AudioFileRow] = []
    with os.scandir(scene_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not is_audio_file(entry.name):
                continue
            audio_path = Path(entry.path)
            channel_label = audio_path.stem
            path, corpus, subset, rel_dir, file_name, extension = common_path_fields(
                audio_path, "DEMAND", scene_code
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
                    scene_code=scene_code,
                    scene_class=scene_class,
                    channel_label=channel_label,
                )
            )
    return rows
