from collections.abc import Callable, Iterator

from src.models.audio_file import AudioFileRow

from . import chime3, demand, librispeech, libritts, vctk

AUDIO_FILE_ROW_ITERATORS: dict[str, Callable[[], Iterator[AudioFileRow]]] = {
    "LibriSpeech": librispeech.iter_rows,
    "LibriTTS": libritts.iter_rows,
    "VCTK": vctk.iter_rows,
    "CHiME3": chime3.iter_rows,
    "DEMAND": demand.iter_rows,
}
