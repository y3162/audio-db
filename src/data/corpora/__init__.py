from collections.abc import Callable, Iterator

from src.data.models.audio_file import AudioFileRow

from . import chime3, demand, generic, librispeech, libritts, vctk

def _generic(name: str) -> Callable[[], Iterator[AudioFileRow]]:
    return lambda: generic.iter_generic(name)


AUDIO_FILE_ROW_ITERATORS: dict[str, Callable[[], Iterator[AudioFileRow]]] = {
    "LibriSpeech": librispeech.iter_rows,
    "LibriTTS": libritts.iter_rows,
    "VCTK": vctk.iter_rows,
    "CHiME3": chime3.iter_rows,
    "DEMAND": demand.iter_rows,
    "DNS5": _generic("DNS5"),
    "DNS2020": _generic("DNS2020"),
    "MS-SNSD": _generic("MS-SNSD"),
    "OpenSLR26": _generic("OpenSLR26"),
    "OpenSLR28": _generic("OpenSLR28"),
    "URGENT2024": _generic("URGENT2024"),
    "VoiceBank-DEMAND": _generic("VoiceBank-DEMAND"),
    "WHAM48": _generic("WHAM48"),
    "musan": _generic("musan"),
}
