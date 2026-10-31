"""Speech recognition models.

``src.asr`` does not import ``src.evaluation`` or ``src.se``.
"""

from src.asr.api import SpeechRecognizer, available_families, load_recognizer, model_catalog
from src.asr.core.result import ASRResult

__all__ = [
    "ASRResult",
    "SpeechRecognizer",
    "available_families",
    "load_recognizer",
    "model_catalog",
]
