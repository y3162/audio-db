"""Recognition result. Text is the decoder output before WER normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ASRResult:
    text: str
    model: str
    language: str | None = None
    model_id: str | None = None
    decode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
