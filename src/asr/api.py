"""Construct a recognizer and transcribe one mono waveform."""

from __future__ import annotations

from src.asr.catalog import FAMILIES, ModelCard, iter_cards, resolve_model
from src.asr.core.recognizer import SpeechRecognizer


def load_recognizer(
    name: str,
    *,
    model_id: str,
    device: str = "cpu",
    language: str | None = None,
    decode: str | None = None,
) -> SpeechRecognizer:
    """Load one published checkpoint.

    Other families are not imported or downloaded. ``device`` is a torch device
    string such as ``cpu`` or ``cuda:1``.
    """
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    card = resolve_model(name, model_id, decode=decode, language=language)
    if card.backend == "nemo":
        from src.asr.models.parakeet import load_parakeet

        return load_parakeet(card, device=device, language=language)
    if card.backend == "whisper":
        from src.asr.models.whisper import load_whisper

        return load_whisper(card, device=device, language=language)
    if card.backend == "transformers":
        from src.asr.models.hf_ctc import load_ctc

        return load_ctc(card, device=device, language=language)
    raise ValueError(f"unknown ASR backend {card.backend!r}")


def available_families() -> tuple[str, ...]:
    return FAMILIES


def model_catalog(family: str | None = None) -> tuple[ModelCard, ...]:
    return iter_cards(family)
