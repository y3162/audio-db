"""OpenAI Whisper. The task is always transcribe, including for multilingual checkpoints."""

from __future__ import annotations

import torch

from src.asr.core.recognizer import SpeechRecognizer
from src.asr.catalog import ModelCard


def load_whisper(card: ModelCard, *, device: str, language: str | None) -> SpeechRecognizer:
    import whisper

    chosen = language or "en"
    if card.languages and chosen not in card.languages:
        raise ValueError(f"{card.model_id} is English-only, but language is {chosen!r}")
    model = whisper.load_model(card.model_id, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    use_fp16 = str(device).startswith("cuda")

    def run(audio: torch.Tensor) -> str:
        model.eval()
        with torch.inference_mode():
            result = model.transcribe(
                audio.numpy(),
                task="transcribe",
                language=chosen,
                temperature=0.0,
                condition_on_previous_text=False,
                fp16=use_fp16,
                verbose=None,
            )
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str):
            raise TypeError("Whisper transcribe did not return text")
        return text.strip()

    return SpeechRecognizer(card, device=device, language=chosen, runner=run, sample_rate=16000)
