"""Hugging Face CTC checkpoints for Wav2Vec2, HuBERT, and WavLM."""

from __future__ import annotations

import torch

from src.asr.core.recognizer import SpeechRecognizer
from src.asr.catalog import ModelCard

_CTC_TYPES = {"wav2vec2", "hubert", "wavlm"}


def load_ctc(card: ModelCard, *, device: str, language: str | None) -> SpeechRecognizer:
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(card.model_id)
    model = AutoModelForCTC.from_pretrained(card.model_id)
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type not in _CTC_TYPES:
        raise RuntimeError(
            f"{card.model_id} loaded model_type {model_type!r}, which is not a Wav2Vec2, HuBERT, or WavLM CTC model"
        )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def run(audio: torch.Tensor) -> str:
        features = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
        features = {key: value.to(device) for key, value in features.items()}
        model.eval()
        with torch.inference_mode():
            logits = model(**features).logits
        predicted = torch.argmax(logits, dim=-1)
        texts = processor.batch_decode(predicted)
        if not texts or not isinstance(texts[0], str):
            raise TypeError("CTC decode did not return text")
        return texts[0].strip()

    return SpeechRecognizer(card, device=device, language=language or "en", runner=run, sample_rate=16000)
