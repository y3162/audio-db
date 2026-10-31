"""NeMo Parakeet. TDT, CTC, and RNN-T each keep the checkpoint's decoder."""

from __future__ import annotations

from typing import Any

import torch

from src.asr.core.recognizer import SpeechRecognizer
from src.asr.catalog import ModelCard


def load_parakeet(card: ModelCard, *, device: str, language: str | None) -> SpeechRecognizer:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=card.model_id)
    _reject_decoder_mismatch(model, card)
    _disable_cuda_graph(model)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def run(audio: torch.Tensor) -> str:
        model.eval()
        array = audio.numpy()
        with torch.inference_mode():
            try:
                output = model.transcribe([array], batch_size=1, verbose=False)
            except TypeError:
                output = model.transcribe([array], batch_size=1)
        return _hypothesis_text(output)

    return SpeechRecognizer(card, device=device, language=language, runner=run, sample_rate=16000)


def _reject_decoder_mismatch(model: Any, card: ModelCard) -> None:
    observed = _observed_decode(model)
    if observed is None or observed == card.decode:
        return
    raise RuntimeError(
        f"{card.model_id} loaded as {observed}, but the catalog decode is {card.decode}. "
        "The checkpoint was not decoded with a different family's algorithm."
    )


def _observed_decode(model: Any) -> str | None:
    class_name = type(model).__name__.lower()
    loss_name = ""
    config = getattr(model, "cfg", None)
    loss = getattr(config, "loss", None)
    if loss is not None:
        target = getattr(loss, "_target_", None)
        loss_name = str(target if target is not None else loss).lower()
    blob = f"{class_name} {loss_name}"
    if "tdt" in blob:
        return "tdt"
    if "ctc" in class_name or "ctc" in loss_name:
        return "ctc"
    if "rnnt" in blob or "rnn_t" in blob or "transducer" in blob:
        return "rnnt"
    return None


def _disable_cuda_graph(model: Any) -> None:
    """Keep the checkpoint strategy. Only turn off CUDA-graph decoding when it is present."""
    try:
        config = getattr(model, "cfg", None)
        decoding = getattr(config, "decoding", None)
        if decoding is None or "greedy" not in decoding or "use_cuda_graph_decoder" not in decoding["greedy"]:
            return
        from omegaconf import open_dict

        with open_dict(decoding):
            decoding.greedy.use_cuda_graph_decoder = False
        change = getattr(model, "change_decoding_strategy", None)
        if change is not None:
            change(decoding)
    except Exception:
        return


def _hypothesis_text(output: Any) -> str:
    value = output
    if isinstance(value, tuple):
        if not value:
            return ""
        value = value[0]
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if isinstance(value, str):
        return value.strip()
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text.strip()
    raise TypeError(f"NeMo transcribe returned {type(output).__name__}, which has no text")
