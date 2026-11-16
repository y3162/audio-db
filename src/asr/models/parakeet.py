"""NeMo Parakeet. TDT, CTC, and RNN-T each keep the checkpoint's decoder.

``nvidia/parakeet-tdt-0.6b-v2`` is the exception: its decode is the fixed TDT
v2 strategy below. Applying it is checked after ``change_decoding_strategy``.
A mismatch or a failed change raises. Other Parakeet checkpoints keep the
previous behavior, which only turns CUDA-graph decoding off when that key is
already present.
"""

from __future__ import annotations

from typing import Any

import torch

from src.asr.core.recognizer import SpeechRecognizer
from src.asr.catalog import ModelCard

TDT_V2_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
TDT_V2_DECODING = {
    "strategy": "greedy_batch",
    "compute_timestamps": False,
    "tdt_include_token_duration": True,
    "use_cuda_graph_decoder": False,
}


def load_parakeet(card: ModelCard, *, device: str, language: str | None) -> SpeechRecognizer:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=card.model_id)
    _reject_decoder_mismatch(model, card)
    extra: dict[str, Any] = {}
    decoding = configure_parakeet_decoding(model, card.model_id)
    if decoding is not None:
        extra["decoding"] = decoding
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

    return SpeechRecognizer(
        card, device=device, language=language, runner=run, sample_rate=16000, extra=extra
    )


def configure_parakeet_decoding(model: Any, model_id: str) -> dict[str, Any] | None:
    """Apply the fixed TDT v2 decode only for that checkpoint."""

    if model_id == TDT_V2_MODEL_ID:
        return apply_tdt_v2_decoding(model)
    _disable_cuda_graph(model)
    return None


def apply_tdt_v2_decoding(model: Any) -> dict[str, Any]:
    """Set the fixed TDT v2 decode and return the effective settings.

    ``change_decoding_strategy`` is required. A raised error is not replaced
    with the checkpoint default, and the effective settings must match
    ``TDT_V2_DECODING`` afterwards. The live decoder flag is checked when the
    model exposes one, because a config write that the decoder ignores is not
    an applied strategy.
    """

    decoding = getattr(getattr(model, "cfg", None), "decoding", None)
    if decoding is None:
        raise RuntimeError("Parakeet TDT v2 decoding config is missing")
    _write_tdt_v2_decoding(decoding)
    change = getattr(model, "change_decoding_strategy", None)
    if not callable(change):
        raise RuntimeError("Parakeet TDT v2 model has no change_decoding_strategy")
    try:
        try:
            change(decoding, verbose=False)
        except TypeError as exc:
            if "verbose" not in str(exc):
                raise
            change(decoding)
    except Exception as exc:
        raise RuntimeError("Parakeet TDT v2 decoding could not be applied") from exc
    effective = effective_tdt_v2_decoding(model)
    if effective != TDT_V2_DECODING:
        raise RuntimeError(f"Parakeet TDT v2 effective decoding differs: {effective}")
    live = _live_cuda_graph(model)
    if live is not None and live is not False:
        raise RuntimeError(f"Parakeet TDT v2 CUDA-graph decoder is still enabled: {live}")
    return dict(effective)


def effective_tdt_v2_decoding(model: Any) -> dict[str, Any]:
    """Read the four TDT v2 fields from the model's decoding config."""

    decoding = model.cfg.decoding
    greedy = _child(decoding, "greedy")
    return {
        "strategy": str(_child(decoding, "strategy")),
        "compute_timestamps": bool(_child(decoding, "compute_timestamps")),
        "tdt_include_token_duration": bool(_child(decoding, "tdt_include_token_duration")),
        "use_cuda_graph_decoder": bool(_child(greedy, "use_cuda_graph_decoder")),
    }


def _write_tdt_v2_decoding(decoding: Any) -> None:
    try:
        from omegaconf import DictConfig, OmegaConf, open_dict
    except ImportError:
        DictConfig = ()  # type: ignore[assignment]
        OmegaConf = None
        open_dict = None
    if open_dict is not None and isinstance(decoding, DictConfig):
        with open_dict(decoding):
            decoding.strategy = TDT_V2_DECODING["strategy"]
            decoding.compute_timestamps = TDT_V2_DECODING["compute_timestamps"]
            decoding.tdt_include_token_duration = TDT_V2_DECODING["tdt_include_token_duration"]
            if decoding.greedy is None:
                decoding.greedy = OmegaConf.create({})
            with open_dict(decoding.greedy):
                decoding.greedy.use_cuda_graph_decoder = TDT_V2_DECODING["use_cuda_graph_decoder"]
        return
    if not isinstance(decoding, dict):
        raise RuntimeError("Parakeet TDT v2 decoding config cannot be updated")
    decoding["strategy"] = TDT_V2_DECODING["strategy"]
    decoding["compute_timestamps"] = TDT_V2_DECODING["compute_timestamps"]
    decoding["tdt_include_token_duration"] = TDT_V2_DECODING["tdt_include_token_duration"]
    greedy = decoding.get("greedy")
    if not isinstance(greedy, dict):
        greedy = {}
        decoding["greedy"] = greedy
    greedy["use_cuda_graph_decoder"] = TDT_V2_DECODING["use_cuda_graph_decoder"]


def _child(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if key not in node:
            raise RuntimeError(f"Parakeet TDT v2 decoding is missing {key}")
        return node[key]
    if key not in node:
        raise RuntimeError(f"Parakeet TDT v2 decoding is missing {key}")
    return node[key]


def _live_cuda_graph(model: Any) -> bool | None:
    """Read the decoder's CUDA-graph switch after the strategy has been applied.

    RNN-T stores ``use_cuda_graph_decoder`` on the decoder. TDT v2 stores the
    same choice as ``allow_cuda_graphs`` on the loop computer and does not
    copy the name onto the wrapper.
    """

    decoding = getattr(model, "decoding", None)
    decoder = getattr(decoding, "decoding", None)
    if decoder is None:
        return None
    if hasattr(decoder, "use_cuda_graph_decoder"):
        return bool(decoder.use_cuda_graph_decoder)
    computer = getattr(decoder, "_decoding_computer", None)
    if computer is not None and hasattr(computer, "allow_cuda_graphs"):
        return bool(computer.allow_cuda_graphs)
    return None


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
    """Keep the checkpoint strategy. Only turn off CUDA-graph decoding when it is present.

    The key must be in the checkpoint's own decoding config. The catalogued
    Parakeet checkpoints do not carry it, so this leaves them untouched and
    NeMo's default applies: CUDA-graph decoding stays on when CUDA is used.
    """
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
