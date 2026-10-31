"""Model ids, backends, and decode methods.

Entries describe published checkpoints. A row here means the id is accepted by
the loader. It does not mean this checkout has executed that checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


FAMILIES = ("parakeet", "whisper", "wav2vec2", "hubert", "wavlm")
BACKENDS = {
    "parakeet": "nemo",
    "whisper": "whisper",
    "wav2vec2": "transformers",
    "hubert": "transformers",
    "wavlm": "transformers",
}
_DECODE_TAGS = ("tdt", "ctc", "rnnt")
_ENGLISH = ("en",)


@dataclass(frozen=True)
class ModelCard:
    family: str
    model_id: str
    backend: str
    decode: str
    languages: tuple[str, ...]
    asr_finetuned: bool
    provider: str
    license: str
    notes: str = ""

    @property
    def checkpoint(self) -> str:
        return self.model_id


def _card(**kwargs) -> ModelCard:
    return ModelCard(**kwargs)


# Official Parakeet ids. Decode is part of the id; it is not chosen globally.
_PARAKEET = (
    _card(
        family="parakeet",
        model_id="nvidia/parakeet-tdt-0.6b-v2",
        backend="nemo",
        decode="tdt",
        languages=_ENGLISH,
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="English TDT 0.6B v2. Representative NeMo checkpoint.",
    ),
    _card(
        family="parakeet",
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        backend="nemo",
        decode="tdt",
        languages=(),
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="Multilingual TDT 0.6B v3. Language coverage follows the model card.",
    ),
    _card(
        family="parakeet",
        model_id="nvidia/parakeet-tdt-1.1b",
        backend="nemo",
        decode="tdt",
        languages=_ENGLISH,
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="English TDT 1.1B.",
    ),
    _card(
        family="parakeet",
        model_id="nvidia/parakeet-ctc-0.6b",
        backend="nemo",
        decode="ctc",
        languages=_ENGLISH,
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="English CTC 0.6B. Loaded with the NeMo CTC model, not the TDT decoder.",
    ),
    _card(
        family="parakeet",
        model_id="nvidia/parakeet-ctc-1.1b",
        backend="nemo",
        decode="ctc",
        languages=_ENGLISH,
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="English CTC 1.1B.",
    ),
    _card(
        family="parakeet",
        model_id="nvidia/parakeet-rnnt-0.6b",
        backend="nemo",
        decode="rnnt",
        languages=_ENGLISH,
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="English RNN-T 0.6B. Loaded with the NeMo RNN-T model, not CTC or TDT.",
    ),
    _card(
        family="parakeet",
        model_id="nvidia/parakeet-rnnt-1.1b",
        backend="nemo",
        decode="rnnt",
        languages=_ENGLISH,
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="English RNN-T 1.1B.",
    ),
)

_WHISPER_MULTILINGUAL = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
    "large-v3-turbo",
)
_WHISPER_ENGLISH = ("tiny.en", "base.en", "small.en", "medium.en")

_WAV2VEC2 = (
    ("facebook/wav2vec2-base-960h", "CTC fine-tune of wav2vec2-base on LibriSpeech 960h."),
    ("facebook/wav2vec2-large-960h", "CTC fine-tune of wav2vec2-large on LibriSpeech 960h."),
    ("facebook/wav2vec2-large-960h-lv60", "CTC fine-tune with self-training on LibriSpeech."),
    ("facebook/wav2vec2-large-960h-lv60-self", "CTC fine-tune; representative large wav2vec2 ASR checkpoint."),
)
_HUBERT = (
    ("facebook/hubert-large-ls960-ft", "ASR fine-tune of HuBERT large on LibriSpeech 960h."),
    ("facebook/hubert-xlarge-ls960-ft", "ASR fine-tune of HuBERT x-large on LibriSpeech 960h."),
)
_WAVLM = (
    (
        "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
        "CTC fine-tune of microsoft/wavlm-base-plus on LibriSpeech clean 100h. "
        "The Microsoft pre-trained WavLM checkpoints are not ASR models.",
    ),
)

# Self-supervised or non-ASR checkpoints that must not be loaded as recognizers.
SSL_ONLY = {
    "facebook/wav2vec2-base": "wav2vec2",
    "facebook/wav2vec2-large": "wav2vec2",
    "facebook/wav2vec2-large-robust": "wav2vec2",
    "facebook/wav2vec2-xls-r-300m": "wav2vec2",
    "facebook/hubert-base-ls960": "hubert",
    "facebook/hubert-large-ls960": "hubert",
    "facebook/hubert-xlarge-ll60k": "hubert",
    "microsoft/wavlm-base": "wavlm",
    "microsoft/wavlm-base-plus": "wavlm",
    "microsoft/wavlm-large": "wavlm",
}


def _whisper_cards() -> tuple[ModelCard, ...]:
    cards = []
    for name in _WHISPER_MULTILINGUAL:
        cards.append(
            _card(
                family="whisper",
                model_id=name,
                backend="whisper",
                decode="whisper",
                languages=(),
                asr_finetuned=True,
                provider="OpenAI",
                license="MIT",
                notes="Multilingual Whisper. task is transcribe.",
            )
        )
    for name in _WHISPER_ENGLISH:
        cards.append(
            _card(
                family="whisper",
                model_id=name,
                backend="whisper",
                decode="whisper",
                languages=_ENGLISH,
                asr_finetuned=True,
                provider="OpenAI",
                license="MIT",
                notes="English-only Whisper. task is transcribe.",
            )
        )
    return tuple(cards)


def _hf_cards() -> tuple[ModelCard, ...]:
    cards = []
    for model_id, notes in _WAV2VEC2:
        cards.append(
            _card(
                family="wav2vec2",
                model_id=model_id,
                backend="transformers",
                decode="ctc",
                languages=_ENGLISH,
                asr_finetuned=True,
                provider="Meta",
                license="MIT",
                notes=notes,
            )
        )
    for model_id, notes in _HUBERT:
        cards.append(
            _card(
                family="hubert",
                model_id=model_id,
                backend="transformers",
                decode="ctc",
                languages=_ENGLISH,
                asr_finetuned=True,
                provider="Meta",
                license="MIT",
                notes=notes,
            )
        )
    for model_id, notes in _WAVLM:
        cards.append(
            _card(
                family="wavlm",
                model_id=model_id,
                backend="transformers",
                decode="ctc",
                languages=_ENGLISH,
                asr_finetuned=True,
                provider="community fine-tune of Microsoft WavLM",
                license="MIT",
                notes=notes,
            )
        )
    return tuple(cards)


CATALOG: tuple[ModelCard, ...] = _PARAKEET + _whisper_cards() + _hf_cards()
_BY_ID = {card.model_id: card for card in CATALOG}


def iter_cards(family: str | None = None) -> tuple[ModelCard, ...]:
    if family is None:
        return CATALOG
    if family not in FAMILIES:
        raise ValueError(f"unknown ASR family {family!r}. Known families: {', '.join(FAMILIES)}")
    return tuple(card for card in CATALOG if card.family == family)


def backend_for(family: str) -> str:
    try:
        return BACKENDS[family]
    except KeyError as exc:
        raise ValueError(f"unknown ASR family {family!r}. Known families: {', '.join(FAMILIES)}") from exc


def resolve_model(
    name: str,
    model_id: str,
    *,
    decode: str | None = None,
    language: str | None = None,
) -> ModelCard:
    """Check a family and model id. This does not download weights."""
    if name not in FAMILIES:
        raise ValueError(f"unknown ASR family {name!r}. Known families: {', '.join(FAMILIES)}")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    model_id = _canonical_id(name, model_id.strip())
    language_code = _language_code(language)
    if model_id in SSL_ONLY:
        owner = SSL_ONLY[model_id]
        raise ValueError(
            f"{model_id} is a self-supervised {owner} checkpoint, not an ASR model. "
            "Choose an ASR fine-tune from the catalog."
        )
    card = _BY_ID.get(model_id)
    if card is None:
        card = _dynamic_card(name, model_id)
    if card.family != name:
        raise ValueError(f"{model_id} belongs to {card.family}, not {name}")
    if decode is not None:
        if not isinstance(decode, str) or not decode.strip():
            raise ValueError("decode must be a non-empty string")
        if decode.strip().lower() != card.decode:
            raise ValueError(
                f"{model_id} uses decode {card.decode!r}, not {decode.strip().lower()!r}"
            )
    if language_code is not None and card.languages and language_code not in card.languages:
        joined = ", ".join(card.languages)
        raise ValueError(f"{model_id} does not list language {language_code!r}. Catalog languages: {joined}")
    return card


def dependency_problem(backend: str) -> str | None:
    """Return a message when the backend package cannot be imported."""
    import importlib.util

    modules = {"nemo": "nemo", "whisper": "whisper", "transformers": "transformers"}
    module = modules.get(backend)
    if module is None:
        return f"unknown ASR backend {backend!r}"
    if importlib.util.find_spec(module) is None:
        return (
            f"ASR backend {backend!r} is not installed in this interpreter "
            f"(missing import {module}). Weights were not downloaded."
        )
    return None


def _canonical_id(name: str, model_id: str) -> str:
    if name != "whisper":
        return model_id
    prefix = "openai/whisper-"
    if model_id.startswith(prefix):
        return model_id[len(prefix) :]
    return model_id


def _dynamic_card(name: str, model_id: str) -> ModelCard:
    if name != "parakeet":
        known = ", ".join(card.model_id for card in iter_cards(name))
        raise ValueError(f"unsupported {name} model id {model_id!r}. Catalog ids: {known}")
    if not model_id.startswith("nvidia/parakeet-"):
        raise ValueError(
            f"unsupported Parakeet model id {model_id!r}. Expected an nvidia/parakeet-* checkpoint."
        )
    tags = _decode_tags(model_id)
    if len(tags) != 1:
        joined = ", ".join(tags) if tags else "none"
        raise ValueError(
            f"cannot choose a decoder for {model_id!r} (found {joined}). "
            "TDT, CTC, and RNN-T are not interchangeable."
        )
    return _card(
        family="parakeet",
        model_id=model_id,
        backend="nemo",
        decode=tags[0],
        languages=(),
        asr_finetuned=True,
        provider="NVIDIA",
        license="CC-BY-4.0",
        notes="Id is outside the built-in list. Decode was taken from the id.",
    )


def _decode_tags(model_id: str) -> tuple[str, ...]:
    found = []
    for tag in _DECODE_TAGS:
        if re.search(rf"(^|[-_]){tag}($|[-_])", model_id):
            found.append(tag)
    return tuple(found)


def _language_code(language: str | None) -> str | None:
    if language is None:
        return None
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")
    lowered = language.strip().lower()
    if lowered in {"en", "english"}:
        return "en"
    return lowered
