"""Standard corpora linked at bootstrap.

``source_name`` is a directory under the dataset root supplied by the caller.
``link_name`` is the directory under ``data/corpora``. Audio corpora are indexed.
Non-audio directories are listed so a new environment records why they are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    source_name: str
    link_name: str
    index: bool
    role: str
    note: str


CATALOG: tuple[CorpusSpec, ...] = (
    CorpusSpec("CHiME3", "CHiME3", True, "noisy-speech-eval", "CHiME-3 challenge audio plus official JSON annotations."),
    CorpusSpec("DEMAND", "DEMAND", True, "noise", "DEMAND environmental noise scenes."),
    CorpusSpec("DNS5", "DNS5", True, "se-source", "DNS Challenge 5 fullband speech and noise. No single official eval mix is defined here."),
    CorpusSpec("LibriSpeech", "LibriSpeech", True, "clean-speech", "LibriSpeech flac with transcripts."),
    CorpusSpec("LibriTTS", "LibriTTS", True, "clean-speech", "LibriTTS wav with transcripts."),
    CorpusSpec("OpenSLR26", "OpenSLR26", True, "rir", "Simulated RIRs (SLR26)."),
    CorpusSpec("OpenSLR28", "OpenSLR28", True, "rir-noise", "RIRs and noises (SLR28)."),
    CorpusSpec("URGENT2024", "URGENT2024", True, "se-challenge", "URGENT 2024 released wavs. Blind test has no clean reference."),
    CorpusSpec("VCTK", "VCTK", True, "clean-speech", "VCTK 0.92 wav48."),
    CorpusSpec("VoiceBank+DEMAND", "VoiceBank-DEMAND", True, "paired-se", "Official noisy/clean pairs. Link name matches existing code."),
    CorpusSpec("WHAM48", "WHAM48", True, "noise", "WHAM! 48 kHz noise. Not a speech mixture by itself."),
    CorpusSpec("musan", "musan", True, "noise-music-speech", "MUSAN music, noise, and speech."),
)

INDEXED = tuple(spec for spec in CATALOG if spec.index)
SKIPPED = tuple(spec for spec in CATALOG if not spec.index)


def corpus_env_var(spec: CorpusSpec) -> str:
    """Environment variable for one corpus path.

    An individual path overrides ``AUDIO_DB_DATASET_ROOT / source_name``.
    """
    token = spec.source_name.upper().replace("+", "_").replace("-", "_")
    return f"AUDIO_DB_CORPUS_{token}"
