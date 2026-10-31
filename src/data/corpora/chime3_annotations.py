"""Read CHiME3 official JSON annotations and point them at isolated audio.

CH0 is the close-talk microphone. CH1–CH6 are the tablet microphones. Neither
is treated as a clean reference for the real recordings. Simulated audio is
also noisy. ``tr05_org`` is the original material and is a separate clean set,
not a time-aligned reference for the real evaluation utterances.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.audio import read_wav_stream_info
from src.evaluation.sample import AudioSample

SPLITS = ("tr05", "dt05", "et05")
CONDITIONS = ("real", "simu", "bth", "org")
CHANNELS = ("CH0", "CH1", "CH2", "CH3", "CH4", "CH5", "CH6")
_TABLET = {f"CH{index}" for index in range(1, 7)}


def load_chime3_samples(
    root: str | Path,
    *,
    split: str,
    condition: str,
    channel: str | None = None,
) -> list[AudioSample]:
    """Build samples from one official annotation file.

    ``root`` is the CHiME3 directory that contains ``data/annotations``.
    Missing audio fails the whole call. Nothing is written to DuckDB here.
    """
    chime_root = Path(root)
    split_name = _choice(split, SPLITS, "split")
    condition_name = _choice(condition, CONDITIONS, "condition")
    channel_name = _channel(condition_name, channel)
    annotation = chime_root / "data" / "annotations" / f"{split_name}_{condition_name}.json"
    if not annotation.is_file():
        raise ValueError(f"CHiME3 annotation not found: {annotation}")
    try:
        payload = json.loads(annotation.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid CHiME3 annotation {annotation}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"CHiME3 annotation must be a non-empty list: {annotation}")
    samples: list[AudioSample] = []
    missing: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{annotation.name} item {index} must be an object")
        sample, audio = _sample(chime_root, annotation.name, split_name, condition_name, channel_name, item, index)
        if sample.utterance_id in seen:
            raise ValueError(f"duplicate CHiME3 utterance id {sample.utterance_id}")
        seen.add(sample.utterance_id)
        if not audio.is_file():
            missing.append(audio.as_posix())
            continue
        rate, _frames, channels = read_wav_stream_info(audio)
        if channels != 1:
            raise ValueError(f"CHiME3 isolated file is not mono: {audio}")
        sample.sample_rate = rate
        samples.append(sample)
    if missing:
        example = missing[0]
        raise ValueError(
            f"CHiME3 set {split_name}_{condition_name} {channel_name or 'org'} is incomplete: "
            f"{len(missing)} of {len(payload)} audio files are missing, for example {example}"
        )
    return samples


def _sample(
    root: Path,
    annotation_name: str,
    split: str,
    condition: str,
    channel: str | None,
    item: dict,
    index: int,
) -> tuple[AudioSample, Path]:
    speaker = _text(item.get("speaker"), f"{annotation_name} item {index} speaker")
    environment = _text(item.get("environment"), f"{annotation_name} item {index} environment").upper()
    wsj_name = item.get("wsj_name")
    if condition != "org":
        wsj_name = _text(wsj_name, f"{annotation_name} item {index} wsj_name")
    elif wsj_name is not None:
        wsj_name = _text(wsj_name, f"{annotation_name} item {index} wsj_name")
    transcript = item.get("prompt") or item.get("dot")
    transcript = _text(transcript, f"{annotation_name} item {index} transcript")
    stem = f"{speaker}_{wsj_name}_{environment}" if wsj_name else f"{speaker}_{environment}"
    if condition == "org":
        folder = root / "data" / "audio" / "16kHz" / "isolated" / f"{split}_org"
        audio = folder / f"{stem}.wav"
        utterance_id = f"chime3:{split}:org:{stem}"
        role = "original_material"
        kind_path = "clean"
    else:
        if channel is None:
            raise ValueError("channel is required")
        folder_name = f"{split}_bth" if condition == "bth" else f"{split}_{environment.lower()}_{condition}"
        folder = root / "data" / "audio" / "16kHz" / "isolated" / folder_name
        audio = folder / f"{stem}.{channel}.wav"
        utterance_id = f"chime3:{split}:{condition}:{stem}.{channel}"
        role = _reference_role(condition, channel)
        kind_path = "noisy"
    metadata = {
        "source": "chime3_annotation",
        "annotation": annotation_name,
        "condition": condition,
        "environment": environment,
        "mic_channel": channel,
        "wsj_name": wsj_name,
        "wavfile": item.get("wavfile"),
        "start": item.get("start"),
        "end": item.get("end"),
        "dot": item.get("dot"),
        "reference_role": role,
        "clean_reference": False,
    }
    sample = AudioSample(
        utterance_id=utterance_id,
        dataset="CHiME3",
        split=split,
        transcript=transcript,
        speaker_id=speaker,
        metadata={key: value for key, value in metadata.items() if value is not None},
        clean_path=audio.as_posix() if kind_path == "clean" else None,
        noisy_path=audio.as_posix() if kind_path == "noisy" else None,
    )
    return sample, audio


def _reference_role(condition: str, channel: str) -> str:
    if condition == "simu":
        return "simulated_noisy"
    if condition == "bth":
        return "booth_recording"
    if channel == "CH0":
        return "close_talk_in_noise"
    if channel in _TABLET:
        return "tablet_microphone"
    raise ValueError(f"unknown CHiME3 channel {channel}")


def _channel(condition: str, channel: str | None) -> str | None:
    if condition == "org":
        if channel not in (None, ""):
            raise ValueError("CHiME3 org has no microphone channel")
        return None
    if channel is None or (isinstance(channel, str) and not channel.strip()):
        raise ValueError("CHiME3 real, simu, and bth require a microphone channel such as CH5")
    if not isinstance(channel, str):
        raise ValueError("channel must be a string")
    text = channel.strip().upper()
    if text.isdigit():
        text = f"CH{text}"
    if text not in CHANNELS:
        raise ValueError(f"CHiME3 channel must be one of: {', '.join(CHANNELS)}")
    return text


def _choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"CHiME3 {label} must be one of: {', '.join(allowed)}")
    return value


def _text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value.strip()
