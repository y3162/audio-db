"""VoiceBank-DEMAND at a −15 dB SNR shift for the ablation runs.

The published snapshot keeps p286 and p287 as validation. These runs reassign
the 28 training speakers in memory: a SHA-256 ranking of the speaker id at
seed 42 puts 3 speakers in validation and the other 25 in train. Test speakers
stay on the snapshot's test list. The snapshot files are not rewritten.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.experiments.calc.dataset import resampled_frame_count
from src.experiments.data.fixed_pair import SnapshotSource
from src.experiments.data.mix.voicebank_snr import dataset_shift, render_case
from src.experiments.data.render.transform import apply_window
from src.experiments.data.render.types import CropSpec
from src.experiments.data.sample import Crop, SampleRef
from src.experiments.data.sampling.planner import fixed_pair_crop
from src.experiments.errors import DataError
from src.experiments.data.errors import ReproductionError

_SOURCE_RATE = 48000
_TARGET_RATE = 16000
_WINDOW = 48000
_SHIFT = -15.0


def ablation_holdout(
    speakers: Sequence[str],
    *,
    seed: int = 42,
    valid_count: int = 3,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Rank speakers by SHA-256 and hold out the first ``valid_count``."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataError("split seed must be a non-negative integer")
    if isinstance(valid_count, bool) or not isinstance(valid_count, int) or valid_count < 1:
        raise DataError("valid_count must be a positive integer")
    labels = [str(speaker) for speaker in speakers]
    unique = tuple(sorted(set(labels)))
    if len(unique) != len(labels) or len(unique) != 28:
        raise DataError(f"VoiceBank ablation expects 28 distinct speakers, got {len(unique)}")
    if valid_count >= len(unique):
        raise DataError("valid_count leaves no training speakers")
    ranked = sorted(
        unique,
        key=lambda speaker: hashlib.sha256(
            f"voicebank-ablation-v1:{seed}:{speaker}".encode("utf-8")
        ).hexdigest(),
    )
    valid = tuple(sorted(ranked[:valid_count]))
    train = tuple(sorted(ranked[valid_count:]))
    return train, valid


class VoicebankAblationSource(SnapshotSource):
    """Snapshot source whose train and valid case lists follow ``ablation_holdout``."""

    def __init__(
        self,
        snapshot: str,
        snapshot_root: str | Path,
        split: str,
        *,
        split_seed: int = 42,
        valid_count: int = 3,
        derived: str | Path | None = None,
        corpora: str | Path | None = None,
        generated: str | Path | None = None,
    ) -> None:
        super().__init__(
            snapshot,
            snapshot_root,
            split,
            derived=derived,
            corpora=corpora,
            generated=generated,
        )
        if isinstance(split_seed, bool) or not isinstance(split_seed, int) or split_seed < 0:
            raise ValueError("split_seed must be a non-negative integer")
        if isinstance(valid_count, bool) or not isinstance(valid_count, int) or valid_count < 1:
            raise ValueError("valid_count must be a positive integer")
        self.split_seed = int(split_seed)
        self.valid_count = int(valid_count)
        self.ablation_record: dict[str, object] | None = None

    def open(self) -> None:
        if self._store is not None:
            return
        super().open()
        self._apply_split()

    def _apply_split(self) -> None:
        store = self.store
        train_set = _binding(store, "train.cases")
        valid_set = _binding(store, "valid.cases")
        test_set = _binding(store, "test.cases")
        pooled = [row["case_id"] for row in store.items[train_set]] + [
            row["case_id"] for row in store.items[valid_set]
        ]
        test_ids = [row["case_id"] for row in store.items[test_set]]
        by_speaker: dict[str, list[str]] = {}
        for case_id in pooled:
            speaker = str(store.case(case_id)["speaker_id"])
            by_speaker.setdefault(speaker, []).append(case_id)
        test_speakers = {str(store.case(case_id)["speaker_id"]) for case_id in test_ids}
        if test_speakers & set(by_speaker):
            raise DataError("VoiceBank train/valid speakers overlap the test speakers")
        train_speakers, valid_speakers = ablation_holdout(
            tuple(by_speaker),
            seed=self.split_seed,
            valid_count=self.valid_count,
        )
        train_ids = _ordered(store, by_speaker, train_speakers)
        valid_ids = _ordered(store, by_speaker, valid_speakers)
        if set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids) or set(valid_ids) & set(test_ids):
            raise DataError("VoiceBank ablation split repeats a case")
        if len(train_ids) + len(valid_ids) != len(pooled):
            raise DataError("VoiceBank ablation split dropped a training case")
        store.items[train_set] = _items(train_set, train_ids)
        store.items[valid_set] = _items(valid_set, valid_ids)
        self.ablation_record = {
            "algorithm": "sha256-rank-v1",
            "seed": self.split_seed,
            "test_speakers": sorted(test_speakers),
            "train_speakers": list(train_speakers),
            "valid_count": self.valid_count,
            "valid_speakers": list(valid_speakers),
        }


class VoicebankAblationProtocol:
    """16 kHz VoiceBank windows scaled by the noisy peak.

    Clean and noisy share that one coefficient, so their relative amplitude
    stays put and a louder clean waveform is not clipped back to 1. Validation
    keeps the full utterance.
    """

    supply = "fixed_pair"

    def __init__(
        self,
        source: VoicebankAblationSource,
        verify_on_read: bool = False,
        *,
        sample_rate: int = _TARGET_RATE,
        window: int = _WINDOW,
    ) -> None:
        if not isinstance(verify_on_read, bool):
            raise ValueError("verify_on_read must be a boolean")
        if int(sample_rate) != _TARGET_RATE:
            raise ValueError("VoiceBank ablation sample_rate must be 16000")
        if int(window) != _WINDOW:
            raise ValueError("VoiceBank ablation window must be 48000")
        self.source = source
        self.verify_on_read = verify_on_read
        self.sample_rate = _TARGET_RATE
        self.window = _WINDOW
        self.normalize = "noisy_peak_v1"
        self.validation_crop = "full_length"
        self.resample = {
            "from_hz": _SOURCE_RATE,
            "method": "torchaudio.functional.resample",
            "to_hz": _TARGET_RATE,
        }
        self.teacher: str | None = None
        self.seed: int | None = None
        self._shift: float | None = None
        self._full: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._utterances: OrderedDict[str, _Utterance] = OrderedDict()

    def prepare(self) -> None:
        dataset = self.source.dataset()
        if str(dataset.get("algorithm")) != "pregenerated.voicebank_snr_shift":
            raise DataError("VoiceBank ablation protocol requires the voicebank snr-shift snapshot")
        rule = dataset.get("rule_json")
        if not isinstance(rule, dict) or int(rule.get("sample_rate") or 0) != _SOURCE_RATE:
            raise DataError("VoiceBank snapshot is not 48 kHz")
        if rule.get("teacher") != "dry_clean":
            raise DataError("VoiceBank teacher is not dry_clean")
        shift = float(dataset_shift(dataset))
        if shift != _SHIFT:
            raise DataError(f"ablation snapshot shift is {shift}, expected {_SHIFT}")
        self._shift = shift
        self.teacher = "dry_clean"
        self._full.clear()
        self._utterances.clear()

    def case_length(self, case_id: str) -> int:
        document = self.source.store.docs.get(case_id)
        output = document.get("output") if isinstance(document, dict) else None
        if not isinstance(output, dict) or "length" not in output:
            raise DataError(f"{case_id} has no planned length")
        try:
            return resampled_frame_count(int(output["length"]), _SOURCE_RATE, _TARGET_RATE)
        except (TypeError, ValueError) as exc:
            raise DataError(f"{case_id} has no planned length") from exc

    def locate(self, case_id: str, length: int, epoch: int) -> Crop:
        if self.seed is None:
            raise DataError("VoiceBank ablation protocol has no seed")
        spec = fixed_pair_crop(case_id, length, self.window, self.seed, epoch, "tail_zero")
        return Crop(start=int(spec.start), length=int(spec.length), pad_to=int(spec.pad_to))

    def render(self, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
        if not sample.case_id:
            raise DataError("VoiceBank rendering requires a case id")
        noisy, clean = self._peak_pair(sample.case_id)
        crop = None
        if sample.crop is not None:
            pad = "none" if sample.crop.length == sample.crop.pad_to else "tail_zero"
            crop = CropSpec(
                start=int(sample.crop.start),
                length=int(sample.crop.length),
                pad_to=int(sample.crop.pad_to),
                pad=pad,
            )
        try:
            noisy_window, clean_window, valid = apply_window(noisy, clean, crop)
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc
        start = 0 if sample.crop is None else int(sample.crop.start)
        self._remember(
            sample.realization_id,
            _Utterance(
                example=_CropExample(start),
                full_clean=np.array(clean, dtype=np.float32, copy=True),
                full_degraded=np.array(noisy, dtype=np.float32, copy=True),
                full_length=int(clean.shape[0]),
            ),
        )
        return noisy_window, clean_window, int(valid)

    def span_for(self, sample: SampleRef) -> tuple[int, int, int, int]:
        full = self.case_length(str(sample.case_id))
        if sample.crop is None:
            return (0, full, full, full)
        return (
            int(sample.crop.start),
            int(sample.crop.length),
            int(sample.crop.pad_to),
            full,
        )

    def utterance_for_id(self, realization_id: str) -> _Utterance:
        cached = self._utterances.get(realization_id)
        if cached is not None:
            self._utterances.move_to_end(realization_id)
            return cached
        self.render(_sample_from_realization(self, realization_id))
        found = self._utterances.get(realization_id)
        if found is None:
            raise DataError("rendered utterance was not cached")
        return found

    def teacher_kind(self, sample: SampleRef) -> str:
        del sample
        if self.teacher is None:
            raise DataError("VoiceBank ablation protocol is not prepared")
        return self.teacher

    def reference_transcript(self, sample_id: str) -> str | None:
        case = self.source.store.case(sample_id)
        speaker = str(case.get("speaker_id") or "")
        utterance = str(case.get("utterance_key") or "")
        if not speaker or not utterance.startswith(speaker + "_"):
            raise DataError(f"{sample_id} has no VoiceBank utterance id")
        path = Path(self.source.corpora) / "VCTK" / "txt" / speaker / f"{utterance}.txt"
        if not path.is_file():
            return None
        text = " ".join(path.read_text(encoding="utf-8").split())
        return text or None

    def _peak_pair(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._full.get(case_id)
        if cached is not None:
            return cached
        if self._shift is None:
            raise DataError("VoiceBank ablation protocol is not prepared")
        try:
            clean, noisy = render_case(
                self.source.store,
                case_id,
                snr_shift_db=self._shift,
                verify_on_read=self.verify_on_read,
            )
        except ReproductionError as exc:
            raise DataError(str(exc)) from exc
        clean_16 = _resample(clean)
        noisy_16 = _resample(noisy)
        expected = self.case_length(case_id)
        if int(clean_16.shape[0]) != expected or clean_16.shape != noisy_16.shape:
            raise DataError(
                f"{case_id} resampled to {clean_16.shape[0]}/{noisy_16.shape[0]}, expected {expected}"
            )
        clean_peak, noisy_peak = _shared_noisy_peak(clean_16, noisy_16)
        pair = (
            np.ascontiguousarray(noisy_peak, dtype=np.float32),
            np.ascontiguousarray(clean_peak, dtype=np.float32),
        )
        self._full[case_id] = pair
        return pair

    def _remember(self, realization_id: str, utterance: _Utterance) -> None:
        self._utterances[realization_id] = utterance
        self._utterances.move_to_end(realization_id)
        while len(self._utterances) > 8:
            self._utterances.popitem(last=False)


@dataclass(frozen=True)
class _CropExample:
    crop_start: int


@dataclass(frozen=True)
class _Utterance:
    example: _CropExample
    full_clean: np.ndarray
    full_degraded: np.ndarray
    full_length: int


def _binding(store: object, slot: str) -> str:
    bindings = getattr(store, "bindings", None)
    if not isinstance(bindings, dict) or slot not in bindings:
        raise DataError(f"snapshot has no {slot} binding")
    return str(bindings[slot])


def _ordered(store: object, by_speaker: dict[str, list[str]], speakers: tuple[str, ...]) -> list[str]:
    chosen: list[str] = []
    for speaker in speakers:
        rows = by_speaker[speaker]
        rows.sort(key=lambda case_id: (str(store.case(case_id)["utterance_key"]), case_id))
        chosen.extend(rows)
    return chosen


def _items(case_set_id: str, case_ids: list[str]) -> list[dict[str, object]]:
    return [
        {"case_set_id": case_set_id, "position": index, "case_id": case_id}
        for index, case_id in enumerate(case_ids)
    ]


def _shared_noisy_peak(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale clean and noisy by the noisy peak.

    The coefficient is ``max(|noisy|) + 1e-9``. Both waveforms use it, clean is
    not scaled again, and values above 1 are left as they are.
    """

    scale = np.max(np.abs(noisy)) + 1e-9
    return clean / scale, noisy / scale


def _resample(audio: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio

    wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).view(1, -1)
    converted = torchaudio.functional.resample(wave, _SOURCE_RATE, _TARGET_RATE)
    return np.ascontiguousarray(converted.view(-1).numpy(), dtype=np.float32)


def _sample_from_realization(protocol: VoicebankAblationProtocol, realization_id: str) -> SampleRef:
    marker = ":epoch-"
    if marker not in realization_id:
        raise DataError("voicebank realization id is invalid")
    case_id, rest = realization_id.split(marker, 1)
    epoch_text, found, crop_text = rest.partition(":crop-")
    if not found:
        raise DataError("voicebank realization id is invalid")
    try:
        int(epoch_text)
    except ValueError as exc:
        raise DataError("voicebank realization id is invalid") from exc
    if crop_text == "full":
        crop = None
    else:
        start_text, separator, length_text = crop_text.partition("+")
        if not separator:
            raise DataError("voicebank realization id is invalid")
        try:
            start = int(start_text)
            length = int(length_text)
        except ValueError as exc:
            raise DataError("voicebank realization id is invalid") from exc
        if length < 1 or length > protocol.window or start < 0:
            raise DataError("voicebank realization crop is outside the window")
        crop = Crop(start, length, protocol.window if length < protocol.window else length)
    return SampleRef(
        sample_id=case_id,
        realization_id=realization_id,
        index=0,
        case_id=case_id,
        crop=crop,
    )
