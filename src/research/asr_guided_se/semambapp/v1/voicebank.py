"""VoiceBank-DEMAND SNR-shift ablations with the official test set used for selection."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.experiments.calc.dataset import resampled_frame_count
from src.experiments.data.errors import ReproductionError
from src.experiments.data.fixed_pair import SnapshotSource
from src.experiments.data.ids import case_set_id_for, dataset_id_for, material_members_key, material_set_id_for
from src.experiments.data.mix.voicebank_snr import dataset_shift, render_case
from src.experiments.data.render.transform import apply_window
from src.experiments.data.render.types import CropSpec
from src.experiments.data.sample import Crop, SampleRef
from src.experiments.data.sampling.planner import fixed_pair_crop
from src.experiments.errors import DataError

_SOURCE_RATE = 48000
_TARGET_RATE = 16000
_WINDOW = 48000


class VoicebankAblationSource(SnapshotSource):
    """Train on the 28-speaker release and bind validation to the published test list."""

    def __init__(
        self,
        snapshot: str,
        snapshot_root: str | Path,
        split: str,
        *,
        allow_test_selection: bool,
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
        if allow_test_selection is not True:
            raise ValueError("VoiceBank v1 requires allow_test_selection: true")
        if split != "train":
            raise ValueError("VoiceBank v1 requires the training split for updates")
        self.ablation_record: dict[str, object] | None = None

    def open(self) -> None:
        if self._store is not None:
            return
        super().open()
        try:
            self._apply_split()
        except Exception:
            self.close()
            raise

    def _apply_split(self) -> None:
        store = self.store
        base = store.dataset
        if base["algorithm"] != "pregenerated.voicebank_snr_shift":
            raise DataError("VoiceBank ablations require a VoiceBank SNR-shift snapshot")
        train_ids = store.case_ids("train.cases") + store.case_ids("valid.cases")
        test_ids = store.case_ids("test.cases")
        train_speakers = {store.case(case_id)["speaker_id"] for case_id in train_ids}
        test_speakers = {store.case(case_id)["speaker_id"] for case_id in test_ids}
        if len(train_ids) != 11_572 or len(set(train_ids)) != len(train_ids) or len(train_speakers) != 28:
            raise DataError("expected 11572 distinct training cases from 28 VoiceBank speakers")
        if len(test_ids) != 824 or len(set(test_ids)) != 824 or test_speakers != {"p232", "p257"}:
            raise DataError("expected 824 distinct test cases from p232 and p257")
        if train_speakers & test_speakers:
            raise DataError("VoiceBank training speakers overlap the official test set")

        # Build new sets for the pooled training data. Validation can use the
        # original test bindings verbatim, including material ids and ordering.
        bindings = {row["slot"]: dict(row) for row in base["bindings"]}
        train_set = case_set_id_for(
            name="voicebank28.train.cases",
            version=1,
            split_role="train",
            rule={"rule": "explicit"},
            case_ids=train_ids,
        )
        bindings["train.cases"]["case_set_id"] = train_set
        store.items[train_set] = [
            {"case_set_id": train_set, "position": i, "case_id": case_id}
            for i, case_id in enumerate(train_ids)
        ]
        for role in ("speech", "noise"):
            bindings[f"train.{role}"]["material_set_id"] = self._pool_training_materials(role)
        for role in ("cases", "speech", "noise"):
            slot = f"valid.{role}"
            bindings[slot] = {**bindings[f"test.{role}"], "slot": slot}

        effective = {
            **base,
            "name": f"{base['name']}.test_selection",
            "bindings": list(bindings.values()),
            "selection_uses_test": True,
            "selection_note": "The official test set supplies validation WER for checkpoint selection and stopping.",
        }
        effective["dataset_id"] = dataset_id_for(
            protocol_id=base["protocol_id"],
            bindings=effective["bindings"],
            profile=base["profile"],
            material_origin=base["material_origin"],
            claim_label=base["claim_label"],
            selection_uses_test=True,
            declared_status=base["declared_status"],
        )
        # The base deviations describe construction of the immutable snapshot.
        # Qualify them so its original split is not presented as the run split.
        effective["deviations_json"] = [
            {**row, "text": f"Base snapshot: {row['text']}"} for row in base["deviations_json"]
        ]
        store.dataset = effective
        store.bindings = {slot: row["case_set_id"] for slot, row in bindings.items()}
        self.ablation_record = {
            "algorithm": "voicebank28-official-test-selection-v1",
            "base_dataset_id": base["dataset_id"],
            "selection_uses_test": True,
            "train_speakers": sorted(train_speakers),
            "valid_speakers": sorted(test_speakers),
            "test_speakers": sorted(test_speakers),
            "train_cases": len(train_ids),
            "valid_cases": len(test_ids),
            "test_cases": len(test_ids),
        }

    def _pool_training_materials(self, role: str) -> str:
        store = self.store
        asset_ids = sorted(set(self.material_ids(f"train.{role}")) | set(self.material_ids(f"valid.{role}")))
        members = []
        for asset_id in asset_ids:
            asset = store.asset(asset_id)
            members.append(material_members_key(asset_id, asset.get("sha256"), asset.get("size_bytes")))
        material_set = material_set_id_for(
            name=f"voicebank28.train.{role}",
            version=1,
            role=role,
            split_role="train",
            rule={"rule": "explicit"},
            members=members,
        )
        store.members.extend(
            {"material_set_id": material_set, "position": position, "asset_id": asset_id}
            for position, asset_id in enumerate(asset_ids)
        )
        return material_set


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
        self._shift = float(dataset_shift(dataset))
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
    full_length: int


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
