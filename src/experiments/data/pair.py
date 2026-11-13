"""Audio pair contract shared by dataset adapters and trainers.

A pair is noisy input, a teacher waveform, a sample rate, a valid length, and
a teacher kind. Dataset names are not part of this contract. ``AudioPairData``
supplies pairs that were not read from a named dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.experiments.data.sample import Batch
from src.experiments.errors import DataError, RuntimeFailure, StateError

TEACHERS = ("dry_clean", "reverberant_clean", "official_mixed_target")
_TENSOR_CLASS = "src.experiments.data.TensorPairData"


def check_waveforms(
    noisy: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    sample_rate: int,
    teacher: str,
    valid_length: int | None = None,
) -> int:
    """Reject a pair whose audio role, rate, length, or values are unusable."""

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
        raise DataError(f"sample_rate {sample_rate!r} is not a positive integer")
    if teacher not in TEACHERS:
        raise DataError(f"teacher {teacher!r} is not a training target")
    noisy_array = _as_float(noisy, "noisy")
    target_array = _as_float(target, "target")
    if noisy_array.shape != target_array.shape or noisy_array.ndim != 1:
        raise DataError(
            f"noisy and target must be matching 1-D waveforms, got {tuple(noisy_array.shape)} and {tuple(target_array.shape)}"
        )
    if noisy_array.shape[0] < 1:
        raise DataError("audio pair is empty")
    if not np.isfinite(noisy_array).all() or not np.isfinite(target_array).all():
        raise DataError("audio pair contains a non-finite sample")
    length = int(noisy_array.shape[0]) if valid_length is None else valid_length
    if isinstance(length, bool) or not isinstance(length, int) or length < 1 or length > int(noisy_array.shape[0]):
        raise DataError(f"valid_length {length} does not fit {int(noisy_array.shape[0])} samples")
    return length


def require_audio_contract(data: object, *, sample_rate: int, teachers: tuple[str, ...]) -> None:
    """Refuse a data module whose declared rate or teacher disagrees with the model.

    Feature-vector ``TensorPairData`` is the only identity that may omit both
    fields. Every other identity, including a snapshot, must declare them.
    """

    if not teachers or any(kind not in TEACHERS for kind in teachers):
        raise RuntimeFailure("model teacher requirements are not training targets")
    identity_fn = getattr(data, "identity", None)
    if not callable(identity_fn):
        raise RuntimeFailure("audio data has no identity")
    identity = identity_fn()
    if not isinstance(identity, dict):
        raise RuntimeFailure("data identity is not a mapping")
    if identity.get("class") == _TENSOR_CLASS:
        return
    rate = identity.get("sample_rate")
    teacher = identity.get("teacher")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate != int(sample_rate):
        raise RuntimeFailure(f"data sample_rate {rate} does not match the model requirement {sample_rate}")
    if teacher not in teachers:
        expected = ", ".join(teachers)
        raise RuntimeFailure(f"data teacher {teacher} is not accepted by this model ({expected})")


def bind_trainer_data(trainer: object) -> None:
    """Check the audio contract and the publication claim before a training step.

    The check lives here so the six trainers do not each reimplement it.
    Forward and the loss do not call it.
    """

    data = getattr(trainer, "data", None)
    rate = getattr(trainer, "audio_sample_rate", None)
    teachers = getattr(trainer, "audio_teachers", None)
    if rate is not None or teachers is not None:
        if data is None or rate is None or not teachers:
            raise RuntimeFailure("trainer audio contract is incomplete")
        require_audio_contract(data, sample_rate=int(rate), teachers=tuple(teachers))
    check_publication(trainer)


def check_publication(trainer: object) -> None:
    """Compare an official claim with the dataset profile. Research does not.

    ``research`` trains on a compatible pair without recording the run as the
    paper's official reproduction. Official and leakage-safe claims keep the
    profile and test-selection checks.
    """

    claim = getattr(trainer, "experiment_claim", None)
    label = getattr(trainer, "claim_label", None)
    if claim is None and label is None:
        return
    if claim == "research" or label == "research":
        return
    data = getattr(trainer, "data", None)
    identity_fn = getattr(data, "identity", None)
    identity = identity_fn() if callable(identity_fn) else {}
    if not isinstance(identity, dict):
        raise RuntimeFailure("data identity is not a mapping")
    if claim == "official_faithful":
        profile = identity.get("profile")
        uses_test = identity.get("selection_uses_test")
        if profile not in (None, "official_faithful"):
            raise RuntimeFailure("official_faithful data profile must be official_faithful")
        if uses_test is False:
            raise RuntimeFailure("official_faithful selection uses the dataset test split as validation")
    elif claim == "leakage_safe":
        profile = identity.get("profile")
        uses_test = identity.get("selection_uses_test")
        if profile not in (None, "leakage_safe"):
            raise RuntimeFailure("leakage_safe data profile must be leakage_safe")
        if uses_test is True:
            raise RuntimeFailure("leakage_safe selection keeps the test split out of best-checkpoint decisions")
    if isinstance(label, str) and label:
        data_claim = identity.get("claim_label")
        if data_claim == "official_reproduction":
            raise RuntimeFailure("semambapp.gsr.official is blocked; refusing official_reproduction")
        if data_claim is not None and data_claim != label:
            raise RuntimeFailure(f"data claim_label {data_claim} does not match trainer claim_label {label}")


def check_training_batch(batch: Batch, *, sample_rate: int, teachers: tuple[str, ...]) -> None:
    """Check one collated micro-batch against the model contract."""

    noisy = batch.inputs
    target = batch.targets
    if not isinstance(noisy, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise RuntimeFailure("training batch is missing waveforms")
    if noisy.ndim != 2 or target.shape != noisy.shape:
        raise RuntimeFailure(f"expected matching [batch, time] waveforms, got {tuple(noisy.shape)} and {tuple(target.shape)}")
    if noisy.shape[0] < 1 or noisy.shape[1] < 1:
        raise RuntimeFailure("training batch is empty")
    if not torch.isfinite(noisy).all() or not torch.isfinite(target).all():
        raise RuntimeFailure("training batch contains a non-finite sample")
    if not noisy.is_floating_point() or not target.is_floating_point():
        raise RuntimeFailure("training batch waveforms must be floating point")
    if batch.lengths is None:
        raise RuntimeFailure("training batch has no valid lengths")
    lengths = batch.lengths
    if lengths.ndim != 1 or int(lengths.shape[0]) != int(noisy.shape[0]):
        raise RuntimeFailure("valid lengths do not match the batch")
    if int(lengths.min()) < 1 or int(lengths.max()) > int(noisy.shape[1]):
        raise RuntimeFailure("valid length does not fit the collated waveform")
    if batch.sample_rate is None or int(batch.sample_rate) != int(sample_rate):
        raise RuntimeFailure(f"batch sample_rate {batch.sample_rate} does not match the model requirement {sample_rate}")
    if batch.teacher_kind not in teachers:
        raise RuntimeFailure(f"batch teacher {batch.teacher_kind} is not accepted by this model")


class _PairView:
    def __init__(self, data: AudioPairData, epoch: int, start_update: int, updates: int) -> None:
        self.updates = updates
        self.plan_hash = data.plan_hash
        self._data = data
        self._epoch = epoch
        self._start = start_update

    def batch(self, update_index: int) -> list[Batch]:
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise StateError(f"update index {update_index!r} is not an integer")
        if update_index < self._start:
            raise StateError(f"update index {update_index} is before start_update {self._start}")
        if update_index < 0 or update_index >= self.updates:
            raise StateError(f"update index {update_index} is outside 0..{self.updates - 1}")
        return [self._data._batch_at(update_index)]


class AudioPairData:
    """Fixed noisy/target waveforms addressed by update index.

    The same index returns the same batch. Nothing in this class reads a
    dataset name, a recipe, or a sample rate other than the one given here.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        sample_rate: int,
        teacher: str,
        batch_size: int = 1,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not pairs:
            raise ValueError("audio pairs are empty")
        stored: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for noisy, target in pairs:
            length = check_waveforms(noisy, target, sample_rate=sample_rate, teacher=teacher)
            stored.append(
                (
                    torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32)).clone(),
                    torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)).clone(),
                    length,
                )
            )
        self.sample_rate = int(sample_rate)
        self.teacher = teacher
        self.batch_size = batch_size
        self.group_size = 1
        self._pairs = tuple(stored)
        if len(self._pairs) < self.batch_size:
            raise ValueError("examples must be at least batch_size")
        self.epoch = 0
        self.update_index = 0
        self.world_size = 1
        self.rank = 0
        self.plan_hash = self._plan_hash()
        self._closed = False

    def identity(self) -> dict[str, object]:
        return {
            "class": "src.experiments.data.pair.AudioPairData",
            "sample_rate": self.sample_rate,
            "teacher": self.teacher,
            "examples": len(self._pairs),
            "batch_size": self.batch_size,
            "world_size": self.world_size,
        }

    def setup(self, runtime: object) -> None:
        device = getattr(runtime, "device", torch.device("cpu"))
        world_size = getattr(runtime, "world_size", 1)
        rank = getattr(runtime, "rank", 0)
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise StateError("runtime world_size must be a positive integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
            raise StateError("runtime rank is outside this world")
        self.world_size = world_size
        self.rank = rank
        self._pairs = tuple(
            (noisy.to(device), target.to(device), length) for noisy, target, length in self._pairs
        )
        self.plan_hash = self._plan_hash()

    def train_epoch(self, epoch: int, *, start_update: int) -> _PairView:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        updates = len(self._pairs) // (self.batch_size * self.world_size)
        if isinstance(start_update, bool) or not isinstance(start_update, int):
            raise StateError(f"start_update {start_update!r} is not an integer")
        if start_update < 0 or start_update > updates:
            raise StateError(f"start_update {start_update} is outside 0..{updates}")
        return _PairView(self, epoch, start_update, updates)

    def validation_epoch(self, epoch: int):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError(f"epoch {epoch!r} is not a valid index")
        updates = len(self._pairs) // (self.batch_size * self.world_size)
        return iter(self._batch_at(index) for index in range(updates))

    def state_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "update_index": self.update_index,
            "plan_hash": self.plan_hash,
            "world_size": self.world_size,
            "identity": self.identity(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("identity") != self.identity() or state.get("world_size") != self.world_size:
            raise StateError("data state does not match this data module")
        if state.get("plan_hash") != self.plan_hash:
            raise StateError("data plan_hash does not match this data module")
        epoch = state.get("epoch")
        update_index = state.get("update_index")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise StateError("data epoch is invalid")
        if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
            raise StateError("data update_index is invalid")
        self.epoch = epoch
        self.update_index = update_index

    def close(self) -> None:
        self._closed = True

    def _batch_at(self, update_index: int) -> Batch:
        stride = self.batch_size * self.world_size
        start = update_index * stride + self.rank * self.batch_size
        chosen = self._pairs[start : start + self.batch_size]
        if len(chosen) != self.batch_size:
            raise DataError("audio pair batch is outside the example list")
        width = max(int(noisy.shape[0]) for noisy, _target, _length in chosen)
        noisy = torch.zeros(self.batch_size, width, dtype=chosen[0][0].dtype, device=chosen[0][0].device)
        target = torch.zeros(self.batch_size, width, dtype=chosen[0][1].dtype, device=chosen[0][1].device)
        lengths = []
        ids = []
        for row, (wave, teacher, length) in enumerate(chosen):
            noisy[row, : wave.shape[0]] = wave
            target[row, : teacher.shape[0]] = teacher
            lengths.append(length)
            ids.append(f"pair-{start + row}")
        return Batch(
            inputs=noisy,
            targets=target,
            sample_id=",".join(ids),
            realization_id=f"epoch-pairs:{','.join(ids)}",
            lengths=torch.tensor(lengths, dtype=torch.long, device=noisy.device),
            sample_ids=tuple(ids),
            realization_ids=tuple(ids),
            teacher_kind=self.teacher,
            sample_rate=self.sample_rate,
        )

    def _plan_hash(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for noisy, target, length in self._pairs:
            digest.update(noisy.detach().cpu().numpy().tobytes())
            digest.update(target.detach().cpu().numpy().tobytes())
            digest.update(str(length).encode("utf-8"))
        digest.update(str(self.batch_size).encode("utf-8"))
        digest.update(str(self.world_size).encode("utf-8"))
        digest.update(str(self.sample_rate).encode("utf-8"))
        digest.update(self.teacher.encode("utf-8"))
        return digest.hexdigest()


class WavePairData(AudioPairData):
    """Paired wavs addressed by filename. The teacher is the caller's declaration.

    Files are read at ``sample_rate``. A different rate is refused instead of
    resampled. This is a dataset adapter: trainers do not grow a name list
    when a directory is added.
    """

    def __init__(self, root: str, *, sample_rate: int, teacher: str, batch_size: int = 1) -> None:
        directory = Path(root)
        pairs = _load_wave_pairs(directory, sample_rate)
        super().__init__(pairs, sample_rate=sample_rate, teacher=teacher, batch_size=batch_size)
        self.root = str(directory)

    def identity(self) -> dict[str, object]:
        recorded = super().identity()
        recorded["class"] = "src.experiments.data.pair.WavePairData"
        recorded["root"] = self.root
        return recorded


def _load_wave_pairs(root: Path, sample_rate: int) -> list[tuple[np.ndarray, np.ndarray]]:
    import soundfile as sf

    noisy_dir = root / "noisy"
    target_dir = root / "target"
    if not noisy_dir.is_dir() or not target_dir.is_dir():
        raise DataError(f"{root} needs noisy/ and target/ wav directories")
    names = sorted(path.name for path in noisy_dir.glob("*.wav"))
    if not names:
        raise DataError(f"{noisy_dir} has no wav files")
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for name in names:
        target_path = target_dir / name
        if not target_path.is_file():
            raise DataError(f"target wav is missing for {name}")
        noisy, noisy_rate = _read_wav(sf, noisy_dir / name)
        target, target_rate = _read_wav(sf, target_path)
        if noisy_rate != sample_rate or target_rate != sample_rate:
            raise DataError(
                f"{name} is {noisy_rate}/{target_rate} Hz; refusing to resample to {sample_rate}"
            )
        if noisy.shape != target.shape:
            raise DataError(f"{name} noisy and target lengths differ")
        pairs.append((noisy, target))
    return pairs


def _read_wav(sf: object, path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)  # type: ignore[attr-defined]
    if audio.size < 1 or int(audio.shape[1]) != 1:
        raise DataError(f"{path} must be a non-empty mono wav")
    mono = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    if not np.isfinite(mono).all():
        raise DataError(f"{path} is non-finite")
    return mono, int(rate)


def _as_float(value: np.ndarray | torch.Tensor, label: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise DataError(f"{label} is not a waveform")
    return np.asarray(array, dtype=np.float64)
