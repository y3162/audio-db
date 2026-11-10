"""Training pairs rendered from recipe JSON.

The teacher is the recipe reference. ``dry_clean``, ``reverberant_clean``, and
``none`` stay distinct: a missing teacher is an error, and one reference is
never substituted for another. Sample identity is the recipe hash (``case_id``),
not a database row number.
"""

from __future__ import annotations

import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.data.constants import CORPORA_ROOT, DUCKDB_MAIN_PATH
from src.data.recipe import case_id_for
from src.evaluation.audio.render import render_recipe
from src.evaluation.datasets.evaldb import EvaluationDB
from src.experiments.calc.sampler import crop_start
from src.experiments.calc.errors import TrainingError

# Named sets registered for evaluation. Training and validation both refuse them.
EVALUATION_SET_NAMES = frozenset(
    {
        "example",
        "libri_demand_snr0",
        "libri_testclean_full_smoke",
        "chime3_et05_real_ch5",
        "chime3_et05_real_ch0",
        "chime3_et05_simu_ch5",
        "chime3_tr05_org",
        "voicebank_demand_test",
    }
)
_TRAIN_FORBIDDEN_SPLITS = frozenset({"test", "test-clean", "et05", "valid", "val", "validation", "dev"})
_VALIDATION_FORBIDDEN_SPLITS = frozenset({"train", "training", "test", "test-clean", "et05"})


@dataclass(frozen=True)
class TrainExample:
    case_id: str
    recipe: dict[str, Any]
    split: str | None
    dataset: str
    reference_name: str


class RecipeDataset:
    """One named split. Waveforms are rendered when a batch is collated."""

    def __init__(
        self,
        examples: list[TrainExample],
        root: Path,
        *,
        name: str,
        role: str,
        seed: int,
        sample_rate: int,
        segment_size: int | None,
        training: bool,
        database: str | None,
        rms_normalize: bool = True,
        audio_catalog: str | Path | None = None,
        corpora_root: str | Path | None = None,
    ) -> None:
        if role not in ("train", "validation"):
            raise TrainingError(f"unknown split role {role!r}")
        if not examples:
            raise TrainingError(f"{role} set {name} is empty")
        self.examples = list(examples)
        self.root = Path(root)
        self.name = name
        self.role = role
        self.seed = seed
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.training = training
        self.database = database
        self.rms_normalize = bool(rms_normalize)
        self.audio_catalog = Path(DUCKDB_MAIN_PATH if audio_catalog is None else audio_catalog)
        self.corpora_root = Path(CORPORA_ROOT if corpora_root is None else corpora_root)
        self.epoch = 0
        # Parallel loads do not draw from a shared RNG. Crop and order are
        # functions of the case id, seed, and epoch.
        self.load_workers = 1
        self._load_pool: ThreadPoolExecutor | None = None
        self._length_cache: list[int] | None = None

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise TrainingError("epoch must be zero or positive")
        self.epoch = epoch

    def identity(self) -> dict[str, Any]:
        ids = [example.case_id for example in self.examples]
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        identity = {
            "name": self.name,
            "role": self.role,
            "database": self.database,
            "size": len(ids),
            "case_ids_sha256": digest,
            "sample_rate": self.sample_rate,
            "segment_size": self.segment_size,
            "training": self.training,
            "seed": self.seed,
        }
        # Omit the default so checkpoints saved before this flag still match.
        # Official TF-GridNet leaves speech_volume_normalize unset, so that
        # mode records the disabled flag and will not resume an RMS run.
        if not self.rms_normalize:
            identity["rms_normalize"] = False
        return identity

    def collate(self, indices: list[int]) -> dict[str, Any]:
        if self.load_workers > 1 and len(indices) > 1:
            loaded = list(self._workers().map(self._load, indices))
        else:
            loaded = [self._load(index) for index in indices]
        clean_rows = []
        noisy_rows = []
        case_ids = []
        for clean, noisy, case_id in loaded:
            clean_rows.append(clean)
            noisy_rows.append(noisy)
            case_ids.append(case_id)
        try:
            clean = torch.stack(clean_rows)
            noisy = torch.stack(noisy_rows)
        except RuntimeError as exc:
            raise TrainingError("batch waveforms have different lengths") from exc
        return {"clean": clean, "noisy": noisy, "case_id": case_ids}

    def case_ids(self) -> list[str]:
        return [example.case_id for example in self.examples]

    def waveform_lengths(self) -> list[int]:
        """Sample counts after resampling, before a crop.

        Lengths come from one ``corpora.audio_files`` query. The files are not
        decoded, and this method does not draw random numbers or change epoch state.
        """
        if self._length_cache is None:
            self._length_cache = _lengths_from_catalog(self)
        return list(self._length_cache)

    def collate_chunks(self, chunks) -> dict[str, Any]:
        """Slice already normalized utterances. Chunks from one case load once."""
        index_of = {example.case_id: index for index, example in enumerate(self.examples)}
        loaded: dict[int, tuple[torch.Tensor, torch.Tensor, str]] = {}
        clean_rows = []
        noisy_rows = []
        case_ids = []
        for chunk in chunks:
            index = index_of.get(chunk.case_id)
            if index is None:
                raise TrainingError(f"chunk case {chunk.case_id} is not in {self.name}")
            if index not in loaded:
                loaded[index] = self._load(index, crop=False)
            clean, noisy, case_id = loaded[index]
            if chunk.stop > int(clean.numel()) or chunk.start < 0 or chunk.stop <= chunk.start:
                raise TrainingError(
                    f"chunk {chunk.start}:{chunk.stop} is outside {case_id} ({clean.numel()} samples)"
                )
            clean_rows.append(clean[chunk.start:chunk.stop].contiguous())
            noisy_rows.append(noisy[chunk.start:chunk.stop].contiguous())
            case_ids.append(case_id)
        return {"clean": torch.stack(clean_rows), "noisy": torch.stack(noisy_rows), "case_id": case_ids}

    def _load(self, index: int, *, crop: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, str]:
        example = self.examples[index]
        if case_id_for(example.recipe) != example.case_id:
            raise TrainingError(f"case_id does not match the recipe hash: {example.case_id}")
        if example.reference_name not in ("dry_clean", "reverberant_clean"):
            raise TrainingError(
                f"case {example.case_id} has no teacher reference ({example.reference_name})"
            )
        rendered = render_recipe(example.recipe, self.root)
        if rendered.reference is None or rendered.reference_name == "none":
            raise TrainingError(f"case {example.case_id} has no teacher waveform")
        if rendered.reference_name != example.reference_name:
            raise TrainingError(
                f"case {example.case_id} rendered {rendered.reference_name}, "
                f"not {example.reference_name}"
            )
        if rendered.sample_rate < 1 or self.sample_rate < 1:
            raise TrainingError(
                f"case {example.case_id} has sample rate {rendered.sample_rate} -> {self.sample_rate}"
            )
        noisy = _mono(rendered.observed)
        clean = _mono(rendered.reference)
        if noisy.shape != clean.shape:
            raise TrainingError(
                f"case {example.case_id} teacher length {tuple(clean.shape)} "
                f"does not match the input {tuple(noisy.shape)}"
            )
        noisy = _match_rate(noisy, rendered.sample_rate, self.sample_rate, example.case_id)
        clean = _match_rate(clean, rendered.sample_rate, self.sample_rate, example.case_id)
        if noisy.shape != clean.shape:
            raise TrainingError(
                f"case {example.case_id} resampled lengths differ: "
                f"{tuple(noisy.shape)} vs {tuple(clean.shape)}"
            )
        if self.rms_normalize:
            noisy, clean = _normalize(noisy, clean)
        if crop is None:
            crop = self.training
        if crop:
            if self.segment_size is None:
                raise TrainingError("training requires segment_size")
            noisy, clean = _segment(noisy, clean, example.case_id, self.seed, self.epoch, self.segment_size)
        return clean, noisy, example.case_id

    def _workers(self) -> ThreadPoolExecutor:
        if self._load_pool is None:
            self._load_pool = ThreadPoolExecutor(
                max_workers=self.load_workers,
                thread_name_prefix="audio-db-load",
            )
        return self._load_pool


def load_recipe_dataset(
    database: str | Path,
    set_name: str,
    *,
    role: str,
    seed: int,
    sample_rate: int,
    segment_size: int | None,
    training: bool,
    rms_normalize: bool = True,
) -> RecipeDataset:
    """Load a named set and reject evaluation data and missing teachers."""
    path = Path(database)
    if path.name == "resource.duckdb":
        raise TrainingError("refusing to train from the production database resource.duckdb")
    database_handle = EvaluationDB(path)
    loaded = database_handle.load_dataset(set_name)
    if not loaded:
        raise TrainingError(f"named set {set_name} is empty")
    root = Path(loaded[0].metadata["audio_root"])
    examples: list[TrainExample] = []
    for sample in loaded:
        recipe = sample.metadata.get("recipe")
        if not isinstance(recipe, dict):
            raise TrainingError(f"case {sample.utterance_id} has no recipe")
        reference = recipe.get("output", {}).get("reference")
        rejection = _set_rejection(
            set_name,
            role=role,
            split=sample.split,
            dataset=sample.dataset,
            metadata=sample.metadata,
        )
        if rejection is not None:
            raise TrainingError(f"{rejection} (case {sample.utterance_id})")
        if reference == "none":
            raise TrainingError(
                f"case {sample.utterance_id} reference is none, so it has no teacher"
            )
        if reference not in ("dry_clean", "reverberant_clean"):
            raise TrainingError(f"case {sample.utterance_id} has unknown reference {reference!r}")
        if case_id_for(recipe) != sample.utterance_id:
            raise TrainingError(f"case {sample.utterance_id} is not the recipe hash")
        examples.append(
            TrainExample(
                case_id=sample.utterance_id,
                recipe=recipe,
                split=sample.split,
                dataset=sample.dataset,
                reference_name=reference,
            )
        )
    return RecipeDataset(
        examples,
        root,
        name=set_name,
        role=role,
        seed=seed,
        sample_rate=sample_rate,
        segment_size=segment_size,
        training=training,
        database=str(database),
        rms_normalize=rms_normalize,
    )


def _set_rejection(
    set_name: str,
    *,
    role: str,
    split: str | None,
    dataset: str,
    metadata: dict[str, Any],
) -> str | None:
    if set_name in EVALUATION_SET_NAMES:
        return f"evaluation set {set_name} cannot be used for {role}"
    split_name = "" if split is None else str(split).casefold()
    if role == "train" and split_name in _TRAIN_FORBIDDEN_SPLITS:
        return f"split {split} cannot be used as training data"
    if role == "validation" and split_name in _VALIDATION_FORBIDDEN_SPLITS:
        return f"split {split} cannot be used as validation data"
    blob = " ".join(
        [
            set_name,
            dataset or "",
            split_name,
            str(metadata.get("subset", "")),
            str(metadata.get("corpus", "")),
        ]
    ).casefold()
    if "librispeech" in blob and "test-clean" in blob:
        return "LibriSpeech test-clean cannot be used for training or validation"
    if "chime3" in blob and "et05" in blob:
        return "CHiME3 et05 cannot be used for training or validation"
    return None


def _match_rate(wave: torch.Tensor, source_rate: int, target_rate: int, case_id: str) -> torch.Tensor:
    """Resample at load time. The recipe and case_id stay at the file rate."""
    if source_rate == target_rate:
        return wave
    if source_rate < 1 or target_rate < 1:
        raise TrainingError(f"case {case_id} cannot resample {source_rate} -> {target_rate}")
    import torchaudio

    converted = torchaudio.functional.resample(wave, source_rate, target_rate)
    expected = int(round(wave.numel() * target_rate / source_rate))
    if abs(int(converted.numel()) - expected) > max(1, target_rate // 100):
        raise TrainingError(
            f"case {case_id} resampled length {converted.numel()} is far from {expected}"
        )
    if converted.numel() < 1 or not torch.isfinite(converted).all():
        raise TrainingError(f"case {case_id} resample produced an empty or non-finite waveform")
    return converted


def _mono(waveform: torch.Tensor) -> torch.Tensor:
    wave = waveform.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if wave.ndim == 2 and wave.shape[0] == 1:
        wave = wave[0]
    if wave.ndim != 1 or wave.numel() == 0:
        raise TrainingError(f"expected a mono waveform, got shape {tuple(waveform.shape)}")
    return wave


def _normalize(noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    energy = torch.sum(noisy * noisy)
    if not torch.isfinite(energy) or float(energy) <= 0.0:
        raise TrainingError("noisy waveform has no energy")
    scale = torch.sqrt(noisy.new_tensor(float(noisy.numel())) / energy)
    return noisy * scale, clean * scale


def _segment(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    case_id: str,
    seed: int,
    epoch: int,
    segment_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noisy.numel() >= segment_size:
        start = crop_start(case_id, seed, epoch, int(noisy.numel()), segment_size)
        stop = start + segment_size
        return noisy[start:stop].contiguous(), clean[start:stop].contiguous()
    pad = segment_size - int(noisy.numel())
    return F.pad(noisy, (0, pad)).contiguous(), F.pad(clean, (0, pad)).contiguous()


def resampled_frame_count(frames: int, source_rate: int, target_rate: int) -> int:
    """Match ``torchaudio.functional.resample`` output length without a waveform.

    After reducing the rates by their gcd, torchaudio keeps
    ``ceil(new_rate * frames / old_rate)`` samples.
    """
    if frames < 1 or source_rate < 1 or target_rate < 1:
        raise TrainingError(f"cannot resample {frames} frames from {source_rate} to {target_rate}")
    if source_rate == target_rate:
        return frames
    divisor = math.gcd(source_rate, target_rate)
    source = source_rate // divisor
    target = target_rate // divisor
    return (target * frames + source - 1) // source


def _lengths_from_catalog(dataset: RecipeDataset) -> list[int]:
    """One catalog query, then lengths in dataset index order."""
    requested: list[tuple[int, str]] = []
    for index, example in enumerate(dataset.examples):
        for name in _duration_sources(example):
            stored = example.recipe["sources"][name]["path"]
            requested.append((index, _corpus_key(dataset.root, stored, dataset.corpora_root)))
    metadata = _audio_metadata(dataset.audio_catalog, [path for _, path in requested])
    lengths: list[int | None] = [None] * len(dataset)
    for index, path in requested:
        example = dataset.examples[index]
        frames, file_rate = metadata[path]
        output = example.recipe["output"]
        used = _frames_before_resample(frames, file_rate, output.get("max_seconds"))
        rendered = resampled_frame_count(used, file_rate, int(output["sample_rate"]))
        length = resampled_frame_count(rendered, int(output["sample_rate"]), dataset.sample_rate)
        current = lengths[index]
        if current is None:
            lengths[index] = length
        elif current != length:
            raise TrainingError(
                f"case {example.case_id} sources resample to different lengths ({current} and {length})"
            )
    if any(item is None for item in lengths):
        raise TrainingError(f"{dataset.name} is missing a waveform length")
    return [int(item) for item in lengths]


def _duration_sources(example: TrainExample) -> list[str]:
    """Sources whose resampled lengths are the utterance length."""
    sources = example.recipe["sources"]
    operations = example.recipe["operations"]
    reference = example.reference_name
    if operations:
        if "clean" not in sources:
            raise TrainingError(f"case {example.case_id} operations have no clean source")
        return ["clean"]
    names: list[str] = []
    if "noisy" in sources:
        names.append("noisy")
    elif "clean" in sources:
        names.append("clean")
    else:
        raise TrainingError(f"case {example.case_id} has no clean or noisy source")
    if reference == "dry_clean" and "clean" not in names:
        if "clean" not in sources:
            raise TrainingError(f"case {example.case_id} has no clean teacher")
        names.append("clean")
    return names


def _frames_before_resample(frames: int, source_rate: int, max_seconds: float | None) -> int:
    if max_seconds is None:
        return frames
    keep = max(1, int(float(max_seconds) * source_rate))
    return min(frames, keep)


def _corpus_key(root: Path, stored: str, corpora_root: Path) -> str:
    audio = Path(os.path.normpath((root / stored).absolute()))
    base = Path(os.path.normpath(corpora_root if corpora_root.is_absolute() else corpora_root.absolute()))
    try:
        relative = audio.relative_to(base)
    except ValueError as exc:
        raise TrainingError(f"audio is outside {base}: {stored}") from exc
    text = relative.as_posix()
    if text == ".." or text.startswith("../"):
        raise TrainingError(f"audio is outside {base}: {stored}")
    return text


def _audio_metadata(catalog: Path, paths: list[str]) -> dict[str, tuple[int, int]]:
    unique = list(dict.fromkeys(paths))
    if not unique:
        return {}
    if not catalog.is_file():
        raise TrainingError(f"audio catalog not found: {catalog}")
    import duckdb

    connection = duckdb.connect(str(catalog), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT path, frames, sample_rate
            FROM corpora.audio_files
            WHERE path IN (SELECT UNNEST(?::VARCHAR[]))
            """,
            [unique],
        ).fetchall()
    except Exception as exc:
        raise TrainingError(f"audio catalog query failed: {catalog}") from exc
    finally:
        connection.close()
    found = {str(path): (int(frames), int(rate)) for path, frames, rate in rows}
    missing = [path for path in unique if path not in found]
    if missing:
        preview = ", ".join(missing[:3])
        raise TrainingError(f"audio catalog is missing {len(missing)} file(s): {preview}")
    return found
