"""Read a pregenerated snapshot in ESPnet shape-file order.

The trainer's chunk iterator calls ``case_ids``, ``waveform_lengths``, and
``collate_chunks``. This module does not open DuckDB and does not shuffle:
shuffle belongs to ``plan_chunk_epoch``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest
from src.reproduction.paths import DEFAULT_CORPORA, DEFAULT_DERIVED, DEFAULT_GENERATED, DEFAULT_SNAPSHOTS
from src.reproduction.pipeline.provenance import snapshot_provenance_parts
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.pipeline.training_data import find_snapshot_dir
from src.reproduction.snapshot_reader import load_snapshot


class GeneratedChunkData:
    """One case-set slot of a pregenerated snapshot, in shape-file order."""

    def __init__(self, store: AssetStore, *, slot: str, settings: dict[str, Any]) -> None:
        dataset = store.dataset
        if dataset.get("algorithm") != "pregenerated.ms_snsd_tfgridnet" or dataset.get("kind") != "pregenerated":
            raise ReproductionError(
                f"slot {slot} expected pregenerated.ms_snsd_tfgridnet, found {dataset.get('algorithm')}"
            )
        if dataset.get("declared_status") != "ready":
            raise ReproductionError(f"dataset {dataset.get('name')} is not ready")
        if (dataset.get("rule_json") or {}).get("normalize") is not None:
            raise ReproductionError("MS-SNSD pregenerated audio is not RMS-normalized at load time")
        self.store = store
        self.slot = slot
        self.ids = store.case_ids(slot)
        self.lengths = [_length(store, case_id) for case_id in self.ids]
        self.load_workers = int(settings.get("load_workers", 1))
        self.verify_on_read = bool(settings.get("verify_on_read", False))
        self.algorithm = str(dataset["algorithm"])
        self.dataset_id = str(dataset["dataset_id"])
        self.name = str(dataset["name"])
        self.version = int(dataset["version"])
        self.snapshot_hash = store.snapshot_hash
        self._verified: set[str] = set()

    def case_ids(self) -> list[str]:
        return list(self.ids)

    def waveform_lengths(self) -> list[int]:
        """Sample counts stored in the snapshot. Files are not decoded."""
        return list(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ReproductionError("epoch must be non-negative")

    def identity(self) -> dict[str, Any]:
        dataset = self.store.dataset
        return {
            "pipeline": "espnet_chunk_v1",
            "source": "pregenerated_snapshot",
            "name": self.name,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "snapshot_hash": self.snapshot_hash,
            "protocol_id": dataset.get("protocol_id"),
            "algorithm": self.algorithm,
            "profile": dataset.get("profile"),
            "claim_label": dataset.get("claim_label"),
            "selection_uses_test": bool(dataset.get("selection_uses_test")),
            "slot": self.slot,
            "case_ids_sha256": digest(self.ids),
            "length_sha256": digest(self.lengths),
            "normalize": None,
            "order": "shape_file",
        }

    def provenance(self) -> dict[str, Any]:
        recorded = self.identity()
        recorded["shape"] = (
            "case order is the ESPnet shape-file order: official utt ids sorted as sort -k1,1 "
            "after the 80/20 path split and the 0.1 s / 31 s duration filter"
        )
        return recorded

    def provenance_parts(self) -> dict[str, Any]:
        cached = getattr(self, "_provenance_parts_cache", None)
        if cached is None:
            cached = snapshot_provenance_parts(self.store)
            cached["pipeline"] = "reproduction"
            self._provenance_parts_cache = cached
        return cached

    def collate_chunks(self, chunks) -> dict[str, Any]:
        """Slice stored PCM_16. There is no utterance RMS."""
        index_of = {case_id: index for index, case_id in enumerate(self.ids)}
        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        clean_rows = []
        noisy_rows = []
        case_ids = []
        for chunk in chunks:
            case_id = chunk.case_id
            index = index_of.get(case_id)
            if index is None:
                raise ReproductionError(f"chunk case {case_id} is not in {self.slot}")
            if case_id not in loaded:
                loaded[case_id] = self._pair(case_id)
            clean, noisy = loaded[case_id]
            if chunk.start < 0 or chunk.stop > int(clean.shape[0]) or chunk.stop <= chunk.start:
                raise ReproductionError(
                    f"chunk {chunk.start}:{chunk.stop} is outside {case_id} ({clean.shape[0]} samples)"
                )
            clean_rows.append(torch.from_numpy(clean[chunk.start:chunk.stop].copy()))
            noisy_rows.append(torch.from_numpy(noisy[chunk.start:chunk.stop].copy()))
            case_ids.append(case_id)
        return {
            "clean": torch.stack(clean_rows),
            "noisy": torch.stack(noisy_rows),
            "case_id": case_ids,
        }

    def close(self) -> None:
        return None

    def _pair(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        case = self.store.case(case_id)
        clean = self._read(case["clean_asset_id"])
        noisy = self._read(case["noisy_asset_id"])
        if clean.shape != noisy.shape:
            raise ReproductionError(f"case {case_id} clean and noisy lengths differ")
        return clean, noisy

    def _read(self, asset_id: str) -> np.ndarray:
        asset = self.store.asset(asset_id)
        path = self.store.path(asset_id)
        if self.verify_on_read and asset_id not in self._verified:
            digest_value = _sha256(path)
            if digest_value != asset.get("sha256"):
                raise ReproductionError(f"sha256 mismatch for {path}")
            self._verified.add(asset_id)
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if int(rate) != int(asset["sample_rate"]):
            raise ReproductionError(f"{path} is {rate} Hz")
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim != 1:
            raise ReproductionError(f"{path} is not mono")
        if int(wave.shape[0]) != int(asset["frames"]):
            raise ReproductionError(f"{path} has {wave.shape[0]} samples; the snapshot says {asset['frames']}")
        if not np.isfinite(wave).all():
            raise ReproductionError(f"{path} is non-finite")
        return wave


def open_chunk_datasets(
    *,
    dataset: str,
    load_workers: int = 1,
    verify_on_read: bool = False,
    snapshot_root: str | Path | None = None,
    derived_root: str | Path | None = None,
    corpora_root: str | Path | None = None,
    generated_root: str | Path | None = None,
):
    """Train and valid views of one snapshot.

    ``delivery: on_demand`` reconstructs PCM from the realization document.
    A snapshot without that key still reads stored generated wavs.
    """
    root = Path(DEFAULT_SNAPSHOTS if snapshot_root is None else snapshot_root)
    loaded = load_snapshot(find_snapshot_dir(root, dataset))
    if (loaded.dataset.get("rule_json") or {}).get("delivery") == "on_demand":
        from src.reproduction.pipeline.ondemand import open_ondemand_datasets

        return open_ondemand_datasets(
            loaded,
            load_workers=load_workers,
            verify_on_read=verify_on_read,
            derived_root=DEFAULT_DERIVED if derived_root is None else derived_root,
            corpora_root=DEFAULT_CORPORA if corpora_root is None else corpora_root,
            generated_root=DEFAULT_GENERATED if generated_root is None else generated_root,
        )
    store = AssetStore.from_snapshot(
        loaded,
        derived=DEFAULT_DERIVED if derived_root is None else derived_root,
        corpora=DEFAULT_CORPORA if corpora_root is None else corpora_root,
        generated=DEFAULT_GENERATED if generated_root is None else generated_root,
    )
    settings = {"load_workers": load_workers, "verify_on_read": verify_on_read}
    return (
        GeneratedChunkData(store, slot="train.cases", settings=settings),
        GeneratedChunkData(store, slot="valid.cases", settings=settings),
    )


def _length(store: AssetStore, case_id: str) -> int:
    case = store.case(case_id)
    clean = int(store.asset(case["clean_asset_id"])["frames"])
    noisy = int(store.asset(case["noisy_asset_id"])["frames"])
    if clean != noisy or clean < 1:
        raise ReproductionError(f"case {case_id} clean and noisy durations differ")
    return clean


def _sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()
