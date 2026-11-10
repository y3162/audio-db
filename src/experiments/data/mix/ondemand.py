"""Reconstruct MS-SNSD mixtures from a snapshot while a batch is collated.

The realization document and the source-file hashes are the stored data.
This module does not write wavs and does not keep a waveform cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.experiments.data.build.ms_snsd import _pcm16_roundtrip, render_realization
from src.experiments.data.errors import ReproductionError
from src.experiments.data.ids import digest
from src.experiments.data.sampling.planner import chunk_case_length
from src.experiments.data.snapshot_provenance import snapshot_provenance_parts
from src.experiments.data.assets.store import AssetStore
from src.experiments.data.snapshot import LoadedSnapshot


class OnDemandChunkData:
    """Shape-file order, with each chunk mixed from the source files."""

    def __init__(self, store: AssetStore, *, slot: str, settings: dict[str, Any]) -> None:
        dataset = store.dataset
        rule = dataset.get("rule_json") or {}
        if dataset.get("algorithm") != "pregenerated.ms_snsd_tfgridnet" or rule.get("delivery") != "on_demand":
            raise ReproductionError(f"slot {slot} is not an on-demand MS-SNSD snapshot")
        if dataset.get("declared_status") != "ready":
            raise ReproductionError(f"dataset {dataset.get('name')} is not ready")
        if rule.get("normalize") is not None:
            raise ReproductionError("MS-SNSD audio is not RMS-normalized at load time")
        self.store = store
        self.slot = slot
        self.ids = store.case_ids(slot)
        self.lengths = [_length(store, case_id) for case_id in self.ids]
        self.split = "test" if slot == "test.cases" else "train"
        self.load_workers = int(settings.get("load_workers", 1))
        self.verify_on_read = bool(settings.get("verify_on_read", False))
        self.speech, self.noise = ms_snsd_rendering(store, slot)
        self.algorithm = str(dataset["algorithm"])
        self.dataset_id = str(dataset["dataset_id"])
        self.name = str(dataset["name"])
        self.version = int(dataset["version"])
        self.snapshot_hash = store.snapshot_hash

    def case_ids(self) -> list[str]:
        return list(self.ids)

    def waveform_lengths(self) -> list[int]:
        """Planned sample counts. Source files are not read."""
        return list(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ReproductionError("epoch must be non-negative")

    def identity(self) -> dict[str, Any]:
        dataset = self.store.dataset
        return {
            "pipeline": "espnet_chunk_v1",
            "source": "ondemand_conditions",
            "delivery": "on_demand",
            "cache": "none",
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
            "case order is the ESPnet shape-file order. Audio is reconstructed from the "
            "realization document and the source wavs. No generated wav is required."
        )
        return recorded

    def provenance_parts(self) -> dict[str, Any]:
        parts = snapshot_provenance_parts(self.store)
        parts["pipeline"] = "reproduction"
        parts["delivery"] = "on_demand"
        return parts

    def collate_chunks(self, chunks) -> dict[str, Any]:
        """Mix each chunk from its source files. Repeated calls repeat the mix."""
        index_of = {case_id: index for index, case_id in enumerate(self.ids)}
        clean_rows = []
        noisy_rows = []
        case_ids = []
        for chunk in chunks:
            case_id = chunk.case_id
            index = index_of.get(case_id)
            if index is None:
                raise ReproductionError(f"chunk case {case_id} is not in {self.slot}")
            clean, noisy = self._pair(case_id)
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
        return render_ms_snsd_pair(
            self.store,
            case_id,
            speech=self.speech,
            noise=self.noise,
            verify_on_read=self.verify_on_read,
        )


def open_ondemand_datasets(
    loaded: LoadedSnapshot,
    *,
    load_workers: int = 1,
    verify_on_read: bool = False,
    derived_root: str | Path,
    corpora_root: str | Path,
    generated_root: str | Path,
) -> tuple[OnDemandChunkData, OnDemandChunkData]:
    algorithm = loaded.dataset.get("algorithm")
    if algorithm == "pregenerated.dns_tflocoformer":
        from src.experiments.data.mix.dns_ondemand import open_dns_datasets

        return open_dns_datasets(
            loaded,
            load_workers=load_workers,
            verify_on_read=verify_on_read,
            derived_root=derived_root,
            corpora_root=corpora_root,
            generated_root=generated_root,
        )
    if algorithm != "pregenerated.ms_snsd_tfgridnet":
        raise ReproductionError(f"no on-demand reader for {algorithm}")
    store = AssetStore.from_snapshot(
        loaded,
        derived=derived_root,
        corpora=corpora_root,
        generated=generated_root,
    )
    settings = {"load_workers": load_workers, "verify_on_read": verify_on_read}
    return (
        OnDemandChunkData(store, slot="train.cases", settings=settings),
        OnDemandChunkData(store, slot="valid.cases", settings=settings),
    )


def ms_snsd_rendering(store: AssetStore, slot: str) -> tuple[dict[str, Path], dict[str, Path]]:
    """Speech and noise paths for one MS-SNSD slot. Validation uses the train files."""

    split = "test" if slot == "test.cases" else "train"
    return _source_map(store, "clean", split), _source_map(store, "noise", split)


def render_ms_snsd_pair(
    store: AssetStore,
    case_id: str,
    *,
    speech: dict[str, Path],
    noise: dict[str, Path],
    verify_on_read: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """PCM_16 clean and noisy for one realization. The caller slices the chunk."""

    document = store.docs.get(case_id)
    if document is None:
        raise ReproductionError(f"case {case_id} has no realization document")
    if verify_on_read:
        _check_source_hashes(store, document, speech, noise)
    rendered = render_realization(document, {"speech": speech, "noise": noise})
    clean = _pcm16_roundtrip(rendered["clean"]).astype(np.float32)
    noisy = _pcm16_roundtrip(rendered["noisy"]).astype(np.float32)
    if clean.shape != noisy.shape or int(clean.shape[0]) != chunk_case_length(store, case_id):
        raise ReproductionError(f"case {case_id} rendered length does not match the plan")
    if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ReproductionError(f"case {case_id} rendered non-finite samples")
    return clean, noisy


def _length(store: AssetStore, case_id: str) -> int:
    return chunk_case_length(store, case_id)


def _source_map(store: AssetStore, kind: str, split: str) -> dict[str, Path]:
    prefix = f"corpora/MS-SNSD/{kind}_{split}/"
    found: dict[str, Path] = {}
    for asset_id in store.assets:
        if not asset_id.startswith(prefix):
            continue
        name = asset_id.rsplit("/", 1)[-1]
        if name in found:
            raise ReproductionError(f"duplicate MS-SNSD source name {name}")
        found[name] = store.path(asset_id)
    if not found:
        raise ReproductionError(f"snapshot has no {kind}_{split} source files")
    return found


def _check_source_hashes(store: AssetStore, document: dict[str, Any], speech: dict[str, Path], noise: dict[str, Path]) -> None:
    import hashlib

    seen: set[str] = set()
    for kind, mapping in (("speech", speech), ("noise", noise)):
        for piece in document["sources"][kind]:
            if "name" not in piece:
                continue
            name = str(piece["name"])
            if name in seen:
                continue
            seen.add(name)
            path = mapping[name]
            digest_value = hashlib.sha256(path.read_bytes()).hexdigest()
            asset = _asset_for_path(store, path)
            if digest_value != asset.get("sha256"):
                raise ReproductionError(f"sha256 mismatch for {path}")


def _asset_for_path(store: AssetStore, path: Path) -> dict[str, Any]:
    for asset_id, asset in store.assets.items():
        if store.path(asset_id) == path:
            return asset
    raise ReproductionError(f"snapshot has no asset for {path}")
