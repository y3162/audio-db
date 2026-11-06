"""Reconstruct DNS mixtures from a snapshot while a batch is collated.

The realization document is the stored plan. Source wavs are read for the
requested span, mixed, and quantized to PCM_16. Nothing is written and the
waveform is not cached.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.reproduction.build.dns2020 import (
    ALGORITHM,
    RENDERER_ACCEPT,
    RENDERER_V1,
    RETRY_ACCEPT,
    RETRY_NONE,
    render_realization,
)
from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest
from src.reproduction.pipeline.provenance import snapshot_provenance_parts
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.snapshot_reader import LoadedSnapshot

_CLEAN_PREFIX = "corpora/DNS2020/datasets/clean/"
_NOISE_PREFIX = "corpora/DNS2020/datasets/noise/"


class DnsOnDemandChunkData:
    """Shape-file order. Each chunk is mixed from the stored DNS plan."""

    def __init__(self, store: AssetStore, *, slot: str, settings: dict[str, Any]) -> None:
        dataset = store.dataset
        rule = dataset.get("rule_json") or {}
        if dataset.get("algorithm") != ALGORITHM or rule.get("delivery") != "on_demand":
            raise ReproductionError(f"slot {slot} is not an on-demand DNS snapshot")
        if dataset.get("declared_status") != "ready":
            raise ReproductionError(f"dataset {dataset.get('name')} is not ready")
        if rule.get("normalize") is not None:
            raise ReproductionError("DNS audio is not RMS-normalized at load time")
        self.renderer = str(rule.get("renderer") or RENDERER_V1)
        self.retry_policy = str(rule.get("retry_policy") or RETRY_NONE)
        if self.renderer == RENDERER_V1 and self.retry_policy != RETRY_NONE:
            raise ReproductionError(f"slot {slot} renderer {self.renderer} does not match {self.retry_policy}")
        if self.renderer == RENDERER_ACCEPT and self.retry_policy != RETRY_ACCEPT:
            raise ReproductionError(f"slot {slot} renderer {self.renderer} does not match {self.retry_policy}")
        if self.renderer not in {RENDERER_V1, RENDERER_ACCEPT}:
            raise ReproductionError(f"slot {slot} has unknown DNS renderer {self.renderer}")
        self.store = store
        self.slot = slot
        self.ids = store.case_ids(slot)
        self.lengths = [_length(store, case_id) for case_id in self.ids]
        self.speech = _source_map(store, _CLEAN_PREFIX)
        self.noise = _source_map(store, _NOISE_PREFIX)
        self.verify_on_read = bool(settings.get("verify_on_read", False))
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
        recorded = {
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
        rule = dataset.get("rule_json") or {}
        if rule.get("renderer"):
            recorded["renderer"] = self.renderer
        if rule.get("retry_policy"):
            recorded["retry_policy"] = self.retry_policy
        return recorded

    def provenance(self) -> dict[str, Any]:
        recorded = self.identity()
        recorded["shape"] = (
            "case order is the dns_data_prep sorted-uttid order. Audio is reconstructed "
            "from the realization and the source wavs. No generated wav is required."
        )
        return recorded

    def provenance_parts(self) -> dict[str, Any]:
        parts = snapshot_provenance_parts(self.store)
        parts["pipeline"] = "reproduction"
        parts["delivery"] = "on_demand"
        return parts

    def collate_chunks(self, chunks) -> dict[str, Any]:
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
            clean_rows.append(torch.from_numpy(np.ascontiguousarray(clean[chunk.start:chunk.stop])))
            noisy_rows.append(torch.from_numpy(np.ascontiguousarray(noisy[chunk.start:chunk.stop])))
            case_ids.append(case_id)
        return {
            "clean": torch.stack(clean_rows),
            "noisy": torch.stack(noisy_rows),
            "case_id": case_ids,
        }

    def close(self) -> None:
        return None

    def _pair(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        document = self.store.docs.get(case_id)
        if document is None:
            raise ReproductionError(f"case {case_id} has no realization document")
        policy = str(document["draws"].get("retry_policy") or RETRY_NONE)
        if policy != self.retry_policy:
            raise ReproductionError(
                f"case {case_id} retry policy {policy} does not match renderer {self.renderer}"
            )
        if self.verify_on_read:
            _check_source_hashes(self.store, document, self.speech, self.noise)
        rendered = render_realization(document, {"speech": self.speech, "noise": self.noise})
        clean = np.asarray(rendered["clean"], dtype=np.float32)
        noisy = np.asarray(rendered["noisy"], dtype=np.float32)
        if clean.shape != noisy.shape or int(clean.shape[0]) != self.lengths[self.ids.index(case_id)]:
            raise ReproductionError(f"case {case_id} rendered length does not match the plan")
        if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
            raise ReproductionError(f"case {case_id} rendered non-finite samples")
        return clean, noisy


def open_dns_datasets(
    loaded: LoadedSnapshot,
    *,
    load_workers: int = 1,
    verify_on_read: bool = False,
    derived_root: str | Path,
    corpora_root: str | Path,
    generated_root: str | Path,
) -> tuple[DnsOnDemandChunkData, DnsOnDemandChunkData]:
    del load_workers
    store = AssetStore.from_snapshot(
        loaded,
        derived=derived_root,
        corpora=corpora_root,
        generated=generated_root,
    )
    settings = {"verify_on_read": verify_on_read}
    return (
        DnsOnDemandChunkData(store, slot="train.cases", settings=settings),
        DnsOnDemandChunkData(store, slot="valid.cases", settings=settings),
    )


def _length(store: AssetStore, case_id: str) -> int:
    case = store.case(case_id)
    document = store.docs.get(case_id)
    if document is None:
        raise ReproductionError(f"case {case_id} has no realization document")
    planned = int(document["output"]["length"])
    recorded = int(case["frames"])
    if planned != recorded or planned < 1:
        raise ReproductionError(f"case {case_id} length is not the planned length")
    return planned


def _source_map(store: AssetStore, prefix: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for asset_id in store.assets:
        if not asset_id.startswith(prefix):
            continue
        name = asset_id.rsplit("/", 1)[-1]
        if name in found:
            raise ReproductionError(f"duplicate DNS source name {name}")
        found[name] = store.path(asset_id)
    if not found:
        raise ReproductionError(f"snapshot has no source files under {prefix}")
    return found


def _check_source_hashes(
    store: AssetStore,
    document: dict[str, Any],
    speech: dict[str, Path],
    noise: dict[str, Path],
) -> None:
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
