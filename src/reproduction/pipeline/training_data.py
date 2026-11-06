"""Fixed-pair training data backed by a snapshot.

The trainer sees ``epoch``, ``validation``, and ``data_state``. It does not
see DuckDB, and this module does not import the trainer.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest, validate_algorithm, validate_rule
from src.reproduction.paths import (
    DEFAULT_CORPORA,
    DEFAULT_DERIVED,
    DEFAULT_GENERATED,
    DEFAULT_SNAPSHOTS,
)
from src.reproduction.pipeline.assemble import assemble
from src.reproduction.pipeline.loader import GroupLoader
from src.reproduction.pipeline.order import resolve_order_policy
from src.reproduction.pipeline.planner import plan_fixed_pair, validation_refs
from src.reproduction.pipeline.renderer import render_sample
from src.reproduction.pipeline.state import require_resume_match
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.pipeline.types import EpochPlan, SampleRef, UpdateGroup
from src.reproduction.snapshot_reader import load_snapshot

CROP_NOTE = (
    "Crop starts use mix_seed('crop', seed, epoch, case_id) on the inclusive "
    "interval [0, length - window]. One case in one epoch has one start, so a "
    "sampler repeat and a per_epoch_random validation crop share it. Official "
    "DataLoader workers draw random.randint instead, and that sequence is not "
    "reproduced. Validation crop 'fixed' uses the epoch key 'valid'."
)


class EpochStream:
    """One rank's updates for one epoch. ``render`` does not advance a cursor."""

    def __init__(
        self,
        updates: int,
        plan_hash: str,
        render: Callable[[int], list[dict]],
        start_update: int,
    ) -> None:
        self.updates = updates
        self.plan_hash = plan_hash
        self._render = render
        self._start = start_update

    def __iter__(self) -> Iterator[list[dict]]:
        for index in range(self._start, self.updates):
            yield self.render(index)

    def render(self, update_index: int) -> list[dict]:
        if update_index < 0 or update_index >= self.updates:
            raise ReproductionError(
                f"update {update_index} is outside an epoch of {self.updates} updates"
            )
        return self._render(update_index)


class ReproductionTrainingData:
    """MP-SENet or MossFormerGAN fixed pairs from one snapshot."""

    def __init__(self, store: AssetStore, *, settings: dict[str, Any]) -> None:
        dataset = store.dataset
        validate_algorithm(
            dataset["algorithm"],
            int(dataset["algorithm_version"]),
            dataset["kind"],
        )
        if dataset["kind"] != "fixed_pair":
            raise ReproductionError(f"{dataset['kind']} pipelines are not implemented")
        if dataset.get("declared_status") != "ready":
            raise ReproductionError(f"dataset {dataset.get('name')} is not ready")
        rule = validate_rule(dataset["algorithm"], dataset["rule_json"])
        self.rule = rule
        self.store = store
        self.allow_test_selection = bool(settings["allow_test_selection"])
        self.selection_uses_test = bool(dataset["selection_uses_test"])
        self.profile = str(dataset["profile"])
        self._reject_test_selection()
        self.seed = int(settings["seed"])
        self.world_size = int(settings["world_size"])
        self.global_batch = int(settings["global_batch"])
        self.micro_batch = int(settings["micro_batch"])
        self.accumulation = int(settings["accumulation"])
        self.sample_rate = int(rule["sample_rate"])
        self.window = int(rule["window"])
        self.normalize = str(rule["normalize"])
        self.official_global_batch = int(rule["global_batch"])
        segment = settings.get("segment_size")
        if segment is not None and int(segment) != self.window:
            raise ReproductionError(
                f"segment_size {segment} does not match the protocol window {self.window}"
            )
        model_rate = settings.get("model_sample_rate")
        if model_rate is not None and int(model_rate) != self.sample_rate:
            raise ReproductionError(
                f"model sampling_rate {model_rate} does not match the protocol rate {self.sample_rate}"
            )
        self.policy = resolve_order_policy(rule["order"], self.world_size)
        self.dataset_id = str(dataset["dataset_id"])
        self.name = str(dataset["name"])
        self.version = int(dataset["version"])
        self.snapshot_hash = store.snapshot_hash
        self.protocol_id = str(dataset["protocol_id"])
        self.algorithm = str(dataset["algorithm"])
        self.algorithm_version = int(dataset["algorithm_version"])
        self.verify_on_read = bool(settings.get("verify_on_read", False))
        self.train_ids = store.case_ids("train.cases")
        self.valid_ids = store.case_ids("valid.cases")
        self.test_ids = store.case_ids("test.cases") if "test.cases" in store.bindings else []
        if self.profile == "leakage_safe" and set(self.valid_ids) & set(self.test_ids):
            raise ReproductionError("leakage_safe validation shares cases with test")
        self.batch_warning = None
        if self.global_batch != self.official_global_batch:
            self.batch_warning = (
                f"warning: global_batch_size {self.global_batch} differs from "
                f"protocol global_batch {self.official_global_batch}"
            )
        self.loader = GroupLoader(int(settings.get("load_workers", 1)))
        self._plans: dict[int, EpochPlan] = {}

    def identity(self) -> dict[str, Any]:
        return {
            "pipeline": "reproduction",
            "name": self.name,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "snapshot_hash": self.snapshot_hash,
            "protocol_id": self.protocol_id,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "profile": self.profile,
            "selection_uses_test": self.selection_uses_test,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "window": self.window,
            "normalize": self.normalize,
            "order_policy": self.policy,
            "world_size": self.world_size,
            "global_batch": self.global_batch,
            "official_global_batch": self.official_global_batch,
            "micro_batch": self.micro_batch,
            "accumulation": self.accumulation,
            "train_size": len(self.train_ids),
            "valid_size": len(self.valid_ids),
            "train_ids_sha256": digest(self.train_ids),
            "valid_ids_sha256": digest(self.valid_ids),
        }

    def provenance(self) -> dict[str, Any]:
        recorded = self.identity()
        recorded["allow_test_selection"] = self.allow_test_selection
        recorded["crop"] = CROP_NOTE
        return recorded

    def plan_for(self, epoch: int) -> EpochPlan:
        cached = self._plans.get(epoch)
        if cached is None:
            cached = plan_fixed_pair(
                self.store,
                seed=self.seed,
                epoch=epoch,
                world_size=self.world_size,
                global_batch=self.global_batch,
                micro_batch=self.micro_batch,
                accumulation=self.accumulation,
            )
            self._plans[epoch] = cached
        return cached

    def epoch(self, epoch: int, *, rank: int, world_size: int, start_update: int = 0) -> EpochStream:
        if world_size != self.world_size or rank < 0 or rank >= world_size:
            raise ReproductionError(
                f"rank {rank} world_size {world_size} does not match this training data"
            )
        plan = self.plan_for(epoch)
        if start_update < 0 or start_update > plan.updates:
            raise ReproductionError(f"start_update {start_update} is outside epoch {epoch}")

        def render(update_index: int) -> list[dict]:
            return self._render_group(plan.rank_groups[rank][update_index])

        return EpochStream(plan.updates, plan.plan_hash, render, start_update)

    def validation(
        self,
        *,
        rank: int,
        world_size: int,
        epoch: int | None = None,
    ) -> Iterator[dict]:
        if world_size != self.world_size or rank < 0 or rank >= world_size:
            raise ReproductionError(
                f"rank {rank} world_size {world_size} does not match this training data"
            )
        del rank
        refs = validation_refs(
            self.store,
            seed=self.seed,
            epoch=0 if epoch is None else epoch,
            world_size=self.world_size,
            policy=self.policy,
        )
        forbidden = set(self.test_ids) - set(self.valid_ids)
        if any(sample.case_id in forbidden for sample in refs):
            raise ReproductionError("validation included a test case that is not the valid set")
        batch_size = int(self.rule["validation"]["batch"])
        if batch_size < 1:
            raise ReproductionError("validation batch must be positive")
        for start in range(0, len(refs), batch_size):
            yield self._render_refs(refs[start : start + batch_size])

    def data_state(self, epoch: int, committed_updates: int) -> dict[str, Any]:
        if epoch < 0 or committed_updates < 0:
            raise ReproductionError("data position must be non-negative")
        plan = self.plan_for(epoch)
        if committed_updates > plan.updates:
            raise ReproductionError(
                f"committed updates {committed_updates} exceed epoch {epoch} ({plan.updates})"
            )
        return {
            "pipeline": "reproduction",
            "dataset_id": self.dataset_id,
            "dataset_name": f"{self.name}@{self.version}",
            "snapshot_hash": self.snapshot_hash,
            "protocol_id": self.protocol_id,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "profile": self.profile,
            "selection_uses_test": self.selection_uses_test,
            "seed": self.seed,
            "epoch": epoch,
            "update_in_epoch": committed_updates,
            "world_size": self.world_size,
            "ranks": list(range(self.world_size)),
            "order_policy": plan.order_policy,
            "order_params": dict(plan.stats["order_params"]),
            "plan_hash": plan.plan_hash,
            "global_batch": self.global_batch,
            "official_global_batch": self.official_global_batch,
            "micro_batch": self.micro_batch,
            "accumulation": self.accumulation,
            "normalize": self.normalize,
            "window": self.window,
        }

    def check_resume(self, saved: dict[str, Any]) -> None:
        current = self.data_state(int(saved["epoch"]), int(saved["update_in_epoch"]))
        require_resume_match(saved, current)

    def close(self) -> None:
        self.loader.close()

    def _render_group(self, group: UpdateGroup) -> list[dict]:
        return [self._render_refs(micro.samples) for micro in group.micro_batches]

    def _render_refs(self, samples: tuple[SampleRef, ...] | list[SampleRef]) -> dict:
        rendered = self.loader.map(self._render_one, list(samples))
        return assemble(rendered, dataset_id=self.dataset_id, snapshot_hash=self.snapshot_hash)

    def _render_one(self, sample: SampleRef):
        return render_sample(
            self.store,
            sample,
            normalize=self.normalize,
            sample_rate=self.sample_rate,
            verify_on_read=self.verify_on_read,
        )

    def _reject_test_selection(self) -> None:
        if self.profile == "leakage_safe" and self.selection_uses_test:
            raise ReproductionError("leakage_safe dataset cannot select on test")
        bindings = self.store.bindings
        if self.profile == "leakage_safe" and bindings.get("valid.cases") == bindings.get("test.cases"):
            raise ReproductionError("leakage_safe validation case set is the test case set")
        if self.selection_uses_test and not self.allow_test_selection:
            raise ReproductionError(
                "dataset uses test for validation or checkpoint selection; "
                "data.allow_test_selection must be true"
            )


def open_training_data(
    *,
    dataset: str,
    allow_test_selection: bool,
    seed: int,
    global_batch: int,
    micro_batch: int,
    accumulation: int,
    world_size: int,
    segment_size: int | None = None,
    model_sample_rate: int | None = None,
    load_workers: int = 1,
    verify_on_read: bool = False,
    snapshot_root: str | Path | None = None,
    derived_root: str | Path | None = None,
    corpora_root: str | Path | None = None,
    generated_root: str | Path | None = None,
) -> ReproductionTrainingData:
    """Load ``name@version`` from the snapshot directory. DuckDB stays closed."""
    root = Path(DEFAULT_SNAPSHOTS if snapshot_root is None else snapshot_root)
    directory = find_snapshot_dir(root, dataset)
    loaded = load_snapshot(directory)
    store = AssetStore.from_snapshot(
        loaded,
        derived=DEFAULT_DERIVED if derived_root is None else derived_root,
        corpora=DEFAULT_CORPORA if corpora_root is None else corpora_root,
        generated=DEFAULT_GENERATED if generated_root is None else generated_root,
    )
    return ReproductionTrainingData(
        store,
        settings={
            "allow_test_selection": allow_test_selection,
            "seed": seed,
            "global_batch": global_batch,
            "micro_batch": micro_batch,
            "accumulation": accumulation,
            "world_size": world_size,
            "segment_size": segment_size,
            "model_sample_rate": model_sample_rate,
            "load_workers": load_workers,
            "verify_on_read": verify_on_read,
        },
    )


def find_snapshot_dir(root: Path, spec: str) -> Path:
    name, separator, version_text = spec.rpartition("@")
    if separator != "@" or not version_text.isdigit() or not name:
        raise ReproductionError("dataset must be name@version")
    version = int(version_text)
    if not root.is_dir():
        raise ReproductionError(f"snapshot root is missing: {root}")
    matches = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        document_path = child / "dataset.json"
        if not document_path.is_file():
            continue
        document = json.loads(document_path.read_text(encoding="utf-8"))
        if document.get("name") == name and int(document.get("version", -1)) == version:
            matches.append(child)
    if len(matches) != 1:
        raise ReproductionError(f"found {len(matches)} snapshots for {spec} under {root}")
    return matches[0]
