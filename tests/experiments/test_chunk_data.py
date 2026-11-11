"""Phase 3-C chunk SnapshotDataModule on CPU synthetic snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.data import Batch, EpochView, TensorPairData
from src.experiments.data.chunk import ChunkProtocol, ChunkSampler
from src.experiments.data.fixed_pair import MicroBatcher, SnapshotDataModule, SnapshotSource
from src.experiments.data.sample import ChunkSpan, SampleRef
from src.experiments.errors import BuildError, CheckpointError, DataError, StateError
from src.experiments.data.build.dns2020 import (
    ALGORITHM as DNS_ALGORITHM,
    RENDERER_ACCEPT,
    RETRY_ACCEPT,
    read_mono,
    resample_to_16k,
)
from src.experiments.data.build.ms_snsd import ALGORITHM as MS_ALGORITHM
from src.experiments.data.ids import canonical_text, digest
from src.experiments.data.render.chunking import PlannedChunk, plan_chunk_epoch
from src.experiments.data.mix.dns_ondemand import DnsOnDemandChunkData
from src.experiments.data.mix.ondemand import OnDemandChunkData
from src.experiments.data.sampling.planner import chunk_case_length
from src.experiments.data.assets.store import AssetStore
from src.experiments.data.snapshot import (
    FORMAT,
    SCHEMA_VERSION,
    arrays_from_assets,
    arrays_from_cases,
    arrays_from_items,
    arrays_from_members,
    file_hashes,
    load_snapshot,
    snapshot_hash,
)

ROOT = Path(__file__).resolve().parents[2]


class _Runtime:
    def __init__(self, world_size: int = 1, rank: int = 0) -> None:
        self.device = torch.device("cpu")
        self.world_size = world_size
        self.rank = rank


class ChunkDataTests(unittest.TestCase):
    def setUp(self) -> None:
        base = Path(os.environ.get("TMPDIR", "/tmp")) / "audio-db-phase3c" / self.id().replace(".", "_")
        if base.exists():
            _remove(base)
        base.mkdir(parents=True)
        self.base = base
        self.derived = base / "derived"
        self.snapshots = base / "snapshots"
        self.ms = _write_snapshot(self.snapshots, derived=self.derived, kind="ms", name="toy.ms")
        self.dns = _write_snapshot(self.snapshots, derived=self.derived, kind="dns", name="toy.dns")

    def tearDown(self) -> None:
        _remove(self.base)

    def test_factory_builds_components_without_duckdb_or_old_readers(self) -> None:
        import src.experiments.data.chunk as chunk_mod
        import src.experiments.data.mix.dns_ondemand as dns_mod
        import src.experiments.data.mix.ondemand as ms_mod
        import src.experiments.data.fixed_pair_stream as training_data

        calls = {"duck": 0, "training": 0, "ms": 0, "dns": 0}
        original_training = training_data.ReproductionTrainingData
        original_ms = ms_mod.OnDemandChunkData
        original_dns = dns_mod.DnsOnDemandChunkData

        class _RefusedTraining:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["training"] += 1
                raise AssertionError("ReproductionTrainingData was constructed")

        class _RefusedMs:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["ms"] += 1
                raise AssertionError("OnDemandChunkData was constructed")

        class _RefusedDns:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["dns"] += 1
                raise AssertionError("DnsOnDemandChunkData was constructed")

        class _Duck:
            @staticmethod
            def connect(*args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["duck"] += 1
                raise AssertionError("duckdb.connect was called")

        previous = sys.modules.get("duckdb")
        sys.modules["duckdb"] = _Duck()  # type: ignore[assignment]
        training_data.ReproductionTrainingData = _RefusedTraining  # type: ignore[misc, assignment]
        ms_mod.OnDemandChunkData = _RefusedMs  # type: ignore[misc, assignment]
        dns_mod.DnsOnDemandChunkData = _RefusedDns  # type: ignore[misc, assignment]
        chunk_mod.OnDemandChunkData = _RefusedMs  # type: ignore[attr-defined]
        chunk_mod.DnsOnDemandChunkData = _RefusedDns  # type: ignore[attr-defined]
        config = _experiment_config(self.base / "build", self.ms)
        try:
            experiment = build_experiment(load_experiment(config))
            try:
                self.assertIsInstance(experiment.get("data"), SnapshotDataModule)
                self.assertIsInstance(experiment.get("sampler"), ChunkSampler)
                self.assertIsInstance(experiment.get("protocol"), ChunkProtocol)
                self.assertIsInstance(experiment.get("source"), SnapshotSource)
                state_ids = set(experiment._graph.states.ids())  # type: ignore[union-attr]
                self.assertIn("data", state_ids)
                for name in ("source", "sampler", "protocol", "batcher"):
                    self.assertNotIn(name, state_ids)
                self.assertIsInstance(TensorPairData(features=2, examples=2, batch_size=1), TensorPairData)
                self.assertTrue(issubclass(EpochView, object))
                self.assertTrue(issubclass(Batch, object))
                identity = experiment.get("data").identity()
                self.assertEqual(identity["order_policy"], "espnet_chunk_v1")
                self.assertEqual(identity["window"], 32)
                self.assertIsNone(identity["normalize"])
                self.assertEqual(identity["group_size"], 1)
            finally:
                experiment.close()
            with self.assertRaises(DataError):
                experiment.get("source").dataset()
        finally:
            training_data.ReproductionTrainingData = original_training
            ms_mod.OnDemandChunkData = original_ms
            dns_mod.DnsOnDemandChunkData = original_dns
            if hasattr(chunk_mod, "OnDemandChunkData"):
                delattr(chunk_mod, "OnDemandChunkData")
            if hasattr(chunk_mod, "DnsOnDemandChunkData"):
                delattr(chunk_mod, "DnsOnDemandChunkData")
            if previous is None:
                sys.modules.pop("duckdb", None)
            else:
                sys.modules["duckdb"] = previous
        self.assertEqual(calls, {"duck": 0, "training": 0, "ms": 0, "dns": 0})

    def test_plan_matches_plan_chunk_epoch(self) -> None:
        for shuffle in (False, True):
            module = _module(self.ms, shuffle=shuffle)
            module.setup(_Runtime())
            try:
                for epoch in (0, 1):
                    _assert_plan(self, module, epoch=epoch, world_size=1, rank=0, shuffle=shuffle)
            finally:
                module.close()
        ordered = _module(self.ms, shuffle=False)
        ordered.setup(_Runtime())
        try:
            samples = _samples(ordered.source, "train.cases")
            plan = plan_chunk_epoch(
                samples,
                espnet_epoch=1,
                seed=7,
                world_size=1,
                num_samples=3,
                chunk_length=32,
                global_batch=2,
                num_cache_chunks=2,
                shuffle=False,
                shift_ratio=0.5,
            )
            self.assertEqual(plan.discarded_tail_chunks[0], 1)
            self.assertEqual(plan.stats["iterator_stop"], "min_rank_batches")
            short = ordered.source.case_ids("train.cases")[0]
            refs = ordered.sampler.sequence(0, 0, 1)
            self.assertNotIn(short, [sample.case_id for sample in refs])
            self.assertTrue(all(sample.realization_id == f"{sample.case_id}:{sample.chunk.start}:{sample.chunk.stop}" for sample in refs))  # type: ignore[union-attr]
        finally:
            ordered.close()

    def test_plan_keeps_process_world_size_and_iterator_stop(self) -> None:
        import src.experiments.data.chunk as chunk_mod

        seen: list[tuple[list[tuple[str, int]], dict[str, object]]] = []
        original = chunk_mod.plan_chunk_epoch

        def wrapped(samples, **kwargs):  # type: ignore[no-untyped-def]
            seen.append((list(samples), dict(kwargs)))
            return original(samples, **kwargs)

        chunk_mod.plan_chunk_epoch = wrapped  # type: ignore[assignment]
        module = _module(self.ms, micro_batch=1, shuffle=False)
        try:
            module.setup(_Runtime(world_size=2, rank=0))
            module.train_epoch(0, start_update=0)
            self.assertTrue(seen)
            for samples_seen, kwargs in seen:
                self.assertEqual([case_id for case_id, _length in samples_seen], list(module.source.case_ids("train.cases")))
                self.assertEqual(kwargs["world_size"], 2)
                self.assertEqual(kwargs["espnet_epoch"], 1)
            other = _module(self.ms, micro_batch=1, shuffle=False)
            other.setup(_Runtime(world_size=2, rank=1))
            try:
                _assert_plan(self, other, epoch=0, world_size=2, rank=1, shuffle=False)
                samples = _samples(other.source, "train.cases")
                plan = plan_chunk_epoch(
                    samples,
                    espnet_epoch=1,
                    seed=7,
                    world_size=2,
                    num_samples=3,
                    chunk_length=32,
                    global_batch=2,
                    num_cache_chunks=2,
                    shuffle=False,
                    shift_ratio=0.5,
                )
                self.assertGreater(plan.discarded_batches[1], 0)
                self.assertEqual(len(other.sampler.sequence(0, 1, 2)), plan.updates * plan.per_rank_batch)
                self.assertEqual(other.plan_hash_for_state({"epoch": 0}), module.plan_hash_for_state({"epoch": 0}))
            finally:
                other.close()
        finally:
            chunk_mod.plan_chunk_epoch = original  # type: ignore[assignment]
            module.close()

    def test_ms_snsd_waveforms_match_existing_pair(self) -> None:
        module = _module(self.ms, shuffle=True)
        module.setup(_Runtime())
        store = _store(self.ms)
        old = OnDemandChunkData(store, slot="train.cases", settings={"verify_on_read": False})
        try:
            _assert_matches_reader(self, module, old, epoch=0)
            case_id = module.source.case_ids("train.cases")[2]
            sample = SampleRef(
                sample_id=case_id,
                realization_id=f"{case_id}:16:48",
                index=0,
                case_id=case_id,
                chunk=ChunkSpan(16, 48),
            )
            noisy, clean, valid = module.protocol.render(sample)
            collated = old.collate_chunks([PlannedChunk(case_id, 16, 48)])
            self.assertEqual(valid, 32)
            self.assertEqual(int(np.max(np.abs(noisy - collated["noisy"][0].numpy()))), 0)
            self.assertEqual(int(np.max(np.abs(clean - collated["clean"][0].numpy()))), 0)
        finally:
            module.close()

    def test_dns_waveforms_match_existing_pair(self) -> None:
        module = _module(self.dns, shuffle=False)
        module.setup(_Runtime())
        store = _store(self.dns)
        old = DnsOnDemandChunkData(store, slot="train.cases", settings={"verify_on_read": False})
        try:
            _assert_matches_reader(self, module, old, epoch=0)
            case_id = str(self.dns["resample_case"])
            document = store.docs[case_id]
            noise = document["sources"]["noise"][0]
            self.assertEqual(int(noise["source_rate"]), 8000)
            span = ChunkSpan(4, 36)
            sample = SampleRef(
                sample_id=case_id,
                realization_id=f"{case_id}:4:36",
                index=0,
                case_id=case_id,
                chunk=span,
            )
            noisy, clean, valid = module.protocol.render(sample)
            collated = old.collate_chunks([PlannedChunk(case_id, span.start, span.stop)])
            self.assertEqual(valid, 32)
            self.assertEqual(int(np.max(np.abs(noisy - collated["noisy"][0].numpy()))), 0)
            self.assertEqual(int(np.max(np.abs(clean - collated["clean"][0].numpy()))), 0)
        finally:
            module.close()

    def test_validation_plan_is_not_the_training_plan(self) -> None:
        module = _module(self.ms, shuffle=True)
        module.setup(_Runtime())
        try:
            train = _reference_plan(module, slot="train.cases", epoch=0, shuffle=True, num_samples=3)
            valid = _reference_plan(module, slot="valid.cases", epoch=0, shuffle=False, num_samples=None)
            batches = list(module.validation_epoch(0))
            again = list(module.validation_epoch(1))
            got = _batch_spans(batches)
            expected = [(chunk.case_id, chunk.start, chunk.stop) for batch in valid.rank_batches[0] for chunk in batch]
            self.assertEqual(got, expected)
            self.assertEqual(len(batches), valid.updates)
            self.assertEqual(_batch_spans(again), expected)
            train_ids = {chunk.case_id for batch in train.rank_batches[0] for chunk in batch}
            valid_ids = {case_id for case_id, _start, _stop in got}
            self.assertTrue(train_ids.isdisjoint(valid_ids))
            self.assertNotEqual(
                [(chunk.case_id, chunk.start, chunk.stop) for batch in train.rank_batches[0] for chunk in batch],
                got,
            )
            view = module.train_epoch(0, start_update=0)
            self.assertEqual(view.updates, train.updates)
        finally:
            module.close()

    def test_group_size_and_global_batch_are_rejected(self) -> None:
        source = _source(self.ms)
        protocol = ChunkProtocol(source)
        sampler = ChunkSampler(
            source,
            seed=7,
            num_samples_per_epoch=3,
            chunk_length=32,
            shift_ratio=0.5,
            num_cache_chunks=2,
            shuffle=False,
        )
        module = SnapshotDataModule(source, sampler, protocol, MicroBatcher(2, 2))
        with self.assertRaises(DataError):
            module.setup(_Runtime())
        module.close()
        wide = ChunkSampler(
            source,
            seed=7,
            num_samples_per_epoch=3,
            chunk_length=32,
            shift_ratio=0.5,
            num_cache_chunks=2,
            shuffle=False,
            global_batch=4,
        )
        refused = SnapshotDataModule(source, wide, protocol, MicroBatcher(2, 1))
        with self.assertRaises(DataError):
            refused.setup(_Runtime())
        refused.close()
        config = _experiment_config(self.base / "group", self.ms, group_size=2)
        from src.experiments.api import Experiment

        experiment = Experiment(load_experiment(config))
        with self.assertRaises(BuildError) as caught:
            experiment.build()
        self.assertIsInstance(caught.exception.__cause__, DataError)

    def test_same_micro_batch_is_stable_on_cpu(self) -> None:
        module = _module(self.ms)
        module.setup(_Runtime())
        try:
            view = module.train_epoch(0, start_update=0)
            first = view.batch(0)[0]
            second = view.batch(0)[0]
            self.assertEqual(int(torch.max(torch.abs(first.inputs - second.inputs))), 0)
            self.assertEqual(int(torch.max(torch.abs(first.targets - second.targets))), 0)
            self.assertEqual(first.sample_id, ",".join(first.sample_ids))
            self.assertEqual(first.realization_id, ",".join(first.realization_ids))
            self.assertEqual(first.inputs.device.type, "cpu")
            self.assertEqual(first.teacher_kind, "dry_clean")
            assert first.lengths is not None
            self.assertTrue(torch.equal(first.lengths, torch.full((first.inputs.shape[0],), first.inputs.shape[-1])))
            with self.assertRaises(StateError):
                view.batch(view.updates)
        finally:
            module.close()

    def test_resume_matches_the_next_chunk_and_rejects_a_bad_plan_hash(self) -> None:
        interrupted = self.base / "interrupted"
        continuous = self.base / "continuous"
        interrupted_config = _experiment_config(
            interrupted,
            self.ms,
            steps=4,
            stop_after=2,
            prefetch=True,
        )
        continuous_config = _experiment_config(continuous, self.ms, steps=4, prefetch=False)
        first = _cli(interrupted_config)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertFalse((interrupted / "complete.json").is_file())
        saved = interrupted / "archive/step_00000002.pt"
        self.assertTrue(saved.is_file())
        resume_config = _experiment_config(interrupted, self.ms, steps=4, prefetch=True)
        whole = _cli(continuous_config)
        self.assertEqual(whole.returncode, 0, whole.stderr)
        resumed = _cli(resume_config, "--resume", str(saved))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(_metrics(interrupted)[2]["sample_id"], _metrics(continuous)[2]["sample_id"])
        fresh = _module(self.ms, prefetch=False)
        fresh.setup(_Runtime())
        experiment = build_experiment(load_experiment(resume_config))
        try:
            experiment._checkpoint.restore(saved, mode="resume")  # type: ignore[union-attr]
            data = experiment.get("data")
            self.assertEqual(data.update_index, 2)
            self.assertEqual(data.epoch, 0)
            self.assertEqual(data.state_dict()["plan_hash"], fresh.plan_hash_for_state({"epoch": 0}))
            continued = data.train_epoch(0, start_update=2).batch(2)[0]
            expected = fresh.train_epoch(0, start_update=0).batch(2)[0]
            self.assertEqual(continued.sample_ids, expected.sample_ids)
            self.assertEqual(continued.realization_ids, expected.realization_ids)
            self.assertEqual(int(torch.max(torch.abs(continued.inputs - expected.inputs))), 0)
            self.assertEqual(int(torch.max(torch.abs(continued.targets - expected.targets))), 0)
            self.assertIsNone(data.state_dict()["weights"])
        finally:
            experiment.close()
            fresh.close()
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        payload["states"]["data"]["plan_hash"] = "f" * 64
        tampered = interrupted / "tampered.pt"
        torch.save(payload, tampered)
        again = build_experiment(load_experiment(resume_config))
        try:
            with self.assertRaises(CheckpointError):
                again._checkpoint.restore(tampered, mode="resume")  # type: ignore[union-attr]
        finally:
            again.close()


def _assert_plan(
    test: ChunkDataTests,
    module: SnapshotDataModule,
    *,
    epoch: int,
    world_size: int,
    rank: int,
    shuffle: bool,
) -> None:
    plan = _reference_plan(
        module,
        slot="train.cases",
        epoch=epoch,
        shuffle=shuffle,
        num_samples=module.sampler.num_samples_per_epoch,
        world_size=world_size,
    )
    refs = module.sampler.sequence(epoch, rank, world_size)
    expected = [(chunk.case_id, chunk.start, chunk.stop) for batch in plan.rank_batches[rank] for chunk in batch]
    test.assertEqual([(sample.case_id, sample.chunk.start, sample.chunk.stop) for sample in refs], expected)  # type: ignore[union-attr]
    groups = module.batcher.groups(refs)
    test.assertEqual(
        [[(sample.case_id, sample.chunk.start, sample.chunk.stop) for sample in group] for group in groups],  # type: ignore[union-attr]
        [[(chunk.case_id, chunk.start, chunk.stop) for chunk in batch] for batch in plan.rank_batches[rank]],
    )
    test.assertEqual(len(groups), plan.updates)
    test.assertEqual(plan.espnet_epoch, epoch + 1)


def _assert_matches_reader(test: ChunkDataTests, module: SnapshotDataModule, old: object, *, epoch: int) -> None:
    groups = module.batcher.groups(module.sampler.sequence(epoch, 0, 1))
    view = module.train_epoch(epoch, start_update=0)
    test.assertEqual(view.updates, len(groups))
    for index, group in enumerate(groups):
        chunks = [PlannedChunk(str(sample.case_id), int(sample.chunk.start), int(sample.chunk.stop)) for sample in group]  # type: ignore[union-attr]
        collated = old.collate_chunks(chunks)  # type: ignore[attr-defined]
        batch = view.batch(index)[0]
        test.assertEqual(int(np.max(np.abs(batch.inputs.numpy() - collated["noisy"].numpy()))), 0)
        test.assertEqual(int(np.max(np.abs(batch.targets.numpy() - collated["clean"].numpy()))), 0)
        test.assertEqual([str(sample.case_id) for sample in group], list(collated["case_id"]))


def _reference_plan(
    module: SnapshotDataModule,
    *,
    slot: str,
    epoch: int,
    shuffle: bool,
    num_samples: int | None,
    world_size: int | None = None,
):
    world = module.world_size if world_size is None else world_size
    return plan_chunk_epoch(
        _samples(module.source, slot),
        espnet_epoch=epoch + 1,
        seed=module.sampler.seed,
        world_size=world,
        num_samples=num_samples,
        chunk_length=module.sampler.chunk_length,  # type: ignore[attr-defined]
        global_batch=module.batcher.micro_batch * world,
        num_cache_chunks=module.sampler.num_cache_chunks,  # type: ignore[attr-defined]
        shuffle=shuffle,
        shift_ratio=module.sampler.shift_ratio,  # type: ignore[attr-defined]
    )


def _samples(source: SnapshotSource, slot: str) -> list[tuple[str, int]]:
    return [(case_id, chunk_case_length(source.store, case_id)) for case_id in source.case_ids(slot)]


def _batch_spans(batches: list[Batch]) -> list[tuple[str, int, int]]:
    spans = []
    for batch in batches:
        for realization_id in batch.realization_ids:
            case_id, start, stop = realization_id.rsplit(":", 2)
            spans.append((case_id, int(start), int(stop)))
    return spans


def _source(spec: dict[str, object]) -> SnapshotSource:
    return SnapshotSource(
        str(spec["selector"]),
        spec["root"],  # type: ignore[arg-type]
        "train",
        derived=spec["derived"],  # type: ignore[arg-type]
        corpora=spec["corpora"],  # type: ignore[arg-type]
        generated=spec["generated"],  # type: ignore[arg-type]
    )


def _store(spec: dict[str, object]) -> AssetStore:
    loaded = load_snapshot(spec["directory"])  # type: ignore[arg-type]
    return AssetStore.from_snapshot(
        loaded,
        derived=spec["derived"],  # type: ignore[arg-type]
        corpora=spec["corpora"],  # type: ignore[arg-type]
        generated=spec["generated"],  # type: ignore[arg-type]
    )


def _module(
    spec: dict[str, object],
    *,
    seed: int = 7,
    shuffle: bool = True,
    micro_batch: int = 2,
    prefetch: bool = False,
) -> SnapshotDataModule:
    source = _source(spec)
    protocol = ChunkProtocol(source)
    sampler = ChunkSampler(
        source,
        seed=seed,
        num_samples_per_epoch=3,
        chunk_length=32,
        shift_ratio=0.5,
        num_cache_chunks=2,
        shuffle=shuffle,
    )
    return SnapshotDataModule(source, sampler, protocol, MicroBatcher(micro_batch, 1), prefetch=prefetch)


def _experiment_config(
    directory: Path,
    spec: dict[str, object],
    *,
    steps: int = 1,
    stop_after: int | None = None,
    prefetch: bool = False,
    group_size: int = 1,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    text = f"""schema: audio-db-experiment/v1
checkpoint:
  selector:
    class: tests.experiments.support.StepArchiveSelector
    kwargs:
      interval: 1
seed: 17
deterministic: false
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
output:
  directory: {directory}
components:
  model:
    role: module
    class: tests.experiments.support.IdentityWave
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: torch.nn.L1Loss
  source:
    role: other
    class: src.experiments.data.SnapshotSource
    kwargs:
      snapshot: {spec["selector"]}
      snapshot_root: {spec["root"]}
      split: train
      derived: {spec["derived"]}
      corpora: {spec["corpora"]}
      generated: {spec["generated"]}
  sampler:
    role: other
    class: src.experiments.data.ChunkSampler
    kwargs:
      seed: 7
      num_samples_per_epoch: 3
      chunk_length: 32
      shift_ratio: 0.5
      num_cache_chunks: 2
      shuffle: true
    inject:
      source: {{ref: source}}
  protocol:
    role: other
    class: src.experiments.data.ChunkProtocol
    kwargs: {{}}
    inject:
      source: {{ref: source}}
  batcher:
    role: other
    class: src.experiments.data.MicroBatcher
    kwargs:
      micro_batch: 2
      group_size: {group_size}
  data:
    role: data
    class: src.experiments.data.SnapshotDataModule
    kwargs:
      prefetch: {str(prefetch).lower()}
    inject:
      source: {{ref: source}}
      sampler: {{ref: sampler}}
      protocol: {{ref: protocol}}
      batcher: {{ref: batcher}}
  optimizer:
    role: optimizer
    class: torch.optim.SGD
    parameters_of: model
    kwargs: {{lr: 0.1}}
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: {group_size}
{stop}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(text)
    return path


def _cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _metrics(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _write_snapshot(root: Path, *, derived: Path, kind: str, name: str) -> dict[str, object]:
    corpora = derived.parent / "corpora"
    generated = derived.parent / "generated"
    corpora.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    if kind == "ms":
        slots = {"train.cases": [32, 80, 96, 128, 160, 112], "valid.cases": [96, 80, 48], "test.cases": [40]}
        algorithm = MS_ALGORITHM
        rule: dict[str, object] = {
            "delivery": "on_demand",
            "normalize": None,
            "order": "espnet_chunk_v1",
            "sample_rate": 16000,
            "teacher": "dry_clean",
        }
    else:
        slots = {"train.cases": [80, 96, 128, 64], "valid.cases": [96, 80], "test.cases": [48]}
        algorithm = DNS_ALGORITHM
        rule = {
            "delivery": "on_demand",
            "normalize": None,
            "order": "espnet_chunk_v1",
            "renderer": RENDERER_ACCEPT,
            "retry_policy": RETRY_ACCEPT,
            "sample_rate": 16000,
            "teacher": "dry_clean",
        }
    assets: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    docs: list[dict[str, object]] = []
    items: list[dict[str, object]] = []
    bindings = []
    resample_case = None
    cursor = 0
    for slot, lengths in slots.items():
        ids: list[str] = []
        for offset, length in enumerate(lengths):
            resample = kind == "dns" and slot == "train.cases" and offset == 0
            case_id, frames = _add_case(
                kind=kind,
                corpora=corpora,
                assets=assets,
                cases=cases,
                docs=docs,
                length=length,
                gen_index=cursor,
                resample=resample,
            )
            if resample:
                resample_case = case_id
            ids.append(case_id)
            cursor += 1
        case_set_id = digest(ids)
        bindings.append({"case_set_id": case_set_id, "material_set_id": None, "slot": slot})
        for position, case_id in enumerate(ids):
            items.append({"case_id": case_id, "case_set_id": case_set_id, "position": position})
    dataset_id = digest({"algorithm": algorithm, "kind": kind, "name": name})
    dataset = {
        "algorithm": algorithm,
        "algorithm_version": 1,
        "bindings": bindings,
        "claim_label": "leakage_safe_comparison",
        "dataset_id": dataset_id,
        "declared_status": "ready",
        "kind": "pregenerated",
        "name": name,
        "profile": "leakage_safe",
        "protocol_id": "protocol",
        "rule_json": rule,
        "selection_uses_test": False,
        "version": 1,
    }
    directory = root / dataset_id
    directory.mkdir(parents=True)
    doc_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in docs)
    (directory / "docs.jsonl").write_text(doc_text, encoding="utf-8")
    parts = {
        "assets": assets,
        "case_set_items": items,
        "cases": cases,
        "dataset": dataset,
        "docs": [json.loads(line) for line in doc_text.splitlines() if line],
        "generated": [],
        "material_set_members": [],
    }
    digest_value = snapshot_hash(parts)
    manifest = {
        "files": file_hashes(parts),
        "provenance": {"format": FORMAT},
        "schema_version": SCHEMA_VERSION,
        "snapshot_hash": digest_value,
    }
    (directory / "dataset.json").write_text(canonical_text(parts["dataset"]) + "\n", encoding="utf-8")
    (directory / "generated.json").write_text(canonical_text(parts["generated"]) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(canonical_text(manifest) + "\n", encoding="utf-8")
    np.savez(directory / "assets.npz", **arrays_from_assets(parts["assets"]))
    np.savez(directory / "cases.npz", **arrays_from_cases(parts["cases"]))
    np.savez(directory / "case_set_items.npz", **arrays_from_items(parts["case_set_items"]))
    np.savez(directory / "material_set_members.npz", **arrays_from_members(parts["material_set_members"]))
    loaded = load_snapshot(directory)
    if loaded.snapshot_hash != digest_value:
        raise RuntimeError("synthetic snapshot hash did not round-trip")
    return {
        "corpora": corpora,
        "derived": derived,
        "directory": directory,
        "generated": generated,
        "resample_case": resample_case,
        "root": root,
        "selector": f"{name}@1",
    }


def _add_case(
    *,
    kind: str,
    corpora: Path,
    assets: list[dict[str, object]],
    cases: list[dict[str, object]],
    docs: list[dict[str, object]],
    length: int,
    gen_index: int,
    resample: bool,
) -> tuple[str, int]:
    speech_name = f"s{gen_index}.wav"
    noise_name = f"n{gen_index}.wav"
    if kind == "ms":
        speech_rel = f"MS-SNSD/clean_train/{speech_name}"
        noise_rel = f"MS-SNSD/noise_train/{noise_name}"
        speech_rate = 16000
        noise_rate = 16000
        frames = length
    else:
        speech_rel = f"DNS2020/datasets/clean/{speech_name}"
        noise_rel = f"DNS2020/datasets/noise/{noise_name}"
        if resample:
            noise_rate = 8000
            raw = np.linspace(-0.35, 0.4, 90, dtype=np.float64)
            noise_path = corpora / noise_rel
            _write_wav(noise_path, raw, noise_rate)
            mono, rate = read_mono(noise_path)
            frames = int(resample_to_16k(mono, rate).shape[0])
            speech_rate = 16000
        else:
            speech_rate = 16000
            noise_rate = 16000
            frames = length
    speech_path = corpora / speech_rel
    noise_path = corpora / noise_rel
    _write_wav(speech_path, _tone(frames, 0.2 + (gen_index % 3) * 0.05), speech_rate)
    if not resample:
        _write_wav(noise_path, _tone(frames if kind == "dns" else length, 0.15 + (gen_index % 4) * 0.04), noise_rate)
    for rel, path, rate, file_frames in (
        (speech_rel, speech_path, speech_rate, frames if kind == "dns" and resample else (frames if speech_rate == 16000 else length)),
        (noise_rel, noise_path, noise_rate, 90 if resample else (frames if kind == "dns" else length)),
    ):
        info = sf.info(path)
        assets.append(
            {
                "asset_id": f"corpora/{rel}",
                "rel_path": rel,
                "sample_rate": int(info.samplerate),
                "frames": int(info.frames),
                "channels": 1,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "integrity": "sha256",
            }
        )
        del rate, file_frames
    if kind == "ms":
        document: dict[str, object] = {
            "algorithm": MS_ALGORITHM,
            "algorithm_version": 1,
            "draws": {
                "build_id": "toy-ms",
                "clip_index": gen_index,
                "gen_index": gen_index,
                "nominal_snr_db": 10,
                "utt_id": f"ms-{gen_index}",
            },
            "output": {"length": frames, "teacher": "dry_clean"},
            "realization_version": 1,
            "sample_rate": 16000,
            "sources": {
                "noise": [{"length": frames, "name": noise_name, "start": 0}],
                "speech": [{"length": frames, "name": speech_name, "start": 0}],
            },
            "steps": ["dbfs_m25", "concat_silence", "snr_mixer_sqrt", "pcm16"],
        }
        build_id = "toy-ms"
    else:
        noise_piece: dict[str, object] = {"length": frames, "name": noise_name, "start": 0}
        if resample:
            noise_piece["source_rate"] = 8000
        document = {
            "algorithm": DNS_ALGORITHM,
            "algorithm_version": 1,
            "draws": {
                "build_id": "toy-dns",
                "gen_index": gen_index,
                "retries": 0,
                "retry_policy": RETRY_ACCEPT,
                "snr_db": 5,
                "target_level_dbfs": -25,
                "utt_id": f"dns-{gen_index}",
            },
            "output": {"length": frames, "teacher": "dry_clean"},
            "realization_version": 1,
            "sample_rate": 16000,
            "sources": {
                "noise": [noise_piece],
                "speech": [{"length": frames, "name": speech_name, "start": 0}],
            },
            "steps": ["concat_silence", "peak_normalize", "rms_-25dbfs", "snr_scale", "target_level", "clip_guard", "pcm16"],
        }
        build_id = "toy-dns"
    case_id = digest(document)
    cases.append(
        {
            "build_id": build_id,
            "case_id": case_id,
            "case_role": "generated",
            "clean_asset_id": None,
            "doc_kind": "realization_v1",
            "frames": frames,
            "gen_index": gen_index,
            "noise_asset_id": None,
            "noisy_asset_id": None,
            "protocol_id": "protocol",
            "sample_rate": 16000,
            "speaker_id": None,
            "teacher_kind": "dry_clean",
            "utterance_key": f"{kind}-{gen_index}",
        }
    )
    docs.append({"case_id": case_id, "doc_json": document})
    return case_id, frames


def _tone(frames: int, scale: float) -> np.ndarray:
    return np.linspace(-scale, scale, frames, dtype=np.float64)


def _write_wav(path: Path, waveform: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, rate, subtype="PCM_16")


def _remove(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()


if __name__ == "__main__":
    unittest.main()
