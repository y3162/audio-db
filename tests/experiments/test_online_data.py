"""Phase 3-D online SnapshotDataModule on CPU synthetic snapshots."""

from __future__ import annotations

import hashlib
import json
import logging
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
from src.experiments.data.fixed_pair import MicroBatcher, OrderSampler, SnapshotDataModule, SnapshotSource
from src.experiments.data.online import Dfn3IndexSampler, Dfn3Protocol, SemambaProtocol
from src.experiments.data.sample import SampleRef
from src.experiments.errors import CheckpointError, DataError, StateError
from src.experiments.data.ids import canonical_text, digest
from src.experiments.data.mix.online_dfn3 import plan_online_epoch
from src.experiments.data.mix.online_semamba import plan_semamba_epoch
from src.experiments.data.sampling.order import rank_order
from src.experiments.data.protocols.online_dfn3 import (
    NUM_SAMPLES,
    render_mixture,
    sample_seed as dfn3_sample_seed,
    speech_key_positions,
)
from src.experiments.data.protocols.online_semamba import (
    SEGMENT_SIZE,
    draw_used,
    render_example,
    sample_seed as semamba_sample_seed,
)
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


class OnlineDataTests(unittest.TestCase):
    def setUp(self) -> None:
        base = Path(os.environ.get("TMPDIR", "/tmp")) / "audio-db-phase3d" / self.id().replace(".", "_")
        if base.exists():
            _remove(base)
        base.mkdir(parents=True)
        self.base = base
        self.derived = base / "derived"
        self.snapshots = base / "snapshots"
        self.dfn = _write_online(
            self.snapshots,
            corpora=base / "corpora-dfn",
            derived=self.derived,
            kind="dfn3",
            name="toy.dfn3",
            speech=5,
            factor=1,
            global_batch=2,
            valid_batch=2,
        )
        self.dfn_factor = _write_online(
            self.snapshots,
            corpora=base / "corpora-dfn-factor",
            derived=self.derived,
            kind="dfn3",
            name="toy.dfn3.factor",
            speech=4,
            factor=1.5,
            global_batch=4,
            valid_batch=3,
        )
        self.semamba = _write_online(
            self.snapshots,
            corpora=base / "corpora-semamba",
            derived=self.derived,
            kind="semamba",
            name="toy.semamba",
            speech=5,
            factor=1,
            global_batch=2,
            valid_batch=2,
        )

    def tearDown(self) -> None:
        _remove(self.base)

    def test_factory_builds_both_modules_without_duckdb_or_old_readers(self) -> None:
        import src.experiments.data.mix.online_dfn3 as dfn_mod
        import src.experiments.data.mix.online_semamba as semamba_mod

        calls = {"duck": 0, "dfn": 0, "semamba": 0}
        original_dfn = dfn_mod.OnlineDFN3TrainingData
        original_semamba = semamba_mod.OnlineSemambaTrainingData

        class _RefusedDfn:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["dfn"] += 1
                raise AssertionError("OnlineDFN3TrainingData was constructed")

        class _RefusedSemamba:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["semamba"] += 1
                raise AssertionError("OnlineSemambaTrainingData was constructed")

        class _Duck:
            @staticmethod
            def connect(*args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["duck"] += 1
                raise AssertionError("duckdb.connect was called")

        previous = sys.modules.get("duckdb")
        sys.modules["duckdb"] = _Duck()  # type: ignore[assignment]
        dfn_mod.OnlineDFN3TrainingData = _RefusedDfn  # type: ignore[misc, assignment]
        semamba_mod.OnlineSemambaTrainingData = _RefusedSemamba  # type: ignore[misc, assignment]
        try:
            for spec, sampler_type, protocol_type, policy in (
                (self.dfn, Dfn3IndexSampler, Dfn3Protocol, "dfn3_online_ceil_v1"),
                (self.semamba, OrderSampler, SemambaProtocol, "per_rank_independent_v1"),
            ):
                config = _experiment_config(self.base / spec["name"], spec)  # type: ignore[index]
                experiment = build_experiment(load_experiment(config))
                try:
                    self.assertIsInstance(experiment.get("data"), SnapshotDataModule)
                    self.assertIsInstance(experiment.get("sampler"), sampler_type)
                    self.assertIsInstance(experiment.get("protocol"), protocol_type)
                    self.assertNotIn("sampler", set(experiment._graph.states.ids()))  # type: ignore[union-attr]
                    self.assertIn("data", set(experiment._graph.states.ids()))  # type: ignore[union-attr]
                    identity = experiment.get("data").identity()
                    self.assertEqual(identity["class"], "src.experiments.data.SnapshotDataModule")
                    self.assertEqual(identity["order_policy"], policy)
                    self.assertEqual(identity["group_size"], 1)
                    self.assertIsNone(experiment.get("data").state_dict()["weights"])
                finally:
                    experiment.close()
            self.assertIsInstance(TensorPairData(features=2, examples=2, batch_size=1), TensorPairData)
            self.assertTrue(issubclass(EpochView, object))
            self.assertTrue(issubclass(Batch, object))
        finally:
            dfn_mod.OnlineDFN3TrainingData = original_dfn
            semamba_mod.OnlineSemambaTrainingData = original_semamba
            if previous is None:
                sys.modules.pop("duckdb", None)
            else:
                sys.modules["duckdb"] = previous
        self.assertEqual(calls, {"duck": 0, "dfn": 0, "semamba": 0})

    def test_dfn3_plan_matches_plan_online_epoch_including_a_short_tail(self) -> None:
        module = _dfn_module(self.dfn, micro_batch=2)
        module.setup(_Runtime())
        fractional = _dfn_module(self.dfn_factor, micro_batch=4)
        fractional.setup(_Runtime())
        try:
            for epoch in (0, 1):
                _assert_dfn_plan(self, module, epoch)
            self.assertEqual(module.sampler.order_parameters()["sampling_factor"], "1.000000")
            self.assertEqual(module.sampler.order_parameters()["speech_key_epoch"], "zero")
            self.assertNotEqual(module.plan_hash, module._epoch_plan(1).plan_hash)  # type: ignore[attr-defined]
            _assert_dfn_plan(self, fractional, 0)
            _assert_dfn_plan(self, fractional, 1)
            self.assertEqual(fractional.sampler.order_parameters()["sampling_factor"], "1.500000")
            self.assertEqual(fractional.sampler.order_parameters()["speech_key_epoch"], "epoch")
            self.assertEqual(fractional.sampler.order_parameters()["count"], 6)
            keys0 = _dfn_keys(fractional, "train", 0)
            keys1 = _dfn_keys(fractional, "train", 1)
            self.assertNotEqual(keys0, keys1)
            self.assertTrue(any(position != index for index, position in enumerate(keys0)))
        finally:
            module.close()
            fractional.close()

    def test_dfn3_waveform_matches_render_mixture(self) -> None:
        module = _dfn_module(self.dfn, micro_batch=2)
        module.setup(_Runtime())
        try:
            keys = _dfn_keys(module, "train", 0)
            sample = next(
                item for item in module.sampler.sequence(0, 0, 1) if keys[item.index] != item.index
            )
            rendered = _render_pair(module, sample)
            direct = _direct_dfn(module, sample, keys[sample.index])
            self.assertEqual(rendered[2], NUM_SAMPLES)
            self.assertEqual(int(np.max(np.abs(rendered[0] - direct.noisy))), 0)
            self.assertEqual(int(np.max(np.abs(rendered[1] - direct.clean))), 0)
            self.assertEqual(module.protocol.teacher_kind(sample), "official_mixed_target")
            self.assertEqual(sample.sample_id, f"{module.source.dataset()['protocol_id']}:train:{sample.index}")
            self.assertIsNone(sample.case_id)
        finally:
            module.close()

    def test_semamba_plan_matches_rank_order_and_drops_the_tail(self) -> None:
        for group_size in (1, 2):
            module = _semamba_module(self.semamba, micro_batch=2, group_size=group_size)
            module.setup(_Runtime())
            try:
                for epoch in (0, 1):
                    order = rank_order(
                        "per_rank_independent_v1",
                        5,
                        seed=7,
                        epoch=epoch,
                        world_size=1,
                        rank=0,
                    )
                    self.assertEqual([sample.index for sample in module.sampler.sequence(epoch, 0, 1)], order)
                    plan = plan_semamba_epoch(
                        split="train",
                        epoch=epoch,
                        count=5,
                        seed=7,
                        world_size=1,
                        micro_batch=2,
                        accumulation=group_size,
                        protocol_id=str(module.source.dataset()["protocol_id"]),
                        snapshot_hash=module.source.snapshot_hash,
                    )
                    grouped = module._epoch_plan(epoch).groups[0]  # type: ignore[attr-defined]
                    expected = [
                        [sample.index for sample in micro.samples]
                        for update in plan.rank_groups[0]
                        for micro in update.micro_batches
                    ]
                    self.assertEqual([[sample.index for sample in group] for group in grouped], expected)
                    self.assertEqual(len(grouped) // group_size, plan.updates)
                    self.assertLess(sum(len(group) for group in grouped), 5)
                self.assertNotEqual(module.plan_hash, module._epoch_plan(1).plan_hash)  # type: ignore[attr-defined]
            finally:
                module.close()

    def test_semamba_waveform_matches_render_example(self) -> None:
        module = _semamba_module(self.semamba, micro_batch=2, group_size=1)
        module.setup(_Runtime())
        try:
            sample = module.sampler.sequence(1, 0, 1)[0]
            rendered = _render_pair(module, sample)
            direct = _direct_semamba(module, sample)
            self.assertEqual(rendered[2], SEGMENT_SIZE)
            self.assertEqual(rendered[0].shape, (SEGMENT_SIZE,))
            self.assertEqual(int(np.max(np.abs(rendered[0] - direct.degraded))), 0)
            self.assertEqual(int(np.max(np.abs(rendered[1] - direct.clean))), 0)
            self.assertEqual(module.protocol.teacher_kind(sample), "dry_clean")
            epoch_key = sample.realization_id.rsplit(":", 2)[-2]
            self.assertEqual(epoch_key, "1")
        finally:
            module.close()

    def test_validation_keeps_epoch_rules_and_supplies_every_row(self) -> None:
        dfn = _dfn_module(self.dfn, micro_batch=2)
        semamba = _semamba_module(self.semamba, micro_batch=2, group_size=2)
        dfn.setup(_Runtime())
        semamba.setup(_Runtime())
        try:
            first = _validation_map(dfn, 0)
            second = _validation_map(dfn, 1)
            self.assertEqual(sorted(first), list(range(5)))
            self.assertEqual(sorted(second), list(range(5)))
            self.assertEqual(len(first), len(second))
            for index, wave in first.items():
                self.assertEqual(int(np.max(np.abs(wave[0] - second[index][0]))), 0)
                self.assertEqual(int(np.max(np.abs(wave[1] - second[index][1]))), 0)
            semamba_first = _validation_map(semamba, 0)
            semamba_second = _validation_map(semamba, 1)
            self.assertEqual(list(semamba_first), list(range(5)))
            self.assertEqual(list(semamba_second), list(semamba_first))
            for index, wave in semamba_first.items():
                self.assertEqual(int(np.max(np.abs(wave[0] - semamba_second[index][0]))), 0)
                self.assertEqual(int(np.max(np.abs(wave[1] - semamba_second[index][1]))), 0)
            valid = next(iter(semamba.validation_epoch(0)))
            self.assertEqual(valid.sample_ids[0], semamba.source.case_ids("valid.cases")[0])
            self.assertIn(":valid:0:", valid.realization_ids[0])
        finally:
            dfn.close()
            semamba.close()

    def test_resume_matches_the_next_batch_and_rejects_tampering(self) -> None:
        for spec, label in ((self.dfn, "dfn3"), (self.semamba, "semamba")):
            with self.subTest(label=label):
                _assert_resume(self, spec)

    def test_rejections(self) -> None:
        module = _dfn_module(self.dfn, micro_batch=2, group_size=2)
        with self.assertRaises(DataError):
            module.setup(_Runtime())
        module.close()
        mismatched = _dfn_module(self.dfn, micro_batch=2, global_batch=8)
        with self.assertRaises(DataError):
            mismatched.setup(_Runtime())
        mismatched.close()
        semamba = _semamba_module(self.semamba, micro_batch=2, group_size=1, global_batch=8)
        with self.assertRaises(DataError):
            semamba.setup(_Runtime())
        semamba.close()
        warned = _dfn_module(self.dfn_factor, micro_batch=2)
        with self.assertLogs("src.experiments.data", level="WARNING") as captured:
            warned.setup(_Runtime())
        self.assertIn("shipped cap", "\n".join(captured.output))
        sample = warned.sampler.sequence(0, 0, 1)[0]
        outside = SampleRef(
            sample.sample_id,
            sample.realization_id,
            99,
            case_id=sample.case_id,
            split=sample.split,
            plan_epoch=sample.plan_epoch,
        )
        with self.assertRaises(DataError):
            warned.protocol.render(outside)
        warned.close()
        bad = _write_online(
            self.snapshots,
            corpora=self.base / "corpora-bad-teacher",
            derived=self.derived,
            kind="dfn3",
            name="toy.dfn3.bad-teacher",
            speech=5,
            factor=1,
            global_batch=2,
            valid_batch=2,
            teacher="dry_clean",
        )
        bad_module = _dfn_module(bad, micro_batch=2)
        bad_module.setup(_Runtime())
        try:
            with self.assertRaises(DataError):
                next(bad_module.validation_epoch(0))
        finally:
            bad_module.close()
        blocked = _write_online(
            self.snapshots,
            corpora=self.base / "corpora-blocked",
            derived=self.derived,
            kind="dfn3",
            name="toy.dfn3.blocked",
            speech=5,
            factor=1,
            global_batch=2,
            valid_batch=2,
            status="blocked",
        )
        blocked_module = _dfn_module(blocked, micro_batch=2)
        with self.assertRaises(DataError):
            blocked_module.setup(_Runtime())
        blocked_module.close()
        missing = _write_online(
            self.snapshots,
            corpora=self.base / "corpora-missing",
            derived=self.derived,
            kind="semamba",
            name="toy.semamba.missing",
            speech=5,
            factor=1,
            global_batch=2,
            valid_batch=2,
            skip_noise=True,
        )
        missing_module = _semamba_module(missing, micro_batch=1, group_size=1)
        missing_module.setup(_Runtime())
        try:
            with self.assertRaises(DataError):
                missing_module.train_epoch(0, start_update=0).batch(0)
        finally:
            missing_module.close()
        live = _dfn_module(self.dfn, micro_batch=2)
        live.setup(_Runtime())
        try:
            state = live.state_dict()
            state["plan_hash"] = "f" * 64
            with self.assertRaises(StateError):
                live.load_state_dict(state)
            state = live.state_dict()
            state["identity"] = dict(state["identity"])
            state["identity"]["dataset_id"] = "f" * 64
            with self.assertRaises(StateError):
                live.load_state_dict(state)
        finally:
            live.close()


def _assert_dfn_plan(test: OnlineDataTests, module: SnapshotDataModule, epoch: int) -> None:
    dataset = module.source.dataset()
    count = int(module.sampler.order_parameters()["count"])
    plan = plan_online_epoch(
        split="train",
        epoch=epoch,
        count=count,
        seed=7,
        world_size=1,
        micro_batch=module.batcher.micro_batch,
        protocol_id=str(dataset["protocol_id"]),
        snapshot_hash=module.source.snapshot_hash,
        case_ids=module.sampler._case_list("train"),  # type: ignore[attr-defined]
    )
    expected = [
        [int(sample.index) for sample in micro.samples]
        for update in plan.rank_groups[0]
        for micro in update.micro_batches
    ]
    grouped = module._epoch_plan(epoch).groups[0]  # type: ignore[attr-defined]
    actual = [[sample.index for sample in group] for group in grouped]
    test.assertEqual(actual, expected)
    test.assertTrue(expected)
    test.assertLess(len(expected[-1]), module.batcher.micro_batch)
    flat = [index for group in expected for index in group]
    keys = _dfn_keys(module, "train", epoch)
    test.assertTrue(all(0 <= index < len(keys) for index in flat))
    covered = sorted(
        int(sample.index)
        for shard_groups in plan.rank_groups
        for update in shard_groups
        for micro in update.micro_batches
        for sample in micro.samples
    )
    test.assertEqual(covered, list(range(count)))


def _assert_resume(test: OnlineDataTests, spec: dict[str, object]) -> None:
    seed = 7 if spec["kind"] == "dfn3" else _safe_semamba_seed(spec)
    interrupted = test.base / f"interrupted-{spec['name']}"
    continuous = test.base / f"continuous-{spec['name']}"
    first = _cli(_experiment_config(interrupted, spec, steps=2, stop_after=1, prefetch=True, seed=seed))
    test.assertEqual(first.returncode, 0, first.stderr)
    saved = interrupted / "step_00000001.pt"
    test.assertTrue(saved.is_file())
    whole = _cli(_experiment_config(continuous, spec, steps=2, prefetch=False, seed=seed))
    test.assertEqual(whole.returncode, 0, whole.stderr)
    resumed = _cli(
        _experiment_config(interrupted, spec, steps=2, prefetch=True, seed=seed),
        "--resume",
        str(saved),
    )
    test.assertEqual(resumed.returncode, 0, resumed.stderr)
    test.assertEqual(_metrics(interrupted)[1]["sample_id"], _metrics(continuous)[1]["sample_id"])
    experiment = build_experiment(
        load_experiment(_experiment_config(interrupted, spec, steps=2, prefetch=True, seed=seed))
    )
    fresh = _module_for(spec, prefetch=False, seed=seed)
    fresh.setup(_Runtime())
    try:
        experiment._checkpoint.restore(saved, mode="resume")  # type: ignore[union-attr]
        data = experiment.get("data")
        test.assertEqual(data.update_index, 1)
        test.assertEqual(data.epoch, 0)
        test.assertEqual(data.state_dict()["plan_hash"], fresh.plan_hash_for_state({"epoch": 0}))
        test.assertEqual(data.identity()["dataset_id"], fresh.identity()["dataset_id"])
        continued = data.train_epoch(0, start_update=1).batch(1)[0]
        expected = fresh.train_epoch(0, start_update=0).batch(1)[0]
        test.assertEqual(continued.sample_ids, expected.sample_ids)
        test.assertEqual(continued.realization_ids, expected.realization_ids)
        test.assertEqual(int(torch.max(torch.abs(continued.inputs - expected.inputs))), 0)
        test.assertEqual(int(torch.max(torch.abs(continued.targets - expected.targets))), 0)
        test.assertEqual(continued.inputs.dtype, torch.float32)
        test.assertEqual(continued.inputs.device.type, "cpu")
    finally:
        experiment.close()
        fresh.close()
    payload = torch.load(saved, map_location="cpu", weights_only=False)
    payload["states"]["data"]["plan_hash"] = "f" * 64
    tampered = interrupted / "tampered.pt"
    torch.save(payload, tampered)
    again = build_experiment(
        load_experiment(_experiment_config(interrupted, spec, steps=2, prefetch=True, seed=seed))
    )
    try:
        with test.assertRaises(CheckpointError):
            again._checkpoint.restore(tampered, mode="resume")  # type: ignore[union-attr]
    finally:
        again.close()


def _dfn_module(
    spec: dict[str, object],
    *,
    micro_batch: int,
    group_size: int = 1,
    global_batch: int | None = None,
    prefetch: bool = False,
    seed: int = 7,
) -> SnapshotDataModule:
    source = _source(spec)
    return SnapshotDataModule(
        source,
        Dfn3IndexSampler(source, seed=seed, global_batch=global_batch),
        Dfn3Protocol(source),
        MicroBatcher(micro_batch, group_size),
        prefetch=prefetch,
    )


def _semamba_module(
    spec: dict[str, object],
    *,
    micro_batch: int,
    group_size: int,
    global_batch: int | None = None,
    prefetch: bool = False,
    seed: int = 7,
) -> SnapshotDataModule:
    source = _source(spec)
    protocol = SemambaProtocol(source)
    return SnapshotDataModule(
        source,
        OrderSampler(source, protocol, seed=seed, global_batch=global_batch),
        protocol,
        MicroBatcher(micro_batch, group_size),
        prefetch=prefetch,
    )


def _module_for(spec: dict[str, object], *, prefetch: bool, seed: int = 7) -> SnapshotDataModule:
    if spec["kind"] == "dfn3":
        return _dfn_module(spec, micro_batch=1, prefetch=prefetch, seed=seed)
    return _semamba_module(spec, micro_batch=1, group_size=1, prefetch=prefetch, seed=seed)


def _safe_semamba_seed(spec: dict[str, object]) -> int:
    """Pick a draw ``.venvs/se`` can render. Kaiser resampling needs ``resampy``."""

    source = _source(spec)
    source.open()
    try:
        protocol_id = str(source.dataset()["protocol_id"])
        snapshot_hash = source.snapshot_hash
        for seed in range(0, 80):
            order = rank_order(
                "per_rank_independent_v1",
                5,
                seed=seed,
                epoch=0,
                world_size=1,
                rank=0,
            )
            if all(not _needs_resampy(protocol_id, snapshot_hash, seed, index) for index in order[:2]):
                return seed
    finally:
        source.close()
    raise AssertionError("no SEMamba seed avoids kaiser resampling")


def _needs_resampy(protocol_id: str, snapshot_hash: str, seed: int, index: int) -> bool:
    rng = np.random.Generator(
        np.random.PCG64(semamba_sample_seed(protocol_id, snapshot_hash, seed, "train", 0, index))
    )
    used, _discarded = draw_used(rng)
    return "bandwidth" in used.selected and used.bandwidth_type in {"kaiser_best", "kaiser_fast"}


def _source(spec: dict[str, object]) -> SnapshotSource:
    return SnapshotSource(
        str(spec["selector"]),
        str(spec["root"]),
        "train",
        derived=str(spec["derived"]),
        corpora=str(spec["corpora"]),
        generated=str(spec["generated"]),
    )


def _dfn_keys(module: SnapshotDataModule, split: str, epoch: int) -> list[int]:
    dataset = module.source.dataset()
    speech = module.source.material_ids(f"{split}.speech")
    return speech_key_positions(
        len(speech),
        float(dataset["rule_json"]["online"]["sampling_factor"]),
        protocol_id=str(dataset["protocol_id"]),
        snapshot_hash=module.source.snapshot_hash,
        seed=module.sampler.seed,
        split=split,
        epoch=epoch,
    )


def _render_pair(module: SnapshotDataModule, sample: SampleRef) -> tuple[np.ndarray, np.ndarray, int]:
    return module.protocol.render(sample)  # type: ignore[no-any-return]


def _direct_dfn(module: SnapshotDataModule, sample: SampleRef, speech_index: int):
    protocol = module.protocol
    assert isinstance(protocol, Dfn3Protocol)
    split, epoch, seed = sample.split, int(sample.plan_epoch), module.sampler.seed  # type: ignore[arg-type]
    rng = np.random.Generator(
        np.random.PCG64(
            dfn3_sample_seed(
                str(module.source.dataset()["protocol_id"]),
                module.source.snapshot_hash,
                seed,
                str(split),
                epoch,
                sample.index,
            )
        )
    )
    return render_mixture(
        protocol.spec,
        protocol._role(str(split), "speech"),
        protocol._role(str(split), "noise"),
        protocol._role(str(split), "rir"),
        speech_index=speech_index,
        split=str(split),
        rng=rng,
    )


def _direct_semamba(module: SnapshotDataModule, sample: SampleRef):
    protocol = module.protocol
    assert isinstance(protocol, SemambaProtocol)
    split, epoch, seed = sample.split, int(sample.plan_epoch), module.sampler.seed  # type: ignore[arg-type]
    rng = np.random.Generator(
        np.random.PCG64(
            semamba_sample_seed(
                str(module.source.dataset()["protocol_id"]),
                module.source.snapshot_hash,
                seed,
                str(split),
                epoch,
                sample.index,
            )
        )
    )
    return render_example(
        protocol._role(str(split), "speech"),
        protocol._role(str(split), "noise"),
        protocol._role(str(split), "rir"),
        speech_index=sample.index,
        split=str(split),
        rng=rng,
    )


def _validation_map(module: SnapshotDataModule, epoch: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    found: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    order: list[int] = []
    for batch in module.validation_epoch(epoch):
        for row, sample_index in enumerate(_batch_indexes(module, epoch, batch)):
            found[sample_index] = (
                batch.inputs[row].numpy().copy(),
                batch.targets[row].numpy().copy(),
            )
            order.append(sample_index)
    return {index: found[index] for index in order}


def _batch_indexes(module: SnapshotDataModule, epoch: int, batch: Batch) -> list[int]:
    ranks = module.sampler.validation_ranks(epoch, 1)[0]  # type: ignore[attr-defined]
    by_realization = {sample.realization_id: sample.index for sample in ranks}
    return [by_realization[realization_id] for realization_id in batch.realization_ids]


def _experiment_config(
    directory: Path,
    spec: dict[str, object],
    *,
    steps: int = 1,
    stop_after: int | None = None,
    prefetch: bool = False,
    seed: int = 7,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    if spec["kind"] == "dfn3":
        sampler = f"""    class: src.experiments.data.Dfn3IndexSampler
    kwargs:
      seed: {seed}
    inject:
      source: {{ref: source}}"""
        protocol = "src.experiments.data.Dfn3Protocol"
    else:
        sampler = f"""    class: src.experiments.data.OrderSampler
    kwargs:
      seed: {seed}
    inject:
      source: {{ref: source}}
      protocol: {{ref: protocol}}"""
        protocol = "src.experiments.data.SemambaProtocol"
    text = f"""schema: audio-db-experiment/v1
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
{sampler}
  protocol:
    role: other
    class: {protocol}
    kwargs: {{}}
    inject:
      source: {{ref: source}}
  batcher:
    role: other
    class: src.experiments.data.MicroBatcher
    kwargs:
      micro_batch: 1
      group_size: 1
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
      accumulation: 1
      checkpoint_interval: 1
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


def _write_online(
    root: Path,
    *,
    corpora: Path,
    derived: Path,
    kind: str,
    name: str,
    speech: int,
    factor: float,
    global_batch: int,
    valid_batch: int,
    teacher: str | None = None,
    status: str = "ready",
    skip_noise: bool = False,
) -> dict[str, object]:
    if kind == "dfn3":
        rate = 48000
        frames = 48000
        algorithm = "online.dfn3"
        default_teacher = "official_mixed_target"
        window = NUM_SAMPLES
        order = "dfn3_online_ceil_v1"
    else:
        rate = 16000
        frames = 24000
        algorithm = "online.semambapp_gsr"
        default_teacher = "dry_clean"
        window = SEGMENT_SIZE
        order = "per_rank_independent_v1"
    teacher_name = default_teacher if teacher is None else teacher
    count = max(int(round(speech * factor)), 1) if kind == "dfn3" else speech
    assets: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    items: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    for split in ("train", "valid"):
        for role in ("speech", "noise", "rir"):
            if skip_noise and role == "noise":
                continue
            material_id = digest({"kind": kind, "name": name, "split": split, "role": role})
            bindings.append({"slot": f"{split}.{role}", "material_set_id": material_id, "case_set_id": None})
            width = speech if role == "speech" else max(speech, 2)
            for position in range(width):
                rel = f"online/{kind}/{split}/{role}/{position}.wav"
                path = corpora / rel
                _write_wav(path, _wave(frames, role, position), rate)
                asset_id = f"corpora/{rel}"
                assets.append(
                    {
                        "asset_id": asset_id,
                        "rel_path": rel,
                        "sample_rate": rate,
                        "frames": frames,
                        "channels": 1,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "integrity": "sha256",
                    }
                )
                members.append(
                    {
                        "material_set_id": material_id,
                        "position": position,
                        "asset_id": asset_id,
                        "group_key": role,
                    }
                )
        if split != "train":
            ids = []
            for position in range(count):
                case_id = digest({"kind": kind, "name": name, "split": split, "position": position, "teacher": teacher_name})
                ids.append(case_id)
                cases.append(
                    {
                        "build_id": name,
                        "case_id": case_id,
                        "case_role": "online",
                        "clean_asset_id": None,
                        "doc_kind": "online_v1",
                        "frames": window,
                        "gen_index": position,
                        "noise_asset_id": None,
                        "noisy_asset_id": None,
                        "protocol_id": "protocol",
                        "sample_rate": rate,
                        "speaker_id": None,
                        "teacher_kind": teacher_name,
                        "utterance_key": f"{kind}-{split}-{position}",
                    }
                )
            case_set_id = digest(ids)
            bindings.append({"slot": f"{split}.cases", "material_set_id": None, "case_set_id": case_set_id})
            for position, case_id in enumerate(ids):
                items.append({"case_id": case_id, "case_set_id": case_set_id, "position": position})
    dataset_id = digest({"algorithm": algorithm, "kind": kind, "name": name, "factor": factor, "status": status})
    dataset = {
        "algorithm": algorithm,
        "algorithm_version": 1,
        "bindings": bindings,
        "claim_label": "protocol_conformant_substitute_materials",
        "dataset_id": dataset_id,
        "declared_status": status,
        "kind": "online_mixing",
        "name": name,
        "profile": "protocol_conformant_substitute_materials",
        "protocol_id": "protocol",
        "rule_json": {
            "delivery": "on_demand",
            "global_batch": global_batch,
            "online": {"sampling_factor": factor},
            "order": order,
            "sample_rate": rate,
            "teacher": default_teacher,
            "validation": {"batch": valid_batch},
            "window": window,
        },
        "selection_uses_test": False,
        "version": 1,
    }
    directory = root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "docs.jsonl").write_text("", encoding="utf-8")
    parts = {
        "assets": assets,
        "case_set_items": items,
        "cases": cases,
        "dataset": dataset,
        "docs": [],
        "generated": [],
        "material_set_members": members,
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
        raise RuntimeError("synthetic online snapshot hash did not round-trip")
    generated = root / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    return {
        "corpora": corpora,
        "derived": derived,
        "directory": directory,
        "generated": generated,
        "kind": kind,
        "name": name,
        "root": root,
        "selector": f"{name}@1",
    }


def _wave(frames: int, role: str, position: int) -> np.ndarray:
    time = np.arange(frames, dtype=np.float64)
    if role == "rir":
        wave = np.zeros(frames, dtype=np.float32)
        wave[0] = 0.6
        wave[min(32, frames - 1)] = 0.2
        return wave
    frequency = 40 + position * 3 + (10 if role == "noise" else 0)
    amplitude = 0.2 if role == "speech" else 0.08
    return np.asarray(amplitude * np.sin(2 * np.pi * frequency * time / frames), dtype=np.float32)


def _write_wav(path: Path, wave: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, rate, subtype="PCM_16")


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
    logging.basicConfig(level=logging.INFO)
    unittest.main()
