"""Phase 3-B fixed-pair SnapshotDataModule on CPU synthetic snapshots."""

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

from src.experiments.api import Experiment, build_experiment, preflight_experiment
from src.experiments.config import load_experiment
from src.experiments.data import Batch, EpochView, TensorPairData
from src.experiments.data.fixed_pair import (
    FixedPairProtocol,
    MicroBatcher,
    OrderSampler,
    SnapshotDataModule,
    SnapshotSource,
)
from src.experiments.data.identity import DISPLAY_IDENTITY_KEYS, comparable_identity
from src.experiments.errors import BuildError, CheckpointError, DataError, RuntimeFailure, StateError
from src.experiments.trainers import _batch_on_device
from src.reproduction.ids import canonical_text, digest
from src.reproduction.pipeline.order import rank_order
from src.reproduction.pipeline.planner import plan_fixed_pair, validation_refs
from src.reproduction.pipeline.renderer import render_sample
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.snapshot_reader import (
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


class FixedPairDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.id().split(".")[-1])
        base = Path(os.environ.get("TMPDIR", "/tmp")) / "audio-db-phase3b" / self.id().replace(".", "_")
        if base.exists():
            _remove(base)
        base.mkdir(parents=True)
        self.base = base
        self.derived = base / "derived"
        self.snapshots = base / "snapshots"
        self.mp = _write_snapshot(
            self.snapshots,
            derived=self.derived,
            name="toy.mp",
            algorithm="fixed_pair.mp_senet",
            normalize="noisy_rms_unit_v1",
            order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            profile="leakage_safe",
            selection_uses_test=False,
            claim_label="leakage_safe_comparison",
            validation_crop="full_length",
        )
        self.moss = _write_snapshot(
            self.snapshots,
            derived=self.derived,
            name="toy.moss",
            algorithm="fixed_pair.mossformergan_se_16k",
            normalize="independent_audio_norm_v1",
            order="rank_mod_fixed_v1",
            profile="leakage_safe",
            selection_uses_test=False,
            claim_label="leakage_safe_comparison",
            validation_crop="per_epoch_random",
        )
        self.official = _write_snapshot(
            self.snapshots,
            derived=self.derived,
            name="toy.official",
            algorithm="fixed_pair.mp_senet",
            normalize="noisy_rms_unit_v1",
            order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            profile="official_faithful",
            selection_uses_test=True,
            claim_label="official_reproduction",
            validation_crop="full_length",
        )

    def tearDown(self) -> None:
        _remove(self.base)

    def test_factory_builds_components_without_duckdb_or_training_data(self) -> None:
        import src.reproduction.pipeline.training_data as training_data

        calls = {"duck": 0, "training": 0}
        original_training = training_data.ReproductionTrainingData

        class _RefusedTraining:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["training"] += 1
                raise AssertionError("ReproductionTrainingData was constructed")

        class _Duck:
            @staticmethod
            def connect(*args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["duck"] += 1
                raise AssertionError("duckdb.connect was called")

        previous = sys.modules.get("duckdb")
        sys.modules["duckdb"] = _Duck()  # type: ignore[assignment]
        training_data.ReproductionTrainingData = _RefusedTraining  # type: ignore[misc, assignment]
        config = _experiment_config(self.base / "build", self.mp, require_profile="leakage_safe")
        try:
            experiment = build_experiment(load_experiment(config))
            try:
                self.assertIsInstance(experiment.get("data"), SnapshotDataModule)
                self.assertIsInstance(experiment.get("source"), SnapshotSource)
                experiment.get("source").dataset()
                state_ids = set(experiment._graph.states.ids())  # type: ignore[union-attr]
                self.assertIn("data", state_ids)
                for name in ("source", "sampler", "protocol", "batcher"):
                    self.assertNotIn(name, state_ids)
                self.assertIsInstance(TensorPairData(features=2, examples=2, batch_size=1), TensorPairData)
                self.assertTrue(issubclass(EpochView, object))
                self.assertTrue(issubclass(Batch, object))
            finally:
                experiment.close()
            with self.assertRaises(DataError):
                experiment.get("source").dataset()
        finally:
            training_data.ReproductionTrainingData = original_training
            if previous is None:
                sys.modules.pop("duckdb", None)
            else:
                sys.modules["duckdb"] = previous
        self.assertEqual(calls, {"duck": 0, "training": 0})

    def test_same_index_is_stable_and_metadata_matches(self) -> None:
        module = _module(self.mp)
        module.setup(_Runtime())
        try:
            view = module.train_epoch(0, start_update=0)
            first = view.batch(0)
            second = view.batch(0)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertEqual(int(torch.max(torch.abs(first[0].inputs - second[0].inputs))), 0)
            self.assertEqual(int(torch.max(torch.abs(first[0].targets - second[0].targets))), 0)
            batch = first[0]
            self.assertEqual(batch.sample_id, ",".join(batch.sample_ids))
            self.assertEqual(batch.realization_id, ",".join(batch.realization_ids))
            self.assertIsNotNone(batch.lengths)
            assert batch.lengths is not None
            for length in batch.lengths.tolist():
                self.assertLessEqual(int(length), batch.inputs.shape[-1])
            self.assertNotIsInstance(batch.sample_ids, np.ndarray)
            with self.assertRaises(StateError):
                view.batch(view.updates)
            with self.assertRaises(StateError):
                view.batch(-1)
            with self.assertRaises(StateError):
                module.train_epoch(0, start_update=1).batch(0)
        finally:
            module.close()
            module.close()

    def test_waveforms_match_plan_fixed_pair_for_both_normalizers(self) -> None:
        import src.reproduction.pipeline.training_data as training_data

        calls = {"training": 0}
        original = training_data.ReproductionTrainingData

        class _Refused:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                calls["training"] += 1
                raise AssertionError("ReproductionTrainingData was constructed")

        training_data.ReproductionTrainingData = _Refused  # type: ignore[misc, assignment]
        try:
            for spec, normalize in ((self.mp, "noisy_rms_unit_v1"), (self.moss, "independent_audio_norm_v1")):
                for epoch in (0, 1):
                    _assert_matches_plan(self, spec, normalize=normalize, epoch=epoch)
        finally:
            training_data.ReproductionTrainingData = original
        self.assertEqual(calls["training"], 0)

    def test_order_follows_rank_order(self) -> None:
        mp = _module(self.mp, seed=7)
        moss = _module(self.moss, seed=7)
        mp.setup(_Runtime())
        moss.setup(_Runtime())
        try:
            mp_cases = mp.source.case_ids("train.cases")
            moss_cases = moss.source.case_ids("train.cases")
            for epoch in (0, 1):
                mp_ids = [sample.case_id for sample in mp.sampler.sequence(epoch, 0, 1)]
                moss_ids = [sample.case_id for sample in moss.sampler.sequence(epoch, 0, 1)]
                self.assertEqual(
                    mp_ids,
                    [mp_cases[index] for index in rank_order("fixed_single_shuffle_v1", len(mp_cases), seed=7, epoch=epoch, world_size=1, rank=0)],
                )
                self.assertEqual(
                    moss_ids,
                    [moss_cases[index] for index in rank_order("rank_mod_fixed_v1", len(moss_cases), seed=7, epoch=epoch, world_size=1, rank=0)],
                )
            self.assertEqual(
                [sample.case_id for sample in mp.sampler.sequence(0, 0, 1)],
                [sample.case_id for sample in mp.sampler.sequence(1, 0, 1)],
            )
        finally:
            mp.close()
            moss.close()

    def test_identity_ignores_display_fields_and_keeps_selection_flag(self) -> None:
        module = _module(self.mp, require_profile="leakage_safe")
        module.setup(_Runtime())
        try:
            first = module.identity()
            second = module.identity()
            self.assertEqual(first["dataset_id"], second["dataset_id"])
            self.assertEqual(first["snapshot_hash"], second["snapshot_hash"])
            compared = comparable_identity(first)
            self.assertTrue(DISPLAY_IDENTITY_KEYS.isdisjoint(compared))
            self.assertIn("dataset_id", compared)
            self.assertIn("snapshot_hash", compared)
            state = module.state_dict()
            renamed = json.loads(json.dumps(state))
            renamed["identity"]["dataset_name"] = "renamed-only"
            module.load_state_dict(renamed)
            renamed["identity"]["snapshot_hash"] = "0" * 64
            with self.assertRaises(StateError):
                module.load_state_dict(renamed)
            renamed = json.loads(json.dumps(state))
            renamed["identity"]["dataset_id"] = "0" * 64
            with self.assertRaises(StateError):
                module.load_state_dict(renamed)
            renamed = json.loads(json.dumps(state))
            renamed["plan_hash"] = renamed["plan_hash"][:-1] + ("0" if renamed["plan_hash"][-1] != "0" else "1")
            with self.assertRaises(StateError):
                module.load_state_dict(renamed)
            self.assertIsNone(module.state_dict()["weights"])
        finally:
            module.close()
        refused = _experiment_config(self.base / "refused", self.official, require_profile="leakage_safe")
        experiment = Experiment(load_experiment(refused))
        with self.assertRaises(BuildError) as caught:
            experiment.build()
        self.assertIsInstance(caught.exception.__cause__, DataError)
        self.assertTrue(experiment._closed)
        source = experiment._partial_components.get("source")  # type: ignore[union-attr]
        data = experiment._partial_components.get("data")  # type: ignore[union-attr]
        self.assertTrue(data._closed)
        with self.assertRaises(DataError):
            source.dataset()
        experiment.close()
        allowed = build_experiment(load_experiment(_experiment_config(self.base / "official", self.official)))
        try:
            identity = allowed.get("data").identity()
            self.assertTrue(identity["selection_uses_test"])
            self.assertEqual(identity["profile"], "official_faithful")
        finally:
            allowed.close()

    def test_resume_discards_prefetched_audio_and_matches_the_next_sample(self) -> None:
        interrupted = self.base / "interrupted"
        continuous = self.base / "continuous"
        interrupted_config = _experiment_config(
            interrupted,
            self.mp,
            steps=4,
            stop_after=2,
            prefetch=True,
            checkpoint_interval=1,
        )
        continuous_config = _experiment_config(
            continuous,
            self.mp,
            steps=4,
            stop_after=None,
            prefetch=False,
            checkpoint_interval=1,
        )
        first = _cli(interrupted_config)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertFalse((interrupted / "complete.json").is_file())
        saved = interrupted / "step_00000002.pt"
        self.assertTrue(saved.is_file())
        resume_config = _experiment_config(
            interrupted,
            self.mp,
            steps=4,
            stop_after=None,
            prefetch=True,
            checkpoint_interval=1,
        )
        whole = _cli(continuous_config)
        self.assertEqual(whole.returncode, 0, whole.stderr)
        resumed = _cli(resume_config, "--resume", str(saved))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(_metrics(resumed_dir := interrupted)[2]["sample_id"], _metrics(continuous)[2]["sample_id"])
        fresh = _module(self.mp, prefetch=False)
        fresh.setup(_Runtime())
        experiment = build_experiment(load_experiment(resume_config))
        try:
            experiment._checkpoint.restore(saved, mode="resume")  # type: ignore[union-attr]
            data = experiment.get("data")
            self.assertEqual(data.update_index, 2)
            self.assertEqual(data.epoch, 0)
            continued = data.train_epoch(0, start_update=2).batch(2)[0]
            expected = fresh.train_epoch(0, start_update=0).batch(2)[0]
            self.assertEqual(continued.sample_id, expected.sample_id)
            self.assertEqual(int(torch.max(torch.abs(continued.inputs - expected.inputs))), 0)
            self.assertEqual(int(torch.max(torch.abs(continued.targets - expected.targets))), 0)
            prefetch = _module(self.mp, prefetch=True)
            prefetch.setup(_Runtime())
            try:
                left = prefetch.train_epoch(0, start_update=0).batch(0)[0]
                right = fresh.train_epoch(0, start_update=0).batch(0)[0]
                self.assertEqual(int(torch.max(torch.abs(left.inputs - right.inputs))), 0)
            finally:
                prefetch.close()
        finally:
            experiment.close()
            fresh.close()
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        payload["states"]["data"]["plan_hash"] = "f" * 64
        tampered = interrupted / "step_00000002.pt"
        torch.save(payload, tampered)
        again = build_experiment(load_experiment(resume_config))
        try:
            with self.assertRaises(CheckpointError):
                again._checkpoint.restore(tampered, mode="resume")  # type: ignore[union-attr]
        finally:
            again.close()
        del resumed_dir

    def test_initialize_from_does_not_restore_the_data_cursor(self) -> None:
        directory = self.base / "init"
        config = _experiment_config(directory, self.mp, steps=1)
        experiment = build_experiment(load_experiment(config))
        data = experiment.get("data")
        data.epoch = 0
        data.update_index = 2
        path = experiment._checkpoint.save(  # type: ignore[union-attr]
            experiment._directory,  # type: ignore[arg-type]
            logical_step=0,
            label="init",
            initialize_allow=["model"],
        )
        experiment.close()
        restored = build_experiment(load_experiment(config))
        try:
            restored._checkpoint.restore(path, mode="initialize")  # type: ignore[union-attr]
            self.assertEqual(restored.get("data").update_index, 0)
            self.assertEqual(restored.get("data").epoch, 0)
        finally:
            restored.close()

    def test_close_covers_preflight_run_failure_and_group_size(self) -> None:
        preflight_config = _experiment_config(self.base / "preflight", self.mp)
        experiment = build_experiment(load_experiment(preflight_config))
        source = experiment.get("source")
        source.dataset()
        preflight_experiment(experiment)
        source.dataset()
        experiment.close()
        with self.assertRaises(DataError):
            source.dataset()
        run_config = _experiment_config(self.base / "run", self.mp, steps=1)
        ran = build_experiment(load_experiment(run_config))
        result = ran.run()
        self.assertEqual(result.logical_step, 1)
        with self.assertRaises(DataError):
            ran.get("source").dataset()
        boom_config = _experiment_config(self.base / "boom", self.mp, loss="tests.experiments.support.BoomLoss")
        boom = build_experiment(load_experiment(boom_config))
        with self.assertRaises(RuntimeError):
            boom.run()
        with self.assertRaises(DataError):
            boom.get("source").dataset()
        mismatch = _experiment_config(self.base / "mismatch", self.mp, accumulation=2)
        mismatched = build_experiment(load_experiment(mismatch))
        with self.assertRaises(RuntimeFailure):
            mismatched.run()
        with self.assertRaises(DataError):
            mismatched.get("source").dataset()

    def test_cpu_batch_stays_on_cpu_and_validation_matches_full_length(self) -> None:
        batch = Batch(
            inputs=torch.ones(1, 4),
            targets=torch.zeros(1, 4),
            sample_id="case",
            realization_id="case",
            lengths=torch.tensor([4]),
        )
        moved = _batch_on_device(batch, torch.device("cpu"))
        self.assertIs(moved.inputs, batch.inputs)
        module = _module(self.mp)
        module.setup(_Runtime())
        try:
            view = module.train_epoch(0, start_update=0)
            trained = view.batch(0)[0]
            self.assertEqual(trained.inputs.device.type, "cpu")
            self.assertEqual(trained.teacher_kind, "dry_clean")
            valid = list(module.validation_epoch(0))
            again = list(module.validation_epoch(1))
            self.assertEqual(len(valid), 1)
            self.assertEqual(int(torch.max(torch.abs(valid[0].inputs - again[0].inputs))), 0)
            self.assertIn("crop-full", valid[0].realization_id)
        finally:
            module.close()
        other = _module(self.mp)
        other.setup(_Runtime(world_size=2, rank=1))
        try:
            self.assertEqual(list(other.validation_epoch(0)), [])
            rank0 = _module(self.mp)
            rank0.setup(_Runtime(world_size=2, rank=0))
            try:
                self.assertEqual(rank0.plan_hash_for_state({"epoch": 0}), other.plan_hash_for_state({"epoch": 0}))
            finally:
                rank0.close()
        finally:
            other.close()

    def test_mossformer_validation_matches_validation_refs(self) -> None:
        loaded = load_snapshot(self.moss["directory"])  # type: ignore[arg-type]
        store = AssetStore.from_snapshot(
            loaded,
            derived=self.moss["derived"],  # type: ignore[arg-type]
            corpora=self.moss["corpora"],  # type: ignore[arg-type]
            generated=self.moss["generated"],  # type: ignore[arg-type]
        )
        module = _module(self.moss)
        module.setup(_Runtime())
        try:
            planned = validation_refs(
                store,
                seed=7,
                epoch=1,
                world_size=1,
                policy="rank_mod_fixed_v1",
                rank=0,
            )
            batches = list(module.validation_epoch(1))
            flat = [row for batch in batches for row in range(batch.inputs.shape[0])]
            self.assertEqual(len(flat), len(planned))
            cursor = 0
            for batch in batches:
                for row in range(batch.inputs.shape[0]):
                    sample = planned[cursor]
                    rendered = render_sample(store, sample, normalize="independent_audio_norm_v1", sample_rate=16000)
                    self.assertEqual(batch.sample_ids[row], sample.case_id)
                    self.assertEqual(int(torch.max(torch.abs(batch.inputs[row] - torch.from_numpy(rendered.noisy)))), 0)
                    self.assertEqual(int(torch.max(torch.abs(batch.targets[row] - torch.from_numpy(rendered.clean)))), 0)
                    cursor += 1
        finally:
            module.close()

    def test_blocked_selector_is_data_error(self) -> None:
        source = SnapshotSource("missing.dataset@1", self.snapshots, "train", derived=self.derived)
        with self.assertRaises(DataError):
            source.open()
        source.close()


def _assert_matches_plan(test: unittest.TestCase, spec: dict[str, object], *, normalize: str, epoch: int) -> None:
    loaded = load_snapshot(spec["directory"])
    store = AssetStore.from_snapshot(
        loaded,
        derived=spec["derived"],
        corpora=spec["corpora"],
        generated=spec["generated"],
    )
    plan = plan_fixed_pair(
        store,
        seed=7,
        epoch=epoch,
        world_size=1,
        global_batch=2,
        micro_batch=2,
        accumulation=1,
    )
    module = _module(spec)
    module.setup(_Runtime())
    try:
        view = module.train_epoch(epoch, start_update=0)
        test.assertEqual(view.updates, plan.updates)
        for update_index, group in enumerate(plan.rank_groups[0]):
            test.assertEqual(len(group.micro_batches), 1)
            planned = group.micro_batches[0].samples
            batch = view.batch(update_index)[0]
            test.assertEqual(list(batch.sample_ids), [sample.case_id for sample in planned])
            for row, sample in enumerate(planned):
                rendered = render_sample(
                    store,
                    sample,
                    normalize=normalize,
                    sample_rate=16000,
                    verify_on_read=False,
                )
                noisy = batch.inputs[row].numpy()
                clean = batch.targets[row].numpy()
                test.assertEqual(int(np.max(np.abs(noisy - rendered.noisy))), 0)
                test.assertEqual(int(np.max(np.abs(clean - rendered.clean))), 0)
                test.assertEqual(int(batch.lengths[row]), int(rendered.valid_length))  # type: ignore[index]
                if sample.crop is not None:
                    test.assertEqual(sample.crop.start, _crop_start(batch.realization_ids[row]))
                    test.assertEqual(sample.crop.length, int(rendered.valid_length))
    finally:
        module.close()


def _crop_start(realization_id: str) -> int:
    crop = realization_id.rsplit("crop-", 1)[1]
    return int(crop.split("+", 1)[0])


def _module(spec: dict[str, object], *, seed: int = 7, require_profile: str | None = "leakage_safe", prefetch: bool = False) -> SnapshotDataModule:
    source = SnapshotSource(
        str(spec["selector"]),
        spec["root"],  # type: ignore[arg-type]
        "train",
        derived=spec["derived"],  # type: ignore[arg-type]
        corpora=spec["corpora"],  # type: ignore[arg-type]
        generated=spec["generated"],  # type: ignore[arg-type]
    )
    protocol = FixedPairProtocol(source)
    sampler = OrderSampler(source, protocol, seed=seed)
    batcher = MicroBatcher(micro_batch=2, group_size=1)
    return SnapshotDataModule(
        source,
        sampler,
        protocol,
        batcher,
        require_profile=require_profile,
        prefetch=prefetch,
    )


def _experiment_config(
    directory: Path,
    spec: dict[str, object],
    *,
    steps: int = 1,
    stop_after: int | None = None,
    prefetch: bool = False,
    checkpoint_interval: int = 1,
    accumulation: int = 1,
    require_profile: str | None = None,
    loss: str = "torch.nn.L1Loss",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    profile = "" if require_profile is None else f"\n      require_profile: {require_profile}"
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
    class: {loss}
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
  protocol:
    role: other
    class: src.experiments.data.FixedPairProtocol
    kwargs: {{}}
    inject:
      source: {{ref: source}}
  sampler:
    role: other
    class: src.experiments.data.OrderSampler
    kwargs:
      seed: 7
    inject:
      source: {{ref: source}}
      protocol: {{ref: protocol}}
  batcher:
    role: other
    class: src.experiments.data.MicroBatcher
    kwargs:
      micro_batch: 2
      group_size: 1
  data:
    role: data
    class: src.experiments.data.SnapshotDataModule
    kwargs:
      prefetch: {str(prefetch).lower()}{profile}
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
      accumulation: {accumulation}
      checkpoint_interval: {checkpoint_interval}
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


def _write_snapshot(
    root: Path,
    *,
    derived: Path,
    name: str,
    algorithm: str,
    normalize: str,
    order: object,
    profile: str,
    selection_uses_test: bool,
    claim_label: str,
    validation_crop: str,
    delivery: str | None = "on_demand",
    train_count: int = 8,
    valid_count: int = 2,
    test_count: int = 2,
) -> dict[str, object]:
    corpora = derived.parent / "corpora"
    generated = derived.parent / "generated"
    corpora.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    assets = []
    cases = []
    total = train_count + valid_count + test_count
    for index in range(total):
        frames = 4 if index == 0 else 16
        noisy = np.linspace(-0.2, 0.5, frames, dtype=np.float32) * ((index % 5) + 1)
        clean = np.linspace(0.4, -0.3, frames, dtype=np.float32) * ((index % 3) + 1)
        noisy_rel = f"toy/{name}/n{index}.wav"
        clean_rel = f"toy/{name}/c{index}.wav"
        noisy_path = derived / noisy_rel
        clean_path = derived / clean_rel
        _write_wav(noisy_path, noisy)
        _write_wav(clean_path, clean)
        for rel, path in ((noisy_rel, noisy_path), (clean_rel, clean_path)):
            assets.append(
                {
                    "asset_id": f"derived/{rel}",
                    "rel_path": rel,
                    "sample_rate": 16000,
                    "frames": frames,
                    "channels": 1,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "integrity": "sha256",
                }
            )
        cases.append(
            {
                "build_id": None,
                "case_id": f"{name}-case-{index:02d}",
                "case_role": "recorded",
                "clean_asset_id": f"derived/{clean_rel}",
                "doc_kind": "case",
                "frames": frames,
                "gen_index": None,
                "noise_asset_id": None,
                "noisy_asset_id": f"derived/{noisy_rel}",
                "protocol_id": "protocol",
                "sample_rate": 16000,
                "speaker_id": None,
                "teacher_kind": "dry_clean",
                "utterance_key": None,
            }
        )
    train_ids = [f"{name}-case-{index:02d}" for index in range(train_count)]
    valid_ids = [f"{name}-case-{index:02d}" for index in range(train_count, train_count + valid_count)]
    test_ids = [f"{name}-case-{index:02d}" for index in range(train_count + valid_count, total)]
    items = []
    bindings = []
    for slot, ids in (("train.cases", train_ids), ("valid.cases", valid_ids), ("test.cases", test_ids)):
        case_set_id = digest(ids)
        bindings.append({"case_set_id": case_set_id, "material_set_id": None, "slot": slot})
        for position, case_id in enumerate(ids):
            items.append({"case_id": case_id, "case_set_id": case_set_id, "position": position})
    rule: dict[str, object] = {
        "epoch_unit": "utterance",
        "global_batch": 2,
        "normalize": normalize,
        "order": order,
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "validation": {"batch": 1, "crop": validation_crop},
        "window": 8,
    }
    if delivery is not None:
        rule["delivery"] = delivery
    dataset_id = digest({"algorithm": algorithm, "name": name, "normalize": normalize, "profile": profile})
    dataset = {
        "algorithm": algorithm,
        "algorithm_version": 1,
        "bindings": bindings,
        "claim_label": claim_label,
        "dataset_id": dataset_id,
        "declared_status": "ready",
        "kind": "fixed_pair",
        "name": name,
        "profile": profile,
        "protocol_id": "protocol",
        "rule_json": rule,
        "selection_uses_test": selection_uses_test,
        "version": 1,
    }
    parts = {
        "assets": assets,
        "case_set_items": items,
        "cases": cases,
        "dataset": dataset,
        "docs": [],
        "generated": [],
        "material_set_members": [],
    }
    directory = root / dataset_id
    directory.mkdir(parents=True)
    digest_value = snapshot_hash(parts)
    manifest = {
        "files": file_hashes(parts),
        "provenance": {"format": FORMAT},
        "schema_version": SCHEMA_VERSION,
        "snapshot_hash": digest_value,
    }
    (directory / "dataset.json").write_text(canonical_text(parts["dataset"]) + "\n", encoding="utf-8")
    (directory / "generated.json").write_text(canonical_text(parts["generated"]) + "\n", encoding="utf-8")
    (directory / "docs.jsonl").write_text("", encoding="utf-8")
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
        "root": root,
        "selector": f"{name}@1",
    }


def _write_wav(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, 16000, subtype="PCM_16")


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
