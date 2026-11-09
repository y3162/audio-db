"""Phase 3-F weighted sampling. Vectors were fixed before this implementation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.data.fixed_pair import SnapshotEpochView
from src.experiments.data.weighted import categorical_draws, rank_sample_positions
from src.experiments.errors import DataError, StateError
from src.reproduction.ids import digest
from src.reproduction.pipeline.order import rank_order
from tests.experiments.test_chunk_data import _module as _chunk_module
from tests.experiments.test_chunk_data import _write_snapshot as _write_chunk
from tests.experiments.test_distributed_data import (
    _losses,
    _max_tree,
    _same_tree,
    _sample_ids,
    _train,
    _write_fixed,
)
from tests.experiments.test_fixed_pair_data import _experiment_config
from tests.experiments.test_fixed_pair_data import _module as _fixed_module
from tests.experiments.test_fixed_pair_data import _write_snapshot as _write_fixed_pair
from tests.experiments.test_online_data import _dfn_module, _semamba_module, _write_online

ONES = ["s1", "s0", "s3", "s2"]
SKEW = ["s1", "s1", "s3", "s1"]
SKEW_EPOCH1 = ["s1", "s1", "s1", "s3"]
SKEW_32 = [
    "s1", "s1", "s3", "s1", "s3", "s3", "s0", "s3",
    "s1", "s1", "s0", "s1", "s1", "s1", "s3", "s1",
    "s1", "s1", "s1", "s1", "s1", "s1", "s3", "s3",
    "s0", "s3", "s3", "s1", "s1", "s3", "s0", "s1",
]
IDS = ["s0", "s1", "s2", "s3"]
SKEW_WEIGHTS = {"s0": 1.0, "s1": 3.0, "s2": 0.0, "s3": 1.0}


class _Runtime:
    def __init__(self, rank: int = 0, world_size: int = 1) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cpu")


def _names(indexes: list[int]) -> list[str]:
    return [IDS[index] for index in indexes]


class WeightedAlgorithmTests(unittest.TestCase):
    def test_fixed_vectors_world_size_one_and_two(self) -> None:
        ones = [1.0, 1.0, 1.0, 1.0]
        skew = [1.0, 3.0, 0.0, 1.0]
        self.assertEqual(_names(categorical_draws(ones, 7, 0, 4)), ONES)
        self.assertEqual(_names(rank_sample_positions(ones, 7, 0, 1, 0)), ONES)
        self.assertEqual(_names(rank_sample_positions(ones, 7, 0, 2, 0)), ["s1", "s3"])
        self.assertEqual(_names(rank_sample_positions(ones, 7, 0, 2, 1)), ["s0", "s2"])
        self.assertEqual(_names(categorical_draws(skew, 7, 0, 4)), SKEW)
        self.assertEqual(_names(categorical_draws(skew, 7, 1, 4)), SKEW_EPOCH1)
        self.assertEqual(_names(rank_sample_positions(skew, 7, 0, 2, 0)), ["s1", "s3"])
        self.assertEqual(_names(rank_sample_positions(skew, 7, 0, 2, 1)), ["s1", "s1"])
        self.assertEqual(_names(categorical_draws(skew, 7, 0, 32)), SKEW_32)
        self.assertNotIn("s2", SKEW_32)
        self.assertEqual(_names(categorical_draws([2.0, 6.0, 0.0, 2.0], 7, 0, 4)), SKEW)
        self.assertEqual(categorical_draws([1e-7, 1.0, 1e-6, 0.0], 7, 0, 4), [1, 1, 1, 1])
        odd = categorical_draws([1.0] * 7, 7, 0, 7)
        self.assertEqual(odd, [5, 6, 6, 5, 6, 1, 1])
        self.assertEqual(rank_sample_positions([1.0] * 7, 7, 0, 2, 0), [5, 6, 6])
        self.assertEqual(rank_sample_positions([1.0] * 7, 7, 0, 2, 1), [6, 5, 1])
        self.assertEqual(digest(SKEW_WEIGHTS), "c5d7db2a48e3978848a888a328c3a497a5eb6320c8674e9e34b52af7d24bfd7c")
        self.assertEqual(
            digest({"s0": 1.0, "s1": 1.0, "s2": 1.0, "s3": 1.0}),
            "47aa23d5bc83a82b29858e45a2fa748c105766822191c9c7ec64807da5e55c42",
        )
        permutation = rank_order("fixed_single_shuffle_v1", 4, seed=7, epoch=0, world_size=1, rank=0)
        self.assertEqual(categorical_draws(ones, 7, 0, 4), [1, 0, 3, 2])
        self.assertNotEqual(permutation, [1, 0, 3, 2])

    def test_draw_frequency_stays_inside_the_prefixed_interval(self) -> None:
        draws = categorical_draws([1.0, 3.0], 11, 0, 4000)
        count = draws.count(0)
        self.assertGreaterEqual(count, 918)
        self.assertLessEqual(count, 1082)
        self.assertEqual(count + draws.count(1), 4000)

    def test_all_zero_and_invalid_values_are_rejected(self) -> None:
        from src.experiments.data.weighted import normalize_weight_table

        ids = ("s0", "s1")
        with self.assertRaises(DataError):
            normalize_weight_table({"s0": 0.0, "s1": 0.0}, ids)
        for value in (True, False, float("nan"), float("inf"), -1.0, "1", None):
            with self.assertRaises(DataError):
                normalize_weight_table({"s0": value, "s1": 1.0}, ids)
        with self.assertRaises(DataError):
            normalize_weight_table({"s0": 1.0}, ids)
        with self.assertRaises(DataError):
            normalize_weight_table({"s0": 1.0, "s1": 1.0, "s2": 1.0}, ids)
        with self.assertRaises(DataError):
            normalize_weight_table([1.0, 1.0], ids)
        tiny = normalize_weight_table({"s0": 1e-7, "s1": 1.0}, ids)
        self.assertEqual(tiny["s0"], 0.0)
        self.assertEqual(tiny["s1"], 1.0)


class WeightedDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.spec = _write_fixed_pair(
            self.base / "snapshots",
            derived=self.base / "derived",
            name="toy.weighted",
            algorithm="fixed_pair.mp_senet",
            normalize="noisy_rms_unit_v1",
            order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            profile="leakage_safe",
            selection_uses_test=False,
            claim_label="leakage_safe_comparison",
            validation_crop="full_length",
            train_count=4,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_unweighted_rows_and_validation_stay_put(self) -> None:
        runtime = _Runtime()
        left = _fixed_module(self.spec, seed=7)
        right = _fixed_module(self.spec, seed=7)
        left.setup(runtime)
        right.setup(runtime)
        try:
            self.assertIsNone(left.state_dict()["weights"])
            self.assertEqual(left.plan_hash, right.plan_hash)
            self.assertEqual(left.identity()["order_policy"], "fixed_single_shuffle_v1")
            view = left.train_epoch(0, start_update=0)
            expected = rank_order(
                "fixed_single_shuffle_v1",
                4,
                seed=7,
                epoch=0,
                world_size=1,
                rank=0,
            )
            ids = left.sampler.training_ids()
            self.assertEqual(list(view.batch(0)[0].sample_ids), [ids[index] for index in expected[:2]])
            valid = [list(batch.sample_ids) for batch in left.validation_epoch(1)]
            view.close()
            left.set_sampling_weights({sample_id: 1.0 for sample_id in ids})
            self.assertEqual([list(batch.sample_ids) for batch in left.validation_epoch(1)], valid)
            waves = list(left.validation_epoch(1))
            again = list(right.validation_epoch(1))
            self.assertEqual(_max_abs(waves[0].inputs, again[0].inputs), 0.0)
            self.assertEqual(_max_abs(waves[0].targets, again[0].targets), 0.0)
        finally:
            left.close()
            right.close()

    def test_weights_follow_the_draw_and_do_not_change_a_realization(self) -> None:
        runtime = _Runtime()
        plain = _fixed_module(self.spec, seed=7)
        weighted = _fixed_module(self.spec, seed=7)
        plain.setup(runtime)
        weighted.setup(runtime)
        try:
            ids = weighted.sampler.training_ids()
            table = {sample_id: 1.0 for sample_id in ids}
            table[ids[2]] = 0.0
            weighted.set_sampling_weights(table)
            self.assertEqual(weighted.identity()["order_policy"], plain.identity()["order_policy"])
            positions = rank_sample_positions([table[sample_id] for sample_id in ids], 7, 0, 1, 0)
            view = weighted.train_epoch(0, start_update=0)
            seen: list[str] = []
            for index in range(view.updates):
                seen.extend(view.batch(index)[0].sample_ids)
            self.assertEqual(seen, [ids[index] for index in positions])
            self.assertNotIn(ids[2], seen)
            self.assertIsNone(weighted.sampler._draw_weights)
            plain_ids = []
            plain_view = plain.train_epoch(0, start_update=0)
            for index in range(plain_view.updates):
                plain_ids.extend(plain_view.batch(index)[0].sample_ids)
            shared = next(sample_id for sample_id in seen if sample_id in plain_ids)
            left = _row(view, shared)
            right = _row(plain_view, shared)
            self.assertEqual(left[0], right[0])
            self.assertEqual(_max_abs(left[1], right[1]), 0.0)
            self.assertEqual(_max_abs(left[2], right[2]), 0.0)
            self.assertNotEqual(weighted.plan_hash, plain.plan_hash)
            self.assertEqual(digest(weighted.state_dict()["weights"]), digest(table))
        finally:
            plain.close()
            weighted.close()

    def test_epoch_boundary_freezes_the_open_plan(self) -> None:
        runtime = _Runtime()
        module = _fixed_module(self.spec, seed=7, prefetch=True)
        module.setup(runtime)
        fresh = _fixed_module(self.spec, seed=7)
        fresh.setup(runtime)
        try:
            ids = module.sampler.training_ids()
            table = {sample_id: (3.0 if index == 1 else 1.0) for index, sample_id in enumerate(ids)}
            view = module.train_epoch(0, start_update=0)
            before = view.batch(0)[0]
            saved = (before.sample_ids, before.inputs.clone(), before.targets.clone())
            with self.assertRaises(DataError):
                module.set_sampling_weights(table)
            after = view.batch(0)[0]
            self.assertEqual(after.sample_ids, saved[0])
            self.assertEqual(_max_abs(after.inputs, saved[1]), 0.0)
            self.assertEqual(_max_abs(after.targets, saved[2]), 0.0)
            old_hash = view.plan_hash
            view.close()
            with self.assertRaises(DataError):
                view.batch(0)
            module.set_sampling_weights(table)
            module.epoch = 0
            module.update_index = view.updates
            frozen = module.state_dict()
            self.assertIsNone(frozen["weights"])
            self.assertEqual(frozen["plan_hash"], old_hash)
            replay = module.train_epoch(0, start_update=0)
            self.assertEqual(replay.batch(0)[0].sample_ids, saved[0])
            replay.close()
            nxt = module.train_epoch(1, start_update=0)
            fresh.set_sampling_weights(table)
            expected = fresh.train_epoch(1, start_update=0)
            self.assertEqual(nxt.batch(0)[0].sample_ids, expected.batch(0)[0].sample_ids)
            self.assertEqual(_max_abs(nxt.batch(0)[0].inputs, expected.batch(0)[0].inputs), 0.0)
            self.assertNotEqual(nxt.plan_hash, old_hash)
            module.epoch = 1
            self.assertEqual(module.state_dict()["weights"], fresh.state_dict()["weights"])
            nxt.close()
            expected.close()
        finally:
            module.close()
            fresh.close()

    def test_resume_matches_weights_without_overwriting_them(self) -> None:
        runtime = _Runtime()
        module = _fixed_module(self.spec, seed=7)
        module.setup(runtime)
        try:
            ids = module.sampler.training_ids()
            table = {sample_id: 1.0 for sample_id in ids}
            module.set_sampling_weights(table)
            view = module.train_epoch(0, start_update=0)
            view.batch(0)
            view.close()
            module.epoch = 0
            module.update_index = 1
            saved = module.state_dict()
            token = module._weights
            module.load_state_dict(saved)
            self.assertIs(module._weights, token)
            self.assertEqual(module.update_index, 1)
            plain = _fixed_module(self.spec, seed=7)
            plain.setup(runtime)
            other = _fixed_module(self.spec, seed=7)
            other.setup(runtime)
            try:
                with self.assertRaises(StateError):
                    plain.load_state_dict(saved)
                self.assertIsNone(plain._weights)
                other.set_sampling_weights({sample_id: 2.0 for sample_id in ids})
                before = dict(other._weights)
                with self.assertRaises(StateError):
                    other.load_state_dict(saved)
                self.assertEqual(other._weights, before)
                unweighted = plain.state_dict()
                with self.assertRaises(StateError):
                    module.load_state_dict(unweighted)
            finally:
                plain.close()
                other.close()
        finally:
            module.close()

    def test_two_ranks_share_one_global_plan(self) -> None:
        modules = []
        try:
            for rank in (0, 1):
                module = _fixed_module(self.spec, seed=7)
                module.setup(_Runtime(rank=rank, world_size=2))
                ids = module.sampler.training_ids()
                module.set_sampling_weights({sample_id: (0.0 if index == 0 else 1.0) for index, sample_id in enumerate(ids)})
                modules.append(module)
            views = [module.train_epoch(0, start_update=0) for module in modules]
            self.assertEqual(views[0].plan_hash, views[1].plan_hash)
            self.assertEqual(views[0].updates, views[1].updates)
            ids = modules[0].sampler.training_ids()
            weights = [0.0 if index == 0 else 1.0 for index in range(len(ids))]
            for rank, view in enumerate(views):
                seen = [sample_id for index in range(view.updates) for sample_id in view.batch(index)[0].sample_ids]
                expected = [ids[index] for index in rank_sample_positions(weights, 7, 0, 2, rank)]
                self.assertEqual(seen, expected)
                self.assertNotIn(ids[0], seen)
        finally:
            for module in modules:
                module.close()


class WeightedRejectionTests(unittest.TestCase):
    def test_chunk_dfn3_and_semamba_ids(self) -> None:
        base = Path(tempfile.mkdtemp())
        try:
            derived = base / "derived"
            chunk_spec = _write_chunk(base / "chunk", derived=derived, kind="ms", name="toy.chunk")
            dfn_spec = _write_online(
                base / "online",
                corpora=base / "corpora-dfn",
                derived=derived,
                kind="dfn3",
                name="toy.dfn",
                speech=4,
                factor=1,
                global_batch=2,
                valid_batch=2,
            )
            semamba_spec = _write_online(
                base / "online",
                corpora=base / "corpora-semamba",
                derived=derived,
                kind="semamba",
                name="toy.semamba",
                speech=4,
                factor=1,
                global_batch=2,
                valid_batch=2,
            )
            runtime = _Runtime()
            chunk = _chunk_module(chunk_spec, micro_batch=1)
            dfn = _dfn_module(dfn_spec, micro_batch=1, group_size=1)
            semamba = _semamba_module(semamba_spec, micro_batch=1, group_size=1, seed=7)
            chunk.setup(runtime)
            dfn.setup(runtime)
            semamba.setup(runtime)
            try:
                with self.assertRaises(DataError):
                    chunk.set_sampling_weights({"x": 1.0})
                with self.assertRaises(DataError):
                    dfn.set_sampling_weights({"x": 1.0})
                ids = semamba.sampler.training_ids()
                self.assertEqual(len(ids), 4)
                self.assertTrue(all(sample_id.endswith(f":train:{index}") for index, sample_id in enumerate(ids)))
                valid_before = [list(batch.sample_ids) for batch in semamba.validation_epoch(0)]
                table = {sample_id: (0.0 if index == 1 else 1.0) for index, sample_id in enumerate(ids)}
                semamba.set_sampling_weights(table)
                view = semamba.train_epoch(0, start_update=0)
                seen = [sample_id for index in range(view.updates) for sample_id in view.batch(index)[0].sample_ids]
                self.assertNotIn(ids[1], seen)
                self.assertEqual(
                    seen,
                    [ids[index] for index in rank_sample_positions([table[sample_id] for sample_id in ids], 7, 0, 1, 0)],
                )
                self.assertEqual([list(batch.sample_ids) for batch in semamba.validation_epoch(0)], valid_before)
                repeated = view.batch(0)[0]
                self.assertEqual(_max_abs(repeated.inputs, view.batch(0)[0].inputs), 0.0)
            finally:
                chunk.close()
                dfn.close()
                semamba.close()
        finally:
            import shutil

            shutil.rmtree(base, ignore_errors=True)


class WeightedTrainerTests(unittest.TestCase):
    def test_standard_trainer_closes_views_on_success_and_failure(self) -> None:
        base = Path(tempfile.mkdtemp())
        spec = _write_fixed(
            base / "snapshots",
            derived=base / "derived",
            name="toy.trainer",
            algorithm="fixed_pair.mp_senet",
            normalize="noisy_rms_unit_v1",
            order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            profile="leakage_safe",
            selection_uses_test=False,
            claim_label="leakage_safe_comparison",
            validation_crop="full_length",
            train_count=4,
        )
        closes: list[int] = []
        original = SnapshotEpochView.close

        def spy(view: SnapshotEpochView) -> None:
            closes.append(view._epoch)
            original(view)

        SnapshotEpochView.close = spy  # type: ignore[method-assign]
        try:
            experiment = build_experiment(load_experiment(_experiment_config(base / "ok", spec, steps=1)))
            experiment.run()
            self.assertEqual(closes, [0])
            closes.clear()
            failed = build_experiment(
                load_experiment(_experiment_config(base / "bad", spec, steps=1, loss="tests.experiments.support.BoomLoss"))
            )
            with self.assertRaises(RuntimeError):
                failed.run()
            self.assertEqual(closes, [0])
        finally:
            SnapshotEpochView.close = original  # type: ignore[method-assign]
            import shutil

            shutil.rmtree(base, ignore_errors=True)


class WeightedDistributedTests(unittest.TestCase):
    def test_world_size_two_resume_and_rank_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            derived = base / "derived"
            spec = _write_fixed(
                base / "snapshots",
                derived=derived,
                name="train.weighted",
                algorithm="fixed_pair.mp_senet",
                normalize="noisy_rms_unit_v1",
                order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
                profile="leakage_safe",
                selection_uses_test=False,
                claim_label="leakage_safe_comparison",
                validation_crop="full_length",
            )
            semamba = _write_online(
                base / "snapshots",
                corpora=base / "corpora",
                derived=derived,
                kind="semamba",
                name="train.semamba",
                speech=4,
                factor=1.0,
                global_batch=2,
                valid_batch=2,
            )
            pipeline = {"extra": {"scheduler": True}, "kind": "fixed", "name": "mp_senet", "spec": spec}
            semamba_pipeline = {"extra": {}, "kind": "semamba", "name": "semamba", "spec": semamba}
            data_class = "tests.experiments.support.WeightedSnapshot"
            continuous = base / "continuous"
            interrupted = base / "interrupted"
            _train(pipeline, continuous, steps=4, stop_after=None, data_class=data_class, timeout_seconds=60)
            _train(pipeline, interrupted, steps=4, stop_after=2, data_class=data_class, timeout_seconds=60)
            saved = interrupted / "run" / "step_00000002.pt"
            _train(pipeline, interrupted, steps=4, stop_after=None, resume=saved, data_class=data_class, timeout_seconds=60)
            left = torch.load(continuous / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
            right = torch.load(interrupted / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
            self.assertEqual(left["states"]["data"]["weights"], right["states"]["data"]["weights"])
            self.assertIsNotNone(left["states"]["data"]["weights"])
            self.assertEqual(left["states"]["data"]["plan_hash"], right["states"]["data"]["plan_hash"])
            self.assertEqual(left["states"]["data"]["epoch"], right["states"]["data"]["epoch"])
            self.assertEqual(left["states"]["data"]["update_index"], right["states"]["data"]["update_index"])
            self.assertEqual(_sample_ids(continuous / "run"), _sample_ids(interrupted / "run"))
            zero_ids = [sample_id for sample_id, weight in left["states"]["data"]["weights"].items() if weight == 0]
            self.assertEqual(len(zero_ids), 1)
            logged = ",".join(",".join(row) for rank_rows in _sample_ids(continuous / "run") for row in rank_rows)
            self.assertNotIn(zero_ids[0], logged)
            self.assertEqual(_max_tree(left["states"]["model"], right["states"]["model"]), 0.0)
            self.assertEqual(_max_tree(left["states"]["optimizer"], right["states"]["optimizer"]), 0.0)
            self.assertEqual(_max_tree(left["states"]["scheduler"], right["states"]["scheduler"]), 0.0)
            self.assertEqual(left["states"]["trainer"], right["states"]["trainer"])
            self.assertTrue(_same_tree(left["rng_by_rank"], right["rng_by_rank"]))
            self.assertEqual(_losses(continuous / "run"), _losses(interrupted / "run"))
            _train(semamba_pipeline, base / "semamba", steps=1, stop_after=None, data_class=data_class, timeout_seconds=60)
            semamba_state = torch.load(base / "semamba" / "run" / "step_00000001.pt", map_location="cpu", weights_only=False)
            self.assertTrue(any(str(key).endswith(":train:0") for key in semamba_state["states"]["data"]["weights"]))
            skew = _train(
                pipeline,
                base / "skew",
                steps=1,
                stop_after=None,
                data_class="tests.experiments.support.SkewWeightModule",
                timeout_seconds=8,
                expect_ok=False,
                process_timeout=25,
            )
            self.assertNotEqual(skew.returncode, 0)
            self.assertIn("ranks disagree", skew.stderr + skew.stdout)
            failed = _train(
                pipeline,
                base / "fail",
                steps=1,
                stop_after=None,
                data_class="tests.experiments.support.RankWeightFailModule",
                timeout_seconds=8,
                expect_ok=False,
                process_timeout=25,
            )
            self.assertNotEqual(failed.returncode, 0)


def _row(view: object, sample_id: str) -> tuple[str, torch.Tensor, torch.Tensor]:
    for index in range(view.updates):  # type: ignore[attr-defined]
        batch = view.batch(index)[0]  # type: ignore[attr-defined]
        for row, name in enumerate(batch.sample_ids):
            if name == sample_id:
                return batch.realization_ids[row], batch.inputs[row], batch.targets[row]
    raise AssertionError(sample_id)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.cpu() - right.cpu())).item())
