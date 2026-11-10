"""P3 fixed-pair pipeline tests. Synthetic audio unless a test names the P2 snapshots."""

from __future__ import annotations

import ast
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.reproduction.errors import ReproductionError
from src.reproduction.ids import digest
from src.reproduction.paths import ROOT
from src.reproduction.pipeline.order import rank_order
from src.reproduction.pipeline.planner import plan_fixed_pair
from src.reproduction.pipeline.seeds import mix_seed
from src.reproduction.pipeline.store import AssetStore
from src.reproduction.pipeline.training_data import ReproductionTrainingData, open_training_data
from src.reproduction.pipeline.transform import apply_window, independent_audio_norm, noisy_rms_unit
from src.reproduction.pipeline.types import CropSpec
from src.experiments.calc.rng import mix_seed as training_mix_seed
from src.experiments.calc.rng import set_determinism
from src.experiments.calc.sampler import crop_start, epoch_indices
from src.experiments.calc.errors import TrainingError
from tests.numerics.support import write_wav

OFFICIAL = ROOT / "data" / "db" / "snapshots"


class OrderTests(unittest.TestCase):
    def test_mix_seed_matches_the_training_helper(self):
        self.assertEqual(mix_seed("crop", 7, 3, "case"), training_mix_seed("crop", 7, 3, "case"))

    def test_global_shuffle_matches_epoch_indices(self):
        for epoch, world, rank in ((0, 1, 0), (2, 2, 1), (5, 4, 3)):
            expected = epoch_indices(32, 11, epoch, world, rank)
            produced = rank_order(
                "global_shuffle_stride_v1", 32, seed=11, epoch=epoch, world_size=world, rank=rank
            )
            self.assertEqual(produced, expected)

    def test_distributed_sampler_matches_torch_drop_last(self):
        from torch.utils.data import DistributedSampler

        class _Length:
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

        for size, world, epoch in ((32, 2, 0), (32, 4, 3), (11, 2, 1), (11572, 2, 0)):
            for rank in range(world):
                sampler = DistributedSampler(
                    _Length(size), num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=True
                )
                sampler.set_epoch(epoch)
                produced = rank_order(
                    "torch_distributed_sampler_v1",
                    size,
                    seed=99,
                    epoch=epoch,
                    world_size=world,
                    rank=rank,
                )
                self.assertEqual(produced, list(sampler))

    def test_fixed_shuffle_is_stable_across_epochs(self):
        expected = list(range(20))
        random.Random(1234).shuffle(expected)
        for epoch in (0, 1, 7):
            self.assertEqual(
                rank_order("fixed_single_shuffle_v1", 20, seed=5, epoch=epoch, world_size=1, rank=0),
                expected,
            )

    def test_official_clearervoice_sampler_and_audio_norm(self):
        import sys

        root = ROOT / "src/se/official/clearervoice-studio/train/speech_enhancement"
        if not (root / "dataloader/dataloader.py").is_file():
            self.skipTest("ClearerVoice checkout is not present")
        inserted = str(root)
        sys.path.insert(0, inserted)
        try:
            from dataloader.dataloader import DistributedSampler, audio_norm
        finally:
            sys.path.remove(inserted)

        class _Length:
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

        for size, world, epoch in ((11, 2, 0), (11572, 2, 1)):
            for rank in range(world):
                sampler = DistributedSampler(
                    _Length(size), num_replicas=world, rank=rank, shuffle=True, seed=0
                )
                sampler.set_epoch(epoch)
                produced = rank_order(
                    "rank_mod_fixed_v1", size, seed=0, epoch=epoch, world_size=world, rank=rank
                )
                self.assertEqual(produced, list(sampler))
        wave = np.array([0.1, -0.2, 0.4, 0.05, -0.3, 0.2], dtype=np.float64)
        reference, _scalar = audio_norm(wave)
        produced = independent_audio_norm(wave)
        self.assertLess(float(np.max(np.abs(reference.astype(np.float32) - produced))), 1e-6)

    def test_rank_mod_matches_the_official_sampler_formula(self):
        for size, world, epoch in ((11, 2, 0), (11572, 2, 4), (8, 3, 1)):
            for rank in range(world):
                self.assertEqual(
                    rank_order("rank_mod_fixed_v1", size, seed=3, epoch=epoch, world_size=world, rank=rank),
                    _clearervoice_sampler(size, world, rank, epoch),
                )
        covered = []
        for rank in range(2):
            covered.extend(_clearervoice_sampler(11, 2, rank, 0))
        self.assertGreater(len(covered) - len(set(covered)), 0)
        self.assertNotIn(10, covered)

    def test_per_rank_independent_can_overlap(self):
        left = rank_order("per_rank_independent_v1", 8, seed=1, epoch=0, world_size=2, rank=0)
        right = rank_order("per_rank_independent_v1", 8, seed=1, epoch=0, world_size=2, rank=1)
        self.assertEqual(len(left), 4)
        self.assertEqual(len(right), 4)
        self.assertTrue(set(left) & set(right) or set(left) | set(right) != set(range(8)))
        again = rank_order("per_rank_independent_v1", 8, seed=1, epoch=0, world_size=2, rank=0)
        self.assertEqual(left, again)

    def test_rank_orders_do_not_drop_or_duplicate_when_divisible(self):
        size = 16
        for policy in ("global_shuffle_stride_v1", "torch_distributed_sampler_v1", "fixed_single_shuffle_v1"):
            found = []
            for rank in range(4):
                found.extend(rank_order(policy, size, seed=2, epoch=1, world_size=4, rank=rank))
            self.assertEqual(sorted(found), list(range(size)))


class PlanAndRenderTests(unittest.TestCase):
    def test_plan_hash_is_stable_and_changes_with_world_and_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(Path(directory), normalize="noisy_rms_unit_v1", order="global_shuffle_stride_v1")
            first = _plan(store, epoch=0, world=1)
            second = _plan(store, epoch=0, world=1)
            self.assertEqual(first.plan_hash, second.plan_hash)
            self.assertNotEqual(first.plan_hash, _plan(store, epoch=1, world=1).plan_hash)
            self.assertNotEqual(first.plan_hash, _plan(store, epoch=0, world=2).plan_hash)

    def test_mp_senet_policy_follows_world_size(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(
                Path(directory),
                normalize="noisy_rms_unit_v1",
                order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
            )
            self.assertEqual(_plan(store, epoch=0, world=1).order_policy, "fixed_single_shuffle_v1")
            self.assertEqual(_plan(store, epoch=0, world=2).order_policy, "torch_distributed_sampler_v1")

    def test_normalization_crop_and_padding(self):
        noisy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        clean = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        scaled_noisy, scaled_clean = noisy_rms_unit(noisy, clean)
        energy = np.sum(scaled_noisy * scaled_noisy, dtype=np.float32)
        self.assertAlmostEqual(float(energy), float(scaled_noisy.size), places=5)
        ratio = clean / noisy
        self.assertTrue(np.allclose(scaled_clean / scaled_noisy, ratio))
        start = crop_start("case", 9, 1, len(noisy), 4)
        window = apply_window(scaled_noisy, scaled_clean, CropSpec(start, 4, 4, "none"))
        self.assertTrue(np.allclose(window[0], scaled_noisy[start : start + 4]))
        self.assertTrue(np.allclose(window[1], scaled_clean[start : start + 4]))
        short_noisy, short_clean, valid = apply_window(
            noisy[:3], clean[:3], CropSpec(0, 3, 8, "tail_zero")
        )
        self.assertEqual(valid, 3)
        self.assertEqual(short_noisy.shape[0], 8)
        self.assertTrue(np.allclose(short_noisy[:3], noisy[:3]))
        self.assertTrue(np.all(short_noisy[3:] == 0))
        self.assertTrue(np.all(short_clean[3:] == 0))
        moss = independent_audio_norm(np.array([0.1, -0.2, 0.4, 0.05, -0.3], dtype=np.float32))
        self.assertTrue(np.isfinite(moss).all())
        reference = _official_audio_norm(np.array([0.1, -0.2, 0.4, 0.05, -0.3], dtype=np.float64))
        self.assertTrue(np.allclose(moss, reference.astype(np.float32)))
        with self.assertRaises(ReproductionError):
            independent_audio_norm(np.ones(4, dtype=np.float32))

    def test_renderer_keeps_time_alignment_and_uses_the_dataset_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp = _training(root / "mp", normalize="noisy_rms_unit_v1", order="fixed_single_shuffle_v1")
            moss = _training(root / "moss", normalize="independent_audio_norm_v1", order="rank_mod_fixed_v1")
            mp_batch = next(iter(mp.epoch(0, rank=0, world_size=1)))[0]
            moss_batch = next(iter(moss.epoch(0, rank=0, world_size=1)))[0]
            self.assertEqual(tuple(mp_batch["clean"].shape), (2, 8))
            self.assertEqual(tuple(mp_batch["clean"].shape), tuple(mp_batch["noisy"].shape))
            self.assertFalse(torch.allclose(mp_batch["noisy"], moss_batch["noisy"]))
            full = next(mp.validation(rank=0, world_size=1, epoch=0))
            rms = torch.sqrt(torch.mean(full["noisy"] ** 2, dim=-1))
            self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))
            short = _training(root / "short", normalize="noisy_rms_unit_v1", order="fixed_single_shuffle_v1", frames=3)
            batch = next(iter(short.epoch(0, rank=0, world_size=1)))[0]
            self.assertEqual(batch["lengths"].tolist(), [3, 3])
            self.assertTrue(torch.all(batch["noisy"][:, 3:] == 0))
            mp.close()
            moss.close()
            short.close()

    def test_validation_is_full_length_or_windowed_and_skips_held_out_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _training(
                root,
                normalize="noisy_rms_unit_v1",
                order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
                validation_crop="full_length",
            )
            batches = list(data.validation(rank=0, world_size=1, epoch=0))
            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0]["clean"].shape[-1], 16)
            ids = [batch["case_id"][0] for batch in batches]
            self.assertTrue(set(ids).isdisjoint(data.test_ids))
            data.close()
            moss = _training(
                root / "moss",
                normalize="independent_audio_norm_v1",
                order="rank_mod_fixed_v1",
                validation_crop="per_epoch_random",
            )
            windowed = next(moss.validation(rank=0, world_size=1, epoch=2))
            self.assertEqual(windowed["clean"].shape[-1], 8)
            moss.close()

    def test_variable_lengths_pad_on_the_right(self):
        from src.reproduction.pipeline.assemble import assemble
        from src.reproduction.pipeline.types import RenderedSample

        samples = [
            RenderedSample(np.ones(3, np.float32), np.full(3, 2, np.float32), 16000, 3, "a", "dry_clean", {}),
            RenderedSample(np.ones(5, np.float32), np.full(5, 2, np.float32), 16000, 5, "b", "dry_clean", {}),
        ]
        batch = assemble(samples, dataset_id="toy", snapshot_hash="abc")
        self.assertEqual(tuple(batch["noisy"].shape), (2, 5))
        self.assertEqual(batch["lengths"].tolist(), [3, 5])
        self.assertEqual(float(batch["noisy"][0, -1]), 0.0)


class GateAndTrainerTests(unittest.TestCase):
    def test_test_selection_requires_both_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = _store(
                root,
                normalize="noisy_rms_unit_v1",
                order=["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
                profile="official_faithful",
                selection_uses_test=True,
                same_valid_and_test=True,
            )
            with self.assertRaises(ReproductionError):
                _open(store, allow=False)
            opened = _open(store, allow=True)
            opened.close()
            safe = _store(root / "safe", normalize="noisy_rms_unit_v1", order="global_shuffle_stride_v1")
            opened = _open(safe, allow=False)
            self.assertTrue(set(opened.valid_ids).isdisjoint(opened.test_ids))
            opened.close()

    def test_pipeline_modules_do_not_import_duckdb_or_training(self):
        forbidden = ("duckdb", "src.training", "src.se", "src.asr", "src.evaluation")
        offenders = []
        for path in (ROOT / "src" / "reproduction" / "pipeline").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                if any(name == item or name.startswith(item + ".") for name in modules for item in forbidden):
                    offenders.append(f"{path.name}:{modules}")
        self.assertEqual(offenders, [])


class RealSnapshotTests(unittest.TestCase):
    def test_p2_datasets_plan_and_render_one_window(self):
        if not (OFFICIAL / "1e78f60c1f8333930bf2b252ccd457b9e83e8739354fadeb3c906136f7845f5b").is_dir():
            self.skipTest("P2 snapshots are not on this machine")
        with self.assertRaises(ReproductionError):
            open_training_data(
                dataset="mp_senet.vb28.official@1",
                allow_test_selection=False,
                seed=1234,
                global_batch=4,
                micro_batch=4,
                accumulation=1,
                world_size=1,
                segment_size=32000,
                model_sample_rate=16000,
                load_workers=1,
            )
        official = open_training_data(
            dataset="mp_senet.vb28.official@1",
            allow_test_selection=True,
            seed=1234,
            global_batch=4,
            micro_batch=4,
            accumulation=1,
            world_size=1,
            segment_size=32000,
            model_sample_rate=16000,
            load_workers=1,
        )
        try:
            self.assertEqual(official.policy, "fixed_single_shuffle_v1")
            self.assertTrue(official.selection_uses_test)
            first = official.plan_for(0)
            self.assertEqual(first.plan_hash, official.plan_for(0).plan_hash)
            self.assertEqual(first.updates, 11572 // 4)
            batch = next(iter(official.epoch(0, rank=0, world_size=1)))[0]
            self.assertEqual(tuple(batch["noisy"].shape), (4, 32000))
            self.assertTrue(torch.isfinite(batch["noisy"]).all())
            self.assertEqual(batch["dataset_id"], official.dataset_id)
        finally:
            official.close()
        safe = open_training_data(
            dataset="mossformergan.vb28.safe@1",
            allow_test_selection=False,
            seed=777,
            global_batch=2,
            micro_batch=1,
            accumulation=2,
            world_size=1,
            segment_size=32000,
            model_sample_rate=16000,
            load_workers=1,
        )
        try:
            self.assertFalse(safe.selection_uses_test)
            self.assertTrue(set(safe.valid_ids).isdisjoint(safe.test_ids))
            self.assertEqual(safe.plan_for(0).order_policy, "rank_mod_fixed_v1")
            window = next(safe.validation(rank=0, world_size=1, epoch=0))
            self.assertEqual(window["clean"].shape[-1], 32000)
            self.assertNotIn(window["case_id"][0], safe.test_ids)
        finally:
            safe.close()


def _clearervoice_sampler(size: int, world: int, rank: int, epoch: int) -> list[int]:
    """Transcribed from ClearerVoice DistributedSampler.__iter__ at the pinned commit."""
    import math

    num_samples = int(math.ceil(size / world))
    total = num_samples * world
    generator = torch.Generator()
    generator.manual_seed(epoch)
    ind = torch.randperm(int(size / world), generator=generator) * world
    indices: list[int] = []
    for offset in range(world):
        indices.extend((ind + offset).tolist())
    indices += indices[: total - len(indices)]
    return indices[rank * num_samples : (rank + 1) * num_samples]


def _official_audio_norm(values: np.ndarray) -> np.ndarray:
    rms = (values ** 2).mean() ** 0.5
    scalar = 10 ** (-25 / 20) / (rms + 1e-6)
    scaled = values * scalar
    power = scaled ** 2
    average = power.mean()
    peak = power[power > average].mean() ** 0.5
    return scaled * (10 ** (-25 / 20) / (peak + 1e-6))


def _store(root: Path, *, normalize: str, order, profile: str = "leakage_safe",
           selection_uses_test: bool = False, same_valid_and_test: bool = False,
           frames: int = 16, count: int = 8) -> AssetStore:
    root.mkdir(parents=True, exist_ok=True)
    derived = root / "derived"
    assets = []
    cases = []
    for index in range(count + 4):
        noisy = np.linspace(-0.2, 0.5, frames, dtype=np.float32) * ((index % 5) + 1)
        clean = np.linspace(0.4, -0.3, frames, dtype=np.float32) * ((index % 3) + 1)
        noisy_id = f"derived/toy/n{index}.wav"
        clean_id = f"derived/toy/c{index}.wav"
        write_wav(derived / "toy" / f"n{index}.wav", torch.from_numpy(noisy))
        write_wav(derived / "toy" / f"c{index}.wav", torch.from_numpy(clean))
        for asset_id, name in ((noisy_id, f"n{index}.wav"), (clean_id, f"c{index}.wav")):
            assets.append({
                "asset_id": asset_id,
                "rel_path": f"toy/{name}",
                "sample_rate": 16000,
                "frames": frames,
                "channels": 1,
                "sha256": "a" * 64,
                "integrity": "sha256",
            })
        cases.append({
            "case_id": f"case-{index:02d}",
            "teacher_kind": "dry_clean",
            "clean_asset_id": clean_id,
            "noisy_asset_id": noisy_id,
            "protocol_id": "shared-mp-senet-case-protocol",
        })
    train_ids = [f"case-{index:02d}" for index in range(count)]
    valid_ids = [f"case-{index:02d}" for index in range(count, count + 2)]
    test_ids = valid_ids if same_valid_and_test else [f"case-{index:02d}" for index in range(count + 2, count + 4)]
    sets = {"train.cases": train_ids, "valid.cases": valid_ids, "test.cases": test_ids}
    items = []
    bindings = []
    seen_sets = set()
    for slot, ids in sets.items():
        case_set_id = digest(ids)
        bindings.append({"slot": slot, "case_set_id": case_set_id, "material_set_id": None})
        if case_set_id in seen_sets:
            continue
        seen_sets.add(case_set_id)
        for position, case_id in enumerate(ids):
            items.append({"case_set_id": case_set_id, "position": position, "case_id": case_id})
    algorithm = "fixed_pair.mp_senet" if normalize == "noisy_rms_unit_v1" else "fixed_pair.mossformergan_se_16k"
    dataset = {
        "dataset_id": digest({"name": root.name, "algorithm": algorithm, "profile": profile}),
        "name": "toy.fixed",
        "version": 1,
        "protocol_id": "protocol",
        "algorithm": algorithm,
        "algorithm_version": 1,
        "kind": "fixed_pair",
        "profile": profile,
        "selection_uses_test": selection_uses_test,
        "declared_status": "ready",
        "rule_json": {
            "sample_rate": 16000,
            "teacher": "dry_clean",
            "normalize": normalize,
            "window": 8,
            "order": order,
            "epoch_unit": "utterance",
            "global_batch": 2,
            "validation": {"crop": "full_length", "batch": 1},
        },
        "bindings": bindings,
    }
    return AssetStore(
        dataset,
        assets,
        cases,
        items,
        snapshot_hash="snapshot",
        derived=derived,
        corpora=root / "corpora",
        generated=root / "generated",
    )


def _training(root: Path, **kwargs) -> ReproductionTrainingData:
    crop = kwargs.pop("validation_crop", None)
    store = _store(root, **kwargs)
    if crop is not None:
        store.dataset["rule_json"]["validation"]["crop"] = crop
    return _open(store, allow=bool(store.dataset["selection_uses_test"]))


def _open(store: AssetStore, *, allow: bool) -> ReproductionTrainingData:
    return ReproductionTrainingData(
        store,
        settings={
            "allow_test_selection": allow,
            "seed": 7,
            "global_batch": 2,
            "micro_batch": 2,
            "accumulation": 1,
            "world_size": 1,
            "segment_size": 8,
            "model_sample_rate": 16000,
            "load_workers": 1,
            "verify_on_read": False,
        },
    )


def _plan(store: AssetStore, *, epoch: int, world: int):
    micro = 1 if world == 2 else 2
    return plan_fixed_pair(
        store,
        seed=7,
        epoch=epoch,
        world_size=world,
        global_batch=2,
        micro_batch=micro,
        accumulation=1,
    )


if __name__ == "__main__":
    unittest.main()
