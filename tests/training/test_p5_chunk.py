"""P5 chunk ranks, virtual rank, resume, and weighted validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reproduction.errors import ReproductionError
from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.reproduction.pipeline.planner import plan_espnet_chunk
from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint
from src.training.core.distributed import ProcessInfo
from src.training.core.trainer import Trainer, _reduce_weighted_reports
from src.training.data.chunk_plan import epoch_chunk_batches
from src.training.errors import TrainingError
from src.training.se.config import load_training_config
from tests.training.support import cpu_process, run_distributed_workers

IDS = [f"c{i}" for i in range(11)]
LENGTHS = [8, 13, 17, 25, 10, 29, 40, 9, 18, 22, 15]
GOLDEN = {
    (1, False): [
        [("c1", 0, 8), ("c1", 4, 12)],
        [("c2", 0, 8), ("c2", 4, 12)],
        [("c2", 8, 16), ("c3", 0, 8)],
        [("c3", 4, 12), ("c3", 8, 16)],
        [("c3", 12, 20), ("c3", 16, 24)],
        [("c4", 0, 8), ("c5", 0, 8)],
        [("c5", 4, 12), ("c5", 8, 16)],
        [("c5", 12, 20), ("c5", 16, 24)],
    ],
    (2, True): [
        [("c10", 5, 13), ("c9", 10, 18)],
        [("c9", 6, 14), ("c10", 1, 9)],
        [("c9", 2, 10), ("c9", 14, 22)],
        [("c6", 16, 24), ("c6", 28, 36)],
        [("c6", 12, 20), ("c6", 20, 28)],
        [("c6", 4, 12), ("c6", 32, 40)],
        [("c6", 24, 32), ("c6", 0, 8)],
        [("c1", 4, 12), ("c1", 0, 8)],
        [("c6", 8, 16), ("c7", 1, 9)],
    ],
}


def _rows(batches):
    return [[(chunk.case_id, chunk.start, chunk.stop) for chunk in batch] for batch in batches]


class PlanTests(unittest.TestCase):
    def test_world_size_one_matches_the_previous_epoch_plan(self):
        for epoch, shuffle in ((1, False), (2, True), (3, True)):
            previous = epoch_chunk_batches(
                IDS, LENGTHS, espnet_epoch=epoch, seed=17, num_samples=6,
                chunk_length=8, batch_size=2, cache_size=5, shuffle=shuffle,
            )
            plan = plan_espnet_chunk(
                list(zip(IDS, LENGTHS)), espnet_epoch=epoch, seed=17, world_size=1,
                num_samples=6, chunk_length=8, global_batch=2, num_cache_chunks=5, shuffle=shuffle,
            )
            self.assertEqual(_rows(previous), _rows(plan.rank_batches[0]))
            if (epoch, shuffle) in GOLDEN:
                self.assertEqual(_rows(plan.rank_batches[0]), GOLDEN[(epoch, shuffle)])

    def test_ranks_cover_without_duplication_and_stop_together(self):
        samples = [(f"u{i}", 20 + (i % 5) * 7) for i in range(11)]
        for world, batch in ((2, 4), (4, 4)):
            plan = plan_chunk_epoch(
                samples, espnet_epoch=1, seed=3, world_size=world, num_samples=6,
                chunk_length=8, global_batch=batch, num_cache_chunks=10, shuffle=True,
            )
            self.assertEqual(plan.per_rank_batch, batch // world)
            self.assertEqual(plan.per_rank_cache, max(10 // world, batch // world))
            self.assertTrue(all(len(batches) == plan.updates for batches in plan.rank_batches))
            assigned = []
            for rank in range(world):
                assigned.extend(case_id for case_id, _length in samples[rank::world])
            self.assertEqual(sorted(assigned), sorted(case_id for case_id, _length in samples))
            self.assertEqual(len(assigned), len(samples))
            self.assertEqual(plan.updates, min(plan.raw_batch_counts))
            self.assertEqual(plan.discarded_batches, tuple(count - plan.updates for count in plan.raw_batch_counts))

    def test_multi_rank_is_not_a_split_of_the_single_process_batches(self):
        samples = list(zip(IDS, LENGTHS))
        single = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=17, world_size=1, num_samples=6,
            chunk_length=8, global_batch=4, num_cache_chunks=8, shuffle=True,
        )
        paired = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=17, world_size=2, num_samples=6,
            chunk_length=8, global_batch=4, num_cache_chunks=8, shuffle=True,
        )
        self.assertNotEqual(_rows(single.raw_rank_batches[0]), _rows(paired.raw_rank_batches[0]))
        self.assertEqual(paired.per_rank_batch, 2)
        self.assertEqual(paired.per_rank_cache, 4)

    def test_epoch_cursor_advances_and_dropped_batches_do_not_carry(self):
        samples = [(f"u{i}", 48) for i in range(8)]
        first = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=4, world_size=2, num_samples=3,
            chunk_length=8, global_batch=4, num_cache_chunks=4, shuffle=False,
        )
        second = plan_chunk_epoch(
            samples, espnet_epoch=2, seed=4, world_size=2, num_samples=3,
            chunk_length=8, global_batch=4, num_cache_chunks=4, shuffle=False,
        )
        self.assertNotEqual(first.selected, second.selected)
        self.assertEqual(first.stats["utterance_cursor"], 3)
        self.assertEqual(second.stats["utterance_cursor"], 6)
        self.assertGreater(sum(first.discarded_tail_chunks), 0)
        again = plan_chunk_epoch(
            samples, espnet_epoch=2, seed=4, world_size=2, num_samples=3,
            chunk_length=8, global_batch=4, num_cache_chunks=4, shuffle=False,
        )
        self.assertEqual(second.plan_hash, again.plan_hash)
        self.assertEqual(second.rank_batches, again.rank_batches)

    def test_virtual_and_recomputed_plans_share_a_hash(self):
        samples = [(f"u{i}", 100 + i * 3) for i in range(16)]
        for world in (2, 4):
            left = plan_chunk_epoch(
                samples, espnet_epoch=2, seed=9, world_size=world, num_samples=5,
                chunk_length=16, global_batch=4, num_cache_chunks=12, shuffle=True,
            )
            right = plan_chunk_epoch(
                samples, espnet_epoch=2, seed=9, world_size=world, num_samples=5,
                chunk_length=16, global_batch=4, num_cache_chunks=12, shuffle=True,
            )
            self.assertEqual(left.plan_hash, right.plan_hash)
            self.assertEqual(len(left.plan_hash), 64)

    def test_uneven_global_batch_is_refused(self):
        with self.assertRaises(ReproductionError):
            plan_chunk_epoch(
                [("a", 30), ("b", 30), ("c", 30)],
                espnet_epoch=1, seed=1, world_size=2, num_samples=2,
                chunk_length=8, global_batch=3, num_cache_chunks=4, shuffle=False,
            )

    def test_recompute_ignores_a_discarded_prefetch(self):
        samples = list(zip(IDS, LENGTHS))
        plan = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=17, world_size=2, num_samples=6,
            chunk_length=8, global_batch=4, num_cache_chunks=8, shuffle=True,
        )
        peeked = plan.rank_batches[0][2]
        del peeked
        again = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=17, world_size=2, num_samples=6,
            chunk_length=8, global_batch=4, num_cache_chunks=8, shuffle=True,
        )
        self.assertEqual(plan.rank_batches[1][2], again.rank_batches[1][2])

    def test_tf_gridnet_and_tf_locoformer_shapes(self):
        grid = plan_chunk_epoch(
            [(f"g{i}", 160000) for i in range(20)],
            espnet_epoch=1, seed=0, world_size=2, num_samples=8,
            chunk_length=48000, global_batch=4, num_cache_chunks=1024, shuffle=True,
        )
        loco = plan_chunk_epoch(
            [(f"l{i}", 480000) for i in range(20)],
            espnet_epoch=1, seed=0, world_size=4, num_samples=4,
            chunk_length=64000, global_batch=4, num_cache_chunks=1024, shuffle=True,
        )
        self.assertEqual(grid.per_rank_batch, 2)
        self.assertEqual(grid.per_rank_cache, 512)
        self.assertEqual(loco.per_rank_batch, 1)
        self.assertEqual(loco.per_rank_cache, 256)
        self.assertTrue(all(chunk.stop - chunk.start == 64000 for batch in loco.rank_batches[0] for chunk in batch))
        self.assertTrue(all(len(batches) == loco.updates for batches in loco.rank_batches))


class WeightedValidationTests(unittest.TestCase):
    def test_uneven_counts_use_sample_weights(self):
        def report(value):
            return {
                "losses": {"generator": {"status": "ok", "value": value, "reason": None}},
                "pesq": {"status": "na", "value": None, "reason": "skip", "count": 0},
                "stoi": {"status": "na", "value": None, "reason": "skip", "count": 0},
            }

        reports = [report(1.0), report(2.0), report(10.0)]
        weights = [1, 1, 8]
        summary = _reduce_weighted_reports(reports, weights, cpu_process())
        self.assertAlmostEqual(summary["losses"]["generator"]["value"], 83 / 10)
        rank_mean = ((1.0 + 2.0) / 2 + 10.0) / 2
        self.assertNotAlmostEqual(summary["losses"]["generator"]["value"], rank_mean)

    def test_kept_batches_match_one_process(self):
        samples = [(f"u{i}", 40 + i) for i in range(9)]
        plan = plan_chunk_epoch(
            samples, espnet_epoch=1, seed=2, world_size=2, num_samples=None,
            chunk_length=8, global_batch=2, num_cache_chunks=6, shuffle=False,
        )
        def value_of(chunks):
            return float(sum(chunk.start for chunk in chunks)) / len(chunks)

        reports = []
        weights = []
        for rank in range(plan.world_size):
            for chunks in plan.rank_batches[rank]:
                reports.append({
                    "losses": {"generator": {"status": "ok", "value": value_of(chunks), "reason": None}},
                    "pesq": {"status": "na", "value": None, "reason": "skip", "count": 0},
                    "stoi": {"status": "na", "value": None, "reason": "skip", "count": 0},
                })
                weights.append(len(chunks))
        summary = _reduce_weighted_reports(reports, weights, cpu_process())
        numerator = 0.0
        denominator = 0
        for rank in range(plan.world_size):
            for chunks in plan.rank_batches[rank]:
                numerator += sum(chunk.start for chunk in chunks)
                denominator += len(chunks)
        self.assertAlmostEqual(summary["losses"]["generator"]["value"], numerator / denominator)


class _Waves:
    def __init__(self, pairs):
        self.pairs = list(pairs)
        self.epoch = 0
        self.name = "synthetic-chunks"

    def case_ids(self):
        return [case_id for case_id, _length in self.pairs]

    def waveform_lengths(self):
        return [length for _case_id, length in self.pairs]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def identity(self):
        return {"name": self.name, "cases": self.case_ids(), "lengths": self.waveform_lengths()}

    def collate_chunks(self, chunks):
        clean, noisy, ids = [], [], []
        for chunk in chunks:
            width = chunk.stop - chunk.start
            clean.append(torch.full((width,), 0.1))
            noisy.append(torch.full((width,), 0.2 + 0.01 * (chunk.start % 7)))
            ids.append(chunk.case_id)
        return {"clean": torch.stack(clean), "noisy": torch.stack(noisy), "case_id": ids}


class _TinyTask:
    def __init__(self):
        torch.manual_seed(5)
        self.layer = torch.nn.Linear(1, 1)
        self.optim = torch.optim.SGD(self.layer.parameters(), lr=0.05)
        self.config = {"model": {"name": "tiny"}, "logging": {"tensorboard": {"audio": {"enabled": False}, "spectrogram": False}}}

    def distribute(self, process):
        from src.training.core.distributed import wrap_module

        self.layer = wrap_module(self.layer, process)

    def train_mode(self):
        self.layer.train()

    def eval_mode(self):
        from src.training.core.distributed import unwrap

        unwrap(self.layer).eval()

    def learning_rates(self):
        return {"generator": float(self.optim.param_groups[0]["lr"])}

    def step_schedulers(self, validation=None):
        del validation

    def state_dict(self):
        from src.training.core.distributed import unwrap

        return {"layer": unwrap(self.layer).state_dict(), "optim": self.optim.state_dict()}

    def load_state_dict(self, state):
        from src.training.core.distributed import unwrap

        unwrap(self.layer).load_state_dict(state["layer"])
        self.optim.load_state_dict(state["optim"])

    def generator_state_dict(self):
        from src.training.core.distributed import unwrap

        return unwrap(self.layer).state_dict()

    def train_update(self, batches, accumulation):
        from src.training.core.distributed import unwrap

        self.train_mode()
        self.optim.zero_grad(set_to_none=True)
        total = 0.0
        for batch in batches:
            noisy = batch["noisy"].mean(dim=1, keepdim=True)
            clean = batch["clean"].mean(dim=1, keepdim=True)
            loss = torch.nn.functional.mse_loss(self.layer(noisy), clean)
            (loss / accumulation).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(unwrap(self.layer).parameters(), 5.0)
        self.optim.step()
        return {"generator": total / len(batches)}

    def validation_report(self, batch):
        value = float(batch["noisy"].mean())
        metric = {"status": "ok", "value": value, "reason": None}
        return {
            "losses": {"generator": metric},
            "pesq": {"status": "na", "value": None, "reason": "skip", "count": 0},
            "stoi": {"status": "na", "value": None, "reason": "skip", "count": 0},
        }


def _chunk_config(directory: Path, *, world: int, steps: int, virtual: int | None = None) -> dict:
    data = {
        "database": "unused.duckdb",
        "train_set": "toy",
        "validation_set": "toy_valid",
        "rms_normalize": False,
    }
    if virtual is not None:
        data["virtual_world_size"] = virtual
    return {
        "seed": 17,
        "deterministic": True,
        "global_batch_size": 4,
        "micro_batch_size_per_gpu": 4,
        "max_steps": steps,
        "segment_size": 8,
        "loss": "official",
        "checkpoint_dir": str(directory),
        "checkpoint_interval": 1,
        "validation_interval": 0,
        "chunk_iterator": {
            "num_samples_per_epoch": 4,
            "chunk_length": 8,
            "chunk_shift_ratio": 0.5,
            "num_cache_chunks": 6,
            "shuffle": True,
            "max_epochs": 3,
            "early_stopping_patience": 5,
            "keep_nbest": 2,
        },
        "model": {"name": "tiny"},
        "data": data,
        "runtime": {"device": "cpu", "allow_busy_gpu": False},
        "logging": {"tensorboard": {"enabled": False, "scalar_interval": 10, "audio": {"enabled": False, "max_cases": 0, "max_seconds": 1}, "spectrogram": False}},
    }


def _fit(directory: Path, dataset, validation, *, world: int, steps: int, resume=None, virtual=None, rank=0, distributed=False):
    config = _chunk_config(directory, world=world, steps=steps, virtual=virtual)
    plan_world = virtual or world
    plan = plan_batches(4, 4, plan_world)
    process = ProcessInfo(
        rank=rank,
        world_size=1 if virtual else world,
        local_rank=rank,
        device=torch.device("cpu"),
        distributed=distributed,
    )
    task = _TinyTask()
    trainer = Trainer(
        task, plan, process, max_steps=steps, seed=17, checkpoint_dir=directory,
        checkpoint_interval=1, validation_interval=0, config=config,
    )
    trainer.fit(dataset, validation, resume=resume)
    return trainer


class ResumeTests(unittest.TestCase):
    def _pair(self):
        train = _Waves([(f"t{i}", 24 + (i % 4) * 5) for i in range(10)])
        valid = _Waves([(f"v{i}", 20 + i * 3) for i in range(6)])
        return train, valid

    def test_virtual_resume_matches_a_continuous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, valid = self._pair()
            continuous_dir = root / "full"
            _fit(continuous_dir, train, valid, world=1, steps=4, virtual=2)
            full_lines = [
                json.loads(line)
                for line in (continuous_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["type"] == "train"
            ]
            stopped = root / "stop"
            _fit(stopped, _Waves(train.pairs), _Waves(valid.pairs), world=1, steps=2, virtual=2)
            checkpoint = stopped / "step_00000002.pt"
            payload = load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(payload["data_state"]["pipeline"], "espnet_chunk_v1")
            self.assertEqual(payload["data_state"]["ddp_mode"], "virtual")
            self.assertEqual(payload["data_state"]["world_size"], 2)
            self.assertEqual(payload["update_in_epoch"], payload["data_state"]["update_in_epoch"])
            resumed = root / "resume"
            resumed.mkdir()
            for name in (
                "step_00000002.pt",
                "data_provenance.json",
                "step_00000002.data_provenance.json",
                "metrics.jsonl",
            ):
                source = stopped / name
                if source.is_file():
                    (resumed / name).write_bytes(source.read_bytes())
            _fit(
                resumed,
                _Waves(train.pairs),
                _Waves(valid.pairs),
                world=1,
                steps=4,
                virtual=2,
                resume=resumed / "step_00000002.pt",
            )
            resumed_lines = [
                json.loads(line)
                for line in (resumed / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["type"] == "train"
            ]
            self.assertEqual(
                [line["case_ids"] for line in full_lines],
                [line["case_ids"] for line in resumed_lines],
            )
            self.assertTrue((resumed / "data_provenance.json").is_file())
            provenance = json.loads((resumed / "data_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["ddp_mode"], "virtual")

    def test_changed_world_size_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, valid = self._pair()
            _fit(root / "run", train, valid, world=1, steps=1, virtual=2)
            other = root / "other"
            other.mkdir()
            for name in ("step_00000001.pt", "data_provenance.json", "step_00000001.data_provenance.json"):
                source = root / "run" / name
                if source.is_file():
                    (other / name).write_bytes(source.read_bytes())
            with self.assertRaises(TrainingError) as caught:
                _fit(
                    other,
                    _Waves(train.pairs),
                    _Waves(valid.pairs),
                    world=1,
                    steps=2,
                    virtual=4,
                    resume=other / "step_00000001.pt",
                )
            self.assertTrue(
                "batch plan" in str(caught.exception) or "chunk plan" in str(caught.exception)
            )


class DistributedChunkTests(unittest.TestCase):
    def test_two_process_plan_validation_and_rank0_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def argv_for_port(port):
                return [
                    [sys.executable, str(ROOT / "tests/training/p5_cpu_worker.py"), str(rank), "2", str(port), str(root)]
                    for rank in range(2)
                ]

            run_distributed_workers(argv_for_port, cwd=ROOT, timeout=120)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["plan_hashes"], [summary["plan_hashes"][0], summary["plan_hashes"][0]])
            self.assertEqual(len(set(summary["updates"])), 1)
            self.assertAlmostEqual(summary["validation"], summary["expected_validation"])
            self.assertEqual(summary["provenance_writers"], [0])
            self.assertTrue(summary["weights_equal"])
            provenance = json.loads((root / "ckpt" / "data_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["ddp_mode"], "real")
            self.assertEqual(provenance["world_size"], 2)


class ConfigTests(unittest.TestCase):
    def test_seed17_configs_keep_their_rms_settings(self):
        official = load_training_config(ROOT / "configs/training/tf_gridnet_chunk_seed17.yaml")
        research = load_training_config(ROOT / "configs/training/tf_gridnet_fixed5000_seed17.yaml")
        self.assertFalse(official["data"]["rms_normalize"])
        self.assertIn("chunk_iterator", official)
        self.assertNotIn("chunk_iterator", research)
        self.assertEqual(research["logical_steps_per_epoch"], 5000)
        self.assertTrue(research["data"]["rms_normalize"])

    def test_virtual_world_size_requires_the_chunk_iterator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                """
seed: 1
deterministic: false
global_batch_size: 4
micro_batch_size_per_gpu: 4
max_steps: 1
segment_size: 8
loss: official
checkpoint_dir: /tmp/unused
checkpoint_interval: 0
validation_interval: 0
optimizer: {name: adam, lr: 0.001, betas: [0.9, 0.999], weight_decay: 0.0}
scheduler: {name: plateau, factor: 0.7, patience: 1}
model: {name: tf_gridnet, variant: ms-snsd, sampling_rate: 16000, architecture: {}, stft: {}}
data:
  database: unused.duckdb
  train_set: toy
  validation_set: toy_valid
  virtual_world_size: 2
runtime: {device: cpu, allow_busy_gpu: false}
""",
                encoding="utf-8",
            )
            with self.assertRaises(TrainingError):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
