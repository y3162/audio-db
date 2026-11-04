import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint
from src.training.core.trainer import Trainer
from src.training.data.chunk_plan import epoch_chunk_batches
from tests.training.support import cpu_process


class Waves:
    def __init__(self, ids, lengths):
        self.ids = list(ids)
        self._lengths = list(lengths)
        self.epoch = None

    def __len__(self):
        return len(self.ids)

    def identity(self):
        return {"name": "waves", "ids": self.ids}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def case_ids(self):
        return list(self.ids)

    def waveform_lengths(self):
        return list(self._lengths)

    def collate(self, indices):
        return {"case_id": [self.ids[i] for i in indices]}

    def collate_chunks(self, chunks):
        return {"case_id": [chunk.case_id for chunk in chunks], "start": [chunk.start for chunk in chunks]}


class Task:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.Adam([self.weight], lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1)
        self.seen = []
        self.sizes = []
        self.notes = []
        self.retargets = []

    def distribute(self, process):
        pass

    def train_update(self, group, accumulation):
        self.seen.extend(group[0]["case_id"])
        self.sizes.append(len(group[0]["case_id"]))
        self.optimizer.zero_grad()
        (self.weight - 1).square().backward()
        self.optimizer.step()
        return {"generator": float(self.weight.detach())}

    def step_schedulers(self, summary):
        if summary is not None:
            self.scheduler.step(summary["losses"]["generator"]["value"])

    def state_dict(self):
        return {
            "weight": self.weight.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state):
        self.weight.data.copy_(state["weight"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

    def generator_state_dict(self):
        return {"weight": self.weight.detach().clone()}

    def eval_mode(self):
        pass

    def train_mode(self):
        pass

    def validation_report(self, batch):
        metric = {"status": "ok", "value": 1.0, "reason": None}
        return {"losses": {"generator": metric}, "pesq": metric, "stoi": metric}

    def learning_rates(self):
        return {"generator": self.optimizer.param_groups[0]["lr"]}

    def retarget_deepfilter_schedule(self, steps):
        self.retargets.append(steps)

    def note_schedule_position(self, index):
        self.notes.append(index)


def _chunk_config():
    return {
        "seed": 17,
        "loss": "official",
        "model": {"name": "tf_gridnet"},
        "micro_batch_size_per_gpu": 2,
        "best_checkpoint_start_epoch": 0,
        "chunk_iterator": {
            "num_samples_per_epoch": 8,
            "chunk_length": 8,
            "chunk_shift_ratio": 0.5,
            "num_cache_chunks": 4,
            "shuffle": False,
            "max_epochs": 3,
            "early_stopping_patience": 1,
            "keep_nbest": 1,
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_chunk_epoch_is_not_a_dataset_pass(self):
        ids = ["a", "b", "c"]
        lengths = [30, 30, 12]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task()
            config = _chunk_config()
            trainer = Trainer(
                task, plan_batches(2, 2, 1), cpu_process(), max_steps=100, seed=17,
                checkpoint_dir=root, checkpoint_interval=0, validation_interval=0, config=config,
            )
            data = Waves(ids, lengths)
            trainer.fit(data, Waves(["v"], [40]))
            planned = epoch_chunk_batches(
                ids, lengths, espnet_epoch=1, seed=17, num_samples=8, chunk_length=8,
                batch_size=2, cache_size=4, shuffle=False,
            )
            self.assertGreater(8, len(ids))
            self.assertEqual(task.seen[:sum(len(batch) for batch in planned)],
                             [chunk.case_id for batch in planned for chunk in batch])
            records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in records if row["type"] == "validation"], [1, 2, 3])
            self.assertEqual(task.scheduler.last_epoch, 3)
            self.assertTrue((root / "1epoch.pt").is_file())
            self.assertFalse((root / "2epoch.pt").is_file())
            self.assertTrue((root / "valid.loss.best.pt").is_file())

    def test_chunk_resume_keeps_the_next_slice(self):
        ids = ["a", "b", "c"]
        lengths = [30, 30, 30]
        config = _chunk_config()
        config["chunk_iterator"]["early_stopping_patience"] = 9
        config["chunk_iterator"]["max_epochs"] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_dir = root / "full"
            task = Task()
            Trainer(
                task, plan_batches(2, 2, 1), cpu_process(), max_steps=100, seed=17,
                checkpoint_dir=full_dir, checkpoint_interval=1, validation_interval=0, config=config,
            ).fit(Waves(ids, lengths), Waves(["v"], [40]))
            partial = Task()
            Trainer(
                partial, plan_batches(2, 2, 1), cpu_process(), max_steps=1, seed=17,
                checkpoint_dir=root / "part", checkpoint_interval=1, validation_interval=0, config=config,
            ).fit(Waves(ids, lengths), Waves(["v"], [40]))
            resumed = Task()
            Trainer(
                resumed, plan_batches(2, 2, 1), cpu_process(), max_steps=100, seed=17,
                checkpoint_dir=root / "resume", checkpoint_interval=1, validation_interval=0, config=config,
            ).fit(Waves(ids, lengths), Waves(["v"], [40]), resume=root / "part" / "step_00000001.pt")
            self.assertEqual(resumed.seen, task.seen[len(partial.seen):])
            self.assertTrue(torch.equal(resumed.weight, task.weight))
            self.assertEqual(resumed.scheduler.state_dict(), task.scheduler.state_dict())
            payload = load_checkpoint(full_dir / "step_00000001.pt", torch.device("cpu"))
            self.assertEqual(payload["epoch"], 0)
            self.assertGreater(payload["micro_batches_consumed"], 0)

    def test_batch_schedule_changes_size_and_cosine_index(self):
        config = {
            "seed": 1,
            "loss": "official",
            "model": {"name": "deepfilternet3"},
            "micro_batch_size_per_gpu": 4,
            "batch_size_limit": 4,
            "batch_schedule": [[0, 2], [1, 4]],
            "best_checkpoint_start_epoch": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task()
            Trainer(
                task, plan_batches(2, 2, 1), cpu_process(), max_steps=4, seed=1,
                checkpoint_dir=root / "full", checkpoint_interval=1, validation_interval=0, config=config,
            ).fit(Waves(["a", "b", "c", "d", "e", "f"], [8] * 6), None)
            self.assertEqual(task.sizes, [2, 2, 2, 4])
            self.assertEqual(task.retargets, [3, 1])
            self.assertEqual(task.notes, [0, 1, 2, 1])
            partial = Task()
            Trainer(
                partial, plan_batches(2, 2, 1), cpu_process(), max_steps=2, seed=1,
                checkpoint_dir=root / "part", checkpoint_interval=1, validation_interval=0, config=config,
            ).fit(Waves(["a", "b", "c", "d", "e", "f"], [8] * 6), None)
            resumed = Task()
            Trainer(
                resumed, plan_batches(2, 2, 1), cpu_process(), max_steps=4, seed=1,
                checkpoint_dir=root / "resume", checkpoint_interval=1, validation_interval=0, config=config,
            ).fit(
                Waves(["a", "b", "c", "d", "e", "f"], [8] * 6), None,
                resume=root / "part" / "step_00000002.pt",
            )
            self.assertEqual(resumed.sizes, [2, 4])
            self.assertEqual(resumed.notes, [2, 1])

    def test_batch_schedule_stops_after_patience(self):
        config = {
            "seed": 1,
            "loss": "official",
            "model": {"name": "deepfilternet3"},
            "micro_batch_size_per_gpu": 4,
            "batch_size_limit": 4,
            "batch_schedule": [[0, 2], [1, 4], [2, 2]],
            "early_stopping_patience": 1,
            "best_checkpoint_start_epoch": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            task = Task()
            Trainer(
                task, plan_batches(2, 2, 1), cpu_process(), max_steps=20, seed=1,
                checkpoint_dir=Path(directory), checkpoint_interval=0, validation_interval=0, config=config,
            ).fit(Waves(["a", "b", "c", "d", "e", "f"], [8] * 6), Waves(["v"], [8]))
            self.assertEqual(task.sizes, [2, 2, 2, 4])
            self.assertEqual(task.scheduler.last_epoch, 2)
