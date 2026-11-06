"""Legacy adapter equivalence with the recipe-dataset trainer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from src.training.data.legacy import LegacyTrainingData
from src.training.data.sampler import epoch_indices
from tests.training.mock_task import LinearPairTask
from tests.training.support import cpu_process, paired_dataset, tiny_config


def _weights(task: LinearPairTask) -> list[torch.Tensor]:
    state = task.state_dict()
    return [tensor.detach().cpu() for tensor in state["generator"].values()] + [
        tensor.detach().cpu() for tensor in state["discriminator"].values()
    ]


class LegacyAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def tearDown(self):
        set_determinism(False, cuda=False)

    def test_order_and_waveforms_match_the_recipe_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = paired_dataset(root / "wav", 8, frames=320, segment_size=80, seed=7)
            plan = plan_batches(4, 2, 1)
            adapter = LegacyTrainingData(
                dataset,
                None,
                seed=7,
                micro_batch=plan.micro_batch_size,
                accumulation=plan.accumulation_steps,
                world_size=1,
            )
            dataset.set_epoch(0)
            indices = epoch_indices(len(dataset), 7, 0, 1, 0)
            expected = []
            cursor = 0
            while cursor + plan.accumulation_steps <= len(indices) // plan.micro_batch_size:
                group = []
                for _ in range(plan.accumulation_steps):
                    start = cursor * plan.micro_batch_size
                    group.append(dataset.collate(indices[start : start + plan.micro_batch_size]))
                    cursor += 1
                expected.append(group)
            produced = list(adapter.epoch(0, rank=0, world_size=1))
        self.assertEqual(len(produced), len(expected))
        for left_group, right_group in zip(produced, expected, strict=True):
            self.assertEqual(len(left_group), len(right_group))
            for left, right in zip(left_group, right_group, strict=True):
                self.assertEqual(left["case_id"], right["case_id"])
                self.assertEqual(left["clean"].shape, right["clean"].shape)
                self.assertTrue(torch.equal(left["clean"], right["clean"]))
                self.assertTrue(torch.equal(left["noisy"], right["noisy"]))

    def test_trainer_matches_a_recipe_dataset_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = _fit(root / "direct", adapter=False)
            wrapped = _fit(root / "wrapped", adapter=True)
        self.assertEqual(direct.case_ids, wrapped.case_ids)
        self.assertEqual(len(direct.case_ids), 2)
        for left, right in zip(_weights(direct), _weights(wrapped), strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_resume_and_prefetch_discard_match_a_continuous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = _fit(root / "continuous", adapter=True, steps=2, prefetch=True)
            partial_dir = root / "partial"
            _fit(partial_dir, adapter=True, steps=2, prefetch=True, segment_steps=1)
            checkpoint = partial_dir / "ckpt" / "step_00000001.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertFalse((partial_dir / "ckpt" / "complete.json").is_file())
            resumed = _fit(root / "resumed", adapter=True, steps=2, prefetch=True, resume=checkpoint, wav=partial_dir / "wav")
            saved = load_checkpoint(checkpoint, torch.device("cpu"))
        self.assertEqual(saved["data_state"]["update_in_epoch"], 1)
        self.assertEqual(saved["micro_batches_consumed"], saved["batch_plan"]["accumulation_steps"])
        self.assertEqual(continuous.case_ids[1:], resumed.case_ids)
        self.assertEqual(len(resumed.case_ids), 1)
        for left, right in zip(_weights(continuous), _weights(resumed), strict=True):
            self.assertTrue(torch.equal(left, right))


def _fit(root: Path, *, adapter: bool, steps: int = 2, prefetch: bool = False,
         segment_steps: int | None = None, resume: Path | None = None, wav: Path | None = None) -> LinearPairTask:
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config(
        root / "ckpt",
        max_steps=steps,
        seed=7,
        global_batch_size=4,
        micro_batch_size_per_gpu=2,
        loss="mock",
        checkpoint_interval=0,
        validation_interval=0,
    )
    config["data"]["prefetch"] = prefetch
    config["data"]["load_workers"] = 1
    plan = plan_batches(4, 2, 1)
    dataset = paired_dataset(root / "wav" if wav is None else wav, 8, frames=320, segment_size=80, seed=7, rewrite=wav is None)
    task = LinearPairTask(config, torch.device("cpu"))
    trainer = Trainer(
        task,
        plan,
        cpu_process(),
        max_steps=steps,
        seed=7,
        checkpoint_dir=root / "ckpt",
        checkpoint_interval=0,
        validation_interval=0,
        config=config,
    )
    if adapter:
        data = LegacyTrainingData(
            dataset,
            None,
            seed=7,
            micro_batch=plan.micro_batch_size,
            accumulation=plan.accumulation_steps,
            world_size=1,
        )
        trainer.fit(data, resume=resume, segment_steps=segment_steps)
        data.close()
    else:
        trainer.fit(dataset, resume=resume, segment_steps=segment_steps)
    return task


if __name__ == "__main__":
    unittest.main()
