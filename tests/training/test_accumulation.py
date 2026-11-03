"""Gradient accumulation, scheduler steps, and a two-process batch."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.training.core.batch import plan_batches
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer, backward
from src.training.se.mp_senet import MPSENetTask
from tests.training.ddp_worker import build_model, shared_batch
from tests.training.support import cpu_process, paired_dataset, run_distributed_workers, tiny_config

ROOT = Path(__file__).resolve().parents[2]


def tearDownModule():
    set_determinism(False, cuda=False)


class AccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_accumulated_gradients_match_the_full_batch(self):
        features, target = shared_batch()
        full = build_model()
        full.zero_grad(set_to_none=True)
        F.mse_loss(full(features), target).backward()
        full_grad = torch.cat([parameter.grad.reshape(-1) for parameter in full.parameters()])

        micro = build_model()
        micro.zero_grad(set_to_none=True)
        for index in range(2):
            chunk_x = features[index * 2 : (index + 1) * 2]
            chunk_y = target[index * 2 : (index + 1) * 2]
            loss = F.mse_loss(micro(chunk_x), chunk_y) / 2
            backward(loss, [micro], last=index == 1)
        micro_grad = torch.cat([parameter.grad.reshape(-1) for parameter in micro.parameters()])
        self.assertTrue(torch.allclose(full_grad, micro_grad, rtol=1e-5, atol=1e-6))

    def test_scheduler_steps_once_per_completed_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accumulated = _run_scheduler(root / "accum", micro_limit=2)
            single = _run_scheduler(root / "single", micro_limit=4)
        self.assertEqual(accumulated[0], 2)
        self.assertEqual(single[0], 2)
        # Eight waveforms and global batch 4 are one epoch at either accumulation.
        expected = 0.001 * 0.5
        self.assertAlmostEqual(accumulated[1], expected)
        self.assertAlmostEqual(single[1], expected)
        self.assertEqual(accumulated[2], 1)
        self.assertEqual(single[2], 1)

    def test_two_processes_keep_the_global_batch_gradient(self):
        features, target = shared_batch()
        reference = build_model()
        F.mse_loss(reference(features), target).backward()
        expected = torch.cat([parameter.grad.reshape(-1) for parameter in reference.parameters()])
        expected_flags = {"sync": [1], "accum": [0, 1], "split": [0, 1], "delay": [1], "manual": [0, 0]}
        for mode, flags in expected_flags.items():
            with self.subTest(mode=mode):
                gradients = _run_workers(mode)
                self.assertEqual(len(gradients), 2)
                for gradient, seen in gradients:
                    self.assertEqual(seen, flags)
                    self.assertTrue(torch.allclose(gradient, expected, rtol=1e-5, atol=1e-6))


def _run_scheduler(root: Path, *, micro_limit: int) -> tuple[int, float, int]:
    config = tiny_config(
        root / "ckpt",
        max_steps=2,
        global_batch_size=4,
        micro_batch_size_per_gpu=micro_limit,
    )
    dataset = paired_dataset(root / "wav", 8, seed=config["seed"])
    task = MPSENetTask(config, torch.device("cpu"))
    plan = plan_batches(4, micro_limit, 1)
    trainer = Trainer(
        task,
        plan,
        cpu_process(),
        max_steps=2,
        seed=config["seed"],
        checkpoint_dir=root / "ckpt",
        checkpoint_interval=0,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset)
    return trainer.global_step, task.learning_rates()["generator"], int(task.sched_g.last_epoch)


def _run_workers(mode: str) -> list[tuple[torch.Tensor, list[int]]]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        destinations = [root / f"{mode}-{rank}.pt" for rank in range(2)]

        def argv_for_port(port):
            return [
                [
                    sys.executable,
                    str(ROOT / "tests/training/ddp_worker.py"),
                    str(rank),
                    "2",
                    str(port),
                    mode,
                    str(destination),
                ]
                for rank, destination in enumerate(destinations)
            ]

        run_distributed_workers(argv_for_port, cwd=ROOT, timeout=120)
        loaded = [torch.load(path, weights_only=True) for path in destinations]
        return [(item["gradient"], item["flags"].tolist()) for item in loaded]


if __name__ == "__main__":
    unittest.main()
