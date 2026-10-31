"""Gradient accumulation, scheduler steps, and a two-process batch."""

import socket
import subprocess
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
from tests.training.support import cpu_process, paired_dataset, tiny_config

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

    def test_scheduler_follows_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config(root / "ckpt", max_steps=2, global_batch_size=4, micro_batch_size_per_gpu=2)
            dataset = paired_dataset(root / "wav", 8, seed=config["seed"])
            task = MPSENetTask(config, torch.device("cpu"))
            plan = plan_batches(4, 2, 1)
            self.assertEqual((plan.micro_batch_size, plan.accumulation_steps), (2, 2))
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
            self.assertEqual(trainer.global_step, 2)
            expected = 0.001 * (0.5**2)
            self.assertAlmostEqual(task.learning_rates()["generator"], expected)

    def test_two_processes_keep_the_global_batch_gradient(self):
        features, target = shared_batch()
        reference = build_model()
        F.mse_loss(reference(features), target).backward()
        expected = torch.cat([parameter.grad.reshape(-1) for parameter in reference.parameters()])
        for mode in ("sync", "accum"):
            with self.subTest(mode=mode):
                gradients = _run_workers(mode)
                self.assertEqual(len(gradients), 2)
                for gradient in gradients:
                    self.assertTrue(torch.allclose(gradient, expected, rtol=1e-5, atol=1e-6))


def _run_workers(mode: str) -> list[torch.Tensor]:
    port = _free_port()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        processes = []
        destinations = []
        for rank in range(2):
            destination = root / f"{mode}-{rank}.pt"
            destinations.append(destination)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "tests/training/ddp_worker.py"),
                        str(rank),
                        "2",
                        str(port),
                        mode,
                        str(destination),
                    ],
                    cwd=str(ROOT),
                )
            )
        codes = [process.wait(timeout=120) for process in processes]
        self_codes = codes
        if any(code != 0 for code in self_codes):
            raise AssertionError(f"ddp workers failed: {self_codes}")
        return [torch.load(path, weights_only=True) for path in destinations]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
