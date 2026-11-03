"""Mock generator and discriminator driven by the real Trainer."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint
from src.training.core.rng import set_determinism
from src.training.core.trainer import Trainer
from tests.training.mock_task import LinearPairTask
from tests.training.support import cpu_process, paired_dataset, run_distributed_workers, tiny_config

def tearDownModule():
    set_determinism(False, cuda=False)


class MockTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_determinism(True, cuda=False)

    def test_discriminator_and_generator_update_in_that_order(self):
        with tempfile.TemporaryDirectory() as directory:
            task, _trainer = _fit(Path(directory), steps=2, micro=4)
        self.assertEqual(task.events, ["disc", "gen", "disc", "gen"])

    def test_accumulated_update_matches_the_full_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, _ = _fit(root / "full", steps=1, micro=4)
            micro, _ = _fit(root / "micro", steps=1, micro=2)
        _assert_close(full.state_dict(), micro.state_dict())

    def test_two_processes_match_the_single_process_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single, _ = _fit(root / "single", steps=1, micro=4, count=4)
            for micro in (2, 1):
                with self.subTest(micro=micro):
                    distributed = _run_ddp(root / f"ddp-{micro}", micro)
                    _assert_close(single.state_dict(), distributed)

    def test_scheduler_advances_once_per_finished_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            task, trainer = _fit(Path(directory), steps=4, micro=4)
        self.assertEqual(trainer.global_step, 4)
        self.assertAlmostEqual(task.learning_rates()["generator"], 0.001 * 0.5**2)
        self.assertAlmostEqual(task.learning_rates()["discriminator"], 0.001 * 0.5**2)

    def test_resume_matches_a_continuous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _continuous, continuous_trainer = _fit(root / "continuous", steps=20, micro=4, interval=10)
            _partial, _partial_trainer = _fit(root / "partial", steps=10, micro=4, interval=10)
            resumed = _resume(root / "partial", root / "resumed", steps=20)
            left = load_checkpoint(root / "continuous" / "ckpt" / "step_00000020.pt", torch.device("cpu"))
            right = load_checkpoint(resumed, torch.device("cpu"))
            self.assertEqual(left["global_step"], right["global_step"])
            self.assertEqual(left["epoch"], right["epoch"])
            self.assertEqual(left["micro_batches_consumed"], right["micro_batches_consumed"])
            _assert_exact(left["task"], right["task"])
            _assert_rng(left["rng"], right["rng"])


def _fit(root: Path, *, steps: int, micro: int, interval: int = 0, count: int = 8):
    root.mkdir(parents=True, exist_ok=True)
    config = tiny_config(
        root / "ckpt",
        max_steps=steps,
        seed=7,
        global_batch_size=4,
        micro_batch_size_per_gpu=micro,
        loss="mock",
    )
    dataset = paired_dataset(root / "wav", count, frames=320, segment_size=80, seed=7)
    task = LinearPairTask(config, torch.device("cpu"))
    trainer = Trainer(
        task,
        plan_batches(4, micro, 1),
        cpu_process(),
        max_steps=steps,
        seed=7,
        checkpoint_dir=root / "ckpt",
        checkpoint_interval=interval,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset)
    return task, trainer


def _resume(partial: Path, destination: Path, *, steps: int) -> Path:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__)),
            "resume",
            str(partial),
            str(destination),
            str(steps),
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return destination / "ckpt" / "step_00000020.pt"


def _resume_main() -> None:
    partial = Path(sys.argv[2])
    destination = Path(sys.argv[3])
    steps = int(sys.argv[4])
    destination.mkdir(parents=True)
    config = tiny_config(
        destination / "ckpt",
        max_steps=steps,
        seed=7,
        global_batch_size=4,
        micro_batch_size_per_gpu=4,
        loss="mock",
    )
    dataset = paired_dataset(partial / "wav", 8, frames=320, segment_size=80, seed=7, rewrite=False)
    task = LinearPairTask(config, torch.device("cpu"))
    trainer = Trainer(
        task,
        plan_batches(4, 4, 1),
        cpu_process(),
        max_steps=steps,
        seed=7,
        checkpoint_dir=destination / "ckpt",
        checkpoint_interval=10,
        validation_interval=0,
        config=config,
    )
    trainer.fit(dataset, resume=partial / "ckpt" / "step_00000010.pt")


def _run_ddp(root: Path, micro: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    paired_dataset(root / "wav", 4, frames=320, segment_size=80, seed=7)
    destination = root / "state.pt"

    def argv_for_port(port):
        return [
            [
                sys.executable,
                str(ROOT / "tests/training/mock_ddp_worker.py"),
                str(rank),
                "2",
                str(port),
                str(root),
                str(destination),
                str(micro),
            ]
            for rank in range(2)
        ]

    run_distributed_workers(argv_for_port, cwd=ROOT, timeout=180)
    return torch.load(destination, weights_only=False)


def _assert_close(left: dict, right: dict) -> None:
    for name in ("generator", "discriminator"):
        for key, value in left[name].items():
            other = right[name][key]
            if not torch.allclose(value, other, rtol=1e-5, atol=1e-6):
                raise AssertionError(f"{name}.{key} max abs {(value - other).abs().max()}")
    for name in ("optim_g", "optim_d", "sched_g", "sched_d"):
        _assert_close_state(left[name], right[name], name)


def _assert_exact(left: dict, right: dict) -> None:
    for name in ("generator", "discriminator"):
        for key, value in left[name].items():
            other = right[name][key]
            if not torch.equal(value, other):
                raise AssertionError(f"{name}.{key} is not identical")
    for name in ("optim_g", "optim_d", "sched_g", "sched_d"):
        _assert_exact_state(left[name], right[name], name)


def _assert_rng(left: dict, right: dict) -> None:
    if left["python"] != right["python"]:
        raise AssertionError("python rng differs")
    if not torch.equal(left["torch"], right["torch"]):
        raise AssertionError("torch rng differs")
    if left["numpy"][0] != right["numpy"][0] or not torch.equal(
        torch.tensor(left["numpy"][1]), torch.tensor(right["numpy"][1])
    ):
        raise AssertionError("numpy rng differs")


def _assert_close_state(left, right, name: str) -> None:
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"{name} keys differ")
        for key, value in left.items():
            _assert_close_state(value, right[key], f"{name}.{key}")
        return
    if isinstance(left, torch.Tensor):
        if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
            raise AssertionError(f"{name} max abs {(left - right).abs().max()}")
        return
    if left != right:
        raise AssertionError(f"{name} differs: {left!r} vs {right!r}")


def _assert_exact_state(left, right, name: str) -> None:
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"{name} keys differ")
        for key, value in left.items():
            _assert_exact_state(value, right[key], f"{name}.{key}")
        return
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError(f"{name} is not identical")
        return
    if left != right:
        raise AssertionError(f"{name} differs: {left!r} vs {right!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        _resume_main()
    else:
        unittest.main()
