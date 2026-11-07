"""SIGTERM on one MossFormerGAN rank stops the other rank at the same boundary."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.training.core.distributed import ProcessInfo
from src.training.core.trainer import Trainer, request_training_stop, reset_training_stop
from tests.training.support import free_tcp_port


def _worker(rank: int, world: int, port: int, result_dir: str, signal_rank0: bool) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        reset_training_stop()
        trainer = Trainer.__new__(Trainer)
        trainer.task = SimpleNamespace(validate_on_data_epoch_end=True)
        trainer.process = ProcessInfo(rank, world, rank, torch.device("cpu"), True)
        trainer._deadline = None
        trainer._segment_end = None
        dist.barrier()
        if signal_rank0 and rank == 0:
            request_training_stop(15, None)
        dist.barrier()
        agreed = trainer._shared_stop_requested()
        Path(result_dir, f"rank{rank}.txt").write_text("1" if agreed else "0", encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


class MossFormerStopAgreementTests(unittest.TestCase):
    def _run(self, signal_rank0: bool) -> list[str]:
        port = free_tcp_port()
        with tempfile.TemporaryDirectory() as root:
            ctx = mp.get_context("spawn")
            processes = []
            for rank in range(2):
                process = ctx.Process(
                    target=_worker,
                    args=(rank, 2, port, root, signal_rank0),
                )
                process.start()
                processes.append(process)
            for process in processes:
                process.join(timeout=60)
            for process in processes:
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
            return [
                Path(root, "rank0.txt").read_text(encoding="utf-8"),
                Path(root, "rank1.txt").read_text(encoding="utf-8"),
            ]

    def test_one_rank_signal_stops_both(self) -> None:
        self.assertEqual(self._run(True), ["1", "1"])

    def test_no_signal_keeps_both_running(self) -> None:
        self.assertEqual(self._run(False), ["0", "0"])


if __name__ == "__main__":
    unittest.main()
