"""SyncBatchNorm on the MossFormerGAN discriminator, and the rank-0 deadlock."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

def _prefer_repo_tests() -> None:
    repo = Path(__file__).resolve().parents[2]
    kept = []
    for entry in list(sys.path):
        base = Path(entry) if entry else Path.cwd()
        foreign = base / "tests" / "__init__.py"
        try:
            drop = foreign.is_file() and base.resolve() != repo
        except OSError:
            drop = False
        if not drop:
            kept.append(entry)
    sys.path[:] = kept
    root = str(repo)
    if root not in sys.path:
        sys.path.insert(0, root)


_prefer_repo_tests()

import torch
import torch.distributed as dist
import torch.nn as nn

from src.experiments.calc.distributed import ProcessInfo
from src.experiments.official.mossformergan.discriminator import Discriminator
from src.experiments.official.mossformergan.reference import MossFormerGANTask
from tests.numerics.support import cpu_process, free_tcp_port, pin_repo_root_for_spawn


def _config(root: Path) -> dict:
    return {
        "seed": 3,
        "loss": "official",
        "grad_clip": 10.0,
        "optimizer": {"name": "adamw", "lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "mossformergan",
            "variant": "se_16k",
            "sampling_rate": 16000,
            "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
        },
        "checkpoint_dir": str(root),
    }


def _sync_count(module: torch.nn.Module) -> int:
    return sum(isinstance(child, torch.nn.SyncBatchNorm) for child in module.modules())


def install_cpu_gloo_syncbn() -> None:
    """Replace SyncBatchNorm.forward with the gloo all_gather it cannot run on CPU.

    ``SyncBatchNorm.forward`` raises ``ValueError`` for a CPU tensor before
    calling ``aten::batch_norm_stats``, which has no CPU kernel. The gloo
    branch of ``torch.nn.modules._functions.SyncBatchNorm`` would
    ``all_gather`` mean, invstd, and count. This stand-in does that gather
    and updates running stats from the gathered batch mean. It is not the
    CUDA SyncBN op.
    """

    original = nn.SyncBatchNorm.forward

    def forward(self, input):
        if (
            input.device.type != "cpu"
            or not self.training
            or not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() < 2
        ):
            return original(self, input)
        dims = tuple(index for index in range(input.dim()) if index != 1)
        count = float(input.numel() // input.shape[1])
        mean = input.mean(dim=dims)
        var = input.var(dim=dims, unbiased=False)
        combined = torch.cat([mean.reshape(-1), var.reshape(-1), mean.new_tensor([count])])
        gathered = [torch.empty_like(combined) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, combined)
        stacked = torch.stack(gathered, 0)
        channels = mean.numel()
        counts = stacked[:, 2 * channels]
        total = counts.sum().clamp_min(1.0)
        global_mean = (stacked[:, :channels] * counts.unsqueeze(1)).sum(0) / total
        global_var = (stacked[:, channels : 2 * channels] * counts.unsqueeze(1)).sum(0) / total
        momentum = 0.1 if self.momentum is None else float(self.momentum)
        if self.running_mean is not None:
            self.running_mean.mul_(1.0 - momentum).add_(global_mean, alpha=momentum)
            self.running_var.mul_(1.0 - momentum).add_(global_var, alpha=momentum)
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
        shape = [1] * input.dim()
        shape[1] = -1
        out = (input - global_mean.reshape(shape)) / torch.sqrt(global_var.reshape(shape) + self.eps)
        if self.weight is not None:
            out = out * self.weight.reshape(shape)
        if self.bias is not None:
            out = out + self.bias.reshape(shape)
        return out

    nn.SyncBatchNorm.forward = forward


def _participate_worker(rank: int, world: int, port: int, result_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log_path = Path(result_dir) / f"part{rank}.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            from datetime import timedelta

            dist.init_process_group(backend="gloo", rank=rank, world_size=world, timeout=timedelta(seconds=120))
            unpatched = nn.SyncBatchNorm(4).train()
            try:
                unpatched(torch.randn(8, 4))
                cpu_error = ""
            except ValueError as exc:
                cpu_error = str(exc)
            task = MossFormerGANTask(_config(Path(result_dir)), torch.device("cpu"))
            task.distribute(ProcessInfo(rank, world, rank, torch.device("cpu"), True))
            discriminator = task.discriminator
            bare = discriminator.module if isinstance(discriminator, torch.nn.parallel.DistributedDataParallel) else discriminator
            try:
                bare.train()(torch.rand(1, 1, 201, 19), torch.rand(1, 1, 201, 19))
                disc_error = ""
            except ValueError as exc:
                disc_error = str(exc)
            install_cpu_gloo_syncbn()
            tiny = nn.SyncBatchNorm(4).train()
            before = tiny.running_mean.detach().clone()
            value = tiny(torch.randn(8, 4) + rank)
            bare.train()
            spectrum = torch.rand(1, 1, 201, 19)
            score = bare(spectrum, spectrum)
            mean = bare.cnn.bn1.running_mean.detach().cpu().clone()
            payload = {
                "cpu_syncbn_error": cpu_error,
                "discriminator_cpu_error": disc_error,
                "tiny_changed": not torch.equal(before, tiny.running_mean.detach().cpu()),
                "tiny_mean": tiny.running_mean.detach().cpu().tolist(),
                "tiny_finite": bool(torch.isfinite(value).all()),
                "validation_requires_all_ranks": task.validation_requires_all_ranks,
                "sync_discriminator": _sync_count(discriminator),
                "sync_generator": _sync_count(task.generator.module),
                "discriminator_is_ddp": isinstance(task.discriminator, torch.nn.parallel.DistributedDataParallel),
                "broadcast_buffers": task.discriminator_ddp_broadcast_buffers,
                "find_unused_parameters": task.discriminator_ddp_find_unused_parameters,
                "running_mean0": float(mean[0]),
                "score_finite": bool(torch.isfinite(score).all()),
            }
            (Path(result_dir) / f"part{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            log.write(traceback.format_exc())
            raise


def _rank0_only_worker(rank: int, world: int, port: int, result_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log_path = Path(result_dir) / f"hang{rank}.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            from datetime import timedelta

            dist.init_process_group(backend="gloo", rank=rank, world_size=world, timeout=timedelta(seconds=8))
            install_cpu_gloo_syncbn()
            module = nn.SyncBatchNorm(4).train()
            if rank == 0:
                module(torch.randn(8, 4))
                log.write("rank0 forward returned\n")
            else:
                log.write("rank1 did not enter the forward\n")
                log.flush()
                import time
                time.sleep(30)
        except Exception:
            log.write(traceback.format_exc())
            raise


class MossFormerSyncBatchNormTests(unittest.TestCase):
    def test_world_size_one_keeps_batchnorm_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = MossFormerGANTask(_config(root), torch.device("cpu"))
            task.distribute(cpu_process())
            self.assertFalse(task.validation_requires_all_ranks)
            self.assertEqual(_sync_count(task.discriminator), 0)
            self.assertEqual(_sync_count(task.generator), 0)
            self.assertGreater(
                sum(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) for module in task.discriminator.modules()),
                0,
            )
            other = MossFormerGANTask(_config(root), torch.device("cpu"))
            other.load_state_dict(task.state_dict())
            for key, value in task.discriminator.state_dict().items():
                self.assertTrue(torch.equal(value, other.discriminator.state_dict()[key]), key)
            fresh = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Discriminator(ndf=16))
            fresh.load_state_dict(task.discriminator.state_dict(), strict=True)
            self.assertTrue(any(key.endswith("running_mean") for key in fresh.state_dict()))
            self.assertTrue(any(key.endswith("running_var") for key in fresh.state_dict()))
            self.assertTrue(torch.equal(fresh.cnn.bn1.running_mean, task.discriminator.cnn.bn1.running_mean))
            self.assertTrue(torch.equal(fresh.cnn.bn1.running_var, task.discriminator.cnn.bn1.running_var))

    def test_both_ranks_run_syncbn_forward(self):
        import multiprocessing as mp

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        pin_repo_root_for_spawn()
        with tempfile.TemporaryDirectory() as directory:
            port = free_tcp_port()
            ctx = mp.get_context("spawn")
            processes = [ctx.Process(target=_participate_worker, args=(rank, 2, port, directory)) for rank in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(180)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.kill()
            for process in processes:
                process.join()
            logs = "\n".join(
                (Path(directory) / f"part{rank}.log").read_text(encoding="utf-8")
                if (Path(directory) / f"part{rank}.log").is_file()
                else ""
                for rank in range(2)
            )
            self.assertFalse(alive, logs)
            self.assertEqual([process.exitcode for process in processes], [0, 0], logs)
            payloads = [json.loads((Path(directory) / f"part{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
            self.assertTrue(all("GPU" in item["cpu_syncbn_error"] for item in payloads))
            self.assertTrue(all(item["discriminator_cpu_error"] == "" for item in payloads), payloads)
            self.assertTrue(all(item["tiny_changed"] and item["tiny_finite"] for item in payloads))
            self.assertEqual(payloads[0]["tiny_mean"], payloads[1]["tiny_mean"])
            self.assertTrue(all(item["validation_requires_all_ranks"] is False for item in payloads))
            self.assertTrue(all(item["sync_discriminator"] == 0 for item in payloads))
            self.assertTrue(all(item["sync_generator"] == 0 for item in payloads))
            self.assertTrue(all(item["discriminator_is_ddp"] is True for item in payloads))
            self.assertTrue(all(item["broadcast_buffers"] is True for item in payloads))
            self.assertTrue(all(item["find_unused_parameters"] is True for item in payloads))
            self.assertTrue(all(item["score_finite"] for item in payloads))
            self.assertTrue(all(item["running_mean0"] == item["running_mean0"] for item in payloads))

    def test_rank0_only_syncbn_forward_does_not_finish(self):
        import multiprocessing as mp

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        pin_repo_root_for_spawn()
        with tempfile.TemporaryDirectory() as directory:
            port = free_tcp_port()
            ctx = mp.get_context("spawn")
            processes = [ctx.Process(target=_rank0_only_worker, args=(rank, 2, port, directory)) for rank in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(25)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.kill()
            for process in processes:
                process.join()
            logs = "\n".join(
                (Path(directory) / f"hang{rank}.log").read_text(encoding="utf-8")
                if (Path(directory) / f"hang{rank}.log").is_file()
                else ""
                for rank in range(2)
            )
            finished_clean = [process.exitcode == 0 for process in processes]
            self.assertFalse(all(finished_clean), logs)
            lowered = logs.lower()
            timed_out = "timed out" in lowered or "timeout" in lowered or bool(alive)
            self.assertTrue(timed_out, logs)


if __name__ == "__main__":
    unittest.main()
