"""MossFormerGAN gradient averaging and clipping against the released step order.

The comparisons use a tiny model so the DDP result is the PyTorch reduction,
not a MossFormerGAN weight. World size 1 of the real model is recorded in
``docs/training/reproduction/preflight/P10_C_FIX2A_NUMERICAL_COMPARISON.md``.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.training.support import free_tcp_port


class Gen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.4, -0.2, 0.7, 0.1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * x


class Disc(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([0.3, -0.5, 0.2, 0.8]))
        self.b = torch.nn.Parameter(torch.tensor([-0.1, 0.4, 0.6, -0.3]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (self.a * z).sum() + (self.b * z * z).sum()


def _flat(module: torch.nn.Module) -> list[float]:
    return torch.cat([parameter.detach().flatten().float().cpu() for parameter in module.parameters()]).tolist()


def _grad(module: torch.nn.Module) -> list[float]:
    parts = []
    for parameter in module.parameters():
        if parameter.grad is None:
            parts.append(torch.zeros_like(parameter).flatten())
        else:
            parts.append(parameter.grad.detach().flatten())
    return torch.cat(parts).float().cpu().tolist()


def _max_abs(left: list[float], right: list[float]) -> float:
    return float((torch.tensor(left) - torch.tensor(right)).abs().max())


def _broadcast(modules: list[torch.nn.Module]) -> None:
    for module in modules:
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)


def _average_new(modules: list[torch.nn.Module], world: int, before: dict[int, torch.Tensor | None]) -> None:
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            old = before[id(parameter)]
            raw = parameter.grad if old is None else parameter.grad - old
            delta = raw.detach().contiguous().clone()
            dist.all_reduce(delta, op=dist.ReduceOp.SUM)
            delta.div_(world)
            parameter.grad = delta if old is None else old + delta


def _before(modules: list[torch.nn.Module]) -> dict[int, torch.Tensor | None]:
    saved: dict[int, torch.Tensor | None] = {}
    for module in modules:
        for parameter in module.parameters():
            saved[id(parameter)] = None if parameter.grad is None else parameter.grad.detach().clone()
    return saved


def _worker(rank: int, world: int, port: int, result_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log_path = Path(result_dir) / f"rank{rank}.log"
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        payload = {
            "synced_clip": _synced_accumulation(rank, world, 0.5),
            "synced_noclip": _synced_accumulation(rank, world, 1e9),
            "three_forward": _three_forward(rank, world),
        }
        if rank == 0:
            (Path(result_dir) / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        dist.destroy_process_group()
    except Exception:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _synced_accumulation(rank: int, world: int, clip: float) -> dict[str, float]:
    """One forward and one backward per micro-batch. Both ranks participate."""
    torch.manual_seed(0)
    official = DDP(Gen(), find_unused_parameters=True)
    delta = DDP(Gen(), find_unused_parameters=True)
    once = DDP(Gen(), find_unused_parameters=True)
    _broadcast([official, delta, once])
    opts = {
        "official": torch.optim.AdamW(official.parameters(), lr=5e-4),
        "delta": torch.optim.AdamW(delta.parameters(), lr=5e-4),
        "once": torch.optim.AdamW(once.parameters(), lr=5e-4),
    }
    # The two micro-batches are not parallel, and each is large enough to cross a clip of 0.5.
    patterns = (
        torch.tensor([8.0, -1.0, 0.5, 2.0]),
        torch.tensor([-1.0, 9.0, -4.0, 0.2]),
    )
    scales = (1.0, 0.7) if rank == 0 else (-0.3, 1.4)
    xs = [pattern * scale for pattern, scale in zip(patterns, scales, strict=True)]
    for index, x in enumerate(xs):
        loss_o = official(x).sum() / 2
        loss_o.backward()
        torch.nn.utils.clip_grad_norm_(official.parameters(), clip)
        saved = _before([delta])
        with delta.no_sync():
            loss_d = delta(x).sum() / 2
            loss_d.backward()
        _average_new([delta], world, saved)
        torch.nn.utils.clip_grad_norm_(delta.parameters(), clip)
        last = index == len(xs) - 1
        context = torch.enable_grad() if last else once.no_sync()
        with context:
            loss_c = once(x).sum() / 2
            loss_c.backward()
        if last:
            torch.nn.utils.clip_grad_norm_(once.parameters(), clip)
    for optimizer in opts.values():
        optimizer.step()
    return {
        "official_delta": _max_abs(_flat(official.module), _flat(delta.module)),
        "official_once": _max_abs(_flat(official.module), _flat(once.module)),
        "official_rank_ready": _flat(official.module)[0],
    }


def _three_forward(rank: int, world: int) -> dict[str, float]:
    """Released discriminator pattern: three forwards, then two backwards."""
    torch.manual_seed(0)
    gen = DDP(Gen(), find_unused_parameters=True, broadcast_buffers=True)
    disc = DDP(Disc(), find_unused_parameters=True, broadcast_buffers=True)
    _broadcast([gen, disc])
    delta_gen = DDP(Gen(), find_unused_parameters=True, broadcast_buffers=True)
    delta_disc = DDP(Disc(), find_unused_parameters=True, broadcast_buffers=True)
    with torch.no_grad():
        delta_gen.module.load_state_dict(gen.module.state_dict())
        delta_disc.module.load_state_dict(disc.module.state_dict())
    _broadcast([delta_gen, delta_disc])
    x = torch.tensor([1.0, -2.0, 0.5, 3.0]) * (1.0 if rank == 0 else -0.5)
    y = gen(x)
    gen_loss = disc(y)
    fake = disc(y.detach())
    real = disc(x)
    disc_loss = (fake - real).square()
    gen_loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 10.0)
    disc_loss.backward()
    official_d = _grad(disc)
    disc_opt = torch.optim.AdamW(disc.parameters(), lr=1e-3)
    disc_opt.step()

    saved = _before([delta_gen, delta_disc])
    with delta_gen.no_sync(), delta_disc.no_sync():
        y2 = delta_gen(x)
        gen_loss2 = delta_disc(y2)
        fake2 = delta_disc(y2.detach())
        real2 = delta_disc(x)
        disc_loss2 = (fake2 - real2).square()
    gen_loss2.backward()
    _average_new([delta_gen, delta_disc], world, saved)
    torch.nn.utils.clip_grad_norm_(delta_gen.parameters(), 10.0)
    saved = _before([delta_gen, delta_disc])
    disc_loss2.backward()
    _average_new([delta_gen, delta_disc], world, saved)
    delta_d = _grad(delta_disc)
    delta_opt = torch.optim.AdamW(delta_disc.parameters(), lr=1e-3)
    delta_opt.step()
    gathered = [None, None]
    dist.all_gather_object(gathered, {
        "official_g": _flat(gen.module),
        "official_d": _flat(disc.module),
        "official_d_grad": official_d,
        "delta_d": _flat(delta_disc.module),
        "delta_d_grad": delta_d,
    })
    first, second = gathered
    return {
        "generator_rank": _max_abs(first["official_g"], second["official_g"]),
        "discriminator_rank": _max_abs(first["official_d"], second["official_d"]),
        "discriminator_grad_rank": _max_abs(first["official_d_grad"], second["official_d_grad"]),
        "delta_discriminator_rank": _max_abs(first["delta_d"], second["delta_d"]),
        "delta_grad_rank": _max_abs(first["delta_d_grad"], second["delta_d_grad"]),
        "official_vs_delta_grad": _max_abs(first["official_d_grad"], first["delta_d_grad"]),
    }


class MossFormerUpdateRuleTests(unittest.TestCase):
    def test_synced_microbatches_match_increment_averaging(self) -> None:
        payload = self._run()
        clipped = payload["synced_clip"]
        self.assertEqual(clipped["official_delta"], 0.0)
        self.assertGreater(clipped["official_once"], 5e-4)
        self.assertEqual(payload["synced_noclip"]["official_delta"], 0.0)
        self.assertEqual(payload["synced_noclip"]["official_once"], 0.0)

    def test_second_discriminator_backward_is_not_averaged(self) -> None:
        row = self._run()["three_forward"]
        self.assertEqual(row["generator_rank"], 0.0)
        self.assertGreater(row["discriminator_rank"], 1e-3)
        self.assertGreater(row["discriminator_grad_rank"], 10.0)
        self.assertEqual(row["delta_discriminator_rank"], 0.0)
        self.assertEqual(row["delta_grad_rank"], 0.0)
        self.assertGreater(row["official_vs_delta_grad"], 10.0)

    def _run(self) -> dict:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        with tempfile.TemporaryDirectory() as directory:
            port = free_tcp_port()
            context = mp.get_context("spawn")
            processes = [
                context.Process(target=_worker, args=(rank, 2, port, directory))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(120)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.kill()
            logs = []
            for rank in range(2):
                path = Path(directory) / f"rank{rank}.log"
                logs.append(path.read_text(encoding="utf-8") if path.is_file() else "")
            self.assertFalse(alive, "\n".join(logs))
            self.assertEqual([process.exitcode for process in processes], [0, 0], "\n".join(logs))
            return json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
