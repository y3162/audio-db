"""Small waveforms and configs for training tests."""

from __future__ import annotations

import math
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import soundfile as sf

from src.data.recipe import canonical_recipe, case_id_for, sha256_file
from src.training.core.distributed import ProcessInfo
from src.training.data.dataset import RecipeDataset, TrainExample


def free_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_distributed_workers(argv_for_port, *, cwd: Path, timeout: float = 180, attempts: int = 5) -> None:
    """Start one process group per port. Retry when the chosen port is taken.

    Rank 0 binds the port after this process has already closed its probe socket,
    so a parallel test can win the race. A failed rank used to leave the other
    rank blocked until the wait timeout.
    """
    last = ""
    for _ in range(attempts):
        port = free_tcp_port()
        logs = []
        processes = []
        for argv in argv_for_port(port):
            handle = tempfile.TemporaryFile()
            logs.append(handle)
            processes.append(subprocess.Popen(argv, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + timeout
        failed = False
        while time.monotonic() < deadline:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                failed = True
                break
            if all(code == 0 for code in codes):
                for handle in logs:
                    handle.close()
                return
            time.sleep(0.05)
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait()
        chunks = []
        for handle in logs:
            handle.seek(0)
            chunks.append(handle.read().decode(errors="replace"))
            handle.close()
        last = "\n".join(chunks)
        if failed and ("EADDRINUSE" in last or "address already in use" in last.lower()):
            continue
        if not failed:
            raise AssertionError(f"distributed workers timed out\n{last}")
        raise AssertionError(f"distributed workers failed\n{last}")
    raise AssertionError(f"distributed port stayed busy\n{last}")


def cpu_process() -> ProcessInfo:
    return ProcessInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"), distributed=False)


def tiny_config(checkpoint_dir: Path, **overrides) -> dict:
    config = {
        "seed": 7,
        "deterministic": True,
        "global_batch_size": 4,
        "micro_batch_size_per_gpu": 2,
        "max_steps": 2,
        "segment_size": 160,
        "loss": "reconstruction",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_interval": 0,
        "validation_interval": 0,
        "optimizer": {"name": "adamw", "lr": 0.001, "betas": [0.8, 0.99], "weight_decay": 0.0},
        "scheduler": {"name": "exponential", "gamma": 0.5},
        "model": {
            "name": "mp_senet",
            "dense_channel": 4,
            "num_tsconformers": 1,
            "n_fft": 32,
            "hop_size": 8,
            "win_size": 32,
            "compress_factor": 0.3,
            "beta": 2.0,
            "sampling_rate": 16000,
        },
        "data": {
            "database": "unused.duckdb",
            "train_set": "toy_train",
            "validation_set": None,
        },
        "runtime": {"device": "cpu", "allow_busy_gpu": False},
    }
    config.update(overrides)
    return config


def tone(frames: int, frequency: float, rate: int = 16000) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32) / rate
    return (0.2 * torch.sin(2 * math.pi * frequency * time)).contiguous()


def write_wav(path: Path, waveform: torch.Tensor, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # WAV headers emitted by some FFmpeg versions contain timestamps, which
    # change recipe hashes when the same fixture is recreated in another run.
    sf.write(str(path), waveform.detach().cpu().numpy(), rate, subtype="PCM_16")


def recorded_example(
    root: Path,
    name: str,
    *,
    frames: int,
    clean_hz: float,
    noisy_hz: float,
    rewrite: bool = True,
) -> TrainExample:
    clean = root / f"{name}-clean.wav"
    noisy = root / f"{name}-noisy.wav"
    if rewrite or not clean.is_file():
        write_wav(clean, tone(frames, clean_hz))
    if rewrite or not noisy.is_file():
        write_wav(noisy, tone(frames, noisy_hz))
    recipe = canonical_recipe(
        {
            "recipe_version": 1,
            "sources": {
                "clean": {"path": clean.name, "sha256": sha256_file(clean)},
                "noisy": {"path": noisy.name, "sha256": sha256_file(noisy)},
            },
            "operations": [],
            "output": {"sample_rate": 16000, "reference": "dry_clean"},
        }
    )
    return TrainExample(case_id_for(recipe), recipe, "train", "toy", "dry_clean")


def paired_dataset(
    root: Path,
    count: int,
    *,
    frames: int = 160,
    segment_size: int = 160,
    seed: int = 7,
    training: bool = True,
    role: str = "train",
    name: str = "toy_train",
    rewrite: bool = True,
) -> RecipeDataset:
    examples = [
        recorded_example(
            root,
            f"item{index}",
            frames=frames,
            clean_hz=180 + 15 * index,
            noisy_hz=90 + 11 * index,
            rewrite=rewrite,
        )
        for index in range(count)
    ]
    return RecipeDataset(
        examples,
        root,
        name=name,
        role=role,
        seed=seed,
        sample_rate=16000,
        segment_size=segment_size,
        training=training,
        database=None,
    )
