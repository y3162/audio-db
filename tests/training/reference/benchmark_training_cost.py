"""Time one official-width training update. Does not run an epoch.

The measured step is the independent task used for training: forward, backward,
gradient clip, and optimizer step. Warmup is reported separately. Results go
to a new directory and do not replace older runs.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))

from src.training.se.run import build_task


OFFICIAL = {
    "mp_senet": {
        "global_batch": 4,
        "segment": 32000,
        "sample_rate": 16000,
        "note": "config.json batch 4, segment 32000, 16 kHz",
    },
    "semambapp": {
        "global_batch": 8,
        "segment": 48000,
        "sample_rate": 16000,
        "note": "config.yaml batch 8, segment 48000, 16 kHz, fused scan",
    },
    "tf_gridnet": {
        "global_batch": 4,
        "segment": 48000,
        "sample_rate": 16000,
        "note": "ms-snsd chunk 48000, batch 4, grad clip 5",
    },
    "deepfilternet3": {
        "global_batch": 16,
        "segment": 144000,
        "sample_rate": 48000,
        "note": "config.ini starts the batch schedule at 16 and max_sample_len_s 3.0",
    },
    "mossformergan": {
        "global_batch": 2,
        "segment": 32000,
        "sample_rate": 16000,
        "note": "yaml batch 1, effec_batch_size 2, max_length 2 s",
    },
}


def config_for(name: str, micro: int, accumulation: int) -> dict:
    spec = OFFICIAL[name]
    common = {
        "seed": 1234,
        "deterministic": True,
        "global_batch_size": spec["global_batch"],
        "micro_batch_size_per_gpu": micro,
        "max_steps": 1,
        "segment_size": spec["segment"],
        "loss": "official",
        "checkpoint_dir": "data/results/training/cost-unused",
        "checkpoint_interval": 0,
        "validation_interval": 0,
        "best_checkpoint_start_epoch": 40,
        "data": {"database": "unused", "train_set": "unused", "validation_set": "unused"},
        "runtime": {"device": "cuda", "allow_busy_gpu": False},
    }
    if name == "mp_senet":
        common.update(
            optimizer={"name": "adamw", "lr": 5e-4, "betas": [0.8, 0.99], "weight_decay": 0.01},
            scheduler={"name": "exponential", "gamma": 0.99},
            model={
                "name": "mp_senet", "dense_channel": 64, "num_tsconformers": 4,
                "n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": 0.3,
                "beta": 2.0, "sampling_rate": 16000,
            },
        )
    elif name == "semambapp":
        common.update(
            optimizer={"name": "adamw", "lr": 2e-4, "betas": [0.8, 0.99], "weight_decay": 0.01},
            scheduler={"name": "exponential", "gamma": 0.99},
            model={
                "name": "semambapp", "sampling_rate": 16000,
                "architecture": {
                    "hid_feature": 48, "num_tfmamba": 4, "d_state": 16, "d_conv": 4,
                    "expand": 2, "n_fft": 400, "scan_backend": "fused",
                },
                "stft": {"hop": 100, "win": 400},
            },
        )
    elif name == "tf_gridnet":
        common.update(
            grad_clip=5.0,
            optimizer={"name": "adam", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0},
            scheduler={"name": "plateau", "factor": 0.5, "patience": 3},
            model={"name": "tf_gridnet", "variant": "ms-snsd", "sampling_rate": 16000, "architecture": {}, "stft": {}},
        )
    elif name == "deepfilternet3":
        common.update(
            grad_clip=1.0,
            optimizer={"name": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 1e-12, "amsgrad": True},
            scheduler={
                "name": "deepfilter_cosine", "steps_per_epoch": 669, "epochs": 120,
                "lr_min": 1e-6, "lr_warmup": 1e-4, "warmup_epochs": 3, "weight_decay_end": 0.01,
            },
            model={"name": "deepfilternet3", "sampling_rate": 48000, "variant": "deepfilternet3", "architecture": {}},
        )
    elif name == "mossformergan":
        common.update(
            grad_clip=10.0,
            optimizer={"name": "adamw", "lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.01},
            scheduler={"name": "none"},
            model={
                "name": "mossformergan", "variant": "se_16k", "sampling_rate": 16000,
                "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
            },
        )
    else:
        raise SystemExit(name)
    common["micro_batch_size_per_gpu"] = micro
    common["accumulation_measured"] = accumulation
    return common


def speech(batch: int, length: int, sample_rate: int, device: torch.device) -> dict[str, torch.Tensor]:
    root = ROOT / "data" / "corpora" / "VoiceBank-DEMAND"
    wavs = sorted(root.glob("clean_trainset_28spk_wav/*.wav"))
    if not wavs:
        raise SystemExit(f"no VoiceBank training wav under {root}")
    import torchaudio
    wave, rate = torchaudio.load(wavs[0])
    wave = wave.mean(0)
    if rate != sample_rate:
        wave = torchaudio.functional.resample(wave, rate, sample_rate)
    if wave.numel() < length:
        wave = wave.repeat((length // wave.numel()) + 1)
    clean = wave[:length].unsqueeze(0).repeat(batch, 1)
    noisy = (clean * 0.6 + 0.01 * torch.randn_like(clean)).clamp(-1, 1)
    return {"clean": clean.to(device), "noisy": noisy.to(device)}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(OFFICIAL))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats(device.index)
    spec = OFFICIAL[args.model]
    micro = args.micro_batch or spec["global_batch"]
    if spec["global_batch"] % micro != 0:
        raise SystemExit("micro batch must divide the official global batch")
    accumulation = spec["global_batch"] // micro
    if args.accumulation != 1 and args.micro_batch is None:
        raise SystemExit("set --micro-batch when changing accumulation")
    if args.micro_batch is not None:
        accumulation = args.accumulation
    cfg = config_for(args.model, micro, accumulation)
    started = time.perf_counter()
    task = build_task(cfg, device)
    sync(device)
    init_seconds = time.perf_counter() - started
    load_started = time.perf_counter()
    batch = speech(micro, spec["segment"], spec["sample_rate"], device)
    sync(device)
    load_seconds = time.perf_counter() - load_started
    batches = [batch for _ in range(accumulation)]

    def one_step():
        sync(device)
        t0 = time.perf_counter()
        losses = task.train_update(batches, accumulation)
        sync(device)
        return time.perf_counter() - t0, {key: float(value) for key, value in losses.items()}

    warmup_seconds, warmup_loss = one_step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index)
    measured = []
    for _ in range(args.measured_steps):
        seconds, losses = one_step()
        measured.append({"seconds": seconds, "loss": losses})
    checkpoint_seconds = None
    checkpoint_path = out.parent / f"{args.model}-checkpoint-timing.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        sync(device)
        t0 = time.perf_counter()
        torch.save(task.state_dict(), checkpoint_path)
        sync(device)
        checkpoint_seconds = time.perf_counter() - t0
    validation_seconds = None
    if hasattr(task, "validation_report"):
        task.eval_mode()
        sync(device)
        t0 = time.perf_counter()
        task.validation_report(batch)
        sync(device)
        validation_seconds = time.perf_counter() - t0
    times = [row["seconds"] for row in measured]
    payload = {
        "model": args.model,
        "official": spec,
        "micro_batch": micro,
        "accumulation": accumulation,
        "global_batch": micro * accumulation,
        "init_seconds": init_seconds,
        "data_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "warmup_loss": warmup_loss,
        "measured": measured,
        "seconds_per_step_median": sorted(times)[len(times) // 2],
        "seconds_per_step_mean": sum(times) / len(times),
        "seconds_per_step_min": min(times),
        "seconds_per_step_max": max(times),
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "validation_seconds_one_batch": validation_seconds,
        "peak_allocated": torch.cuda.max_memory_allocated(device.index) if device.type == "cuda" else None,
        "peak_reserved": torch.cuda.max_memory_reserved(device.index) if device.type == "cuda" else None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tf32": False,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
        "train_cases": 10700,
        "repeated_utterance": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "model": args.model,
        "median_seconds": payload["seconds_per_step_median"],
        "peak_allocated": payload["peak_allocated"],
        "global_batch": payload["global_batch"],
        "accumulation": accumulation,
    }, indent=2))


if __name__ == "__main__":
    main()
