"""Short real-model DDP smoke for the ESPnet chunk iterator.

Run under torchrun on a node that actually has the requested GPU count.
``gridnet`` expects 2 GPUs. ``locoformer`` expects 4 GPUs and 64,000-sample
chunks. The networks are smaller than the published recipes; the data plan
uses the published rank, batch, and chunk rules. This does not claim that
virtual-rank gradients match NCCL.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reproduction.pipeline.chunking import plan_chunk_epoch
from src.training.core.batch import plan_batches
from src.training.core.checkpoint import load_checkpoint
from src.training.core.distributed import barrier, setup_process, unwrap
from src.training.core.trainer import Trainer
from src.training.se.tf_gridnet import TFGridNetTask
from src.training.se.tf_locoformer import TFLocoformerTask


PROFILES = {
    "gridnet": {
        "world": 2,
        "chunk_length": 48000,
        "length": 80000,
        "num_samples": 1,
        "utterances": 4,
        "valid": 4,
        "cache": 16,
        "model_name": "tf_gridnet",
    },
    "locoformer": {
        "world": 4,
        "chunk_length": 64000,
        "length": 70000,
        "num_samples": 1,
        "utterances": 8,
        "valid": 8,
        "cache": 16,
        "model_name": "tf_locoformer",
    },
}


class SineSet:
    def __init__(self, pairs, *, name: str) -> None:
        self.pairs = list(pairs)
        self.name = name
        self.sample_rate = 16000
        self.waves = {}
        for index, (case_id, length) in enumerate(self.pairs):
            time = torch.arange(length, dtype=torch.float32) / self.sample_rate
            self.waves[case_id] = 0.2 * torch.sin(2 * math.pi * (80 + 15 * index) * time)

    def case_ids(self):
        return [case_id for case_id, _length in self.pairs]

    def waveform_lengths(self):
        return [length for _case_id, length in self.pairs]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def identity(self):
        return {"name": self.name, "cases": self.case_ids(), "lengths": self.waveform_lengths(), "sample_rate": self.sample_rate}

    def collate_chunks(self, chunks):
        clean, noisy, ids = [], [], []
        for chunk in chunks:
            piece = self.waves[chunk.case_id][chunk.start:chunk.stop].contiguous()
            clean.append(piece)
            noisy.append((0.8 * piece).contiguous())
            ids.append(chunk.case_id)
        return {"clean": torch.stack(clean), "noisy": torch.stack(noisy), "case_id": ids}


def _config(profile: str, directory: Path, spec: dict, *, max_steps: int) -> dict:
    chunk = {
        "num_samples_per_epoch": spec["num_samples"],
        "chunk_length": spec["chunk_length"],
        "chunk_shift_ratio": 0.5,
        "num_cache_chunks": spec["cache"],
        "shuffle": True,
        "max_epochs": 2,
        "early_stopping_patience": 5,
        "keep_nbest": 1,
    }
    if profile == "gridnet":
        model = {
            "name": "tf_gridnet",
            "variant": "ms-snsd",
            "sampling_rate": 16000,
            "architecture": {
                "n_fft": 256,
                "stride": 128,
                "n_layers": 1,
                "lstm_hidden_units": 32,
                "attn_n_head": 4,
                "attn_approx_qk_dim": 128,
                "emb_dim": 16,
                "emb_ks": 4,
                "emb_hs": 4,
                "activation": "prelu",
                "eps": 1.0e-5,
            },
            "stft": {},
        }
        optimizer = {"name": "adam", "lr": 0.001, "betas": [0.9, 0.999], "weight_decay": 0.0}
        scheduler = {"name": "plateau", "factor": 0.7, "patience": 1}
    else:
        model = {
            "name": "tf_locoformer",
            "variant": "dns",
            "sampling_rate": 16000,
            "architecture": {
                "n_layers": 1,
                "emb_dim": 32,
                "num_groups": 4,
                "n_heads": 4,
                "attention_dim": 32,
                "ffn_hidden_dim": [64, 64],
                "dropout": 0.0,
                "n_fft": 256,
                "hop_length": 128,
            },
            "stft": {},
        }
        optimizer = {"name": "adamw", "lr": 0.001, "betas": [0.9, 0.999], "weight_decay": 0.01}
        scheduler = {
            "name": "warmup_reduce_on_plateau",
            "warmup_steps": 4000,
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0.0,
            "eps": 1e-8,
        }
    return {
        "seed": 17,
        "deterministic": False,
        "global_batch_size": 4,
        "micro_batch_size_per_gpu": 4,
        "max_steps": max_steps,
        "segment_size": spec["chunk_length"],
        "loss": "official",
        "grad_clip": 5.0,
        "checkpoint_dir": str(directory),
        "checkpoint_interval": 1,
        "validation_interval": 0,
        "best_checkpoint_start_epoch": 0,
        "chunk_iterator": chunk,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "model": model,
        "data": {
            "database": "unused.duckdb",
            "train_set": f"p5-{profile}-train",
            "validation_set": f"p5-{profile}-valid",
            "rms_normalize": False,
        },
        "runtime": {"device": "cuda", "allow_busy_gpu": True},
        "logging": {
            "tensorboard": {
                "enabled": False,
                "scalar_interval": 10,
                "audio": {"enabled": False, "max_cases": 0, "max_seconds": 1},
                "spectrogram": False,
            }
        },
    }


def _pairs(prefix: str, count: int, length: int):
    return [(f"{prefix}{index}", length) for index in range(count)]


def _reference_plan(train: SineSet, spec: dict):
    return plan_chunk_epoch(
        list(zip(train.case_ids(), train.waveform_lengths())),
        espnet_epoch=1,
        seed=17,
        world_size=spec["world"],
        num_samples=spec["num_samples"],
        chunk_length=spec["chunk_length"],
        global_batch=4,
        num_cache_chunks=spec["cache"],
        shuffle=True,
    )


def _run(profile: str, directory: Path, spec: dict, train: SineSet, valid: SineSet, resume: Path | None, *, max_steps: int) -> Trainer:
    process = setup_process("cuda")
    if process.world_size != spec["world"]:
        raise SystemExit(f"{profile} requires world_size {spec['world']}, found {process.world_size}")
    if torch.cuda.device_count() < spec["world"]:
        raise SystemExit(f"{profile} sees {torch.cuda.device_count()} CUDA devices, needs {spec['world']}")
    if process.rank == 0 and resume is None and directory.exists():
        shutil.rmtree(directory)
    barrier(process)
    config = _config(profile, directory, spec, max_steps=max_steps)
    task = TFGridNetTask(config, process.device) if profile == "gridnet" else TFLocoformerTask(config, process.device)
    trainer = Trainer(
        task,
        plan_batches(4, 4, spec["world"]),
        process,
        max_steps=max_steps,
        seed=17,
        checkpoint_dir=directory,
        checkpoint_interval=1,
        validation_interval=0,
        config=config,
    )
    trainer.fit(train, valid, resume=resume)
    return trainer


def main() -> None:
    profile = sys.argv[1]
    if profile not in PROFILES:
        raise SystemExit("profile must be gridnet or locoformer")
    spec = PROFILES[profile]
    if os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) != spec["world"]:
        raise SystemExit(f"WORLD_SIZE {os.environ['WORLD_SIZE']} does not match {profile}")
    root = ROOT / "data" / "results" / "training" / "p5-smoke" / profile
    if int(os.environ.get("RANK", "0")) == 0:
        root.mkdir(parents=True, exist_ok=True)
    train = SineSet(_pairs("t", spec["utterances"], spec["length"]), name=f"p5-{profile}-train")
    valid = SineSet(_pairs("v", spec["valid"], spec["length"]), name=f"p5-{profile}-valid")
    reference = _reference_plan(train, spec)
    first_dir = root / "first"
    _run(profile, first_dir, spec, train, valid, None, max_steps=1)
    rank = int(os.environ.get("RANK", "0"))
    process_world = int(os.environ.get("WORLD_SIZE", "1"))
    names = []
    if rank == 0:
        payload = load_checkpoint(first_dir / "step_00000001.pt", torch.device("cpu"))
        state = payload["data_state"]
        next_plan = plan_chunk_epoch(
            list(zip(train.case_ids(), train.waveform_lengths())),
            espnet_epoch=2,
            seed=17,
            world_size=spec["world"],
            num_samples=spec["num_samples"],
            chunk_length=spec["chunk_length"],
            global_batch=4,
            num_cache_chunks=spec["cache"],
            shuffle=True,
        )
        if state["plan_hash"] != next_plan.plan_hash:
            raise SystemExit("saved plan_hash is not the next epoch of the CPU plan")
        if state["world_size"] != spec["world"] or state["ddp_mode"] != "real":
            raise SystemExit(f"unexpected data_state {state['ddp_mode']} world {state['world_size']}")
        provenance = json.loads((first_dir / "data_provenance.json").read_text(encoding="utf-8"))
        if provenance["ddp_mode"] != "real" or provenance["world_size"] != spec["world"]:
            raise SystemExit("provenance was not written for this DDP run")
        lines = [
            json.loads(line)
            for line in (first_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        train_lines = [line for line in lines if line["type"] == "train"]
        valid_lines = [line for line in lines if line["type"] == "validation"]
        if len(train_lines) != 1 or not valid_lines:
            raise SystemExit("expected one train step and a validation record")
        names = train_lines[0]["case_ids"]
        expected = [chunk.case_id for chunk in reference.rank_batches[0][0]]
        if names != expected:
            raise SystemExit(f"rank 0 batch {names} != plan {expected}")
        resume_dir = root / "resume"
        if resume_dir.exists():
            shutil.rmtree(resume_dir)
        resume_dir.mkdir()
        for name in ("step_00000001.pt", "data_provenance.json", "step_00000001.data_provenance.json", "metrics.jsonl"):
            source = first_dir / name
            if source.is_file():
                shutil.copy2(source, resume_dir / name)
    torch.distributed.barrier()
    resume_dir = root / "resume"
    resumed = _run(profile, resume_dir, spec, train, valid, resume_dir / "step_00000001.pt", max_steps=2)
    vector = torch.nn.utils.parameters_to_vector(unwrap(resumed.task.generator).parameters()).detach()
    gathered = [torch.empty_like(vector) for _ in range(process_world)]
    torch.distributed.all_gather(gathered, vector)
    if rank == 0:
        if not all(torch.equal(gathered[0], item) for item in gathered):
            raise SystemExit("ranks diverged after the synchronized step")
        second = [
            json.loads(line)
            for line in (resume_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["type"] == "train"
        ]
        resumed_ids = second[-1]["case_ids"]
        expected_next = [chunk.case_id for chunk in next_plan.rank_batches[0][0]]
        if resumed_ids != expected_next:
            raise SystemExit(f"resumed batch {resumed_ids} != next plan {expected_next}")
        summary = {
            "profile": profile,
            "world_size": spec["world"],
            "gpus": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "chunk_length": spec["chunk_length"],
            "per_rank_batch": reference.per_rank_batch,
            "per_rank_cache": reference.per_rank_cache,
            "updates": reference.updates,
            "plan_hash": reference.plan_hash,
            "resumed_case_ids": resumed_ids,
            "weights_equal": True,
            "model": "reduced architecture; data plan uses the published rank and chunk rules",
        }
        for line in (first_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("type") == "validation":
                summary["validation_loss"] = item["losses"]["generator"]["value"]
                summary["validation_weight"] = item.get("weight")
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("P5_GPU_SMOKE_OK", profile, reference.plan_hash, flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
