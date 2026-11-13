"""Short production-path training on real snapshots.

The launcher, SnapshotDataModule, and the official trainers are the same code
as a full run. Models are the reduced architectures used by the trainer tests.
AudioPairBridge is not used. Writes stay outside data/.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/configs/experiments"
OUT = Path(os.environ.get("PIPELINE_SMOKE_DIR", "/tmp/audio-db-pipeline-smoke"))
PYTHON = sys.executable

RESAMPLE = {
    "tool": "soxr",
    "quality": "HQ",
    "source_rate": 48000,
    "target_rate": 16000,
    "length": "source_frames_plus_one_div_3",
    "order": ["read_float32", "soxr_hq", "protocol_normalize", "protocol_window"],
}

DATASETS = {
    "voicebank": {
        "snapshot": "mp_senet.vb28.ondemand.official@1",
        "snapshot_root": str(ROOT / "data/db/snapshots-voicebank-float-v1"),
        "resample": RESAMPLE,
    },
    "ms_snsd": {
        "snapshot": "tf_gridnet.ms_snsd.espnet_recipe@1",
        "snapshot_root": str(ROOT / "data/db/snapshots"),
        "resample": None,
    },
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("PIPELINE_SMOKE_ONLY")
    jobs = [job for job in _jobs() if only is None or job[0] == only]
    report = []
    failed = False
    for name, config in jobs:
        row = _run(name, config)
        report.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["status"] != "PASS":
            failed = True
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report {path}", flush=True)
    return 1 if failed else 0


def _jobs() -> list[tuple[str, dict]]:
    jobs: list[tuple[str, dict]] = []
    for dataset, spec in DATASETS.items():
        jobs.append((f"mp_senet-{dataset}", _mp(spec)))
        jobs.append((f"mossformergan-{dataset}", _moss(spec)))
        jobs.append((f"semambapp-{dataset}", _semamba(spec)))
        jobs.append((f"tf_gridnet-{dataset}", _grid(spec)))
        jobs.append((f"tf_locoformer-{dataset}", _loco(spec)))
    jobs.append(("deepfilternet3-official", _dfn_official()))
    jobs.append(("deepfilternet3-wave", _dfn_wave()))
    return jobs


def _mp(spec: dict) -> dict:
    config = _load(FIX / "mp_senet_gan.yaml")
    for name in ("generator", "metric_discriminator"):
        config["components"][name]["kwargs"].update(
            {"dense_channel": 4, "num_tsconformers": 1, "n_fft": 400, "hop_size": 100, "win_size": 400}
        )
        config["components"][name]["prepare"] = {"trainable": True, "distribute": "none"}
    _point(config, spec, fixed=True)
    config["components"]["sampler"]["kwargs"]["max_epoch_samples"] = 1
    config["components"]["trainer"]["kwargs"].update(
        {"experiment_claim": "research", "validation_interval": 1, "max_logical_steps": 1, "accumulation": 1}
    )
    config["components"]["batcher"]["kwargs"] = {"micro_batch": 1, "group_size": 1}
    return config


def _moss(spec: dict) -> dict:
    config = _load(FIX / "mossformergan_se_16k.yaml")
    for name in ("generator", "discriminator"):
        config["components"][name]["prepare"] = {
            "trainable": True,
            "distribute": "none",
            "find_unused_parameters": True,
            "broadcast_buffers": True,
            "sync_batchnorm": False,
        }
    _point(config, spec, fixed=True)
    config["components"]["sampler"]["kwargs"]["max_epoch_samples"] = 1
    config["components"]["trainer"]["kwargs"].update(
        {
            "experiment_claim": "research",
            "max_logical_steps": 2,
            "accumulation": 1,
            "validate_on_epoch_end": True,
        }
    )
    config["components"]["batcher"]["kwargs"] = {"micro_batch": 1, "group_size": 1}
    return config


def _semamba(spec: dict) -> dict:
    config = _load(FIX / "semambapp_gsr.yaml")
    config["deterministic"] = False
    for name in ("generator", "mrd", "cqt"):
        kwargs = config["components"][name]["kwargs"]
        kwargs.update({"hid_feature": 8, "num_tfmamba": 1, "d_state": 4, "d_conv": 2, "expand": 2})
        config["components"][name]["prepare"] = {"trainable": True, "distribute": "none"}
    _point(config, spec, fixed=True)
    config["components"]["protocol"]["kwargs"] = {
        "normalize": "independent_peak_v1",
        "window": 48000,
        "sample_rate": 16000,
        "validation_crop": "full_length",
        "resample": spec["resample"],
    }
    config["components"]["sampler"]["class"] = "src.experiments.data.OrderSampler"
    config["components"]["sampler"]["kwargs"] = {
        "seed": 1234,
        "order": "per_rank_independent_v1",
        "validation_scope": "rank0",
        "validation_batch": 1,
        "max_epoch_samples": 1,
    }
    config["components"]["protocol"]["class"] = "src.experiments.data.FixedPairProtocol"
    config["components"]["trainer"]["kwargs"].update(
        {"claim_label": "research", "validation_interval": 1, "max_logical_steps": 1, "accumulation": 1}
    )
    config["components"]["batcher"]["kwargs"] = {"micro_batch": 1, "group_size": 1}
    return config


def _grid(spec: dict) -> dict:
    config = _load(FIX / "tf_gridnet_ms_snsd.yaml")
    config["components"]["model"]["kwargs"]["architecture"] = {
        "activation": "prelu",
        "attn_approx_qk_dim": 8,
        "attn_n_head": 1,
        "emb_dim": 4,
        "emb_hs": 1,
        "emb_ks": 1,
        "eps": 1e-5,
        "lstm_hidden_units": 4,
        "n_fft": 16,
        "n_layers": 1,
        "stride": 8,
    }
    config["components"]["model"]["prepare"] = {"trainable": True, "distribute": "none"}
    config["components"]["loss"]["kwargs"]["windows"] = [8, 16]
    _point(config, spec, fixed=False)
    config["components"]["sampler"]["kwargs"].update(
        {
            "num_samples_per_epoch": 64,
            "chunk_length": 8000,
            "num_cache_chunks": 4,
            "global_batch": 1,
            "shuffle": False,
            "max_chunks": 1,
            "validation_scope": "per_rank",
        }
    )
    config["components"]["batcher"]["kwargs"] = {"micro_batch": 1, "group_size": 1}
    config["components"]["trainer"]["kwargs"]["max_logical_steps"] = 1
    config["components"]["trainer"]["kwargs"]["max_epochs"] = 1
    return config


def _loco(spec: dict) -> dict:
    config = _load(FIX / "tf_locoformer_dns.yaml")
    config["components"]["model"]["kwargs"]["architecture"] = {
        "attention_dim": 4,
        "conv1d_kernel": 4,
        "conv1d_shift": 1,
        "dropout": 0.0,
        "emb_dim": 4,
        "eps": 1e-5,
        "ffn_hidden_dim": [8, 8],
        "ffn_type": ["swiglu_conv1d", "swiglu_conv1d"],
        "flash_attention": False,
        "hop_length": 8,
        "n_fft": 16,
        "n_heads": 1,
        "n_layers": 1,
        "norm_type": "rmsgroupnorm",
        "normalize_variance": True,
        "num_groups": 4,
        "num_spk": 1,
        "pos_enc": "rope",
        "tf_order": "ft",
    }
    config["components"]["model"]["prepare"] = {"trainable": True, "distribute": "none"}
    config["components"]["loss"]["kwargs"]["windows"] = [8, 16]
    _point(config, spec, fixed=False)
    config["components"]["sampler"]["kwargs"].update(
        {
            "num_samples_per_epoch": 64,
            "chunk_length": 8000,
            "num_cache_chunks": 4,
            "global_batch": 1,
            "shuffle": False,
            "max_chunks": 1,
            "validation_scope": "per_rank",
        }
    )
    config["components"]["batcher"]["kwargs"] = {"micro_batch": 1, "group_size": 1}
    config["components"]["trainer"]["kwargs"]["max_logical_steps"] = 1
    config["components"]["trainer"]["kwargs"]["max_epochs"] = 1
    config["components"]["trainer"]["kwargs"]["validation_profile"] = "research"
    return config


def _dfn_official() -> dict:
    config = _load(FIX / "deepfilternet3.yaml")
    config["runtime"]["kwargs"]["device"] = "cuda"
    config["components"]["model"]["kwargs"]["architecture"] = {
        "conv_ch": 16,
        "nb_erb": 8,
        "nb_df": 8,
        "emb_hidden_dim": 16,
        "emb_num_layers": 2,
        "df_hidden_dim": 16,
        "df_num_layers": 1,
        "enc_lin_groups": 8,
        "lin_groups": 8,
        "df_order": 5,
    }
    config["components"]["model"]["prepare"]["distribute"] = "none"
    config["components"]["data"]["kwargs"]["max_validation_batches"] = 1
    config["components"]["trainer"]["kwargs"]["max_logical_steps"] = 16
    config["components"]["trainer"]["kwargs"]["max_epochs"] = 1
    config["checkpoint"]["selector"]["kwargs"]["interval"] = 1
    return config


def _dfn_wave() -> dict:
    root = OUT / "wave-pair"
    if not (root / "noisy" / "pair.wav").is_file():
        _write_mixed_pair(root)
    config = _dfn_official()
    config["components"].pop("source", None)
    config["components"].pop("sampler", None)
    config["components"].pop("protocol", None)
    config["components"].pop("batcher", None)
    config["components"]["data"] = {
        "role": "data",
        "class": "src.experiments.data.pair.WavePairData",
        "kwargs": {"root": str(root), "sample_rate": 48000, "teacher": "official_mixed_target", "batch_size": 1},
    }
    config["components"]["trainer"]["kwargs"]["batch_schedule"] = None
    config["components"]["trainer"]["kwargs"]["max_logical_steps"] = 1
    config["components"]["trainer"]["kwargs"]["max_epochs"] = 1
    return config


def _point(config: dict, spec: dict, *, fixed: bool) -> None:
    config["runtime"]["kwargs"]["device"] = "cuda"
    config["components"]["source"]["kwargs"]["snapshot"] = spec["snapshot"]
    config["components"]["source"]["kwargs"]["snapshot_root"] = spec["snapshot_root"]
    if fixed:
        config["components"]["protocol"]["kwargs"]["resample"] = spec["resample"]
    else:
        config["components"]["protocol"]["kwargs"]["sample_rate"] = 16000
        config["components"]["protocol"]["kwargs"]["resample"] = spec["resample"]
    config["components"]["data"]["kwargs"]["max_validation_batches"] = 1
    config["components"]["data"]["kwargs"]["prefetch"] = False
    config["checkpoint"]["selector"]["kwargs"]["interval"] = 1


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run(name: str, config: dict) -> dict:
    directory = OUT / name
    row: dict = {"name": name, "status": "FAIL", "directory": str(directory)}
    if directory.exists():
        row["error"] = "directory already exists"
        return row
    directory.mkdir(parents=True)
    config = copy.deepcopy(config)
    config["output"] = {"directory": str(directory)}
    # Several official CUDA backward kernels have no deterministic implementation.
    # The short GPU proof still uses the production trainer. CPU tests keep determinism.
    config["deterministic"] = False
    trainer = config["components"]["trainer"]["kwargs"]
    if int(trainer.get("max_logical_steps") or 1) < 2:
        trainer["max_logical_steps"] = 2
    trainer["stop_after_logical_steps"] = 1
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    first = _launch(path)
    row["returncode"] = first.returncode
    checkpoint = _checkpoint(directory)
    row["checkpoint"] = None if checkpoint is None else str(checkpoint)
    if first.returncode not in {0, 3} or checkpoint is None:
        row["error"] = _tail(first.stderr + first.stdout)
        return row
    trainer.pop("stop_after_logical_steps", None)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    mismatch = copy.deepcopy(config)
    _retune(mismatch)
    mismatch_path = directory / "mismatch.yaml"
    mismatch_path.write_text(yaml.safe_dump(mismatch, sort_keys=False), encoding="utf-8")
    refused = _launch(mismatch_path, resume=checkpoint)
    row["mismatch_returncode"] = refused.returncode
    row["mismatch_tail"] = _tail(refused.stderr + refused.stdout)
    if refused.returncode == 0 or not _refused_pipeline(row["mismatch_tail"]):
        row["error"] = "resume did not refuse the changed pipeline"
        return row
    resumed = _launch(path, resume=checkpoint)
    row["resume_returncode"] = resumed.returncode
    if resumed.returncode != 0:
        row["error"] = _tail(resumed.stderr + resumed.stdout)
        return row
    row["status"] = "PASS"
    return row


def _refused_pipeline(text: str) -> bool:
    return (
        "does not match" in text
        or "is not accepted" in text
        or "config identity" in text
    )


def _retune(config: dict) -> None:
    protocol = config["components"].get("protocol")
    if protocol is None:
        config["components"]["data"]["kwargs"]["teacher"] = "dry_clean"
        return
    kwargs = protocol["kwargs"]
    if "normalize" in kwargs:
        kwargs["normalize"] = "independent_peak_v1" if kwargs.get("normalize") != "independent_peak_v1" else "noisy_rms_unit_v1"
        kwargs["window"] = int(kwargs.get("window") or 32000) + 1
    elif "chunk_length" in config["components"].get("sampler", {}).get("kwargs", {}):
        config["components"]["sampler"]["kwargs"]["chunk_length"] = int(config["components"]["sampler"]["kwargs"]["chunk_length"]) + 8
    else:
        klass = str(protocol.get("class", ""))
        if "Dfn3" in klass or "Semamba" in klass:
            sampler = config["components"]["sampler"]["kwargs"]
            sampler["seed"] = int(sampler.get("seed", 0)) + 1
        else:
            kwargs["window"] = int(kwargs.get("window") or 144000) + 1


def _launch(config: Path, resume: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [PYTHON, "-m", "src.experiments.launcher", "--config", str(config), "--nproc", "1"]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("PIPELINE_SMOKE_GPU", "2")
    return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False, timeout=1800)


def _checkpoint(directory: Path) -> Path | None:
    archive = sorted((directory / "archive").glob("*.pt"))
    if archive:
        return archive[-1]
    latest = directory / "latest.pt"
    if latest.is_file():
        return latest
    return None


def _tail(text: str) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-40:])


def _write_mixed_pair(root: Path) -> None:
    from src.experiments.data.protocols.online_dfn3 import mix_signals

    rng = np.random.default_rng(0)
    speech = rng.normal(0.0, 0.05, size=144000).astype(np.float64)
    noise = rng.normal(0.0, 0.02, size=144000).astype(np.float64)
    clean, _scaled, mixture = mix_signals(speech, speech, noise, snr_db=5.0, gain_db=-6.0)
    (root / "noisy").mkdir(parents=True)
    (root / "target").mkdir(parents=True)
    _wav(root / "noisy" / "pair.wav", np.asarray(mixture, dtype=np.float32))
    _wav(root / "target" / "pair.wav", np.asarray(clean, dtype=np.float32))


def _wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(pcm.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
