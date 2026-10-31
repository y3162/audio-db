"""Flat training configuration.

Unknown keys are rejected so a misspelled setting cannot be ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.training.errors import TrainingError

_TOP = {
    "seed",
    "deterministic",
    "global_batch_size",
    "micro_batch_size_per_gpu",
    "max_steps",
    "segment_size",
    "loss",
    "checkpoint_dir",
    "checkpoint_interval",
    "validation_interval",
    "best_checkpoint_start_epoch",
    "grad_clip",
    "optimizer",
    "scheduler",
    "model",
    "data",
    "runtime",
}
_OPTIMIZER = {"name", "lr", "betas", "weight_decay"}
_SCHEDULER = {"name", "gamma", "step_size", "factor", "patience"}
_LOSSES = {"official", "reconstruction", "waveform_l1"}
_MODEL_NAMES = {
    "mp_senet",
    "tf_gridnet",
    "semambapp",
    "mossformergan",
    "deepfilternet3",
    "unipase",
    "stupase",
}
_MODEL = {
    "name",
    "dense_channel",
    "num_tsconformers",
    "n_fft",
    "hop_size",
    "win_size",
    "compress_factor",
    "beta",
    "sampling_rate",
}
_DATA = {"database", "train_set", "validation_set"}
_RUNTIME = {"device", "allow_busy_gpu"}


def load_training_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TrainingError("training config must be a mapping")
    _unknown(raw, _TOP, "training config")
    config = {
        "seed": _int(raw["seed"], "seed", minimum=0),
        "deterministic": _bool(raw["deterministic"], "deterministic"),
        "global_batch_size": _int(raw["global_batch_size"], "global_batch_size"),
        "micro_batch_size_per_gpu": _int(raw["micro_batch_size_per_gpu"], "micro_batch_size_per_gpu"),
        "max_steps": _int(raw["max_steps"], "max_steps"),
        "segment_size": _int(raw["segment_size"], "segment_size"),
        "loss": _choice(raw["loss"], "loss", _LOSSES),
        "checkpoint_dir": str(raw["checkpoint_dir"]),
        "checkpoint_interval": _int(raw["checkpoint_interval"], "checkpoint_interval", minimum=0),
        "validation_interval": _int(raw["validation_interval"], "validation_interval", minimum=0),
        "best_checkpoint_start_epoch": _int(
            raw.get("best_checkpoint_start_epoch", 40),
            "best_checkpoint_start_epoch",
            minimum=0,
        ),
        "grad_clip": _optional_float(raw.get("grad_clip"), "grad_clip"),
        "optimizer": _optimizer(raw["optimizer"]),
        "scheduler": _scheduler(raw["scheduler"]),
        "model": _model(raw["model"], raw["loss"]),
        "data": _data(raw["data"]),
        "runtime": _runtime(raw["runtime"]),
    }
    if config["validation_interval"] > 0 and not config["data"]["validation_set"]:
        raise TrainingError("validation_interval requires data.validation_set")
    if config["data"]["validation_set"] == config["data"]["train_set"]:
        raise TrainingError("train_set and validation_set must be different named sets")
    return config


def _optimizer(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingError("optimizer must be a mapping")
    _unknown(raw, _OPTIMIZER, "optimizer")
    if raw["name"] not in ("adamw", "adam"):
        raise TrainingError("optimizer.name must be adamw or adam")
    betas = raw["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise TrainingError("optimizer.betas must be a pair")
    return {
        "name": raw["name"],
        "lr": _float(raw["lr"], "optimizer.lr"),
        "betas": [_float(betas[0], "optimizer.betas[0]"), _float(betas[1], "optimizer.betas[1]")],
        "weight_decay": _float(raw["weight_decay"], "optimizer.weight_decay", minimum=0.0),
    }


def _scheduler(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingError("scheduler must be a mapping")
    _unknown(raw, _SCHEDULER, "scheduler")
    name = raw["name"]
    if name == "exponential":
        return {"name": name, "gamma": _float(raw["gamma"], "scheduler.gamma")}
    if name == "steplr":
        return {
            "name": name,
            "gamma": _float(raw["gamma"], "scheduler.gamma"),
            "step_size": _int(raw["step_size"], "scheduler.step_size"),
        }
    if name == "plateau":
        return {
            "name": name,
            "factor": _float(raw.get("factor", 0.7), "scheduler.factor"),
            "patience": _int(raw.get("patience", 1), "scheduler.patience", minimum=0),
        }
    if name == "none":
        return {"name": name}
    raise TrainingError("scheduler.name must be exponential, steplr, plateau, or none")


def _model(raw: Any, loss: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingError("model must be a mapping")
    name = raw.get("name")
    if name not in _MODEL_NAMES:
        raise TrainingError(f"model.name must be one of {', '.join(sorted(_MODEL_NAMES))}")
    if name == "mp_senet":
        if loss == "waveform_l1":
            raise TrainingError("mp_senet does not use loss waveform_l1")
        _unknown(raw, _MODEL, "model")
        return _mp_senet_model(raw)
    if name == "tf_gridnet" and loss not in ("official", "waveform_l1"):
        raise TrainingError("tf_gridnet loss must be official or waveform_l1")
    if name != "tf_gridnet" and loss != "official":
        raise TrainingError(f"{name} training uses loss official")
    allowed = {"name", "variant", "stage", "sampling_rate", "architecture", "stft", "components"}
    _unknown(raw, allowed, "model")
    architecture = raw.get("architecture") or {}
    stft = raw.get("stft") or {}
    components = raw.get("components") or {}
    if not isinstance(architecture, dict) or not isinstance(stft, dict) or not isinstance(components, dict):
        raise TrainingError("model.architecture, model.stft, and model.components must be mappings")
    for key, value in components.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            raise TrainingError("model.components values must be checkpoint paths")
    variant = raw.get("variant")
    stage = raw.get("stage")
    if variant is not None and not isinstance(variant, str):
        raise TrainingError("model.variant must be a string")
    if stage is not None and not isinstance(stage, str):
        raise TrainingError("model.stage must be a string")
    if "sampling_rate" not in raw:
        raise TrainingError("model.sampling_rate is required")
    return {
        "name": name,
        "variant": variant,
        "stage": stage,
        "sampling_rate": _int(raw["sampling_rate"], "model.sampling_rate"),
        "architecture": dict(architecture),
        "stft": dict(stft),
        "components": dict(components),
    }


def _mp_senet_model(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "mp_senet",
        "dense_channel": _int(raw["dense_channel"], "model.dense_channel"),
        "num_tsconformers": _int(raw["num_tsconformers"], "model.num_tsconformers"),
        "n_fft": _int(raw["n_fft"], "model.n_fft"),
        "hop_size": _int(raw["hop_size"], "model.hop_size"),
        "win_size": _int(raw["win_size"], "model.win_size"),
        "compress_factor": _float(raw["compress_factor"], "model.compress_factor"),
        "beta": _float(raw["beta"], "model.beta"),
        "sampling_rate": _int(raw["sampling_rate"], "model.sampling_rate"),
    }


def _data(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingError("data must be a mapping")
    _unknown(raw, _DATA, "data")
    validation = raw.get("validation_set")
    if validation is not None and not isinstance(validation, str):
        raise TrainingError("data.validation_set must be a string or null")
    train_set = raw["train_set"]
    if not isinstance(train_set, str) or not train_set:
        raise TrainingError("data.train_set must be a non-empty string")
    database = raw["database"]
    if not isinstance(database, str) or not database:
        raise TrainingError("data.database must be a path")
    return {"database": database, "train_set": train_set, "validation_set": validation}


def _runtime(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingError("runtime must be a mapping")
    _unknown(raw, _RUNTIME, "runtime")
    device = raw["device"]
    if device not in ("cpu", "cuda"):
        raise TrainingError("runtime.device must be cpu or cuda")
    return {"device": device, "allow_busy_gpu": _bool(raw["allow_busy_gpu"], "runtime.allow_busy_gpu")}


def _unknown(raw: dict, allowed: set[str], label: str) -> None:
    extra = set(raw) - allowed
    if extra:
        raise TrainingError(f"unknown {label} fields: {', '.join(sorted(extra))}")


def _int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingError(f"{name} must be an integer >= {minimum}")
    return value


def _float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{name} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise TrainingError(f"{name} must be >= {minimum}")
    return number


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _float(value, name, minimum=0.0)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingError(f"{name} must be a boolean")
    return value


def _choice(value: Any, name: str, options: set[str]) -> str:
    if value not in options:
        raise TrainingError(f"{name} must be one of {', '.join(sorted(options))}")
    return str(value)
