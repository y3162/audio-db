"""Flat training configuration.

Unknown keys are rejected so a misspelled setting cannot be ignored.
"""

from __future__ import annotations

import math
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
    "logical_steps_per_epoch",
    "chunk_iterator",
    "batch_schedule",
    "batch_size_limit",
    "early_stopping_patience",
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
    "logging",
}
_OPTIMIZER = {"name", "lr", "betas", "weight_decay", "amsgrad"}
_SCHEDULER = {
    "name",
    "gamma",
    "step_size",
    "factor",
    "patience",
    "steps_per_epoch",
    "epochs",
    "lr_min",
    "lr_warmup",
    "warmup_epochs",
    "weight_decay_end",
    "warmup_steps",
    "mode",
    "threshold",
    "threshold_mode",
    "cooldown",
    "min_lr",
    "eps",
}
_LOSSES = {"official", "reconstruction", "waveform_l1"}
_MODEL_NAMES = {
    "mp_senet",
    "tf_gridnet",
    "tf_locoformer",
    "semambapp",
    "mossformergan",
    "deepfilternet3",
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
_LEGACY_DATA = {
    "database",
    "train_set",
    "validation_set",
    "load_workers",
    "prefetch",
    "rms_normalize",
    "virtual_world_size",
}
_REPRODUCTION_DATA = {
    "pipeline",
    "dataset",
    "allow_test_selection",
    "virtual_world_size",
    "load_workers",
    "prefetch",
    "verify_on_read",
}
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
    if "logging" in raw:
        logging = raw["logging"]
        if not isinstance(logging, dict):
            raise TrainingError("logging must be a mapping")
        _unknown(logging, {"tensorboard"}, "logging")
        tb = logging.get("tensorboard", {})
        if not isinstance(tb, dict):
            raise TrainingError("logging.tensorboard must be a mapping")
        _unknown(tb, {"enabled", "scalar_interval", "audio", "spectrogram"}, "logging.tensorboard")
        audio = tb.get("audio", {})
        if not isinstance(audio, dict):
            raise TrainingError("logging.tensorboard.audio must be a mapping")
        _unknown(audio, {"enabled", "max_cases", "max_seconds"}, "logging.tensorboard.audio")
        seconds = audio.get("max_seconds", 5)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 3600:
            raise TrainingError("logging.tensorboard.audio.max_seconds must be positive and at most 3600")
        config["logging"] = {"tensorboard": {
            "enabled": _bool(tb.get("enabled", False), "logging.tensorboard.enabled"),
            "scalar_interval": _int(tb.get("scalar_interval", 10), "logging.tensorboard.scalar_interval"),
            "audio": {"enabled": _bool(audio.get("enabled", False), "logging.tensorboard.audio.enabled"),
                      "max_cases": _int(audio.get("max_cases", 3), "logging.tensorboard.audio.max_cases", minimum=0),
                      "max_seconds": float(seconds)},
            "spectrogram": _bool(tb.get("spectrogram", False), "logging.tensorboard.spectrogram"),
        }}
    reproduction = config["data"].get("pipeline") == "reproduction"
    if reproduction and ("logical_steps_per_epoch" in raw or "batch_schedule" in raw):
        raise TrainingError(
            "reproduction data cannot be combined with logical_steps_per_epoch or batch_schedule"
        )
    if "logical_steps_per_epoch" in raw:
        config["logical_steps_per_epoch"] = _int(raw["logical_steps_per_epoch"], "logical_steps_per_epoch")
        if not config["data"]["validation_set"]:
            raise TrainingError("logical_steps_per_epoch requires validation_set")
    if "chunk_iterator" in raw:
        config["chunk_iterator"] = _chunk_iterator(raw["chunk_iterator"])
        if not reproduction and not config["data"]["validation_set"]:
            raise TrainingError("chunk_iterator requires validation_set")
        if "logical_steps_per_epoch" in config:
            raise TrainingError("chunk_iterator cannot be combined with logical_steps_per_epoch")
    if config["data"].get("virtual_world_size") is not None and "chunk_iterator" not in config:
        raise TrainingError("data.virtual_world_size requires chunk_iterator")
    if "batch_schedule" in raw:
        config["batch_schedule"] = _batch_schedule(raw["batch_schedule"])
        config["batch_size_limit"] = _int(
            raw.get("batch_size_limit", config["global_batch_size"]),
            "batch_size_limit",
        )
        if "chunk_iterator" in config:
            raise TrainingError("batch_schedule cannot be combined with chunk_iterator")
    if "early_stopping_patience" in raw:
        config["early_stopping_patience"] = _int(
            raw["early_stopping_patience"], "early_stopping_patience", minimum=1,
        )
    if not reproduction:
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
        "amsgrad": bool(raw.get("amsgrad", False)),
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
    if name == "warmup_reduce_on_plateau":
        mode = raw.get("mode", "min")
        threshold_mode = raw.get("threshold_mode", "rel")
        if mode not in ("min", "max"):
            raise TrainingError("scheduler.mode must be min or max")
        if threshold_mode not in ("rel", "abs"):
            raise TrainingError("scheduler.threshold_mode must be rel or abs")
        factor = _float(raw["factor"], "scheduler.factor")
        if factor >= 1.0:
            raise TrainingError("scheduler.factor must be < 1")
        return {
            "name": name,
            "warmup_steps": _int(raw["warmup_steps"], "scheduler.warmup_steps"),
            "mode": mode,
            "factor": factor,
            "patience": _int(raw["patience"], "scheduler.patience", minimum=0),
            "threshold": _float(raw.get("threshold", 1e-4), "scheduler.threshold", minimum=0.0),
            "threshold_mode": threshold_mode,
            "cooldown": _int(raw.get("cooldown", 0), "scheduler.cooldown", minimum=0),
            "min_lr": _float(raw.get("min_lr", 0.0), "scheduler.min_lr", minimum=0.0),
            "eps": _float(raw.get("eps", 1e-8), "scheduler.eps", minimum=0.0),
        }
    if name == "deepfilter_cosine":
        return {
            "name": name,
            "steps_per_epoch": _int(raw["steps_per_epoch"], "scheduler.steps_per_epoch"),
            "epochs": _int(raw["epochs"], "scheduler.epochs"),
            "lr_min": _float(raw["lr_min"], "scheduler.lr_min", minimum=0.0),
            "lr_warmup": _float(raw["lr_warmup"], "scheduler.lr_warmup", minimum=0.0),
            "warmup_epochs": _int(raw["warmup_epochs"], "scheduler.warmup_epochs", minimum=0),
            "weight_decay_end": _float(raw["weight_decay_end"], "scheduler.weight_decay_end", minimum=0.0),
        }
    raise TrainingError(
        "scheduler.name must be exponential, steplr, plateau, warmup_reduce_on_plateau, "
        "deepfilter_cosine, or none"
    )


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
    allowed = {"name", "variant", "stage", "sampling_rate", "architecture", "stft", "components", "init"}
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
    init = raw.get("init")
    if init is not None and init != "xavier_uniform":
        raise TrainingError("model.init must be xavier_uniform or null")
    parsed = {
        "name": name,
        "variant": variant,
        "stage": stage,
        "sampling_rate": _int(raw["sampling_rate"], "model.sampling_rate"),
        "architecture": dict(architecture),
        "stft": dict(stft),
        "components": dict(components),
    }
    if init is not None:
        parsed["init"] = init
    return parsed


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


def _chunk_iterator(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingError("chunk_iterator must be a mapping")
    allowed = {
        "num_samples_per_epoch",
        "chunk_length",
        "chunk_shift_ratio",
        "num_cache_chunks",
        "shuffle",
        "max_epochs",
        "early_stopping_patience",
        "keep_nbest",
    }
    _unknown(raw, allowed, "chunk_iterator")
    return {
        "num_samples_per_epoch": _int(raw["num_samples_per_epoch"], "chunk_iterator.num_samples_per_epoch"),
        "chunk_length": _int(raw["chunk_length"], "chunk_iterator.chunk_length", minimum=2),
        "chunk_shift_ratio": _float(raw.get("chunk_shift_ratio", 0.5), "chunk_iterator.chunk_shift_ratio", minimum=0.0),
        "num_cache_chunks": _int(raw.get("num_cache_chunks", 1024), "chunk_iterator.num_cache_chunks"),
        "shuffle": _bool(raw.get("shuffle", True), "chunk_iterator.shuffle"),
        "max_epochs": _int(raw["max_epochs"], "chunk_iterator.max_epochs"),
        "early_stopping_patience": _int(
            raw.get("early_stopping_patience", 5),
            "chunk_iterator.early_stopping_patience",
            minimum=0,
        ),
        "keep_nbest": _int(raw.get("keep_nbest", 5), "chunk_iterator.keep_nbest"),
    }


def _batch_schedule(raw: Any) -> list[list[int]]:
    if not isinstance(raw, list) or not raw:
        raise TrainingError("batch_schedule must be a non-empty list of [epoch, batch_size]")
    schedule = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise TrainingError("batch_schedule entries must be [epoch, batch_size]")
        schedule.append([
            _int(item[0], "batch_schedule epoch", minimum=0),
            _int(item[1], "batch_schedule batch_size"),
        ])
    if schedule[0][0] != 0:
        raise TrainingError("batch_schedule must start at epoch 0")
    return schedule


def _data(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingError("data must be a mapping")
    pipeline = raw.get("pipeline", "legacy")
    if pipeline not in ("legacy", "reproduction"):
        raise TrainingError("data.pipeline must be legacy or reproduction")
    if pipeline == "reproduction":
        return _reproduction_data(raw)
    allowed = set(_LEGACY_DATA)
    if "pipeline" in raw:
        allowed.add("pipeline")
    _unknown(raw, allowed, "data")
    parsed = _legacy_data(raw)
    if "pipeline" in raw:
        parsed["pipeline"] = "legacy"
    return parsed


def _legacy_data(raw: dict[str, Any]) -> dict[str, Any]:
    validation = raw.get("validation_set")
    if validation is not None and not isinstance(validation, str):
        raise TrainingError("data.validation_set must be a string or null")
    train_set = raw["train_set"]
    if not isinstance(train_set, str) or not train_set:
        raise TrainingError("data.train_set must be a non-empty string")
    database = raw["database"]
    if not isinstance(database, str) or not database:
        raise TrainingError("data.database must be a path")
    workers = _int(raw.get("load_workers", 4), "data.load_workers", minimum=1)
    parsed = {
        "database": database,
        "train_set": train_set,
        "validation_set": validation,
        "load_workers": workers,
        "prefetch": _bool(raw.get("prefetch", True), "data.prefetch"),
        "rms_normalize": _bool(raw.get("rms_normalize", True), "data.rms_normalize"),
    }
    if raw.get("virtual_world_size") is not None:
        parsed["virtual_world_size"] = _int(raw["virtual_world_size"], "data.virtual_world_size", minimum=1)
    return parsed


def _reproduction_data(raw: dict[str, Any]) -> dict[str, Any]:
    _unknown(raw, _REPRODUCTION_DATA, "data")
    dataset = raw.get("dataset")
    if not isinstance(dataset, str):
        raise TrainingError("data.dataset must be name@version")
    name, separator, version = dataset.rpartition("@")
    if separator != "@" or not name or not version.isdigit() or int(version) < 1:
        raise TrainingError("data.dataset must be name@version")
    virtual = raw.get("virtual_world_size")
    return {
        "pipeline": "reproduction",
        "dataset": dataset,
        "allow_test_selection": _bool(
            raw.get("allow_test_selection", False),
            "data.allow_test_selection",
        ),
        "virtual_world_size": None if virtual is None else _int(virtual, "data.virtual_world_size", minimum=1),
        "load_workers": _int(raw.get("load_workers", 4), "data.load_workers", minimum=1),
        "prefetch": _bool(raw.get("prefetch", True), "data.prefetch"),
        "verify_on_read": _bool(raw.get("verify_on_read", False), "data.verify_on_read"),
        "database": None,
        "train_set": None,
        "validation_set": None,
        "rms_normalize": False,
    }


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
