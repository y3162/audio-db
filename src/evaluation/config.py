"""Load and validate evaluation experiment files."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.models.audio_file import AUDIO_FILE_COLUMNS
from src.evaluation.errors import ConfigError
from src.evaluation.identity import sha256_json

PROTOCOL_RMS_V1 = "rms_v1"
WER_NORM_V1 = "wer-norm-v1"
MOCK_MODEL = "mock"
ASR_CONDITIONS = ("noisy", "enhanced", "clean")
PLANNED_ASR = {
    "parakeet": "Parakeet TDT 0.6B v2",
    "whisper": "Whisper Large-v3",
    "wav2vec2": "Wav2Vec 2.0",
}
DATASET_SOURCES = ("duckdb", "synthetic")
DATASET_ROLES = ("clean", "noisy", "noise")
PAIR_MODES = ("hash", "cycle")


@dataclass
class ExperimentSection:
    name: str
    seed: int = 0
    limit: int | None = None


@dataclass
class DatasetConfig:
    source: str
    path: str | None = None
    set_name: str | None = None
    root: str | None = None
    corpora_root: str | None = None
    role: str = "clean"
    filters: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    noise_filters: dict[str, Any] | None = None
    noise_limit: int | None = None
    pair: str = "hash"
    count: int = 2
    sample_rate: int = 16000
    seconds: float = 1.0
    seed: int | None = None
    max_seconds: float | None = None


@dataclass
class MixingConfig:
    enabled: bool = False
    protocol: str = PROTOCOL_RMS_V1
    snr_db: list[float] = field(default_factory=list)
    prevent_clip: bool = False
    hard_clip: bool = False


@dataclass
class ModelConfig:
    name: str
    variant: str | None = None
    checkpoint: str | None = None
    checkpoints: dict[str, str] | None = None
    model_config: Any = None


@dataclass
class ASRConfig:
    enabled: bool = False
    name: str = "mock"
    default_text: str = ""
    scripts: dict[str, Any] | None = None
    conditions: list[str] = field(default_factory=lambda: ["noisy", "enhanced"])
    sample_rate: int | None = None
    device: str | None = None
    language: str | None = None
    decode: str | None = None
    checkpoint: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    wer: bool = False
    normalization: str = WER_NORM_V1
    pesq: bool = False
    stoi: bool = False
    estoi: bool = False
    si_sdr: bool = False


@dataclass
class OutputConfig:
    root: str = "data/results/se-evaluation"
    run_id: str | None = None
    save_audio: bool = False
    resume: bool = False


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    allow_busy_gpu: bool = False


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection
    dataset: DatasetConfig
    mixing: MixingConfig
    models: list[ModelConfig]
    asr: ASRConfig
    metrics: MetricsConfig
    output: OutputConfig
    runtime: RuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "experiment": asdict(self.experiment),
            "dataset": asdict(self.dataset),
            "mixing": asdict(self.mixing),
            "models": [_model_dict(model) for model in self.models],
            "asr": asdict(self.asr),
            "metrics": asdict(self.metrics),
            "output": asdict(self.output),
            "runtime": asdict(self.runtime),
        }
        return payload

    def canonical_dict(self) -> dict[str, Any]:
        """Settings that change results. Output location and the execution cap are omitted."""
        dataset = asdict(self.dataset)
        return {
            "experiment": {"name": self.experiment.name, "seed": self.experiment.seed},
            "dataset": dataset,
            "mixing": asdict(self.mixing),
            "models": [_model_dict(model) for model in self.models],
            "asr": asdict(self.asr),
            "metrics": asdict(self.metrics),
            "device": self.runtime.device,
        }


def config_hash(config: ExperimentConfig) -> str:
    return sha256_json(config.canonical_dict())


def load_config(path: str | Path) -> ExperimentConfig:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        values = _load_yaml(text, file_path)
    elif suffix == ".json":
        try:
            values = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON config {file_path}: {exc}") from exc
    else:
        raise ConfigError(f"unsupported config format: {file_path.suffix}")
    if not isinstance(values, dict):
        raise ConfigError("configuration must be a mapping")
    return config_from_dict(values)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    _reject_unknown(
        raw,
        {"experiment", "dataset", "mixing", "models", "asr", "metrics", "output", "runtime"},
        "config",
    )
    _require(raw, ("experiment", "dataset", "models"), "config")
    experiment = _parse_experiment(_mapping(raw["experiment"], "experiment"))
    dataset = _parse_dataset(_mapping(raw["dataset"], "dataset"))
    mixing = _parse_mixing(_mapping(raw.get("mixing", {}), "mixing"))
    models = _parse_models(raw["models"])
    asr = _parse_asr(_mapping(raw.get("asr", {}), "asr"))
    metrics = _parse_metrics(_mapping(raw.get("metrics", {}), "metrics"))
    output = _parse_output(_mapping(raw.get("output", {}), "output"))
    runtime = _parse_runtime(_mapping(raw.get("runtime", {}), "runtime"))
    if metrics.wer and not asr.enabled:
        raise ConfigError("metrics.wer requires asr.enabled")
    if dataset.set_name and mixing.enabled:
        raise ConfigError(
            "mixing.enabled must be false when dataset.set is used. "
            "Mixture conditions are stored on the evaluation cases."
        )
    if mixing.enabled and dataset.source == "duckdb" and not dataset.set_name:
        if dataset.role != "clean":
            raise ConfigError("duckdb mixing requires dataset.role=clean")
        if not dataset.noise_filters:
            raise ConfigError("duckdb mixing requires dataset.noise_filters")
    _reject_duplicate_models(models)
    return ExperimentConfig(
        experiment=experiment,
        dataset=dataset,
        mixing=mixing,
        models=models,
        asr=asr,
        metrics=metrics,
        output=output,
        runtime=runtime,
    )


def apply_overrides(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    output: str | None = None,
    run_id: str | None = None,
    resume: bool = False,
    models: str | None = None,
) -> ExperimentConfig:
    if limit is not None:
        config.experiment.limit = _int(limit, "--limit", minimum=0)
    if device is not None:
        if not str(device).strip():
            raise ConfigError("--device must be a non-empty string")
        config.runtime.device = str(device)
    if seed is not None:
        config.experiment.seed = _int(seed, "--seed", minimum=0)
    if output is not None:
        config.output.root = str(output)
    if run_id is not None:
        config.output.run_id = _run_id(run_id)
    if resume:
        config.output.resume = True
    if models:
        names = [part.strip() for part in models.split(",") if part.strip()]
        known = {model.name for model in config.models}
        missing = [name for name in names if name not in known]
        if missing:
            raise ConfigError(
                "requested models are not in the config: " + ", ".join(missing)
            )
        kept = [model for model in config.models if model.name in names]
        if not kept:
            raise ConfigError("model filter removed every model")
        config.models = kept
        _reject_duplicate_models(config.models)
    return config


def validate_registered_models(models: list[ModelConfig]) -> None:
    """Resolve aliases against the SE registry. Mock models stay local."""
    if not any(model.name != MOCK_MODEL for model in models):
        return
    from src.se import model_metadata

    for model in models:
        if model.name == MOCK_MODEL:
            continue
        try:
            spec = model_metadata(model.name)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        model.name = spec.name
        if len(spec.variants) > 1 and model.variant is None:
            joined = ", ".join(spec.variants)
            raise ConfigError(f"{model.name} requires variant, one of: {joined}")
        if model.variant is not None and spec.variants and model.variant not in spec.variants:
            joined = ", ".join(spec.variants)
            raise ConfigError(
                f"unknown variant {model.variant!r} for {model.name}; expected one of: {joined}"
            )


def _model_dict(model: ModelConfig) -> dict[str, Any]:
    return {
        "name": model.name,
        "variant": model.variant,
        "checkpoint": model.checkpoint,
        "checkpoints": model.checkpoints,
        "config": model.model_config,
    }


def _parse_experiment(raw: dict[str, Any]) -> ExperimentSection:
    _reject_unknown(raw, {"name", "seed", "limit"}, "experiment")
    _require(raw, ("name",), "experiment")
    name = _str(raw["name"], "experiment.name")
    seed = _int(raw.get("seed", 0), "experiment.seed", minimum=0)
    limit = raw.get("limit")
    parsed_limit = None if limit is None else _int(limit, "experiment.limit", minimum=0)
    return ExperimentSection(name=name, seed=seed, limit=parsed_limit)


def _parse_dataset(raw: dict[str, Any]) -> DatasetConfig:
    _reject_unknown(
        raw,
        {
            "source",
            "path",
            "set",
            "manifest",
            "root",
            "corpora_root",
            "role",
            "filters",
            "limit",
            "noise_filters",
            "noise_limit",
            "pair",
            "count",
            "sample_rate",
            "seconds",
            "seed",
            "max_seconds",
        },
        "dataset",
    )
    _require(raw, ("source",), "dataset")
    source = _str(raw["source"], "dataset.source")
    if source == "manifest" or "manifest" in raw:
        raise ConfigError(
            "dataset.source=manifest has been removed. "
            "Import the manifest with make db-init and set dataset.set."
        )
    if source not in DATASET_SOURCES:
        raise ConfigError(
            f"dataset.source must be one of: {', '.join(DATASET_SOURCES)}"
        )
    path = raw.get("path")
    if source == "duckdb" and not path:
        raise ConfigError("dataset.path is required when source is duckdb")
    if path is not None:
        path = _str(path, "dataset.path")
    role = _str(raw.get("role", "clean"), "dataset.role")
    if role not in DATASET_ROLES:
        raise ConfigError(f"dataset.role must be one of: {', '.join(DATASET_ROLES)}")
    pair = _str(raw.get("pair", "hash"), "dataset.pair")
    if pair not in PAIR_MODES:
        raise ConfigError(f"dataset.pair must be one of: {', '.join(PAIR_MODES)}")
    filters = _filters(raw.get("filters", {}), "dataset.filters")
    noise_filters = raw.get("noise_filters")
    parsed_noise = None if noise_filters is None else _filters(noise_filters, "dataset.noise_filters")
    set_name = None if raw.get("set") is None else _str(raw["set"], "dataset.set")
    if set_name and source != "duckdb":
        raise ConfigError("dataset.set requires dataset.source=duckdb")
    if set_name and (filters or parsed_noise):
        raise ConfigError("dataset.set cannot be combined with filters or noise_filters")
    return DatasetConfig(
        source=source,
        path=path,
        set_name=set_name,
        root=None if raw.get("root") is None else _str(raw["root"], "dataset.root"),
        corpora_root=(
            None if raw.get("corpora_root") is None else _str(raw["corpora_root"], "dataset.corpora_root")
        ),
        role=role,
        filters=filters,
        limit=None if raw.get("limit") is None else _int(raw["limit"], "dataset.limit", minimum=0),
        noise_filters=parsed_noise,
        noise_limit=(
            None if raw.get("noise_limit") is None else _int(raw["noise_limit"], "dataset.noise_limit", minimum=0)
        ),
        pair=pair,
        count=_int(raw.get("count", 2), "dataset.count", minimum=1),
        sample_rate=_int(raw.get("sample_rate", 16000), "dataset.sample_rate", minimum=1),
        seconds=_float(raw.get("seconds", 1.0), "dataset.seconds", minimum=0.001),
        seed=None if raw.get("seed") is None else _int(raw["seed"], "dataset.seed", minimum=0),
        max_seconds=(
            None if raw.get("max_seconds") is None else _float(raw["max_seconds"], "dataset.max_seconds", minimum=0.001)
        ),
    )


def _parse_mixing(raw: dict[str, Any]) -> MixingConfig:
    _reject_unknown(raw, {"enabled", "protocol", "snr_db", "prevent_clip", "hard_clip"}, "mixing")
    enabled = _bool(raw.get("enabled", False), "mixing.enabled")
    protocol = _str(raw.get("protocol", PROTOCOL_RMS_V1), "mixing.protocol")
    if protocol != PROTOCOL_RMS_V1:
        raise ConfigError(
            f"unknown mixing protocol {protocol!r}. Implemented: {PROTOCOL_RMS_V1}. "
            "This protocol is not the VoiceBank-DEMAND or URGENT mixture procedure."
        )
    snrs = [_float(value, "mixing.snr_db") for value in _list(raw.get("snr_db", []), "mixing.snr_db")]
    if enabled and not snrs:
        raise ConfigError("mixing.snr_db must list at least one SNR when mixing is enabled")
    return MixingConfig(
        enabled=enabled,
        protocol=protocol,
        snr_db=snrs,
        prevent_clip=_bool(raw.get("prevent_clip", False), "mixing.prevent_clip"),
        hard_clip=_bool(raw.get("hard_clip", False), "mixing.hard_clip"),
    )


def _parse_models(raw: Any) -> list[ModelConfig]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("models must be a non-empty list")
    models = []
    for index, item in enumerate(raw):
        where = f"models[{index}]"
        mapping = _mapping(item, where)
        _reject_unknown(mapping, {"name", "variant", "checkpoint", "checkpoints", "config"}, where)
        _require(mapping, ("name",), where)
        name = _str(mapping["name"], f"{where}.name")
        checkpoint = mapping.get("checkpoint")
        checkpoints = mapping.get("checkpoints")
        if checkpoint is not None and checkpoints is not None:
            raise ConfigError(f"{where} accepts checkpoint or checkpoints, not both")
        if name == MOCK_MODEL:
            if checkpoint is not None or checkpoints is not None:
                raise ConfigError(f"{where} mock does not take a checkpoint")
        elif checkpoint is None and checkpoints is None:
            raise ConfigError(
                f"{where} {name} requires checkpoint or checkpoints. "
                "The pipeline does not choose weights on its own."
            )
        parsed_many = None
        if checkpoints is not None:
            if not isinstance(checkpoints, dict) or not checkpoints:
                raise ConfigError(f"{where}.checkpoints must be a non-empty mapping")
            parsed_many = {}
            for key, value in checkpoints.items():
                parsed_many[_str(key, f"{where}.checkpoints key")] = _str(
                    value, f"{where}.checkpoints.{key}"
                )
        model_config = mapping.get("config")
        if model_config is not None:
            _jsonable(model_config, f"{where}.config")
        variant = None if mapping.get("variant") is None else _str(mapping["variant"], f"{where}.variant")
        models.append(
            ModelConfig(
                name=name,
                variant=variant,
                checkpoint=None if checkpoint is None else _str(checkpoint, f"{where}.checkpoint"),
                checkpoints=parsed_many,
                model_config=model_config,
            )
        )
    return models


def _parse_asr(raw: dict[str, Any]) -> ASRConfig:
    _reject_unknown(
        raw,
        {
            "enabled",
            "name",
            "default_text",
            "scripts",
            "conditions",
            "sample_rate",
            "device",
            "language",
            "decode",
            "checkpoint",
            "params",
        },
        "asr",
    )
    conditions = list(raw.get("conditions", ["noisy", "enhanced"]))
    if not isinstance(conditions, list) or not conditions:
        raise ConfigError("asr.conditions must be a non-empty list")
    parsed_conditions = []
    for item in conditions:
        name = _str(item, "asr.conditions")
        if name not in ASR_CONDITIONS:
            raise ConfigError(f"unknown ASR condition {name!r}")
        parsed_conditions.append(name)
    scripts = raw.get("scripts")
    if scripts is not None:
        if not isinstance(scripts, dict):
            raise ConfigError("asr.scripts must be a mapping")
        _jsonable(scripts, "asr.scripts")
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError("asr.params must be a mapping")
    _jsonable(params, "asr.params")
    name = _str(raw.get("name", "mock"), "asr.name")
    if name not in {MOCK_MODEL, *PLANNED_ASR}:
        raise ConfigError(
            f"unknown ASR model {name!r}. Known names: mock, {', '.join(PLANNED_ASR)}"
        )
    return ASRConfig(
        enabled=_bool(raw.get("enabled", False), "asr.enabled"),
        name=name,
        default_text="" if raw.get("default_text") is None else _str(raw["default_text"], "asr.default_text"),
        scripts=scripts,
        conditions=parsed_conditions,
        sample_rate=(
            None if raw.get("sample_rate") is None else _int(raw["sample_rate"], "asr.sample_rate", minimum=1)
        ),
        device=None if raw.get("device") is None else _str(raw["device"], "asr.device"),
        language=None if raw.get("language") is None else _str(raw["language"], "asr.language"),
        decode=None if raw.get("decode") is None else _str(raw["decode"], "asr.decode"),
        checkpoint=None if raw.get("checkpoint") is None else _str(raw["checkpoint"], "asr.checkpoint"),
        params=dict(params),
    )


def _parse_metrics(raw: dict[str, Any]) -> MetricsConfig:
    _reject_unknown(raw, {"wer", "normalization", "pesq", "stoi", "estoi", "si_sdr"}, "metrics")
    normalization = _str(raw.get("normalization", WER_NORM_V1), "metrics.normalization")
    if normalization != WER_NORM_V1:
        raise ConfigError(f"unknown WER normalization {normalization!r}. Implemented: {WER_NORM_V1}")
    return MetricsConfig(
        wer=_bool(raw.get("wer", False), "metrics.wer"),
        normalization=normalization,
        pesq=_bool(raw.get("pesq", False), "metrics.pesq"),
        stoi=_bool(raw.get("stoi", False), "metrics.stoi"),
        estoi=_bool(raw.get("estoi", False), "metrics.estoi"),
        si_sdr=_bool(raw.get("si_sdr", False), "metrics.si_sdr"),
    )


def _parse_output(raw: dict[str, Any]) -> OutputConfig:
    _reject_unknown(raw, {"root", "run_id", "save_audio", "resume"}, "output")
    run_id = raw.get("run_id")
    return OutputConfig(
        root=_str(raw.get("root", "data/results/se-evaluation"), "output.root"),
        run_id=None if run_id is None else _run_id(run_id),
        save_audio=_bool(raw.get("save_audio", False), "output.save_audio"),
        resume=_bool(raw.get("resume", False), "output.resume"),
    )


def _parse_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    _reject_unknown(raw, {"device", "allow_busy_gpu"}, "runtime")
    return RuntimeConfig(
        device=_str(raw.get("device", "cpu"), "runtime.device"),
        allow_busy_gpu=_bool(raw.get("allow_busy_gpu", False), "runtime.allow_busy_gpu"),
    )


def _filters(raw: Any, where: str) -> dict[str, Any]:
    mapping = _mapping(raw, where)
    unknown = sorted(set(mapping) - set(AUDIO_FILE_COLUMNS))
    if unknown:
        raise ConfigError(f"unknown columns in {where}: {', '.join(unknown)}")
    cleaned: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, (bool, list, dict)):
            raise ConfigError(f"{where}.{key} must be a scalar")
        if not isinstance(value, (str, int, float)):
            raise ConfigError(f"{where}.{key} must be a string or number")
        cleaned[key] = value
    _jsonable(cleaned, where)
    return cleaned


def _reject_duplicate_models(models: list[ModelConfig]) -> None:
    seen = set()
    for model in models:
        if model.checkpoints:
            checkpoint_key = tuple(sorted(model.checkpoints.items()))
        else:
            checkpoint_key = model.checkpoint
        key = (model.name, model.variant, checkpoint_key)
        if key in seen:
            raise ConfigError(
                f"duplicate model entry for {model.name!r} variant {model.variant!r}. "
                "Use a different variant or checkpoint."
            )
        seen.add(key)


def _load_yaml(text: str, path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "YAML configuration requires PyYAML. Install it in .venvs/lab: "
            "uv pip install --python .venvs/lab/bin/python pyyaml"
        ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML config {path}: {exc}") from exc


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown fields in {where}: {', '.join(unknown)}")


def _require(value: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ConfigError(f"missing fields in {where}: {', '.join(missing)}")


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a boolean")
    return value


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}")
    return value


def _float(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{where} must be finite")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{where} must be >= {minimum}")
    return number


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list")
    return value


def _run_id(value: Any) -> str:
    text = _str(value, "output.run_id")
    if len(text) > 128 or not all(char.isalnum() or char in "._-" for char in text):
        raise ConfigError(
            "output.run_id may contain only letters, digits, '.', '_' and '-', up to 128 characters"
        )
    return text


def _jsonable(value: Any, where: str) -> None:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ConfigError(f"{where} must be JSON-serializable") from exc
