"""Load and validate an ``audio-db-experiment/v1`` document.

Loading does not construct components and does not create directories.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from src.experiments.errors import ConfigError
from src.experiments.execution.runtime import PreparePolicy

SCHEMA = "audio-db-experiment/v1"
TOP_KEYS = frozenset(
    {
        "schema",
        "seed",
        "deterministic",
        "entry",
        "runtime",
        "components",
        "output",
        "checkpoint",
        "initialize",
        "publication",
    }
)
DEFAULT_SELECTOR = "src.experiments.selectors.StandardSelector"
CONCRETE_KEYS = frozenset({"role", "class", "kwargs", "inject", "parameters_of", "prepare", "state"})
ROLES = frozenset(
    {"module", "optimizer", "scheduler", "trainer", "data", "loss", "metric", "evaluator", "other", "stopping"}
)
PREPARE_KEYS = frozenset(
    {"trainable", "distribute", "find_unused_parameters", "broadcast_buffers", "sync_batchnorm"}
)
PRECISIONS = frozenset({"fp32", "amp_fp16"})
_COMPONENT_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


@dataclass(frozen=True)
class InjectRef:
    """One explicit reference to a component id."""

    ref: str


@dataclass(frozen=True)
class ParameterOf:
    """One module whose parameters feed an optimizer."""

    component: str
    lr: float | None = None
    weight_decay: float | None = None


@dataclass(frozen=True)
class ComponentConfig:
    """One YAML component, either an alias or a concrete class."""

    component_id: str
    ref: str | None
    role: str | None
    class_path: str | None
    kwargs: dict[str, Any]
    inject: dict[str, InjectRef | tuple[InjectRef, ...]]
    parameters_of: tuple[ParameterOf, ...] | None
    parameters_flat: bool
    prepare: PreparePolicy | None
    save_state: bool | None


@dataclass(frozen=True)
class InitializeConfig:
    """Checkpoint that supplies the initial state of a new experiment."""

    checkpoint: Path
    states: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """A validated experiment document. Instances are not created yet."""

    schema: str
    seed: int
    deterministic: bool
    entry: str
    runtime_class: str
    runtime_kwargs: dict[str, Any]
    components: dict[str, ComponentConfig]
    output_directory: Path
    source_path: Path | None = None
    selector_class: str = DEFAULT_SELECTOR
    selector_kwargs: dict[str, Any] = field(default_factory=dict)
    initialize: InitializeConfig | None = None
    publication: dict[str, Any] | None = None


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"duplicate key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Read YAML and validate the schema. Do not construct components."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read experiment config {source}: {exc}") from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("experiment config must be a mapping")
    return replace(_parse_document(raw), source_path=source)


def _parse_document(raw: dict[str, Any]) -> ExperimentConfig:
    _reject_unknown(raw, TOP_KEYS, "experiment")
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ConfigError(f"unsupported schema {schema!r}")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("seed must be an integer >= 0")
    deterministic = raw.get("deterministic")
    if not isinstance(deterministic, bool):
        raise ConfigError("deterministic must be a bool")
    entry = raw.get("entry")
    if not isinstance(entry, str) or _COMPONENT_ID.fullmatch(entry) is None:
        raise ConfigError("entry must be a component id")
    runtime_class, runtime_kwargs = _parse_runtime(raw.get("runtime"))
    output_directory = _parse_output(raw.get("output"))
    selector_class, selector_kwargs = _parse_checkpoint(raw.get("checkpoint"))
    initialize = _parse_initialize(raw.get("initialize"))
    publication = _parse_publication(raw.get("publication"))
    components = _parse_components(raw.get("components"))
    _reject_differentiable(components)
    trainers = [
        component_id
        for component_id, spec in components.items()
        if spec.ref is None and spec.role == "trainer"
    ]
    if not trainers:
        raise ConfigError("at least one trainer is required")
    entry_spec = components.get(entry)
    if entry_spec is None:
        raise ConfigError(f"entry {entry} is not a component")
    if entry_spec.ref is not None or entry_spec.role != "trainer":
        raise ConfigError(f"entry {entry} must have role trainer")
    return ExperimentConfig(
        schema=SCHEMA,
        seed=seed,
        deterministic=deterministic,
        entry=entry,
        runtime_class=runtime_class,
        runtime_kwargs=runtime_kwargs,
        components=components,
        output_directory=output_directory,
        selector_class=selector_class,
        selector_kwargs=selector_kwargs,
        initialize=initialize,
        publication=publication,
    )


def _parse_runtime(raw: object) -> tuple[str, dict[str, Any]]:
    mapping = _expect_mapping(raw, "runtime")
    _reject_unknown(mapping, frozenset({"class", "kwargs"}), "runtime")
    class_path = mapping.get("class")
    if not isinstance(class_path, str) or "." not in class_path:
        raise ConfigError("runtime class must be a fully qualified class name")
    kwargs_raw = mapping.get("kwargs", {})
    kwargs = _expect_mapping(kwargs_raw, "runtime.kwargs")
    _reject_nested_ref(kwargs, "runtime")
    precision = kwargs.get("precision")
    if precision is not None and precision not in PRECISIONS:
        raise ConfigError(f"unsupported precision {precision!r}")
    device = kwargs.get("device")
    if device is not None and not isinstance(device, str):
        raise ConfigError("runtime device must be a string")
    return class_path, dict(kwargs)


def _parse_output(raw: object) -> Path:
    mapping = _expect_mapping(raw, "output")
    _reject_unknown(mapping, frozenset({"directory"}), "output")
    directory = mapping.get("directory")
    if not isinstance(directory, str) or not directory:
        raise ConfigError("output.directory must be a non-empty string")
    return Path(directory)


def _parse_checkpoint(raw: object) -> tuple[str, dict[str, Any]]:
    """Read ``checkpoint.selector``. Without the section, StandardSelector uses its default interval."""

    if raw is None:
        return DEFAULT_SELECTOR, {}
    mapping = _expect_mapping(raw, "checkpoint")
    _reject_unknown(mapping, frozenset({"selector"}), "checkpoint")
    selector = _expect_mapping(mapping.get("selector"), "checkpoint.selector")
    _reject_unknown(selector, frozenset({"class", "kwargs"}), "checkpoint.selector")
    class_path = selector.get("class")
    if not isinstance(class_path, str) or "." not in class_path:
        raise ConfigError("checkpoint.selector class must be a fully qualified class name")
    kwargs = dict(_expect_mapping(selector.get("kwargs", {}), "checkpoint.selector.kwargs"))
    _reject_nested_ref(kwargs, "checkpoint.selector")
    return class_path, kwargs


def _parse_initialize(raw: object) -> InitializeConfig | None:
    """Read ``initialize``: the checkpoint and the state ids a new experiment starts from."""

    if raw is None:
        return None
    mapping = _expect_mapping(raw, "initialize")
    _reject_unknown(mapping, frozenset({"checkpoint", "states"}), "initialize")
    checkpoint = mapping.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ConfigError("initialize.checkpoint must be a non-empty string")
    states = mapping.get("states")
    if not isinstance(states, list) or not states:
        raise ConfigError("initialize.states must be a non-empty list of state ids")
    cleaned: list[str] = []
    for state_id in states:
        if not isinstance(state_id, str) or _COMPONENT_ID.fullmatch(state_id) is None:
            raise ConfigError(f"initialize.states has an invalid state id {state_id!r}")
        if state_id in cleaned:
            raise ConfigError(f"initialize.states repeats {state_id}")
        cleaned.append(state_id)
    return InitializeConfig(checkpoint=Path(checkpoint), states=tuple(cleaned))


def _parse_publication(raw: object) -> dict[str, Any] | None:
    """Read the official dataset and pipeline declaration. Absence is allowed here.

    ``official_faithful`` refuses a missing or different block when the graph
    is built. Other claims do not become official reproductions by carrying one.
    """

    if raw is None:
        return None
    from src.experiments.data.publication import PUBLICATION_KEYS

    mapping = _expect_mapping(raw, "publication")
    allowed = frozenset(PUBLICATION_KEYS)
    _reject_unknown(mapping, allowed, "publication")
    missing = [key for key in PUBLICATION_KEYS if key not in mapping]
    if missing:
        raise ConfigError(f"publication is missing {missing[0]}")
    cleaned: dict[str, Any] = {}
    for key in PUBLICATION_KEYS:
        value = mapping[key]
        if key in {
            "dataset_id",
            "protocol_id",
            "snapshot_hash",
            "profile",
            "claim_label",
            "teacher",
            "algorithm",
            "normalize",
            "validation_crop",
            "validation_scope",
            "split",
        }:
            if not isinstance(value, str) or not value:
                raise ConfigError(f"publication.{key} must be a non-empty string")
            cleaned[key] = value
            continue
        if key in {"sample_rate", "window"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"publication.{key} must be a positive integer")
            cleaned[key] = value
            continue
        if key == "selection_uses_test":
            if not isinstance(value, bool):
                raise ConfigError("publication.selection_uses_test must be a bool")
            cleaned[key] = value
            continue
        if key == "resample":
            cleaned[key] = _parse_resample(value)
            continue
        raise ConfigError(f"publication.{key} is not parsed")
    return cleaned


def _parse_resample(raw: object) -> dict[str, Any]:
    mapping = _expect_mapping(raw, "publication.resample")
    allowed = frozenset({"tool", "quality", "source_rate", "target_rate", "length", "order"})
    _reject_unknown(mapping, allowed, "publication.resample")
    missing = [key for key in ("tool", "quality", "source_rate", "target_rate", "length", "order") if key not in mapping]
    if missing:
        raise ConfigError(f"publication.resample is missing {missing[0]}")
    cleaned: dict[str, Any] = {}
    for key in ("tool", "quality", "length"):
        value = mapping[key]
        if not isinstance(value, str) or not value:
            raise ConfigError(f"publication.resample.{key} must be a non-empty string")
        cleaned[key] = value
    for key in ("source_rate", "target_rate"):
        value = mapping[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"publication.resample.{key} must be a positive integer")
        cleaned[key] = value
    order = mapping["order"]
    if not isinstance(order, list) or not order or any(not isinstance(item, str) or not item for item in order):
        raise ConfigError("publication.resample.order must be a non-empty list of strings")
    cleaned["order"] = list(order)
    return cleaned


def _parse_components(raw: object) -> dict[str, ComponentConfig]:
    mapping = _expect_mapping(raw, "components")
    if not mapping:
        raise ConfigError("components must not be empty")
    components: dict[str, ComponentConfig] = {}
    for component_id, body in mapping.items():
        if not isinstance(component_id, str) or _COMPONENT_ID.fullmatch(component_id) is None:
            raise ConfigError(f"invalid component id {component_id!r}")
        components[component_id] = _parse_component(component_id, body)
    return components


def _parse_component(component_id: str, raw: object) -> ComponentConfig:
    mapping = _expect_mapping(raw, f"component {component_id}")
    if "ref" in mapping:
        _reject_unknown(mapping, frozenset({"ref"}), f"component {component_id}")
        ref = mapping["ref"]
        if not isinstance(ref, str) or _COMPONENT_ID.fullmatch(ref) is None:
            raise ConfigError(f"component {component_id}: ref must be a component id")
        return ComponentConfig(
            component_id=component_id,
            ref=ref,
            role=None,
            class_path=None,
            kwargs={},
            inject={},
            parameters_of=None,
            parameters_flat=False,
            prepare=None,
            save_state=None,
        )
    _reject_unknown(mapping, CONCRETE_KEYS, f"component {component_id}")
    role = mapping.get("role")
    if role not in ROLES:
        raise ConfigError(f"component {component_id}: invalid role {role!r}")
    class_path = mapping.get("class")
    if not isinstance(class_path, str) or "." not in class_path:
        raise ConfigError(f"component {component_id}: class must be a fully qualified class name")
    kwargs = dict(_expect_mapping(mapping.get("kwargs", {}), f"component {component_id} kwargs"))
    _reject_nested_ref(kwargs, component_id)
    _reject_differentiable_value(kwargs, component_id)
    inject = _parse_inject(mapping.get("inject", {}), component_id)
    parameters_of, parameters_flat = _parse_parameters_of(mapping.get("parameters_of"), component_id, role)
    prepare = _parse_prepare(mapping.get("prepare"), component_id, role)
    save_state = _parse_state(mapping.get("state"), component_id)
    return ComponentConfig(
        component_id=component_id,
        ref=None,
        role=str(role),
        class_path=class_path,
        kwargs=kwargs,
        inject=inject,
        parameters_of=parameters_of,
        parameters_flat=parameters_flat,
        prepare=prepare,
        save_state=save_state,
    )


def _parse_inject(
    raw: object,
    component_id: str,
) -> dict[str, InjectRef | tuple[InjectRef, ...]]:
    mapping = _expect_mapping(raw, f"component {component_id} inject")
    parsed: dict[str, InjectRef | tuple[InjectRef, ...]] = {}
    for name, value in mapping.items():
        if not isinstance(name, str) or not name:
            raise ConfigError(f"component {component_id}: inject names must be strings")
        parsed[name] = _parse_inject_value(value, component_id, name)
    return parsed


def _parse_inject_value(value: object, component_id: str, name: str) -> InjectRef | tuple[InjectRef, ...]:
    if isinstance(value, dict):
        return _parse_ref_mapping(value, component_id, name)
    if isinstance(value, list):
        if not value:
            raise ConfigError(f"component {component_id}: inject {name} is empty")
        refs = tuple(_parse_ref_mapping(item, component_id, name) for item in value)
        return refs
    raise ConfigError(f"component {component_id}: inject {name} must be a ref or a list of refs")


def _parse_ref_mapping(value: object, component_id: str, name: str) -> InjectRef:
    mapping = _expect_mapping(value, f"component {component_id} inject {name}")
    _reject_unknown(mapping, frozenset({"ref"}), f"component {component_id} inject {name}")
    ref = mapping.get("ref")
    if not isinstance(ref, str) or _COMPONENT_ID.fullmatch(ref) is None:
        raise ConfigError(f"component {component_id}: inject {name} ref must be a component id")
    return InjectRef(ref)


def _parse_parameters_of(
    raw: object,
    component_id: str,
    role: object,
) -> tuple[tuple[ParameterOf, ...] | None, bool]:
    if raw is None:
        return None, False
    if role != "optimizer":
        raise ConfigError(f"component {component_id}: parameters_of is only valid on an optimizer")
    if isinstance(raw, str):
        _check_component_id(raw, component_id, "parameters_of")
        return (ParameterOf(raw),), True
    if isinstance(raw, dict):
        return (_parse_parameter_group(raw, component_id),), False
    if isinstance(raw, list):
        if not raw:
            raise ConfigError(f"component {component_id}: parameters_of is empty")
        groups = tuple(_parse_parameter_item(item, component_id) for item in raw)
        return groups, False
    raise ConfigError(f"component {component_id}: parameters_of must be a component id, mapping, or list")


def _parse_parameter_item(raw: object, component_id: str) -> ParameterOf:
    if isinstance(raw, str):
        _check_component_id(raw, component_id, "parameters_of")
        return ParameterOf(raw)
    if isinstance(raw, dict):
        return _parse_parameter_group(raw, component_id)
    raise ConfigError(f"component {component_id}: parameters_of entries must be ids or mappings")


def _parse_parameter_group(raw: dict[str, Any], component_id: str) -> ParameterOf:
    _reject_unknown(raw, frozenset({"component", "lr", "weight_decay"}), f"component {component_id} parameters_of")
    component = raw.get("component")
    if not isinstance(component, str):
        raise ConfigError(f"component {component_id}: parameters_of.component must be a component id")
    _check_component_id(component, component_id, "parameters_of")
    return ParameterOf(
        component,
        _optional_number(raw.get("lr"), component_id, "lr"),
        _optional_number(raw.get("weight_decay"), component_id, "weight_decay"),
    )


def _parse_prepare(raw: object, component_id: str, role: object) -> PreparePolicy | None:
    if raw is None:
        if role == "module":
            return PreparePolicy()
        return None
    if role != "module":
        raise ConfigError(f"component {component_id}: prepare is only valid on a module")
    mapping = _expect_mapping(raw, f"component {component_id} prepare")
    _reject_unknown(mapping, PREPARE_KEYS, f"component {component_id} prepare")
    trainable = mapping.get("trainable", True)
    distribute = mapping.get("distribute", "none")
    find_unused = mapping.get("find_unused_parameters", False)
    broadcast = mapping.get("broadcast_buffers", True)
    sync_batchnorm = mapping.get("sync_batchnorm", False)
    if not isinstance(trainable, bool) or not isinstance(find_unused, bool):
        raise ConfigError(f"component {component_id}: prepare flags must be bools")
    if not isinstance(broadcast, bool) or not isinstance(sync_batchnorm, bool):
        raise ConfigError(f"component {component_id}: prepare flags must be bools")
    if distribute not in {"ddp", "none", "fsdp"}:
        raise ConfigError(f"component {component_id}: invalid distribute {distribute!r}")
    return PreparePolicy(
        trainable=trainable,
        distribute=distribute,  # type: ignore[arg-type]
        find_unused_parameters=find_unused,
        broadcast_buffers=broadcast,
        sync_batchnorm=sync_batchnorm,
    )


def _parse_state(raw: object, component_id: str) -> bool | None:
    if raw is None:
        return None
    mapping = _expect_mapping(raw, f"component {component_id} state")
    _reject_unknown(mapping, frozenset({"save"}), f"component {component_id} state")
    save = mapping.get("save")
    if not isinstance(save, bool):
        raise ConfigError(f"component {component_id}: state.save must be a bool")
    return save


def _optional_number(value: object, component_id: str, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"component {component_id}: {name} must be a number")
    return float(value)


def _check_component_id(value: str, component_id: str, field: str) -> None:
    if _COMPONENT_ID.fullmatch(value) is None:
        raise ConfigError(f"component {component_id}: {field} has invalid id {value!r}")


def _expect_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ConfigError(f"unknown key {unknown[0]} in {label}")


def _reject_nested_ref(value: object, component_id: str) -> None:
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            raise ConfigError(f"component {component_id}: nested ref in kwargs is not allowed")
        for item in value.values():
            _reject_nested_ref(item, component_id)
    elif isinstance(value, list):
        for item in value:
            _reject_nested_ref(item, component_id)


def _reject_differentiable(components: dict[str, ComponentConfig]) -> None:
    for spec in components.values():
        _reject_differentiable_value(spec.kwargs, spec.component_id)


def config_sha256(config: ExperimentConfig) -> str:
    """Hash the experiment identity.

    ``output`` is excluded. Resume refuses a different directory by comparing
    the stored path, not this hash. ``stop_after_logical_steps`` is an
    execution cap and is excluded, so a capped run can continue under the
    same algorithm configuration. ``checkpoint`` is excluded: a selector
    decides which states are kept and does not change a training step.
    ``initialize`` is included when present, because another initial state is
    another experiment. ``publication`` is included when present, because an
    official dataset declaration is part of the experiment. A document without
    either field hashes as before.
    """

    payload: dict[str, Any] = {
        "components": {
            component_id: _component_identity(spec)
            for component_id, spec in config.components.items()
        },
        "deterministic": config.deterministic,
        "entry": config.entry,
        "runtime": {"class": config.runtime_class, "kwargs": _jsonable(config.runtime_kwargs)},
        "schema": config.schema,
        "seed": config.seed,
    }
    if config.initialize is not None:
        payload["initialize"] = {
            "checkpoint": str(config.initialize.checkpoint),
            "states": list(config.initialize.states),
        }
    if config.publication is not None:
        payload["publication"] = _jsonable(config.publication)
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _component_identity(spec: ComponentConfig) -> dict[str, Any]:
    if spec.ref is not None:
        return {"ref": spec.ref}
    kwargs = _jsonable(spec.kwargs)
    if isinstance(kwargs, dict):
        kwargs.pop("stop_after_logical_steps", None)
    payload: dict[str, Any] = {
        "class": spec.class_path,
        "kwargs": kwargs,
        "role": spec.role,
    }
    if spec.inject:
        payload["inject"] = {
            name: _inject_identity(value) for name, value in spec.inject.items()
        }
    if spec.parameters_of is not None:
        payload["parameters_flat"] = spec.parameters_flat
        payload["parameters_of"] = [
            {
                "component": binding.component,
                "lr": binding.lr,
                "weight_decay": binding.weight_decay,
            }
            for binding in spec.parameters_of
        ]
    if spec.prepare is not None:
        payload["prepare"] = asdict(spec.prepare)
    if spec.save_state is not None:
        payload["state"] = {"save": spec.save_state}
    return payload


def _inject_identity(value: InjectRef | tuple[InjectRef, ...]) -> dict[str, str] | list[dict[str, str]]:
    if isinstance(value, InjectRef):
        return {"ref": value.ref}
    return [{"ref": item.ref} for item in value]


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ConfigError(f"config identity cannot encode {type(value).__name__}")


def _reject_differentiable_value(value: object, component_id: str) -> None:
    if isinstance(value, dict):
        if value.get("differentiable") is True:
            raise ConfigError(f"component {component_id}: differentiable true is not available")
        for item in value.values():
            _reject_differentiable_value(item, component_id)
    elif isinstance(value, list):
        for item in value:
            _reject_differentiable_value(item, component_id)
