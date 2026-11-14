"""Build a named component graph from explicit references.

Construction order is the role stage, then dependencies, then component id.
Argument names and type annotations are not treated as dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from src.experiments.config import ComponentConfig, ExperimentConfig, InjectRef
from src.experiments.errors import BuildError, PrepareError
from src.experiments.core.factory import ComponentFactory, ComponentSpec
from src.experiments.core.registry import ComponentRegistry, StateRegistry
from src.experiments.core.state import ComponentState, RuntimeState
from src.experiments.execution.runtime import PreparedModel, ProcessRuntime
from src.experiments.calc.rng import set_determinism

_STAGE = {
    "module": 0,
    "loss": 0,
    "data": 0,
    "metric": 0,
    "other": 0,
    "stopping": 0,
    "optimizer": 1,
    "scheduler": 1,
    "trainer": 2,
}
_SAVE_ROLES = frozenset({"module", "optimizer", "scheduler", "trainer", "data", "stopping"})


@dataclass
class BuiltGraph:
    """Objects produced by one successful build."""

    components: ComponentRegistry
    states: StateRegistry
    trainer: object
    runtime: ProcessRuntime


class ExperimentBuilder:
    """Create, prepare, and register the components of one experiment."""

    def __init__(self, factory: ComponentFactory | None = None) -> None:
        self._factory = factory or ComponentFactory()
        self.components: ComponentRegistry | None = None

    def build(self, config: ExperimentConfig, runtime: ProcessRuntime) -> BuiltGraph:
        """Build through state registration. Do not run a learning step."""

        self.components = None
        self._validate(config)
        set_determinism(config.deterministic, cuda=runtime.device.type == "cuda")
        _seed(config.seed)
        registry = ComponentRegistry()
        self.components = registry
        for component_id, spec in config.components.items():
            if spec.ref is not None:
                registry.add_alias(component_id, spec.ref)
        self._build_stage(config, registry, 0, runtime)
        self._prepare_modules(config, registry, runtime)
        self._build_stage(config, registry, 1, runtime)
        self._assert_unique_parameters(config, registry)
        self._build_stage(config, registry, 2, runtime)
        trainer = registry.get(config.entry)
        if not callable(getattr(trainer, "run", None)):
            raise BuildError(f"component {config.entry}: trainer has no run method")
        from src.experiments.data.pair import bind_trainer_data
        from src.experiments.data.publication import require_declared_publication
        from src.experiments.errors import DataError, RuntimeFailure

        try:
            if getattr(trainer, "experiment_claim", None) == "official_faithful":
                require_declared_publication(trainer, config.publication)
            bind_trainer_data(trainer)
        except (DataError, RuntimeFailure) as exc:
            raise BuildError(f"component {config.entry}: {exc}") from exc
        stops = [i for i, spec in config.components.items() if spec.role == "stopping" and spec.ref is None]
        if stops:
            from src.experiments.stopping import TrainingControl
            policy = registry.get(stops[0])
            if getattr(trainer, "experiment_claim", None) == "official_faithful":
                raise BuildError("a stopping policy is incompatible with official_faithful")
            for component_id, spec in config.components.items():
                if spec.role != "data" or spec.ref is not None:
                    continue
                lineage = getattr(registry.get(component_id), "lineage", None)
                if callable(lineage):
                    details = lineage()
                    if isinstance(details, dict) and details.get("claim_label") == "official_reproduction":
                        raise BuildError("a stopping policy is incompatible with official_reproduction")
            if not getattr(trainer, "supports_training_control", False):
                raise BuildError(f"component {config.entry}: trainer does not support stopping policies")
            from src.experiments.core.identity import policy_identity

            if policy_identity(policy) != config.components[stops[0]].class_path:
                raise BuildError("stopping checkpoint_class_id must equal its public class path")
            try:
                sources = getattr(trainer, "training_metric_sources", ("validation",))
                names = getattr(trainer, "training_metric_names", None)
                metric = getattr(policy, "metric", None)
                if names is not None and isinstance(metric, str) and metric not in names:
                    raise ValueError(f"trainer does not emit metric {metric}")
                trainer.training_control = TrainingControl(policy, runtime, supported_sources=sources)
            except ValueError as exc:
                raise BuildError(f"component {stops[0]}: {exc}") from exc
        states = self._register_states(config, registry, runtime, getattr(trainer, "training_control", None))
        return BuiltGraph(components=registry, states=states, trainer=trainer, runtime=runtime)

    def _validate(self, config: ExperimentConfig) -> None:
        stops = [spec for spec in config.components.values() if spec.role == "stopping" and spec.ref is None]
        if len(stops) > 1:
            raise BuildError("only one stopping policy is supported")
        if stops and (stops[0].save_state is False or stops[0].inject):
            raise BuildError("stopping policy must save its state and cannot inject components")
        if stops and config.components[config.entry].kwargs.get("experiment_claim") == "official_faithful":
            raise BuildError("a stopping policy is incompatible with official_faithful")
        self._require_defined(config)
        self._require_no_collision(config)
        cycle = _find_cycle(config)
        if cycle:
            raise BuildError("cycle in components " + ", ".join(cycle))
        self._require_stage_order(config)
        self._require_parameter_targets(config)

    def _require_defined(self, config: ExperimentConfig) -> None:
        for component_id in sorted(config.components):
            for ref in _dependency_ids(config.components[component_id]):
                if ref not in config.components:
                    raise BuildError(f"component {component_id}: undefined reference {ref}")

    def _require_no_collision(self, config: ExperimentConfig) -> None:
        for component_id in sorted(config.components):
            spec = config.components[component_id]
            if spec.ref is not None:
                continue
            overlap = sorted(set(spec.kwargs) & set(spec.inject))
            if overlap:
                raise BuildError(f"component {component_id}: argument collision on {overlap[0]}")
            if spec.parameters_of is not None and "params" in spec.kwargs:
                raise BuildError(f"component {component_id}: argument collision on params")

    def _require_stage_order(self, config: ExperimentConfig) -> None:
        for component_id in sorted(config.components):
            spec = config.components[component_id]
            if spec.ref is not None or spec.role is None:
                continue
            for ref in _dependency_ids(spec):
                target = _concrete_id(config, ref)
                target_role = config.components[target].role
                if target_role is None:
                    raise BuildError(f"component {component_id}: dependency {ref} has no role")
                if _STAGE[target_role] > _STAGE[spec.role]:
                    raise BuildError(f"component {component_id}: dependency {ref} is in a later build stage")

    def _require_parameter_targets(self, config: ExperimentConfig) -> None:
        owners: dict[str, str] = {}
        for component_id in sorted(config.components):
            spec = config.components[component_id]
            if spec.parameters_of is None:
                continue
            seen: set[str] = set()
            for binding in spec.parameters_of:
                target = _concrete_id(config, binding.component)
                role = config.components[target].role
                if role != "module":
                    raise BuildError(f"component {component_id}: parameters_of {binding.component} is not a module")
                policy = config.components[target].prepare
                if policy is not None and not policy.trainable:
                    raise BuildError(f"component {component_id}: parameters_of {binding.component} is not trainable")
                if target in seen:
                    raise BuildError(f"component {component_id}: parameters_of repeats {binding.component}")
                seen.add(target)
                previous = owners.get(target)
                if previous is not None:
                    raise BuildError(f"components {previous} and {component_id} share parameters of {target}")
                owners[target] = component_id

    def _build_stage(
        self,
        config: ExperimentConfig,
        registry: ComponentRegistry,
        stage: int,
        runtime: ProcessRuntime,
    ) -> None:
        ids = [
            component_id
            for component_id, spec in config.components.items()
            if spec.ref is None and spec.role is not None and _STAGE[spec.role] == stage
        ]
        stage_set = set(ids)
        dependencies = {
            component_id: [
                _concrete_id(config, ref)
                for ref in _dependency_ids(config.components[component_id])
                if _concrete_id(config, ref) in stage_set
            ]
            for component_id in ids
        }
        for component_id in _topological_order(ids, dependencies):
            obj = self._create(config, registry, component_id)
            registry.add(component_id, obj)
            if config.components[component_id].role == "data":
                setup = getattr(obj, "setup", None)
                if callable(setup):
                    try:
                        setup(runtime)
                    except Exception as exc:
                        raise BuildError(f"component {component_id}: data setup failed: {exc}") from exc

    def _create(self, config: ExperimentConfig, registry: ComponentRegistry, component_id: str) -> object:
        spec = config.components[component_id]
        if spec.class_path is None:
            raise BuildError(f"component {component_id}: missing class")
        kwargs = dict(spec.kwargs)
        kwargs.update(self._resolve_inject(spec, registry, component_id))
        if spec.parameters_of is not None:
            kwargs["params"] = self._parameter_argument(spec, registry, component_id)
        try:
            obj = self._factory.create(ComponentSpec(spec.class_path, kwargs))
        except BuildError as exc:
            message = str(exc)
            if message.startswith(f"component {component_id}:"):
                raise
            raise BuildError(f"component {component_id}: {exc}") from exc
        try:
            setattr(obj, "_experiment_component_id", component_id)
        except (AttributeError, TypeError) as exc:
            raise BuildError(f"component {component_id}: cannot attach component id: {exc}") from exc
        return obj

    def _resolve_inject(
        self,
        spec: ComponentConfig,
        registry: ComponentRegistry,
        component_id: str,
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for name, value in spec.inject.items():
            try:
                resolved[name] = _resolve_ref(value, registry)
            except BuildError as exc:
                raise BuildError(f"component {component_id}: {exc}") from exc
        return resolved

    def _parameter_argument(
        self,
        spec: ComponentConfig,
        registry: ComponentRegistry,
        component_id: str,
    ) -> list[Any]:
        assert spec.parameters_of is not None
        if spec.parameters_flat:
            binding = spec.parameters_of[0]
            return list(self._prepared_parameters(binding.component, registry, component_id))
        groups: list[dict[str, Any]] = []
        for binding in spec.parameters_of:
            group: dict[str, Any] = {
                "params": list(self._prepared_parameters(binding.component, registry, component_id))
            }
            if binding.lr is not None:
                group["lr"] = binding.lr
            if binding.weight_decay is not None:
                group["weight_decay"] = binding.weight_decay
            groups.append(group)
        return groups

    def _prepared_parameters(
        self,
        component_id: str,
        registry: ComponentRegistry,
        owner_id: str,
    ) -> tuple[torch.nn.Parameter, ...]:
        handle = registry.get(component_id)
        if not isinstance(handle, PreparedModel):
            raise BuildError(f"component {owner_id}: parameters_of {component_id} is not a prepared module")
        if not handle.policy.trainable:
            raise BuildError(f"component {owner_id}: parameters_of {component_id} is not trainable")
        return handle.parameters

    def _prepare_modules(
        self,
        config: ExperimentConfig,
        registry: ComponentRegistry,
        runtime: ProcessRuntime,
    ) -> None:
        module_ids = sorted(
            component_id
            for component_id, spec in config.components.items()
            if spec.ref is None and spec.role == "module"
        )
        for component_id in module_ids:
            spec = config.components[component_id]
            if spec.prepare is None:
                raise BuildError(f"component {component_id}: module has no prepare policy")
            module = registry.get(component_id)
            try:
                prepared = runtime.prepare_model(module, spec.prepare)  # type: ignore[arg-type]
            except PrepareError as exc:
                raise PrepareError(f"component {component_id}: {exc}") from exc
            registry.replace(component_id, replace(prepared, component_id=component_id))

    def _assert_unique_parameters(self, config: ExperimentConfig, registry: ComponentRegistry) -> None:
        owners: dict[int, str] = {}
        for component_id in sorted(config.components):
            spec = config.components[component_id]
            if spec.role != "optimizer":
                continue
            optimizer = registry.get(component_id)
            groups = getattr(optimizer, "param_groups", None)
            if not isinstance(groups, list):
                raise BuildError(f"component {component_id}: optimizer does not expose param_groups")
            local: set[int] = set()
            for group in groups:
                for parameter in group["params"]:
                    identity = id(parameter)
                    if identity in local:
                        raise BuildError(f"component {component_id}: optimizer repeats a parameter")
                    local.add(identity)
                    previous = owners.get(identity)
                    if previous is not None:
                        raise BuildError(f"components {previous} and {component_id} share a parameter")
                    owners[identity] = component_id

    def _register_states(
        self,
        config: ExperimentConfig,
        registry: ComponentRegistry,
        runtime: ProcessRuntime,
        control: object | None = None,
    ) -> StateRegistry:
        states = StateRegistry()
        for component_id in sorted(config.components):
            spec = config.components[component_id]
            if spec.ref is not None or spec.role is None:
                continue
            if not _should_save(spec):
                continue
            target = control if spec.role == "stopping" else registry.get(component_id)
            states.register(ComponentState(component_id, spec.role, target))
        states.register(RuntimeState(runtime))
        return states


def _should_save(spec: ComponentConfig) -> bool:
    if spec.save_state is not None:
        return spec.save_state
    return spec.role in _SAVE_ROLES


def _dependency_ids(spec: ComponentConfig) -> list[str]:
    refs: list[str] = []
    if spec.ref is not None:
        refs.append(spec.ref)
    for value in spec.inject.values():
        refs.extend(_inject_ids(value))
    if spec.parameters_of is not None:
        refs.extend(binding.component for binding in spec.parameters_of)
    return refs


def _inject_ids(value: InjectRef | tuple[InjectRef, ...]) -> list[str]:
    if isinstance(value, InjectRef):
        return [value.ref]
    return [item.ref for item in value]


def _resolve_ref(value: InjectRef | tuple[InjectRef, ...], registry: ComponentRegistry) -> object:
    if isinstance(value, InjectRef):
        return registry.get(value.ref)
    return [_resolve_ref(item, registry) for item in value]


def _concrete_id(config: ExperimentConfig, component_id: str) -> str:
    seen: list[str] = []
    current = component_id
    while True:
        if current in seen:
            raise BuildError("cycle in components " + ", ".join([*seen, current]))
        seen.append(current)
        spec = config.components[current]
        if spec.ref is None:
            return current
        current = spec.ref


def _find_cycle(config: ExperimentConfig) -> list[str] | None:
    nodes = list(config.components)
    edges = {component_id: _dependency_ids(spec) for component_id, spec in config.components.items()}
    color = {node: 0 for node in nodes}

    def visit(node: str, stack: list[str]) -> list[str] | None:
        color[node] = 1
        stack.append(node)
        for dependency in edges[node]:
            if dependency not in color:
                continue
            if color[dependency] == 1:
                start = stack.index(dependency)
                return [*stack[start:], dependency]
            if color[dependency] == 0:
                found = visit(dependency, stack)
                if found:
                    return found
        stack.pop()
        color[node] = 2
        return None

    for node in sorted(nodes):
        if color[node] == 0:
            found = visit(node, [])
            if found:
                return found
    return None


def _topological_order(ids: list[str], dependencies: dict[str, list[str]]) -> list[str]:
    indegree = {component_id: 0 for component_id in ids}
    children: dict[str, list[str]] = {component_id: [] for component_id in ids}
    for component_id in ids:
        for dependency in dependencies[component_id]:
            indegree[component_id] += 1
            children[dependency].append(component_id)
    ready = sorted(component_id for component_id in ids if indegree[component_id] == 0)
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        unlocked: list[str] = []
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                unlocked.append(child)
        ready.extend(unlocked)
        ready.sort()
    if len(ordered) != len(ids):
        raise BuildError("cycle in components " + ", ".join(sorted(ids)))
    return ordered


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
