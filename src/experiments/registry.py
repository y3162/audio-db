"""Live component lookup and state-owner registration.

Component identity and checkpoint identity are separate. An alias is one
component name for an existing object, and it is not a second saved state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.experiments.errors import BuildError, StateError


class ComponentRegistry:
    """Map component ids to the objects a built graph exposes."""

    def __init__(self) -> None:
        self._objects: dict[str, object] = {}
        self._aliases: dict[str, str] = {}

    def add(self, component_id: str, obj: object) -> None:
        if component_id in self._objects or component_id in self._aliases:
            raise BuildError(f"duplicate component id {component_id}")
        self._objects[component_id] = obj

    def add_alias(self, component_id: str, target_id: str) -> None:
        if component_id in self._objects or component_id in self._aliases:
            raise BuildError(f"duplicate component id {component_id}")
        self._aliases[component_id] = target_id

    def canonical(self, component_id: str) -> str:
        seen: list[str] = []
        current = component_id
        while current in self._aliases:
            if current in seen:
                joined = ", ".join([*seen, current])
                raise BuildError(f"cycle in components {joined}")
            seen.append(current)
            current = self._aliases[current]
        return current

    def get(self, component_id: str) -> object:
        if component_id not in self._objects and component_id not in self._aliases:
            raise BuildError(f"unknown component {component_id}")
        canonical = self.canonical(component_id)
        try:
            return self._objects[canonical]
        except KeyError as exc:
            raise BuildError(f"component {component_id} is not built") from exc

    def replace(self, component_id: str, obj: object) -> None:
        canonical = self.canonical(component_id)
        if canonical not in self._objects:
            raise BuildError(f"component {component_id} is not built")
        self._objects[canonical] = obj

    def __contains__(self, component_id: object) -> bool:
        return isinstance(component_id, str) and (
            component_id in self._objects or component_id in self._aliases
        )


class StateRegistry:
    """Register state owners and reject conflicting ids.

    Registering the same Python object again keeps a single saved entry. A
    different object that claims an existing id is an error. ``seal`` freezes
    the set at the start of ``run``.
    """

    def __init__(self) -> None:
        self._owners: dict[str, Any] = {}
        self._owner_keys: dict[int, str] = {}
        self._aliases: dict[str, str] = {}
        self._sealed = False

    def register(self, owner: Any) -> None:
        state_id = owner.state_id
        owner_key = owner.owner_key
        if not isinstance(state_id, str) or not state_id:
            raise StateError("state id must be a non-empty string")
        if not isinstance(owner_key, int):
            raise StateError(f"state {state_id} has no owner key")
        if self._sealed:
            raise StateError(f"cannot register {state_id} after run has started")
        existing = self._owner_keys.get(owner_key)
        if existing is not None:
            if state_id != existing:
                self._aliases[state_id] = existing
            return
        if state_id in self._owners:
            raise StateError(f"state id {state_id} is already registered")
        self._owners[state_id] = owner
        self._owner_keys[owner_key] = state_id

    def seal(self) -> None:
        self._sealed = True

    def ids(self) -> tuple[str, ...]:
        """State ids in lexicographic order. Aliases are not included."""

        return tuple(sorted(self._owners))

    def aliases(self) -> Mapping[str, str]:
        return dict(self._aliases)

    def owner(self, state_id: str) -> Any:
        try:
            return self._owners[state_id]
        except KeyError as exc:
            raise StateError(f"unknown state id {state_id}") from exc
