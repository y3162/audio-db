"""Registry of implemented model families."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable
    loader: Callable
    sample_rate: int
    aliases: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    task: str = 'speech_enhancement'


class ModelRegistry:
    def __init__(self):
        self._specs = {}
        self._aliases = {}

    def register(self, spec):
        names = (spec.name, *spec.aliases)
        if len(set(names)) != len(names) or any(n in self._aliases for n in names):
            raise ValueError(f'duplicate model registration: {spec.name}')
        self._specs[spec.name] = spec
        self._aliases.update({name: spec.name for name in names})

    def get(self, name):
        try:
            return self._specs[self._aliases[name]]
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Unknown or not yet implemented model: {name!r}') from exc

    def names(self):
        return tuple(sorted(self._specs))


registry = ModelRegistry()
