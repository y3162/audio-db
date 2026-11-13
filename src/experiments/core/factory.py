"""Create one object from a fully qualified class name and keyword arguments.

The factory does not resolve references, share instances, or choose a
construction order. Those belong to the experiment builder.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.experiments.errors import BuildError, ExperimentError


@dataclass(frozen=True)
class ComponentSpec:
    """One concrete object the factory can construct.

    ``kwargs`` are already resolved. References and ``parameters_of`` are not
    visible here.
    """

    class_path: str
    kwargs: Mapping[str, Any]


class ComponentFactory:
    """Import a class and call it once. No registry is updated."""

    def create(self, spec: ComponentSpec) -> object:
        """Return ``class_path(**kwargs)``.

        A missing module, a missing class, or a constructor failure becomes
        :class:`BuildError`.
        """

        cls = self.resolve(spec.class_path)
        try:
            return cls(**dict(spec.kwargs))
        except ExperimentError:
            raise
        except (TypeError, ValueError) as exc:
            raise BuildError(f"failed to construct {spec.class_path}: {exc}") from exc

    @staticmethod
    def resolve(class_path: str) -> type:
        """Resolve a fully qualified path to exactly one class."""
        if not isinstance(class_path, str) or "." not in class_path:
            raise BuildError(f"invalid class path {class_path!r}")
        module_name, class_name = class_path.rsplit(".", 1)
        if not module_name or not class_name:
            raise BuildError(f"invalid class path {class_path!r}")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise BuildError(f"cannot import {class_path}: {exc}") from exc
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise BuildError(f"class not found: {class_path}") from exc
        if not isinstance(cls, type):
            raise BuildError(f"{class_path} is not a class")
        return cls
