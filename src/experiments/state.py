"""State owners registered by the builder.

Phase 1 records who owns state. It does not write a checkpoint file.
"""

from __future__ import annotations

from typing import Any

import torch

from src.experiments.errors import CheckpointError, RuntimeFailure, StateError
from src.experiments.runtime import PreparedModel, ProcessRuntime


class ComponentState:
    """Saved view of one concrete component."""

    def __init__(self, state_id: str, kind: str, target: object) -> None:
        self.state_id = state_id
        self.kind = kind
        self.target = target
        if isinstance(target, PreparedModel):
            self.owner_key = id(target.state_module)
        else:
            self.owner_key = id(target)

    def state_dict(self) -> dict[str, Any]:
        target = self.target.state_module if isinstance(self.target, PreparedModel) else self.target
        method = getattr(target, "state_dict", None)
        if not callable(method):
            raise StateError(f"state {self.state_id} has no state_dict")
        state = method()
        if not isinstance(state, dict):
            raise StateError(f"state {self.state_id} did not return a mapping")
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        target = self.target.state_module if isinstance(self.target, PreparedModel) else self.target
        method = getattr(target, "load_state_dict", None)
        if not callable(method):
            raise StateError(f"state {self.state_id} has no load_state_dict")
        try:
            if isinstance(target, torch.nn.Module):
                method(state, strict=True)
            else:
                method(state)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise CheckpointError(f"state {self.state_id} failed to load: {exc}") from exc


class RuntimeState:
    """Process runtime state, including the GradScaler when precision is AMP."""

    def __init__(self, runtime: ProcessRuntime) -> None:
        self.state_id = "runtime"
        self.kind = "runtime"
        self.runtime = runtime
        self.owner_key = id(runtime)

    def state_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.runtime.world_size,
            "precision": self.runtime.precision,
            "device": self.runtime.device.type,
            "grad_scaler": self.runtime.scaler_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # Checkpoints written before CUDA support omit device and mean CPU.
        saved_device = state.get("device", "cpu")
        if (
            state.get("world_size") != self.runtime.world_size
            or state.get("precision") != self.runtime.precision
            or saved_device != self.runtime.device.type
        ):
            raise StateError("runtime state does not match this process")
        saved_scaler = state.get("grad_scaler")
        if self.runtime.precision != "amp_fp16":
            if saved_scaler is not None:
                raise CheckpointError("fp32 runtime cannot restore a grad scaler")
            return
        if not isinstance(saved_scaler, dict):
            raise CheckpointError("amp runtime requires a grad scaler state")
        try:
            self.runtime.load_scaler(saved_scaler)
        except RuntimeFailure as exc:
            raise CheckpointError(f"grad scaler state is invalid: {exc}") from exc
