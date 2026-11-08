"""CPU float32 runtime for one process.

Distributed wrapping, mixed precision, and gradient scaling are refused.
Phase 2 owns those paths. Optimizer updates still go through :meth:`step`
so a later scaler can keep every rank on the same decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal

import torch

from src.experiments.errors import PrepareError, RuntimeFailure
from src.training.core.rng import capture_rank_rng, restore_rank_rng


@dataclass(frozen=True)
class PreparePolicy:
    """How :meth:`ProcessRuntime.prepare_model` places one module."""

    trainable: bool = True
    distribute: Literal["ddp", "none"] = "none"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    sync_batchnorm: bool = False


@dataclass(frozen=True)
class PreparedModel:
    """Handles a trainer uses after preparation.

    ``module`` is what forward calls. ``parameters`` is the view an optimizer
    must hold. ``state_module`` is what a checkpoint reads. On CPU without DDP
    the first and the third are the same object.
    """

    component_id: str
    module: torch.nn.Module
    parameters: tuple[torch.nn.Parameter, ...]
    state_module: torch.nn.Module
    policy: PreparePolicy


@dataclass(frozen=True)
class StepResult:
    """Whether every rank applied an optimizer update."""

    applied: bool


class ProcessRuntime:
    """Single-process CPU runtime.

    ``device`` must be ``cpu`` and ``precision`` must be ``fp32``. A world
    size other than 1 is refused so a distributed launch cannot silently skip
    gradient synchronization.
    """

    def __init__(self, *, device: str = "cpu", precision: str = "fp32") -> None:
        import os

        if not isinstance(device, str) or not isinstance(precision, str):
            raise PrepareError("device and precision must be strings")
        if device != "cpu":
            raise PrepareError(f"Phase 1 runtime only supports device cpu, got {device}")
        if precision != "fp32":
            raise PrepareError(f"Phase 1 runtime only supports precision fp32, got {precision}")
        world = os.environ.get("WORLD_SIZE")
        if world is not None and world != "1":
            raise PrepareError(f"Phase 1 runtime supports WORLD_SIZE 1 only, got {world}")
        rank = os.environ.get("RANK")
        if rank is not None and rank != "0":
            raise PrepareError(f"Phase 1 runtime supports RANK 0 only, got {rank}")
        self.device = torch.device("cpu")
        self.precision = "fp32"
        self.world_size = 1
        self._clipped: set[frozenset[int]] = set()

    def prepare_model(self, module: torch.nn.Module, policy: PreparePolicy) -> PreparedModel:
        """Move ``module`` to CPU and return the optimizer parameter view."""

        if not isinstance(module, torch.nn.Module):
            raise PrepareError(f"module is not a torch.nn.Module: {type(module).__name__}")
        if not isinstance(policy, PreparePolicy):
            raise PrepareError("prepare policy is missing")
        if policy.distribute == "fsdp":
            raise PrepareError("distribute fsdp is not available")
        if policy.distribute != "none":
            raise PrepareError(f"distribute {policy.distribute} is not implemented")
        try:
            placed = module.to(self.device)
        except Exception as exc:
            raise PrepareError(f"failed to place module on cpu: {exc}") from exc
        if not isinstance(placed, torch.nn.Module):
            raise PrepareError("placed module is not a torch.nn.Module")
        if policy.trainable:
            placed.train()
            parameters = tuple(placed.parameters())
        else:
            for parameter in placed.parameters():
                parameter.requires_grad_(False)
            placed.eval()
            parameters = ()
        return PreparedModel(
            component_id="",
            module=placed,
            parameters=parameters,
            state_module=placed,
            policy=policy,
        )

    @contextmanager
    def sync_context(
        self,
        modules: list[PreparedModel],
        *,
        sync: bool,
    ) -> Iterator[None]:
        """Enter a forward scope. CPU has no gradient bucket to delay."""

        del modules
        if not isinstance(sync, bool):
            raise RuntimeFailure("sync must be a bool")
        yield

    def autocast(self) -> AbstractContextManager[None]:
        """Return a no-op context. Float32 does not cast."""

        return nullcontext()

    def backward(self, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        """Backpropagate a finite loss. Non-finite values never call backward."""

        if not isinstance(loss, torch.Tensor):
            raise RuntimeFailure("loss must be a tensor")
        if loss.numel() != 1:
            raise RuntimeFailure("loss must be a scalar")
        if not bool(torch.isfinite(loss.detach()).all()):
            raise RuntimeFailure("loss is not finite")
        loss.backward(retain_graph=retain_graph)

    def clip(self, parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
        """Clip gradients once per parameter set in the current step."""

        params = [parameter for parameter in parameters if parameter.requires_grad]
        key = frozenset(id(parameter) for parameter in params)
        if key in self._clipped:
            raise RuntimeFailure("parameters were already unscaled in this step")
        self._clipped.add(key)
        if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)) or max_norm <= 0:
            raise RuntimeFailure("max_norm must be a positive number")
        norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm))
        return float(norm)

    def step(self, optimizer: torch.optim.Optimizer) -> StepResult:
        """Apply ``optimizer.step``. Float32 always reports the update as applied.

        This does not zero gradients and does not update a scaler.
        """

        optimizer.step()
        return StepResult(applied=True)

    def end_step(self) -> None:
        """Finish one logical step. Float32 has no scaler update."""

        self._clipped.clear()

    def average_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Average gradients across ranks. One rank has nothing to average."""

        del parameters

    def capture_rng(self) -> dict[str, object]:
        """Capture this rank's Python, NumPy, and Torch CPU generators.

        CUDA generators are not read. Phase 2-A never initializes a device.
        A later distributed runtime gathers one blob per rank before save.
        """

        return capture_rank_rng(cuda=False)

    def restore_rng(self, state: dict[str, object]) -> None:
        """Restore one rank blob. World size 1 uses ``rng_by_rank[0]``."""

        restore_rank_rng(state, cuda=False)

    def agree(self, flag: bool, *, policy: str = "identical") -> bool:
        """Return ``flag``. One rank is already unanimous."""

        if not isinstance(flag, bool):
            raise RuntimeFailure("flag must be a bool")
        if policy not in {"identical", "any", "all"}:
            raise RuntimeFailure(f"unknown agree policy {policy}")
        return flag
