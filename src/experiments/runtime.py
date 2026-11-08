"""CPU float32 runtime for one process or a Gloo process group.

World size 1 keeps the Phase 1 path and does not initialize a process group.
World size 2 or more uses the CPU Gloo backend and ``DistributedDataParallel``.
CUDA, NCCL, AMP, and FSDP stay refused. A launch environment that does not
match the requested world size is an error; this runtime does not fall back
to a single process.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

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

    ``module`` is what forward calls. Under DDP it is the wrapper.
    ``parameters`` is the view an optimizer must hold. ``state_module`` is the
    unwrapped module a checkpoint reads, so state keys do not grow a
    ``module.`` prefix. On CPU, DDP does not replace parameter objects:
    ``parameters`` and the wrapper's parameters are the same objects.
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
    """CPU float32 runtime.

    ``device`` must be ``cpu`` and ``precision`` must be ``fp32``. ``world_size``
    in the constructor, when set, must equal the ``WORLD_SIZE`` of this launch.
    An unset environment variable means one process. Ranks above 0 require
    ``RANK`` and ``LOCAL_RANK`` from ``torchrun``.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        precision: str = "fp32",
        world_size: int | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        if not isinstance(device, str) or not isinstance(precision, str):
            raise PrepareError("device and precision must be strings")
        if device != "cpu":
            raise PrepareError(f"runtime only supports device cpu, got {device}")
        if precision != "fp32":
            raise PrepareError(f"runtime only supports precision fp32, got {precision}")
        if world_size is not None and (isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1):
            raise PrepareError("world_size must be a positive integer")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise PrepareError("timeout_seconds must be a positive number")
        resolved_world, rank, local_rank = _launch_world(world_size)
        self.device = torch.device("cpu")
        self.precision = "fp32"
        self.world_size = resolved_world
        self.rank = rank
        self.local_rank = local_rank
        self.timeout_seconds = float(timeout_seconds)
        self.backend = "none"
        self._clipped: set[frozenset[int]] = set()
        self._group_started = False
        if self.world_size > 1:
            self._init_gloo()

    def prepare_model(self, module: torch.nn.Module, policy: PreparePolicy) -> PreparedModel:
        """Move ``module`` to CPU and, for a multi-rank launch, wrap it with DDP.

        World size 1 accepts ``distribute: ddp`` and leaves the module
        unwrapped. A trainable module in a multi-rank launch must ask for
        DDP; otherwise its gradients would never be synchronized.
        ``sync_batchnorm`` is applied only for CUDA, which this runtime
        refuses, so CPU ranks keep ``BatchNorm``.
        """

        if not isinstance(module, torch.nn.Module):
            raise PrepareError(f"module is not a torch.nn.Module: {type(module).__name__}")
        if not isinstance(policy, PreparePolicy):
            raise PrepareError("prepare policy is missing")
        if policy.distribute == "fsdp":
            raise PrepareError("distribute fsdp is not available")
        if policy.distribute not in {"none", "ddp"}:
            raise PrepareError(f"distribute {policy.distribute} is not implemented")
        if self.world_size > 1 and policy.trainable and policy.distribute != "ddp":
            raise PrepareError(
                f"trainable module requires distribute ddp when WORLD_SIZE is {self.world_size}"
            )
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
        execution: torch.nn.Module = placed
        if policy.distribute == "ddp" and self.world_size > 1 and policy.trainable:
            if not dist.is_initialized():
                raise PrepareError("process group is not initialized")
            try:
                execution = DistributedDataParallel(
                    placed,
                    device_ids=None,
                    output_device=None,
                    find_unused_parameters=policy.find_unused_parameters,
                    broadcast_buffers=policy.broadcast_buffers,
                )
            except Exception as exc:
                raise PrepareError(f"failed to wrap module with DDP: {exc}") from exc
        return PreparedModel(
            component_id="",
            module=execution,
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
        """Enter a forward scope.

        DDP reads the sync flag during forward. The matching backward must
        run inside this same scope. Turning synchronization off after forward
        does not cancel an all-reduce the forward already installed.
        """

        if not isinstance(sync, bool):
            raise RuntimeFailure("sync must be a bool")
        with ExitStack() as stack:
            if not sync:
                for prepared in modules:
                    module = prepared.module
                    if isinstance(module, DistributedDataParallel):
                        stack.enter_context(module.no_sync())
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
        """Apply ``optimizer.step`` after every rank has entered this call.

        Float32 always reports the update as applied. This does not zero
        gradients and does not update a scaler.
        """

        self.agree(True, policy="identical")
        optimizer.step()
        return StepResult(applied=True)

    def end_step(self) -> None:
        """Finish one logical step. Float32 has no scaler update."""

        self._clipped.clear()

    def average_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Average gradients across ranks.

        Missing gradients participate as zeros so every rank reduces the same
        tensors. World size 1 does nothing. DDP's own all-reduce is separate;
        callers that already synchronized through :meth:`sync_context` must
        not call this again.
        """

        params = [parameter for parameter in parameters if parameter.requires_grad]
        if self.world_size == 1:
            return
        self._require_group()
        for parameter in params:
            gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(self.world_size)
            parameter.grad = gradient

    def capture_rng(self) -> dict[str, object]:
        """Capture this rank's Python, NumPy, and Torch CPU generators.

        CUDA generators are not read. A distributed save gathers one blob per
        rank before writing the checkpoint.
        """

        return capture_rank_rng(cuda=False)

    def restore_rng(self, state: dict[str, object]) -> None:
        """Restore one rank blob. The caller selects ``rng_by_rank[rank]``."""

        restore_rank_rng(state, cuda=False)

    def agree(self, flag: bool, *, policy: str = "identical") -> bool:
        """Combine one boolean from every rank.

        ``identical`` raises :class:`RuntimeFailure` when the values differ,
        so a skipped update cannot land on only one rank. ``any`` and ``all``
        reduce the gathered values. Every rank must call this together.
        """

        if not isinstance(flag, bool):
            raise RuntimeFailure("flag must be a bool")
        if policy not in {"identical", "any", "all"}:
            raise RuntimeFailure(f"unknown agree policy {policy}")
        if self.world_size == 1:
            return flag
        gathered = self.gather_objects(flag)
        if not all(isinstance(item, bool) for item in gathered):
            raise RuntimeFailure("agree received a non-bool from another rank")
        if policy == "identical" and any(item != flag for item in gathered):
            raise RuntimeFailure("ranks disagree on the update decision")
        if policy == "any":
            return any(gathered)
        if policy == "all":
            return all(gathered)
        return flag

    def barrier(self) -> None:
        """Wait until every rank reaches this call. World size 1 returns."""

        if self.world_size == 1:
            return
        self._require_group()
        dist.barrier()

    def gather_objects(self, value: object) -> list[object]:
        """Return ``value`` from rank 0 through rank ``world_size - 1``.

        World size 1 returns a one-element list and does not touch a process
        group. Object identity is not preserved across ranks; callers compare
        values.
        """

        if self.world_size == 1:
            return [value]
        self._require_group()
        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def shutdown(self) -> None:
        """Destroy the process group when this runtime created one."""

        if self.world_size == 1 or not self._group_started:
            return
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self._group_started = False

    def _init_gloo(self) -> None:
        if dist.is_initialized():
            if dist.get_world_size() != self.world_size or dist.get_rank() != self.rank:
                raise PrepareError("process group does not match RANK and WORLD_SIZE")
            backend = dist.get_backend()
            if backend != "gloo":
                raise PrepareError(f"runtime only supports the gloo backend, got {backend}")
            self.backend = "gloo"
            self._group_started = True
            return
        try:
            dist.init_process_group(
                backend="gloo",
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=self.timeout_seconds),
            )
        except Exception as exc:
            raise PrepareError(f"failed to initialize the gloo process group: {exc}") from exc
        self.backend = "gloo"
        self._group_started = True

    def _require_group(self) -> None:
        if not dist.is_initialized():
            raise RuntimeFailure("process group is not initialized")


def _launch_world(requested: int | None) -> tuple[int, int, int]:
    raw_world = os.environ.get("WORLD_SIZE")
    raw_rank = os.environ.get("RANK")
    raw_local = os.environ.get("LOCAL_RANK")
    if raw_world is None:
        if raw_rank not in (None, "0") or raw_local not in (None, "0"):
            raise PrepareError("RANK is set without WORLD_SIZE; refusing to fall back to a single process")
        if requested not in (None, 1):
            raise PrepareError(
                f"runtime world_size {requested} does not match WORLD_SIZE (unset); refusing to fall back"
            )
        return 1, 0, 0
    world = _env_int("WORLD_SIZE", raw_world)
    if requested is not None and requested != world:
        raise PrepareError(
            f"runtime world_size {requested} does not match WORLD_SIZE {world}; refusing to fall back"
        )
    if world == 1:
        rank = 0 if raw_rank is None else _env_int("RANK", raw_rank)
        if rank != 0:
            raise PrepareError(f"WORLD_SIZE 1 does not match RANK {rank}")
        return 1, 0, 0
    if raw_rank is None or raw_local is None:
        raise PrepareError(
            f"WORLD_SIZE {world} requires RANK and LOCAL_RANK; refusing to fall back to a single process"
        )
    rank = _env_int("RANK", raw_rank)
    local_rank = _env_int("LOCAL_RANK", raw_local)
    if rank < 0 or rank >= world or local_rank < 0:
        raise PrepareError(
            f"invalid distributed rank {rank} local_rank {local_rank} WORLD_SIZE {world}"
        )
    return world, rank, local_rank


def _env_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise PrepareError(f"{name} must be an integer, got {raw!r}") from exc
