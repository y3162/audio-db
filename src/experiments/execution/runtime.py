"""Process runtime for CPU/FP32, CUDA/FP32, and CUDA AMP FP16.

World size 1 does not initialize a process group. CPU world size 2 or more
uses Gloo. CUDA world size 2 or more uses NCCL, one GPU per rank. A launch
environment that does not match the requested device, precision, or world
size is an error; this runtime does not fall back to CPU or to one process.

AMP owns one ``torch.amp.GradScaler`` per process. The trainer calls
``autocast``, ``backward``, ``clip``, ``step``, and ``end_step``. The scaler
is updated once at ``end_step``, after every optimizer in that logical step
has been unscaled and either stepped or skipped together on all ranks.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.experiments.errors import PrepareError, RuntimeFailure
from src.experiments.calc.rng import capture_rank_rng, restore_rank_rng

_SCALER_KEYS = frozenset(
    {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
)


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
    ``module.`` prefix. Single-device DDP, on CPU and on CUDA, does not
    replace parameter objects: ``parameters`` and the wrapper's parameters
    are the same objects.
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
    """CPU or CUDA runtime for one process or a matching process group.

    ``precision`` is ``fp32`` or, on CUDA only, ``amp_fp16``. ``world_size``
    in the constructor, when set, must equal the ``WORLD_SIZE`` of this launch.
    An unset environment variable means one process. Ranks above 0 require
    ``RANK`` and ``LOCAL_RANK`` from ``torchrun``. CUDA selects
    ``cuda:LOCAL_RANK`` and refuses a launch where two ranks would share a GPU.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        precision: str = "fp32",
        world_size: int | None = None,
        timeout_seconds: float = 120,
        allow_busy_gpu: bool = False,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        if not isinstance(device, str) or not isinstance(precision, str):
            raise PrepareError("device and precision must be strings")
        if device not in {"cpu", "cuda"}:
            raise PrepareError(f"runtime only supports device cpu or cuda, got {device}")
        if precision not in {"fp32", "amp_fp16"}:
            raise PrepareError(f"runtime only supports precision fp32 or amp_fp16, got {precision}")
        if precision == "amp_fp16" and device != "cuda":
            raise PrepareError("amp_fp16 requires device cuda; refusing to fall back to fp32")
        if world_size is not None and (isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1):
            raise PrepareError("world_size must be a positive integer")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise PrepareError("timeout_seconds must be a positive number")
        if not isinstance(allow_busy_gpu, bool):
            raise PrepareError("allow_busy_gpu must be a boolean")
        _check_scaler_config(init_scale, growth_factor, backoff_factor, growth_interval)
        # cuBLAS reads this when the CUDA context is created. Set it before set_device.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        resolved_world, rank, local_rank = _launch_world(world_size)
        self.precision = precision
        self.world_size = resolved_world
        self.rank = rank
        self.local_rank = local_rank
        self.timeout_seconds = float(timeout_seconds)
        self.backend = "none"
        self._clipped: set[frozenset[int]] = set()
        self._unscaled: set[int] = set()
        self._stepped: set[int] = set()
        self._loss_finite: torch.Tensor | None = None
        self._group_started = False
        self._scaler: torch.amp.GradScaler | None = None
        self.allow_busy_gpu = bool(allow_busy_gpu)
        self.device = _resolve_device(device, local_rank, resolved_world)
        if self.device.type == "cuda":
            from src.experiments.calc.errors import TrainingError
            from src.experiments.calc.gpu import assert_gpu_idle

            try:
                assert_gpu_idle(self.local_rank, allow_busy=self.allow_busy_gpu)
            except TrainingError as exc:
                raise PrepareError(str(exc)) from exc
        if precision == "amp_fp16":
            self._scaler = torch.amp.GradScaler(
                "cuda",
                init_scale=float(init_scale),
                growth_factor=float(growth_factor),
                backoff_factor=float(backoff_factor),
                growth_interval=int(growth_interval),
                enabled=True,
            )
            if not self._scaler.is_enabled():
                raise PrepareError("amp_fp16 GradScaler was disabled; refusing to fall back to fp32")
        if self.world_size > 1:
            try:
                self._init_distributed()
            except Exception:
                self.shutdown()
                raise

    def prepare_model(self, module: torch.nn.Module, policy: PreparePolicy) -> PreparedModel:
        """Move ``module`` onto this rank's device and maybe wrap it with DDP.

        World size 1 accepts ``distribute: ddp`` and leaves the module
        unwrapped. A trainable module in a multi-rank launch must ask for
        DDP. ``sync_batchnorm`` converts BatchNorm only for CUDA world size
        2 or more. CPU ranks keep BatchNorm.
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
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        try:
            placed = module.to(self.device)
        except Exception as exc:
            raise PrepareError(f"failed to place module on {self.device}: {exc}") from exc
        if not isinstance(placed, torch.nn.Module):
            raise PrepareError("placed module is not a torch.nn.Module")
        if policy.sync_batchnorm and self.device.type == "cuda" and self.world_size > 1:
            placed = torch.nn.SyncBatchNorm.convert_sync_batchnorm(placed)
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
            device_ids = None if self.device.type == "cpu" else [int(self.device.index)]
            output_device = None if self.device.type == "cpu" else int(self.device.index)
            try:
                execution = DistributedDataParallel(
                    placed,
                    device_ids=device_ids,
                    output_device=output_device,
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
        """Return CUDA FP16 autocast, or a no-op context for FP32."""

        if self.precision == "amp_fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def backward(self, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        """Backpropagate a loss. A non-finite loss is rejected by step() before the update.

        AMP scales the loss and does not change the scale during gradient
        accumulation. A finite loss can still overflow in the backward; that
        is decided later by :meth:`step`, not rejected here.
        """

        if not isinstance(loss, torch.Tensor):
            raise RuntimeFailure("loss must be a tensor")
        if loss.numel() != 1:
            raise RuntimeFailure("loss must be a scalar")
        # Reading this flag here would finish the forward before backward is launched.
        # step() reads it once, before the optimizer applies the gradients.
        flag = torch.isfinite(loss.detach()).all()
        if self._loss_finite is None:
            self._loss_finite = flag
        else:
            self._loss_finite = torch.logical_and(self._loss_finite, flag)
        from src.experiments.execution.profile import Span

        with Span("backward", self.device):
            if self._scaler is None:
                loss.backward(retain_graph=retain_graph)
                return
            self._scaler.scale(loss).backward(retain_graph=retain_graph)

    def clip(
        self,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.Tensor:
        """Clip gradients once per parameter set in the current step.

        AMP unscales ``optimizer`` first and refuses a second unscale of that
        optimizer. FP32 ignores ``optimizer``. The returned norm is the
        unscaled norm.
        """

        params = [parameter for parameter in parameters if parameter.requires_grad]
        if self.precision == "amp_fp16":
            if optimizer is None:
                raise RuntimeFailure("amp clip requires the optimizer that owns these parameters")
            self._unscale(optimizer)
        key = frozenset(id(parameter) for parameter in params)
        if key in self._clipped:
            raise RuntimeFailure("parameters were already unscaled in this step")
        self._clipped.add(key)
        if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)) or max_norm <= 0:
            raise RuntimeFailure("max_norm must be a positive number")
        norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm))
        # Reading the norm here would drain the stream before the optimizer step.
        return norm

    def clip_repeated(self, parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
        """Clip gradients on every call.

        :meth:`clip` still refuses a second clip of the same parameters in
        one logical step, and it still unscales AMP at most once.
        MossFormerGAN clips after every successful micro-batch, so one
        accumulation window clips the same parameters more than once.
        This method does not record a clip and does not unscale. It refuses
        ``amp_fp16`` rather than weakening that one-unscale rule.
        """

        if self.precision == "amp_fp16":
            raise RuntimeFailure("clip_repeated refuses amp_fp16; clip once per step after unscale")
        params = [parameter for parameter in parameters if parameter.requires_grad]
        if not params:
            raise RuntimeFailure("clip_repeated received no trainable parameters")
        if (
            isinstance(max_norm, bool)
            or not isinstance(max_norm, (int, float))
            or not math.isfinite(float(max_norm))
            or float(max_norm) <= 0
        ):
            raise RuntimeFailure("max_norm must be a positive finite number")
        norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm))
        return float(norm)

    def host_scalar(self, value: torch.Tensor) -> torch.Tensor:
        """Queue a host copy of one scalar.

        Call this before :meth:`step`. That method waits for work already
        queued, which finishes this copy, and only then launches the optimizer.
        Reading the returned CPU tensor does not wait for the optimizer.
        """

        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise RuntimeFailure("host_scalar requires a scalar tensor")
        if value.device.type != "cuda":
            return value.detach()
        host = torch.empty((), dtype=torch.float32, pin_memory=True)
        host.copy_(value.detach().to(dtype=torch.float32), non_blocking=True)
        return host

    def step(self, optimizer: torch.optim.Optimizer) -> StepResult:
        """Apply one optimizer update, or skip it on every rank.

        FP32 always steps after every rank has entered this call. AMP unscales
        when :meth:`clip` did not, then ORs the overflow flag across ranks.
        ``applied`` is True only when ``optimizer.step`` itself ran. The
        return value of ``GradScaler.step`` is not used: standard optimizers
        return ``None`` even when the update ran. This does not zero
        gradients and does not update the scaler.
        """

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeFailure("optimizer must be a torch optimizer")
        if id(optimizer) in self._stepped:
            raise RuntimeFailure("optimizer was already stepped in this logical step")
        self._require_finite_loss()
        from src.experiments.execution.profile import Span

        if self._scaler is None:
            self.agree(True, policy="identical")
            with Span("optimizer_step", self.device):
                applied = self._call_optimizer_step(optimizer)
            self._stepped.add(id(optimizer))
            return StepResult(applied=applied)
        if id(optimizer) not in self._unscaled:
            self._unscale(optimizer)
        finite = self._coordinate_finite(optimizer)
        with Span("optimizer_step", self.device):
            applied = self._call_optimizer_step(optimizer) if finite else False
        self._stepped.add(id(optimizer))
        return StepResult(applied=applied)

    def _require_finite_loss(self) -> None:
        flag = self._loss_finite
        self._loss_finite = None
        if flag is None:
            return
        if not self.agree(bool(flag), policy="all"):
            raise RuntimeFailure("loss is not finite")

    def end_step(self) -> None:
        """Finish one logical step. AMP updates the scaler once."""

        try:
            if self._scaler is not None:
                if not self._stepped:
                    raise RuntimeFailure("end_step requires runtime.step before the scaler update")
                try:
                    self._scaler.update()
                except (AssertionError, RuntimeError) as exc:
                    raise RuntimeFailure(f"grad scaler update failed: {exc}") from exc
        finally:
            self._clipped.clear()
            self._unscaled.clear()
            self._stepped.clear()
            self._loss_finite = None

    def set_grad_scale(self, scale: float) -> None:
        """Replace the AMP scale between logical steps.

        Tests use this to force a finite loss whose backward overflows.
        The growth tracker is left unchanged. Calling this during unscale
        or after ``step`` in the same logical step is a failure.
        """

        if self._scaler is None:
            raise RuntimeFailure("grad scale exists only for amp_fp16")
        if self._unscaled or self._stepped:
            raise RuntimeFailure("grad scale cannot change during a logical step")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise RuntimeFailure("grad scale must be a positive finite number")
        value = float(scale)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeFailure("grad scale must be a positive finite number")
        # GradScaler initializes its scale tensor on the first ``scale()`` call.
        if self._scaler._scale is None:  # noqa: SLF001
            self._scaler._init_scale = value  # noqa: SLF001
            return
        self._scaler.update(new_scale=value)

    def scaler_state(self) -> dict[str, Any] | None:
        """Return the public GradScaler state, or ``None`` for FP32."""

        if self._scaler is None:
            return None
        return dict(self._scaler.state_dict())

    def load_scaler(self, state: dict[str, Any]) -> None:
        """Restore a GradScaler. Incomplete or FP32 state is refused."""

        if self._scaler is None:
            raise RuntimeFailure("fp32 runtime cannot restore a grad scaler")
        if set(state) != _SCALER_KEYS:
            missing = sorted(_SCALER_KEYS - set(state))
            extra = sorted(set(state) - _SCALER_KEYS)
            raise RuntimeFailure(f"grad scaler state keys missing {missing} extra {extra}")
        for name in ("scale", "growth_factor", "backoff_factor"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeFailure(f"grad scaler {name} is not a finite number")
        interval = state["growth_interval"]
        tracker = state["_growth_tracker"]
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise RuntimeFailure("grad scaler growth_interval is invalid")
        if isinstance(tracker, bool) or not isinstance(tracker, int) or tracker < 0:
            raise RuntimeFailure("grad scaler growth tracker is invalid")
        if float(state["growth_factor"]) <= 1.0 or not 0.0 < float(state["backoff_factor"]) < 1.0:
            raise RuntimeFailure("grad scaler factors are invalid")
        if float(state["scale"]) <= 0.0:
            raise RuntimeFailure("grad scaler scale is invalid")
        try:
            self._scaler.load_state_dict(
                {
                    "scale": float(state["scale"]),
                    "growth_factor": float(state["growth_factor"]),
                    "backoff_factor": float(state["backoff_factor"]),
                    "growth_interval": int(interval),
                    "_growth_tracker": int(tracker),
                }
            )
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise RuntimeFailure(f"grad scaler state is invalid: {exc}") from exc

    def average_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Average gradients across ranks.

        Missing gradients participate as zeros so every rank reduces the same
        tensors. World size 1 does nothing. DDP's own all-reduce is separate;
        callers that already synchronized through :meth:`sync_context` must
        not call this again. NCCL reduces CUDA tensors on this rank's device.
        """

        params = [parameter for parameter in parameters if parameter.requires_grad]
        if self.world_size == 1:
            return
        self._require_group()
        for parameter in params:
            gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            if self.backend == "nccl" and gradient.device != self.device:
                raise RuntimeFailure("nccl all_reduce requires the rank's CUDA device")
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(self.world_size)
            parameter.grad = gradient

    def average_new_gradients(
        self,
        parameters: Iterable[torch.nn.Parameter],
        baselines: Mapping[int, torch.Tensor | None],
    ) -> None:
        """Average only the gradient added since ``baselines``.

        ``baselines`` maps ``id(parameter)`` to a detached clone of ``.grad``
        taken before this backward, or None when ``.grad`` was None. That
        running sum is not reduced again. A parameter that still has no
        gradient stays None; this does not invent a zero tensor. Every rank
        passes the same parameters in the same order and must agree on which
        gradients exist. World size 1 returns without rewriting ``.grad``.
        :meth:`average_gradients` is unchanged and still averages the stored
        gradient, using zeros where it is missing.
        """

        if not isinstance(baselines, Mapping):
            raise RuntimeFailure("baselines must map parameter ids to gradients")
        params = [parameter for parameter in parameters if parameter.requires_grad]
        for parameter in params:
            if id(parameter) not in baselines:
                raise RuntimeFailure("baseline is missing a parameter")
        mask = tuple(parameter.grad is not None for parameter in params)
        masks = self.gather_objects(mask)
        if any(item != mask for item in masks):
            raise RuntimeFailure("ranks disagree on which parameters received a new gradient")
        if self.world_size == 1:
            return
        self._require_group()
        for parameter, present in zip(params, mask, strict=True):
            if not present:
                continue
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeFailure("gradient disappeared after the rank check")
            old = baselines[id(parameter)]
            if old is None:
                delta = gradient.detach().contiguous()
            else:
                if tuple(old.shape) != tuple(gradient.shape) or old.dtype != gradient.dtype:
                    raise RuntimeFailure("baseline gradient does not match the parameter")
                delta = (gradient - old.to(device=gradient.device, dtype=gradient.dtype)).detach().contiguous()
            if self.backend == "nccl" and delta.device != self.device:
                raise RuntimeFailure("nccl all_reduce requires the rank's CUDA device")
            dist.all_reduce(delta, op=dist.ReduceOp.SUM)
            delta.div_(self.world_size)
            if old is None:
                parameter.grad = delta
            else:
                parameter.grad = old.to(device=delta.device, dtype=delta.dtype) + delta

    def broadcast_buffers(self, module: torch.nn.Module, *, src: int = 0) -> None:
        """Copy one module's buffers from ``src`` to every rank.

        World size 1 returns. A forward that updates buffers outside DDP
        leaves ranks different; a later checkpoint compares those buffers.
        Broadcasting after the optimizer step does not change parameters.
        """

        if not isinstance(module, torch.nn.Module):
            raise RuntimeFailure("broadcast_buffers requires a module")
        if isinstance(src, bool) or not isinstance(src, int) or src < 0 or src >= self.world_size:
            raise RuntimeFailure("broadcast src is outside the process group")
        if self.world_size == 1:
            return
        self._require_group()
        for buffer in module.buffers():
            if self.backend == "nccl" and buffer.device != self.device:
                raise RuntimeFailure("nccl broadcast requires the rank's CUDA device")
            dist.broadcast(buffer, src=src)

    def capture_rng(self) -> dict[str, object]:
        """Capture this rank's Python, NumPy, Torch, and current CUDA generators.

        The CUDA blob also records the device index and UUID. A later resume
        on a different GPU is refused.
        """

        cuda = self.device.type == "cuda"
        state = capture_rank_rng(cuda=cuda)
        if cuda:
            props = torch.cuda.get_device_properties(self.device)
            state["cuda_device_identity"] = {
                "index": int(self.device.index),
                "name": str(props.name),
                "uuid": str(props.uuid),
            }
        return state

    def restore_rng(self, state: dict[str, object]) -> None:
        """Restore one rank blob. The caller selects ``rng_by_rank[rank]``."""

        restore_rank_rng(state, cuda=self.device.type == "cuda")
        if self.device.type != "cuda":
            return
        identity = state.get("cuda_device_identity")
        if not isinstance(identity, dict):
            raise RuntimeFailure("checkpoint has no CUDA device identity; refusing to resume")
        props = torch.cuda.get_device_properties(self.device)
        saved_index = identity.get("index")
        saved_uuid = identity.get("uuid")
        if not isinstance(saved_index, int) or isinstance(saved_index, bool):
            raise RuntimeFailure("checkpoint CUDA device index is invalid")
        if saved_uuid != str(props.uuid) or saved_index != int(self.device.index):
            raise RuntimeFailure(
                "checkpoint CUDA device "
                f"{saved_uuid} index {saved_index} does not match "
                f"{props.uuid} index {int(self.device.index)}"
            )

    def agree(self, flag: bool, *, policy: str = "identical") -> bool:
        """Combine one boolean from every rank.

        ``identical`` raises :class:`RuntimeFailure` when the values differ,
        so a skipped update cannot land on only one rank. ``any`` and ``all``
        reduce the gathered values. Every rank must call this together.
        AMP overflow does not use ``identical``: a rank-local overflow becomes
        a collective skip inside :meth:`step`.
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
        values. On NCCL this pickles the object into a collective and is not
        a large-model state transfer.
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

    def _init_distributed(self) -> None:
        expected = "nccl" if self.device.type == "cuda" else "gloo"
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if dist.is_initialized():
            if dist.get_world_size() != self.world_size or dist.get_rank() != self.rank:
                raise PrepareError("process group does not match RANK and WORLD_SIZE")
            backend = dist.get_backend()
            if backend != expected:
                raise PrepareError(
                    f"runtime requires the {expected} backend, got {backend}; refusing to fall back"
                )
            self.backend = expected
            self._group_started = True
            if self.device.type == "cuda":
                self._assert_unique_cuda_devices()
            return
        try:
            dist.init_process_group(
                backend=expected,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=self.timeout_seconds),
                device_id=self.device if self.device.type == "cuda" else None,
            )
        except Exception as exc:
            raise PrepareError(f"failed to initialize the {expected} process group: {exc}") from exc
        self.backend = expected
        self._group_started = True
        if self.device.type == "cuda":
            self._assert_unique_cuda_devices()

    def _assert_unique_cuda_devices(self) -> None:
        props = torch.cuda.get_device_properties(self.device)
        identity = {"uuid": str(props.uuid), "index": int(self.device.index)}
        gathered = self.gather_objects(identity)
        uuids = [item.get("uuid") if isinstance(item, dict) else None for item in gathered]
        if len(set(uuids)) != self.world_size:
            raise PrepareError("two ranks selected the same CUDA device; refusing to share one GPU")

    def _require_group(self) -> None:
        if not dist.is_initialized():
            raise RuntimeFailure("process group is not initialized")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    def _unscale(self, optimizer: torch.optim.Optimizer) -> None:
        if self._scaler is None:
            return
        if id(optimizer) in self._unscaled:
            raise RuntimeFailure("optimizer was already unscaled in this step")
        try:
            self._scaler.unscale_(optimizer)
        except RuntimeError as exc:
            raise RuntimeFailure(f"grad scaler unscale failed: {exc}") from exc
        self._unscaled.add(id(optimizer))

    def _coordinate_finite(self, optimizer: torch.optim.Optimizer) -> bool:
        """Return whether every rank can step this optimizer.

        PyTorch 2.6 records the overflow flag on
        ``GradScaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]``.
        ``GradScaler.update`` reads that private tensor and there is no public
        hook to OR it across ranks. This method all-reduces those tensors in
        place, then writes 0 or 1 back so every rank's scaler sees the same
        flag. Tested with ``torch==2.6.0+cu124``.
        """

        tensors = self._found_inf_tensors(optimizer)
        combined = tensors[0].detach().clone()
        for extra in tensors[1:]:
            combined = combined + extra.detach().to(combined.device)
        if self.world_size > 1:
            self._require_group()
            if combined.device.type != "cuda":
                raise RuntimeFailure("nccl overflow consensus requires a CUDA flag")
            dist.all_reduce(combined, op=dist.ReduceOp.MAX)
        overflow = bool(combined.item() > 0)
        value = torch.ones((), dtype=combined.dtype, device=combined.device)
        if not overflow:
            value = torch.zeros((), dtype=combined.dtype, device=combined.device)
        for tensor in tensors:
            tensor.copy_(value)
        return not overflow

    def _found_inf_tensors(self, optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
        if self._scaler is None:
            raise RuntimeFailure("grad scaler is not enabled")
        states = getattr(self._scaler, "_per_optimizer_states", None)
        record = states.get(id(optimizer)) if isinstance(states, dict) else None
        found = record.get("found_inf_per_device") if isinstance(record, dict) else None
        if not isinstance(found, dict) or not found:
            raise RuntimeFailure("PyTorch 2.6 GradScaler did not record found_inf for this optimizer")
        tensors: list[torch.Tensor] = []
        for value in found.values():
            if not isinstance(value, torch.Tensor):
                raise RuntimeFailure("grad scaler found_inf is not a tensor")
            tensors.append(value)
        return tensors

    def _call_optimizer_step(self, optimizer: torch.optim.Optimizer) -> bool:
        called = False
        original = optimizer.step

        def wrapped(*args: object, **kwargs: object) -> object:
            nonlocal called
            called = True
            return original(*args, **kwargs)

        optimizer.step = wrapped  # type: ignore[method-assign]
        try:
            optimizer.step()
        finally:
            optimizer.step = original  # type: ignore[method-assign]
        return called


def _check_scaler_config(
    init_scale: object,
    growth_factor: object,
    backoff_factor: object,
    growth_interval: object,
) -> None:
    if isinstance(init_scale, bool) or not isinstance(init_scale, (int, float)) or float(init_scale) <= 0:
        raise PrepareError("init_scale must be a positive number")
    if isinstance(growth_factor, bool) or not isinstance(growth_factor, (int, float)) or float(growth_factor) <= 1:
        raise PrepareError("growth_factor must be greater than 1")
    if (
        isinstance(backoff_factor, bool)
        or not isinstance(backoff_factor, (int, float))
        or not 0 < float(backoff_factor) < 1
    ):
        raise PrepareError("backoff_factor must be between 0 and 1")
    if isinstance(growth_interval, bool) or not isinstance(growth_interval, int) or growth_interval < 1:
        raise PrepareError("growth_interval must be a positive integer")


def _resolve_device(device: str, local_rank: int, world_size: int) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise PrepareError("CUDA was requested but CUDA is not available; refusing to fall back to cpu")
    count = torch.cuda.device_count()
    if count < 1:
        raise PrepareError("CUDA was requested but no CUDA device is visible; refusing to fall back to cpu")
    if local_rank >= count:
        raise PrepareError(
            f"LOCAL_RANK {local_rank} is outside the {count} visible CUDA devices; refusing to fall back"
        )
    if world_size > count:
        raise PrepareError(
            f"WORLD_SIZE {world_size} exceeds {count} visible CUDA devices; refusing to share one GPU"
        )
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


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
