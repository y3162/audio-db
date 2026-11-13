"""Helpers for experiment-framework tests."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import torch
import yaml
from torch import nn

from src.experiments.data.fixed_pair import SnapshotDataModule
from src.experiments.errors import DataError
from src.experiments.runtime import PreparePolicy, PreparedModel, ProcessRuntime
from src.experiments.selectors import Decision, SavePoint, StandardSelector
from src.experiments.trainers import StandardTrainer


class StepArchiveSelector(StandardSelector):
    """Keep every state that replaces Latest as ``archive/step_XXXXXXXX.pt``.

    Tests compare the state at individual steps, so each saved state also
    stays as an Archive. ``interval`` is the Latest period inside an epoch.
    """

    checkpoint_class_id = "tests.experiments.support.StepArchiveSelector"

    def __init__(self, interval: int = 1, partial: bool = False) -> None:
        super().__init__(latest_every=interval, partial=partial)

    def decide(self, point: SavePoint) -> Decision:
        if not self.wants_latest(point):
            return Decision()
        return Decision(latest=True, archive=(f"step_{point.logical_step:08d}.pt",))


class Timeline:
    events: list[str] = []
    constructed: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.events = []
        cls.constructed = 0


class Marker:
    """Small class the factory must be able to import without naming it."""

    created = 0

    def __init__(self, value: int = 1) -> None:
        type(self).created += 1
        self.value = value


class OrderProbe:
    labels: list[str] = []

    def __init__(self, label: str) -> None:
        OrderProbe.labels.append(label)

    @classmethod
    def reset(cls) -> None:
        cls.labels = []


class NeedsExplicit:
    def __init__(self, loss: nn.Module) -> None:
        Timeline.constructed += 1
        self.loss = loss


class NeedsOther:
    def __init__(self, other: object) -> None:
        Timeline.constructed += 1
        self.other = other


class TakesLr:
    def __init__(self, lr: float = 0.1) -> None:
        Timeline.constructed += 1
        self.lr = lr


class NullTrainer:
    def __init__(self) -> None:
        Timeline.events.append("trainer")

    def run(self, runtime: object, checkpoint: object) -> None:
        del runtime, checkpoint
        raise RuntimeError("NullTrainer cannot train")

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state


class CaptureTrainer:
    def __init__(
        self,
        model: object | None = None,
        loss: object | None = None,
        loss_for_log: object | None = None,
        data: object | None = None,
        optimizer: object | None = None,
        max_logical_steps: int = 1,
    ) -> None:
        Timeline.events.append("trainer")
        self.model = model
        self.loss = loss
        self.loss_for_log = loss_for_log
        self.data = data
        self.optimizer = optimizer
        self.max_logical_steps = max_logical_steps

    def run(self, runtime: object, checkpoint: object) -> None:
        del runtime, checkpoint
        raise RuntimeError("CaptureTrainer cannot train")


class ProbeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        Timeline.events.append("module")


class ProbeOptimizer:
    def __init__(self, params: object, lr: float = 0.1) -> None:
        materialized = list(params)  # type: ignore[arg-type]
        self.param_groups = [{"params": materialized, "lr": lr}]
        Timeline.events.append("optimizer")
        for parameter in materialized:
            if not getattr(parameter, "_prepared_by_runtime", False):
                raise RuntimeError("optimizer received an unprepared parameter")

    def zero_grad(self, set_to_none: bool = False) -> None:
        del set_to_none

    def step(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state


class ModuleWithExtra(nn.Module):
    def __init__(self, extra: object, in_features: int = 1, out_features: int = 1) -> None:
        super().__init__()
        self.inner = nn.Linear(in_features, out_features)
        self.extra = extra
        Timeline.constructed += 1


class RecordingRuntime(ProcessRuntime):
    def prepare_model(self, module: nn.Module, policy: PreparePolicy) -> PreparedModel:
        Timeline.events.append("prepare")
        prepared = super().prepare_model(module, policy)
        for parameter in prepared.parameters:
            setattr(parameter, "_prepared_by_runtime", True)
        return prepared


def write_config(
    directory: Path,
    components: dict[str, object],
    *,
    seed: int = 17,
    entry: str = "trainer",
    runtime_class: str = "src.experiments.runtime.ProcessRuntime",
    filename: str = "experiment.yaml",
) -> Path:
    """Write a valid experiment document whose output is ``directory / 'out'``."""

    directory = Path(directory)
    document = {
        "schema": "audio-db-experiment/v1",
        "seed": seed,
        "deterministic": True,
        "entry": entry,
        "runtime": {
            "class": runtime_class,
            "kwargs": {"device": "cpu", "precision": "fp32"},
        },
        "output": {"directory": str(directory / "out")},
        "components": components,
    }
    path = directory / filename
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def null_trainer() -> dict[str, object]:
    return {"role": "trainer", "class": "tests.experiments.support.NullTrainer"}


class InspectTrainer:
    """Checks DDP parameter identity, then runs :class:`StandardTrainer`."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        from src.experiments.trainers import StandardTrainer

        self._trainer = StandardTrainer(*args, **kwargs)  # type: ignore[arg-type]

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        from torch.nn.parallel import DistributedDataParallel

        handle = self._trainer.model
        if not isinstance(handle.module, DistributedDataParallel):
            raise RuntimeError("model was not wrapped with DistributedDataParallel")
        if handle.module.module is not handle.state_module:
            raise RuntimeError("checkpoint state is not the unwrapped module")
        wrapped_ids = [id(parameter) for parameter in handle.module.parameters()]
        raw_ids = [id(parameter) for parameter in handle.state_module.parameters()]
        if wrapped_ids != raw_ids:
            raise RuntimeError("DDP replaced parameter objects")
        optimizer_ids = {
            id(parameter)
            for group in self._trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != set(raw_ids):
            raise RuntimeError("optimizer does not hold the prepared parameters")
        device = getattr(runtime, "device", None)
        expected = "nccl" if getattr(device, "type", "cpu") == "cuda" else "gloo"
        if getattr(runtime, "backend", None) != expected:
            raise RuntimeError(f"backend is {getattr(runtime, 'backend', None)}")
        return self._trainer.run(runtime, checkpoint)  # type: ignore[arg-type]

    def state_dict(self) -> dict[str, object]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._trainer.load_state_dict(state)

    def __getattr__(self, name: str) -> object:
        return getattr(self._trainer, name)


class TamperTrainer:
    """Runs the standard loop and corrupts rank 1 before the second save."""

    def __init__(self, *args: object, tamper: str, **kwargs: object) -> None:
        from src.experiments.trainers import StandardTrainer

        if tamper not in {"weight", "buffer", "optimizer", "scheduler", "trainer"}:
            raise ValueError("tamper must be weight, buffer, optimizer, scheduler, or trainer")
        self._trainer = StandardTrainer(*args, **kwargs)  # type: ignore[arg-type]
        self.tamper = tamper

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        self._trainer._before_checkpoint = self._before_checkpoint  # type: ignore[method-assign]
        return self._trainer.run(runtime, checkpoint)  # type: ignore[arg-type]

    def _before_checkpoint(self, runtime: object, checkpoint: object) -> None:
        import hashlib

        trainer = self._trainer
        if trainer.logical_step != 2:
            return
        directory = checkpoint.directory  # type: ignore[attr-defined]
        if int(getattr(runtime, "rank", 0)) == 0:
            previous = directory / "archive/step_00000001.pt"
            digest = hashlib.sha256(previous.read_bytes()).hexdigest()
            (directory / "previous.sha256").write_text(digest + "\n", encoding="utf-8")
            return
        module = trainer.model.state_module
        with torch.no_grad():
            if self.tamper == "weight":
                module.linear.weight.add_(1)
            elif self.tamper == "buffer":
                module.marker.add_(1)
            elif self.tamper == "optimizer":
                found = False
                for parameter in module.parameters():
                    slot = trainer.optimizer.state.get(parameter)
                    if not slot:
                        continue
                    for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                        value = slot.get(key)
                        if torch.is_tensor(value):
                            value.add_(1)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise RuntimeError("optimizer has no tensor state to tamper")
            elif self.tamper == "scheduler":
                trainer.scheduler.last_epoch += 3
            else:
                trainer.logical_step += 7

    def state_dict(self) -> dict[str, object]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._trainer.load_state_dict(state)

    def __getattr__(self, name: str) -> object:
        return getattr(self._trainer, name)


class _KillOnPickle:
    """Kills this process with ``SIGKILL`` while it is being pickled, once per marker file."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        marker = Path(self.marker)
        if not marker.exists():
            marker.write_text("killed\n", encoding="utf-8")
            os.kill(os.getpid(), signal.SIGKILL)
        return (str, ("save-probe",))


class KillDuringSaveTrainer:
    """Runs the standard loop and dies inside ``torch.save`` of ``kill_step``.

    The process is killed after the temporary checkpoint file was opened and
    before it is complete. The marker file makes that happen once, so the
    same YAML resumes without being killed again.
    """

    def __init__(self, *args: object, kill_step: int, marker: str, **kwargs: object) -> None:
        from src.experiments.trainers import StandardTrainer

        self._trainer = StandardTrainer(*args, **kwargs)  # type: ignore[arg-type]
        self.kill_step = int(kill_step)
        self.marker = str(marker)

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        return self._trainer.run(runtime, checkpoint)  # type: ignore[arg-type]

    def state_dict(self) -> dict[str, object]:
        state = self._trainer.state_dict()
        if self._trainer.logical_step == self.kill_step:
            state["save_probe"] = _KillOnPickle(self.marker)
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        cleaned = dict(state)
        cleaned.pop("save_probe", None)
        self._trainer.load_state_dict(cleaned)

    def __getattr__(self, name: str) -> object:
        return getattr(self._trainer, name)


class FaultTrainer:
    """Runs the standard loop and fails once, on chosen ranks, before logical step ``step + 1``.

    ``fault``:

    - ``exception`` raises ``RuntimeError``.
    - ``oom`` raises what PyTorch raises when a CUDA allocation fails. No GPU is needed.
    - ``cuda_oom`` asks the runtime device for more memory than it has.
    - ``sigkill`` kills the rank, as the kernel does when the host runs out of memory.
    - ``hang`` stops making progress and ignores a stop request.
    - ``exit_late`` trains to the end, then exits with code 7.
    - ``none`` does nothing.

    ``rank`` is the rank that fails, or -1 for every rank. ``delay`` slows
    every step so a test can signal a run that is under way. The marker file
    makes the fault happen once, so the same YAML resumes. Every rank writes
    ``<marker>.pid<rank>`` when it starts.
    """

    FAULTS = ("none", "exception", "oom", "cuda_oom", "sigkill", "hang", "exit_late")

    def __init__(
        self,
        *args: object,
        fault: str = "none",
        step: int = 0,
        rank: int = 0,
        marker: str,
        delay: float = 0.0,
        **kwargs: object,
    ) -> None:
        from src.experiments.trainers import StandardTrainer

        if fault not in self.FAULTS:
            raise ValueError(f"fault must be one of {self.FAULTS}")
        self._trainer = StandardTrainer(*args, **kwargs)  # type: ignore[arg-type]
        self.fault = fault
        self.step = int(step)
        self.rank = int(rank)
        self.marker = Path(marker)
        self.delay = float(delay)

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        import time

        trainer = self._trainer
        rank = int(getattr(runtime, "rank", 0))
        self.marker.with_name(f"{self.marker.name}.pid{rank}").write_text(str(os.getpid()), encoding="utf-8")
        # Decided before the first step, so every chosen rank fails in the same launch.
        armed = self.rank in (-1, rank) and not self.marker.exists()

        def before(runtime_arg: object) -> None:
            if self.delay > 0:
                time.sleep(self.delay)
            if not armed or trainer.logical_step != self.step or self.fault in ("none", "exit_late"):
                return
            self.marker.write_text(self.fault, encoding="utf-8")
            if self.fault == "exception":
                raise RuntimeError(f"injected failure on rank {rank}")
            if self.fault == "oom":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 GiB. (injected)")
            if self.fault == "cuda_oom":
                device = getattr(runtime_arg, "device")
                total = torch.cuda.get_device_properties(device).total_memory
                torch.empty(2 * total, dtype=torch.uint8, device=device)
                raise RuntimeError("CUDA granted more memory than the device has")
            if self.fault == "sigkill":
                os.kill(os.getpid(), signal.SIGKILL)
            while True:
                time.sleep(3600)

        trainer._before_logical_step = before  # type: ignore[method-assign]
        result = trainer.run(runtime, checkpoint)  # type: ignore[arg-type]
        if self.fault == "exit_late" and armed:
            import atexit

            self.marker.write_text(self.fault, encoding="utf-8")
            atexit.register(os._exit, 7)
        return result

    def state_dict(self) -> dict[str, object]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._trainer.load_state_dict(state)

    def __getattr__(self, name: str) -> object:
        return getattr(self._trainer, name)


class RankSplitSelector(StandardSelector):
    """Decides differently on rank 1. The manager must refuse to write."""

    checkpoint_class_id = "tests.experiments.support.RankSplitSelector"

    def decide(self, point: SavePoint) -> Decision:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if point.logical_step == 1:
            return Decision(latest=True)
        if rank == 1 and point.logical_step == 2:
            return Decision(latest=True, archive=("rank1.pt",))
        return super().decide(point)


class DisagreeTrainer:
    """Calls ``agree`` with a different flag on each rank."""

    def __init__(self, max_logical_steps: int = 1) -> None:
        self.max_logical_steps = max_logical_steps

    def run(self, runtime: object, checkpoint: object) -> None:
        del checkpoint
        flag = int(getattr(runtime, "rank", 0)) == 0
        runtime.agree(flag, policy="identical")  # type: ignore[attr-defined]
        raise RuntimeError("agree returned despite a disagreement")

    def state_dict(self) -> dict[str, object]:
        return {"logical_step": 0, "optimizer_updates": {}, "stop_reason": "initialized"}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state


class DropoutLinear(nn.Module):
    """Linear layer whose dropout consumes the process RNG.

    The buffer changes on every forward so a checkpoint has to restore it.
    """

    def __init__(self, in_features: int, out_features: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(p)
        self.register_buffer("marker", torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.marker += 1
        return self.drop(self.linear(inputs))


class AutocastLinear(nn.Module):
    """Linear layer that records whether forward ran inside autocast."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("autocast_on", torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            self.autocast_on.fill_(1)
        return self.linear(inputs)


class ScaleBumpTrainer:
    """Raises the grad scale once, then optionally restores a finite scale.

    ``bump_step`` is the logical step about to run. ``restore_step`` puts the
    scale back before a later step so the batches after the skip stay finite.
    """

    def __init__(
        self,
        *args: object,
        bump_step: int = 1,
        bump_scale: float = 131072.0,
        restore_step: int | None = None,
        restore_scale: float | None = None,
        **kwargs: object,
    ) -> None:
        from src.experiments.trainers import StandardTrainer

        self._trainer = StandardTrainer(*args, **kwargs)  # type: ignore[arg-type]
        self.bump_step = int(bump_step)
        self.bump_scale = float(bump_scale)
        self.restore_step = None if restore_step is None else int(restore_step)
        self.restore_scale = None if restore_scale is None else float(restore_scale)

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        trainer = self._trainer

        def before(runtime_arg: object) -> None:
            if trainer.logical_step == self.bump_step:
                runtime_arg.set_grad_scale(self.bump_scale)  # type: ignore[attr-defined]
            if self.restore_step is not None and trainer.logical_step == self.restore_step:
                runtime_arg.set_grad_scale(self.restore_scale)  # type: ignore[attr-defined]

        trainer._before_logical_step = before  # type: ignore[method-assign]
        return trainer.run(runtime, checkpoint)  # type: ignore[arg-type]

    def state_dict(self) -> dict[str, object]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._trainer.load_state_dict(state)

    def __getattr__(self, name: str) -> object:
        return getattr(self._trainer, name)


class RankOverflowTrainer:
    """Makes rank 1's gradient non-finite on one logical step.

    The loss stays finite. The hook runs before unscale, so only that rank's
    GradScaler sees a local overflow.
    """

    def __init__(self, *args: object, overflow_step: int = 1, **kwargs: object) -> None:
        from src.experiments.trainers import StandardTrainer

        self._trainer = StandardTrainer(*args, **kwargs)  # type: ignore[arg-type]
        self.overflow_step = int(overflow_step)

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        trainer = self._trainer

        def before(runtime_arg: object) -> None:
            if trainer.logical_step != self.overflow_step or int(getattr(runtime_arg, "rank", 0)) != 1:
                return
            parameter = trainer.model.parameters[0]
            parameter.grad = torch.full_like(parameter, float("inf"))

        trainer._before_optimizer_step = before  # type: ignore[method-assign]
        return trainer.run(runtime, checkpoint)  # type: ignore[arg-type]

    def state_dict(self) -> dict[str, object]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._trainer.load_state_dict(state)

    def __getattr__(self, name: str) -> object:
        return getattr(self._trainer, name)


class CudaConsumeTrainer:
    """Draws a different CUDA random count on each rank before every save."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        from src.experiments.trainers import StandardTrainer

        self._trainer = StandardTrainer(*args, **kwargs)  # type: ignore[arg-type]

    def run(self, runtime: object, checkpoint: object):  # type: ignore[no-untyped-def]
        trainer = self._trainer
        original = trainer._before_checkpoint

        def before(runtime_arg: object, checkpoint_arg: object) -> None:
            device = getattr(runtime_arg, "device", None)
            if getattr(device, "type", None) == "cuda":
                torch.rand(2 + int(getattr(runtime_arg, "rank", 0)), device=device)
            original(runtime_arg, checkpoint_arg)  # type: ignore[arg-type]

        trainer._before_checkpoint = before  # type: ignore[method-assign]
        return trainer.run(runtime, checkpoint)  # type: ignore[arg-type]

    def state_dict(self) -> dict[str, object]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._trainer.load_state_dict(state)

    def __getattr__(self, name: str) -> object:
        return getattr(self._trainer, name)


class IdentityWave(nn.Module):
    """Passes a waveform through. ``anchor`` gives the optimizer one parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.anchor


class BoomLoss(nn.Module):
    """Raises on the first forward so lifecycle tests can leave the run."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del prediction, target
        raise RuntimeError("boom")


class GainWave(nn.Module):
    """Length-independent gain. Phase 3-E uses it instead of an SE model."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.gain


class CursorSkewTrainer(StandardTrainer):
    """Moves rank 1's data cursor before the first checkpoint."""

    def _before_checkpoint(self, runtime: ProcessRuntime, checkpoint: object) -> None:
        del checkpoint
        if int(runtime.rank) == 1 and self.logical_step == 1:
            self.data.update_index = int(self.data.update_index) + 1


class PlanSkewTrainer(StandardTrainer):
    """Reports a different plan hash from rank 1 at the first checkpoint."""

    def _before_checkpoint(self, runtime: ProcessRuntime, checkpoint: object) -> None:
        del checkpoint
        if int(runtime.rank) != 1 or self.logical_step != 1:
            return
        original = self.data.state_dict

        def skewed() -> dict[str, object]:
            payload = dict(original())
            payload["plan_hash"] = "0" * 64
            return payload

        self.data.state_dict = skewed  # type: ignore[method-assign]


class RankBuildFailModule(SnapshotDataModule):
    """Fails setup on the rank named by ``AUDIO_DB_FAIL_BUILD_RANK``."""

    def setup(self, runtime: object) -> None:
        rank = int(getattr(runtime, "rank", 0))
        if os.environ.get("AUDIO_DB_FAIL_BUILD_RANK") == str(rank):
            raise DataError(f"injected build failure on rank {rank}")
        super().setup(runtime)


def research_weights(sample_ids: tuple[str, ...] | list[str]) -> dict[str, float]:
    """One zero weight and a 1:3 pattern. The total stays positive when two or more ids exist."""

    table: dict[str, float] = {}
    for index, sample_id in enumerate(sample_ids):
        if index == 0:
            table[sample_id] = 0.0
        elif index % 2:
            table[sample_id] = 3.0
        else:
            table[sample_id] = 1.0
    if len(sample_ids) == 1:
        table[sample_ids[0]] = 1.0
    return table


class WeightedSnapshot(SnapshotDataModule):
    """Sets the same research weight table on every rank during setup."""

    def setup(self, runtime: object) -> None:
        super().setup(runtime)
        sample_ids = self.sampler.training_ids()  # type: ignore[attr-defined]
        self.set_sampling_weights(research_weights(sample_ids))


class SkewWeightModule(SnapshotDataModule):
    """Rank 1 changes one positive weight. Ranks do not exchange the table."""

    def setup(self, runtime: object) -> None:
        super().setup(runtime)
        sample_ids = self.sampler.training_ids()  # type: ignore[attr-defined]
        table = research_weights(sample_ids)
        if int(getattr(runtime, "rank", 0)) == 1:
            table[sample_ids[1]] = 9.0
        self.set_sampling_weights(table)


class RankWeightFailModule(SnapshotDataModule):
    """Rank 0 rejects an all-zero table before the epoch plan is gathered."""

    def setup(self, runtime: object) -> None:
        super().setup(runtime)
        sample_ids = self.sampler.training_ids()  # type: ignore[attr-defined]
        if int(getattr(runtime, "rank", 0)) == 0:
            self.set_sampling_weights({sample_id: 0.0 for sample_id in sample_ids})
        else:
            self.set_sampling_weights(research_weights(sample_ids))


class RankRenderFailModule(SnapshotDataModule):
    """Fails the first micro-batch on the rank named by ``AUDIO_DB_FAIL_RENDER_RANK``."""

    def materialize(self, epoch: int, index: int, groups: tuple[tuple[object, ...], ...]) -> object:
        if os.environ.get("AUDIO_DB_FAIL_RENDER_RANK") == str(self.rank) and index == 0:
            raise DataError(f"injected render failure on rank {self.rank}")
        return super().materialize(epoch, index, groups)
