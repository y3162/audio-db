"""Helpers for experiment-framework tests."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from src.experiments.runtime import PreparePolicy, PreparedModel, ProcessRuntime


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
