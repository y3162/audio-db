"""A new trainer can finish without a logical step, epoch, or phase."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.experiments.api import Experiment
from src.experiments.config import load_experiment
from src.experiments.evaluation.boundary import ResearchSession
from src.experiments.results import RunResult
from tests.experiments.support import write_config


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> int:
    found: list[int] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state or "step" not in state:
                continue
            found.append(int(state["step"]))
    return max(found) if found else 0


class EventSE:
    """One optimizer and a real forward. The checkpoint counter stays absent."""

    def __init__(
        self,
        model: object,
        loss: object,
        data: object,
        optimizer: torch.optim.Optimizer,
        updates: int = 1,
        model_selection: object | None = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.data = data
        self.optimizer = optimizer
        self.updates = int(updates)
        self.model_selection = model_selection
        self.research = ResearchSession(None, model_selection, data=data, module=model)
        self.evaluators = self.research.evaluators

    def fit(self) -> RunResult:
        self.lifecycle.start()
        checkpoint = self.lifecycle.checkpoint
        done = _optimizer_steps(self.optimizer)
        if done >= self.updates:
            self.lifecycle.save_if_needed("final")
            return self.lifecycle.finish(
                RunResult("completed", None, {"optimizer": done}, checkpoint.directory)
            )
        loss = _supervised_step(self.runtime, self.model, self.loss, self.data, self.optimizer)
        done = _optimizer_steps(self.optimizer)
        extra = self.research.metrics(
            self.runtime,
            checkpoint,
            kind="boundary",
            epoch=None,
            progress={},
            official={"loss": loss},
        )
        metrics = {"loss": loss, **extra}
        self.lifecycle.save_if_needed("boundary", metrics=metrics)
        if done >= self.updates:
            self.lifecycle.save_if_needed("final", metrics=metrics)
            return self.lifecycle.finish(
                RunResult("completed", None, {"optimizer": done}, checkpoint.directory)
            )
        return self.lifecycle.finish(
            RunResult("interrupted", None, {"optimizer": done}, checkpoint.directory)
        )


class EventGAN:
    """Two optimizers. Neither the pair nor the finish invents a logical step."""

    def __init__(
        self,
        generator: object,
        discriminator: object,
        data: object,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        loss: object,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.loss = loss

    def fit(self) -> RunResult:
        self.lifecycle.start()
        checkpoint = self.lifecycle.checkpoint
        _supervised_step(self.runtime, self.generator, self.loss, self.data, self.optim_g)
        _supervised_step(self.runtime, self.discriminator, self.loss, self.data, self.optim_d)
        self.lifecycle.save_if_needed("boundary")
        self.lifecycle.save_if_needed("final")
        return self.lifecycle.finish(
            RunResult(
                "completed",
                None,
                {
                    "optim_d": _optimizer_steps(self.optim_d),
                    "optim_g": _optimizer_steps(self.optim_g),
                },
                checkpoint.directory,
            )
        )


def _supervised_step(runtime: object, model: object, loss: object, data: object, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    view = data.train_epoch(0, start_update=0)
    try:
        batch = view.batch(0)[0]
    finally:
        close = getattr(view, "close", None)
        if callable(close):
            close()
    batch = runtime.move(batch)
    optimizer.zero_grad(set_to_none=True)
    value = loss(model(batch.inputs), batch.targets)
    runtime.backward(value)
    runtime.step(optimizer)
    runtime.end_step()
    return float(value.detach())


def _components(trainer: str, *, updates: int | None = None, selection: bool = False) -> dict[str, object]:
    trainer_spec: dict[str, object] = {
        "role": "trainer",
        "class": trainer,
        "inject": {
            "model": {"ref": "model"},
            "loss": {"ref": "loss"},
            "data": {"ref": "data"},
            "optimizer": {"ref": "optimizer"},
        },
    }
    if updates is not None:
        trainer_spec["kwargs"] = {"updates": updates}
    components: dict[str, object] = {
        "model": {
            "role": "module",
            "class": "torch.nn.Linear",
            "kwargs": {"in_features": 8, "out_features": 8},
            "prepare": {"trainable": True, "distribute": "none"},
        },
        "loss": {"role": "loss", "class": "torch.nn.MSELoss"},
        "data": {
            "role": "data",
            "class": "src.experiments.data.TensorPairData",
            "kwargs": {"features": 8, "examples": 32, "batch_size": 4},
        },
        "optimizer": {
            "role": "optimizer",
            "class": "torch.optim.AdamW",
            "parameters_of": "model",
            "kwargs": {"lr": 0.01},
        },
        "trainer": trainer_spec,
    }
    if selection:
        components["selection"] = {
            "role": "other",
            "class": "src.experiments.selection.BestMetricSelection",
            "kwargs": {"metric": "validation.loss", "mode": "min"},
            "state": {"save": True},
        }
        trainer_spec["inject"]["model_selection"] = {"ref": "selection"}
    return components


class CounterlessTests(unittest.TestCase):
    def test_one_optimizer_finishes_without_a_counter(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        result = Experiment(load_experiment(write_config(directory, _components(EventSE.__module__ + ".EventSE")))).run()
        self.assertIsNone(result.logical_step)
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(result.optimizer_updates["optimizer"], 1)
        payload = torch.load(result.output_dir / "latest.pt", map_location="cpu", weights_only=False)
        self.assertIsNone(payload["logical_step"])
        self.assertEqual(payload["progress"], {})
        complete = json.loads((result.output_dir / "complete.json").read_text(encoding="utf-8"))
        self.assertIsNone(complete["logical_step"])

    def test_resume_continues_from_the_optimizer(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = write_config(
            directory,
            _components(EventSE.__module__ + ".EventSE", updates=2),
        )
        first = Experiment(load_experiment(config)).run()
        self.assertEqual(first.stop_reason, "interrupted")
        self.assertIsNone(first.logical_step)
        resumed = Experiment(load_experiment(config))
        second = resumed.run(resume=first.output_dir)
        self.assertEqual(second.stop_reason, "completed")
        self.assertIsNone(second.logical_step)
        self.assertEqual(second.optimizer_updates["optimizer"], 2)
        self.assertEqual(_optimizer_steps(resumed.get("optimizer")), 2)
        payload = torch.load(second.output_dir / "latest.pt", map_location="cpu", weights_only=False)
        self.assertIsNone(payload["logical_step"])

    def test_two_optimizers_finish_without_a_counter(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        components = _components("tests.experiments.test_counterless.EventGAN")
        components["discriminator"] = {
            "role": "module",
            "class": "torch.nn.Linear",
            "kwargs": {"in_features": 8, "out_features": 8},
            "prepare": {"trainable": True, "distribute": "none"},
        }
        components["optim_d"] = {
            "role": "optimizer",
            "class": "torch.optim.AdamW",
            "parameters_of": "discriminator",
            "kwargs": {"lr": 0.01},
        }
        trainer = components["trainer"]
        assert isinstance(trainer, dict)
        trainer["inject"] = {
            "generator": {"ref": "model"},
            "discriminator": {"ref": "discriminator"},
            "data": {"ref": "data"},
            "optim_g": {"ref": "optimizer"},
            "optim_d": {"ref": "optim_d"},
            "loss": {"ref": "loss"},
        }
        result = Experiment(load_experiment(write_config(directory, components))).run()
        self.assertIsNone(result.logical_step)
        self.assertEqual(result.optimizer_updates, {"optim_d": 1, "optim_g": 1})

    def test_research_selection_does_not_invent_a_step(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        experiment = Experiment(
            load_experiment(
                write_config(
                    directory,
                    _components(EventSE.__module__ + ".EventSE", selection=True),
                )
            )
        )
        result = experiment.run()
        self.assertIsNone(result.logical_step)
        selection = experiment.get("selection")
        self.assertIsNone(selection.best_step)
        self.assertIsNotNone(selection.best_observation)
        rows = (result.output_dir / "research_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertTrue(rows)
        self.assertNotIn("logical_step", json.loads(rows[0]))

    def _directory(self):
        import tempfile

        return tempfile.TemporaryDirectory()
