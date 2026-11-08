"""Preparation happens before optimizers, and failed builds do not run."""

from __future__ import annotations

import os
import unittest

import torch

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import BuildError, ConfigError, PrepareError, StateError
from src.experiments.runtime import PreparedModel, ProcessRuntime
from tests.experiments.support import Timeline, null_trainer, write_config


def _parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        Timeline.reset()

    def test_optimizer_is_created_after_prepare_and_sees_the_handle(self) -> None:
        path = write_config(
            self.enterContext(self._directory()),
            {
                "model": {"role": "module", "class": "tests.experiments.support.ProbeModule"},
                "optimizer": {
                    "role": "optimizer",
                    "class": "tests.experiments.support.ProbeOptimizer",
                    "parameters_of": "model",
                    "kwargs": {"lr": 0.1},
                },
                "trainer": {
                    "role": "trainer",
                    "class": "tests.experiments.support.CaptureTrainer",
                    "inject": {"model": {"ref": "model"}, "optimizer": {"ref": "optimizer"}},
                },
            },
            runtime_class="tests.experiments.support.RecordingRuntime",
        )
        experiment = build_experiment(load_experiment(path))
        self.assertEqual(Timeline.events, ["module", "prepare", "optimizer", "trainer"])
        handle = experiment.get("model")
        trainer = experiment.get("trainer")
        self.assertIsInstance(handle, PreparedModel)
        self.assertIs(trainer.model, handle)
        self.assertIs(handle.module, handle.state_module)
        self.assertEqual(_parameter_ids(trainer.optimizer), {id(parameter) for parameter in handle.parameters})
        self.assertEqual(
            {id(parameter) for parameter in handle.parameters},
            {id(parameter) for parameter in handle.module.parameters()},
        )
        self.assertIn("model", experiment._graph.states.ids())
        self.assertIn("runtime", experiment._graph.states.ids())
        self.assertIn("trainer", experiment._graph.states.ids())

    def test_linear_parameters_match_and_frozen_parameters_are_refused(self) -> None:
        built_dir = self.enterContext(self._directory())
        frozen_dir = self.enterContext(self._directory())
        built = build_experiment(
            load_experiment(
                write_config(
                    built_dir,
                    {
                        "model": {
                            "role": "module",
                            "class": "torch.nn.Linear",
                            "kwargs": {"in_features": 4, "out_features": 4},
                            "prepare": {"trainable": True, "distribute": "none"},
                        },
                        "optimizer": {
                            "role": "optimizer",
                            "class": "torch.optim.AdamW",
                            "parameters_of": "model",
                            "kwargs": {"lr": 0.01},
                        },
                        "trainer": null_trainer(),
                    },
                )
            )
        )
        handle = built.get("model")
        self.assertEqual(_parameter_ids(built.get("optimizer")), {id(parameter) for parameter in handle.parameters})
        frozen = write_config(
            frozen_dir,
            {
                "model": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 2, "out_features": 2},
                    "prepare": {"trainable": False, "distribute": "none"},
                },
                "optimizer": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": "model",
                    "kwargs": {"lr": 0.01},
                },
                "trainer": null_trainer(),
            },
            filename="frozen.yaml",
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(frozen))
        self.assertIn("model", str(caught.exception))

    def test_gan_optimizers_bind_disjoint_parameters(self) -> None:
        path = write_config(
            self.enterContext(self._directory()),
            {
                "generator": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 8, "out_features": 8},
                    "prepare": {"trainable": True, "distribute": "none"},
                },
                "discriminator": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 8, "out_features": 1},
                    "prepare": {"trainable": True, "distribute": "none"},
                },
                "data": {
                    "role": "data",
                    "class": "src.experiments.data.TensorPairData",
                    "kwargs": {"features": 8, "examples": 32, "batch_size": 4},
                },
                "optim_g": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": {"component": "generator", "lr": 0.0001},
                },
                "optim_d": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": {"component": "discriminator", "lr": 0.0002},
                },
                "trainer": {
                    "role": "trainer",
                    "class": "src.experiments.trainers.GanTrainer",
                    "kwargs": {"max_logical_steps": 2},
                    "inject": {
                        "generator": {"ref": "generator"},
                        "discriminator": {"ref": "discriminator"},
                        "data": {"ref": "data"},
                        "optim_g": {"ref": "optim_g"},
                        "optim_d": {"ref": "optim_d"},
                    },
                },
            },
        )
        experiment = build_experiment(load_experiment(path))
        generator_ids = {id(parameter) for parameter in experiment.get("generator").parameters}
        discriminator_ids = {id(parameter) for parameter in experiment.get("discriminator").parameters}
        optim_g = _parameter_ids(experiment.get("optim_g"))
        optim_d = _parameter_ids(experiment.get("optim_d"))
        self.assertEqual(optim_g, generator_ids)
        self.assertEqual(optim_d, discriminator_ids)
        self.assertEqual(optim_g & optim_d, set())

    def test_experts_are_distinct_and_keep_inject_order(self) -> None:
        path = write_config(
            self.enterContext(self._directory()),
            {
                "expert_b": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 8, "out_features": 8},
                },
                "expert_a": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 8, "out_features": 8},
                },
                "router": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 8, "out_features": 2},
                },
                "data": {
                    "role": "data",
                    "class": "src.experiments.data.TensorPairData",
                    "kwargs": {"features": 8, "examples": 16, "batch_size": 4},
                },
                "optim_experts": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": [
                        {"component": "expert_a", "lr": 0.001},
                        {"component": "expert_b", "lr": 0.001},
                    ],
                },
                "optim_router": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": "router",
                    "kwargs": {"lr": 0.0005},
                },
                "trainer": {
                    "role": "trainer",
                    "class": "src.experiments.trainers.MixtureTrainer",
                    "kwargs": {"max_logical_steps": 2},
                    "inject": {
                        "experts": [{"ref": "expert_a"}, {"ref": "expert_b"}],
                        "router": {"ref": "router"},
                        "data": {"ref": "data"},
                        "optim_experts": {"ref": "optim_experts"},
                        "optim_router": {"ref": "optim_router"},
                    },
                },
            },
        )
        experiment = build_experiment(load_experiment(path))
        trainer = experiment.get("trainer")
        expert_a = experiment.get("expert_a")
        expert_b = experiment.get("expert_b")
        self.assertIsNot(expert_a, expert_b)
        self.assertIs(trainer.experts[0], expert_a)
        self.assertIs(trainer.experts[1], expert_b)
        expert_ids = {id(parameter) for parameter in expert_a.parameters}
        expert_ids.update(id(parameter) for parameter in expert_b.parameters)
        router_ids = {id(parameter) for parameter in experiment.get("router").parameters}
        self.assertEqual(_parameter_ids(experiment.get("optim_experts")), expert_ids)
        self.assertEqual(_parameter_ids(experiment.get("optim_router")) & router_ids, router_ids)
        self.assertEqual(_parameter_ids(experiment.get("optim_experts")) & router_ids, set())

    def test_shared_module_parameters_are_rejected(self) -> None:
        path = write_config(
            self.enterContext(self._directory()),
            {
                "model": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 2, "out_features": 2},
                },
                "optim_a": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": "model",
                    "kwargs": {"lr": 0.01},
                },
                "optim_b": {
                    "role": "optimizer",
                    "class": "torch.optim.AdamW",
                    "parameters_of": "model",
                    "kwargs": {"lr": 0.02},
                },
                "trainer": null_trainer(),
            },
        )
        with self.assertRaises(BuildError) as caught:
            build_experiment(load_experiment(path))
        message = str(caught.exception)
        self.assertIn("optim_a", message)
        self.assertIn("optim_b", message)

    def test_second_build_is_state_error(self) -> None:
        path = write_config(self.enterContext(self._directory()), {"trainer": null_trainer()})
        experiment = build_experiment(load_experiment(path))
        with self.assertRaises(StateError):
            experiment.build()

    def test_runtime_refuses_cpu_amp_unavailable_cuda_and_fsdp(self) -> None:
        with self.assertRaises(PrepareError) as caught_cpu_amp:
            ProcessRuntime(device="cpu", precision="amp_fp16")
        self.assertIn("refusing to fall back", str(caught_cpu_amp.exception))
        with self.assertRaises(PrepareError):
            ProcessRuntime(device="cuda", precision="amp_bf16")
        if not torch.cuda.is_available():
            with self.assertRaises(PrepareError) as caught_cuda:
                ProcessRuntime(device="cuda", precision="fp32")
            self.assertIn("refusing to fall back", str(caught_cuda.exception))
        else:
            runtime = ProcessRuntime(device="cuda", precision="fp32")
            self.assertEqual(runtime.device.type, "cuda")
            self.assertEqual(runtime.backend, "none")
            self.assertIsNone(runtime.scaler_state())
            runtime.shutdown()
        directory = self.enterContext(self._directory())
        distributed = write_config(
            directory,
            {
                "model": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 2, "out_features": 2},
                    "prepare": {"trainable": True, "distribute": "ddp"},
                },
                "trainer": null_trainer(),
            },
        )
        distributed_experiment = build_experiment(load_experiment(distributed))
        handle = distributed_experiment.get("model")
        self.assertIs(handle.module, handle.state_module)  # type: ignore[attr-defined]
        distributed_experiment.close()
        sharded = write_config(
            directory,
            {
                "model": {
                    "role": "module",
                    "class": "torch.nn.Linear",
                    "kwargs": {"in_features": 2, "out_features": 2},
                    "prepare": {"trainable": True, "distribute": "fsdp"},
                },
                "trainer": null_trainer(),
            },
            filename="fsdp.yaml",
        )
        with self.assertRaises(PrepareError) as caught_fsdp:
            build_experiment(load_experiment(sharded))
        self.assertIn("fsdp", str(caught_fsdp.exception))
        bf16 = write_config(
            directory,
            {"trainer": null_trainer()},
            filename="bf16.yaml",
        )
        text = bf16.read_text(encoding="utf-8").replace("fp32", "amp_bf16")
        bf16.write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError) as caught_bf16:
            load_experiment(bf16)
        self.assertIn("amp_bf16", str(caught_bf16.exception))

    def test_incomplete_distributed_launch_is_refused(self) -> None:
        previous = os.environ.get("WORLD_SIZE")
        os.environ["WORLD_SIZE"] = "2"
        try:
            with self.assertRaises(PrepareError) as caught:
                ProcessRuntime()
            self.assertIn("WORLD_SIZE", str(caught.exception))
            self.assertIn("refusing to fall back", str(caught.exception))
        finally:
            if previous is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = previous

    def _directory(self):
        import tempfile

        return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
