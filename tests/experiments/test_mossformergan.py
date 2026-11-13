"""MossFormerGAN official GAN training on the experiment framework.

CPU / FP32 comparisons use the current ``MossFormerGANTask``. A PESQ stub is
the numerical path. One case calls the real wideband PESQ library. Gloo world
size 2 checks incremental averaging and a rank-agreed skip. SyncBatchNorm is
not claimed here: this process is CPU.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.checkpoint import CheckpointError
from src.experiments.data import Batch
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.official.mossformergan.trainer import MossFormerGANTrainer
from src.experiments.runtime import PreparePolicy, ProcessRuntime
from src.experiments.calc.rng import set_determinism
from src.experiments.official.mossformergan.reference import (
    MossFormerGANTask,
    local_validation_totals,
    official_validation_mean,
)

ROOT = Path(__file__).resolve().parents[2]
LENGTH = 2400
PESQ_LENGTH = 8000


def _task_config() -> dict:
    return {
        "seed": 7,
        "loss": "official",
        "grad_clip": 10.0,
        "optimizer": {"name": "adamw", "lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.01},
        "scheduler": {"name": "none"},
        "model": {
            "name": "mossformergan",
            "variant": "se_16k",
            "sampling_rate": 16000,
            "architecture": {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
        },
    }


def _noise(index: int, length: int = LENGTH) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1000 + index)
    return {
        "clean": torch.randn(1, length, generator=generator) * 0.05,
        "noisy": torch.randn(1, length, generator=generator) * 0.05,
    }


def _tone(length: int, frequency: float) -> torch.Tensor:
    time = torch.arange(length, dtype=torch.float32) / 16000
    return (0.2 * torch.sin(2 * math.pi * frequency * time)).unsqueeze(0)


def _as_batch(item: dict[str, torch.Tensor], index: int) -> Batch:
    return Batch(
        inputs=item["noisy"],
        targets=item["clean"],
        sample_id=f"s{index}",
        realization_id=f"s{index}",
        sample_ids=(f"s{index}",),
        realization_ids=(f"s{index}",),
    )


def _metric_script(fail_calls: set[int]):
    state = {"n": 0}

    def metric(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
        del enhanced, sample_rate
        state["n"] += 1
        if state["n"] in fail_calls:
            return None
        return torch.full((clean.shape[0],), 0.4)

    def reset() -> None:
        state["n"] = 0

    metric.reset = reset  # type: ignore[attr-defined]
    return metric


def _max_tree(left: object, right: object, path: str = "state") -> float:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(path)
        if left.dtype.is_floating_point or left.dtype.is_complex:
            return float((left.detach().cpu() - right.detach().cpu()).abs().max())
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise AssertionError(path)
        return 0.0
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(path)
        return max((_max_tree(left[key], right[key], f"{path}.{key}") for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError(path)
        return max(
            (_max_tree(lhs, rhs, f"{path}[{index}]") for index, (lhs, rhs) in enumerate(zip(left, right))),
            default=0.0,
        )
    if isinstance(left, float) and isinstance(right, (int, float)):
        if math.isnan(left) and math.isnan(float(right)):
            return 0.0
        return abs(left - float(right))
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")
    return 0.0


def _clone_grads(module: torch.nn.Module) -> list[torch.Tensor | None]:
    return [None if parameter.grad is None else parameter.grad.detach().cpu().clone() for parameter in module.parameters()]


def _yaml(directory: Path, **kwargs: object) -> Path:
    updates = int(kwargs.get("updates", 4))
    accumulation = int(kwargs.get("accumulation", 2))
    steps = int(kwargs.get("steps", 1))
    length = int(kwargs.get("length", LENGTH))
    partial = "true" if kwargs.get("partial", False) else "false"
    validate = "true" if kwargs.get("validate", False) else "false"
    distribute = str(kwargs.get("distribute", "none"))
    timeout = int(kwargs.get("timeout", 120))
    device = str(kwargs.get("device", "cpu"))
    deterministic = "true" if kwargs.get("deterministic", True) else "false"
    text = f"""
schema: audio-db-experiment/v1
checkpoint:
  selector:
    class: tests.experiments.support.StepArchiveSelector
    kwargs:
      interval: 1
      partial: {partial}
seed: 7
deterministic: {deterministic}
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: {device}
    precision: fp32
    timeout_seconds: {timeout}
output:
  directory: {directory}
components:
  generator:
    role: module
    class: src.experiments.official.mossformergan.models.MossFormerGANGenerator
    kwargs:
      seed: 7
      variant: se_16k
      n_fft: 400
      win_len: 400
      hop: 100
      num_channel: 64
      sampling_rate: 16000
    prepare:
      trainable: true
      distribute: {distribute}
      find_unused_parameters: true
      broadcast_buffers: true
      sync_batchnorm: true
  discriminator:
    role: module
    class: src.experiments.official.mossformergan.models.MossFormerGANDiscriminator
    kwargs:
      seed: 7
      variant: se_16k
      n_fft: 400
      win_len: 400
      hop: 100
      num_channel: 64
      sampling_rate: 16000
    prepare:
      trainable: true
      distribute: {distribute}
      find_unused_parameters: true
      broadcast_buffers: true
      sync_batchnorm: true
  data:
    role: data
    class: tests.experiments.mossformergan_data.WavePairData
    kwargs:
      updates_per_epoch: {updates}
      length: {length}
      seed: 17
      group_size: {accumulation}
      profile: official_faithful
      selection_uses_test: true
  optim_g:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: generator
    kwargs:
      lr: 0.0005
      betas: [0.9, 0.999]
      weight_decay: 0.01
  optim_d:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: discriminator
    kwargs:
      lr: 0.001
      betas: [0.9, 0.999]
      weight_decay: 0.01
  trainer:
    role: trainer
    class: src.experiments.official.mossformergan.trainer.MossFormerGANTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: {accumulation}
      grad_clip: 10.0
      experiment_claim: official_faithful
      validate_on_epoch_end: {validate}
    inject:
      generator: {{ref: generator}}
      discriminator: {{ref: discriminator}}
      data: {{ref: data}}
      optim_g: {{ref: optim_g}}
      optim_d: {{ref: optim_d}}
"""
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(path.name)
    return payload


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _torchrun(directory: Path, mode: str, timeout: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "LOCAL_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        env.pop(name, None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "2",
            "--master_port",
            str(_free_port()),
            "-m",
            "tests.experiments.mossformergan_worker",
            str(directory),
            mode,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class ClipRuntimeTests(unittest.TestCase):
    def test_repeated_clip_does_not_weaken_the_one_shot_guard(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([100.0, 0.0])
        runtime.clip([parameter], 1.0)
        with self.assertRaises(RuntimeFailure):
            runtime.clip([parameter], 1.0)
        runtime.end_step()
        parameter.grad = torch.tensor([100.0, 0.0])
        runtime.clip_repeated([parameter], 1.0)
        self.assertLessEqual(float(parameter.grad.detach().norm()), 1.0 + 1e-5)
        runtime.clip_repeated([parameter], 1.0)
        self.assertLessEqual(float(parameter.grad.detach().norm()), 1.0 + 1e-5)
        self.assertEqual(runtime._clipped, set())
        runtime.precision = "amp_fp16"
        with self.assertRaises(RuntimeFailure):
            runtime.clip_repeated([parameter], 1.0)
        runtime.precision = "fp32"

    def test_new_gradient_average_leaves_missing_grads_absent(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        missing = torch.nn.Parameter(torch.ones(3))
        present = torch.nn.Parameter(torch.ones(3))
        present.grad = torch.full((3,), 2.0)
        baseline = present.grad.detach().clone()
        runtime.average_new_gradients([missing, present], {id(missing): None, id(present): baseline})
        self.assertIsNone(missing.grad)
        self.assertTrue(torch.equal(present.grad, torch.full((3,), 2.0)))
        with self.assertRaises(RuntimeFailure):
            runtime.average_new_gradients([present], {})


class MossFormerGANAlgorithmTests(unittest.TestCase):
    runtime: ProcessRuntime
    task: MossFormerGANTask
    generator: object
    discriminator: object
    optim_g: torch.optim.AdamW
    optim_d: torch.optim.AdamW
    initial: dict[str, object]

    @classmethod
    def setUpClass(cls) -> None:
        set_determinism(True)
        from src.experiments.official.mossformergan.models import (
            MossFormerGANDiscriminator,
            MossFormerGANGenerator,
        )

        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        discriminator = MossFormerGANDiscriminator(seed=7)
        generator = MossFormerGANGenerator(seed=7)
        if not torch.equal(before, torch.random.get_rng_state()):
            raise AssertionError("model init did not restore the RNG")
        torch.manual_seed(123)
        task = MossFormerGANTask(_task_config(), torch.device("cpu"))
        if not torch.equal(before, torch.random.get_rng_state()):
            raise AssertionError("task init did not restore the RNG")
        generator_diff = _max_tree(task.generator.state_dict(), generator.generator.state_dict(), "generator")
        discriminator_diff = _max_tree(
            task.discriminator.state_dict(), discriminator.discriminator.state_dict(), "discriminator"
        )
        if generator_diff != 0.0 or discriminator_diff != 0.0:
            raise AssertionError(f"init diff generator {generator_diff} discriminator {discriminator_diff}")
        cls.runtime = ProcessRuntime(device="cpu", precision="fp32")
        policy = PreparePolicy(trainable=True, distribute="none")
        cls.generator = cls.runtime.prepare_model(generator, policy)
        cls.discriminator = cls.runtime.prepare_model(discriminator, policy)
        cls.optim_g = torch.optim.AdamW(cls.generator.parameters, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
        cls.optim_d = torch.optim.AdamW(cls.discriminator.parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        cls.optim_g._experiment_component_id = "optim_g"  # type: ignore[attr-defined]
        cls.optim_d._experiment_component_id = "optim_d"  # type: ignore[attr-defined]
        cls.task = task
        cls.initial = {
            "disc": {key: value.detach().cpu().clone() for key, value in cls.discriminator.state_module.state_dict().items()},
            "gen": {key: value.detach().cpu().clone() for key, value in cls.generator.state_module.state_dict().items()},
            "optim_d": cls.optim_d.state_dict(),
            "optim_g": cls.optim_g.state_dict(),
            "task_disc": {key: value.detach().cpu().clone() for key, value in task.discriminator.state_dict().items()},
            "task_gen": {key: value.detach().cpu().clone() for key, value in task.generator.state_dict().items()},
            "task_optim_d": task.optim_d.state_dict(),
            "task_optim_g": task.optim_g.state_dict(),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime.shutdown()

    def setUp(self) -> None:
        self._reload()

    def _reload(self) -> None:
        self.generator.state_module.load_state_dict(self.initial["gen"])
        self.discriminator.state_module.load_state_dict(self.initial["disc"])
        self.optim_g.load_state_dict(self.initial["optim_g"])
        self.optim_d.load_state_dict(self.initial["optim_d"])
        self.optim_g.zero_grad(set_to_none=True)
        self.optim_d.zero_grad(set_to_none=True)
        self.task.generator.load_state_dict(self.initial["task_gen"])
        self.task.discriminator.load_state_dict(self.initial["task_disc"])
        self.task.optim_g.load_state_dict(self.initial["task_optim_g"])
        self.task.optim_d.load_state_dict(self.initial["task_optim_d"])
        self.task.optim_g.zero_grad(set_to_none=True)
        self.task.optim_d.zero_grad(set_to_none=True)
        self.task._accu_count = 0
        self.task.counters = {
            "micro_batches_failed": 0,
            "micro_batches_processed": 0,
            "micro_batches_succeeded": 0,
            "official_step": 0,
            "optimizer_updates": 0,
        }
        self.task._best_val = float("inf")
        self.task._stalls = 0
        self.task._best_snapshot = None
        self.task.stop_training = False
        self.generator.state_module.train()
        self.discriminator.state_module.train()
        self.task.train_mode()

    def _trainer(self, accumulation: int, **kwargs: object) -> MossFormerGANTrainer:
        claim = str(kwargs.get("experiment_claim", "official_faithful"))
        return MossFormerGANTrainer(
            self.generator,
            self.discriminator,
            _IdleData(),
            self.optim_g,
            self.optim_d,
            max_logical_steps=4,
            accumulation=accumulation,
            grad_clip=float(kwargs.get("grad_clip", 10.0)),
            experiment_claim=claim,
        )

    def _exercise(self, batches: list[dict[str, torch.Tensor]], accumulation: int, fail_calls: set[int]) -> None:
        metric = _metric_script(fail_calls)
        self._reload()
        torch.manual_seed(19)
        with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
            self.task.train_update(batches, accumulation)
        expected_gen = {key: value.detach().cpu().clone() for key, value in self.task.generator.state_dict().items()}
        expected_disc = {
            key: value.detach().cpu().clone() for key, value in self.task.discriminator.state_dict().items()
        }
        expected_grads = _clone_grads(self.task.generator)
        expected_disc_grads = _clone_grads(self.task.discriminator)
        expected_optim_g = self.task.optim_g.state_dict()
        expected_optim_d = self.task.optim_d.state_dict()
        expected_counters = dict(self.task.counters)
        expected_accu = int(self.task._accu_count)
        self._reload()
        metric.reset()
        trainer = self._trainer(accumulation)
        phases: list[str] = []
        trainer._on_update_phase = lambda phase: phases.append(phase)  # type: ignore[method-assign]
        torch.manual_seed(19)
        with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
            for index, batch in enumerate(batches):
                trainer.consume_micro_batch(_as_batch(batch, index), self.runtime, update_index=index)
        gen_diff = _max_tree(expected_gen, trainer.generator.state_module.generator.state_dict(), "generator")
        disc_diff = _max_tree(
            expected_disc, trainer.discriminator.state_module.discriminator.state_dict(), "discriminator"
        )
        optim_g_diff = _max_tree(expected_optim_g, trainer.optim_g.state_dict(), "optim_g")
        optim_d_diff = _max_tree(expected_optim_d, trainer.optim_d.state_dict(), "optim_d")
        self.assertEqual(gen_diff, 0.0, gen_diff)
        self.assertEqual(disc_diff, 0.0, disc_diff)
        self.assertEqual(optim_g_diff, 0.0, optim_g_diff)
        self.assertEqual(optim_d_diff, 0.0, optim_d_diff)
        self.assertEqual(trainer.counters, expected_counters)
        self.assertEqual(trainer._accu_count, expected_accu)
        self.assertEqual(
            trainer.optimizer_updates["optim_g"],
            trainer.optimizer_updates["optim_d"],
        )
        self.assertEqual(trainer.optimizer_updates["optim_g"], expected_counters["optimizer_updates"])
        self._assert_grads(expected_grads, trainer.generator.state_module)
        self._assert_grads(expected_disc_grads, trainer.discriminator.state_module)
        if expected_counters["optimizer_updates"]:
            self.assertLess(phases.index("after_discriminator_clip"), phases.index("before_generator_step"))
            self.assertLess(phases.index("after_generator_step"), phases.index("before_discriminator_step"))

    def _assert_grads(self, expected: list[torch.Tensor | None], module: torch.nn.Module) -> None:
        live = list(module.parameters())
        self.assertEqual(len(expected), len(live))
        for index, (left, parameter) in enumerate(zip(expected, live, strict=True)):
            if left is None or parameter.grad is None:
                self.assertIsNone(left, index)
                self.assertIsNone(parameter.grad, index)
                continue
            self.assertEqual(_max_tree(left, parameter.grad, f"grad{index}"), 0.0)

    def test_component_losses_match_in_eval(self) -> None:
        batch = _noise(0)
        self.task.generator.eval()
        self.task.discriminator.eval()
        self.generator.state_module.eval()
        self.discriminator.state_module.eval()
        torch.manual_seed(3)
        old = MossFormerGANTask._generator_terms(
            self.task, self.task.generator, self.task.discriminator, batch["clean"], batch["noisy"]
        )
        trainer = self._trainer(1)
        torch.manual_seed(3)
        new = MossFormerGANTask._generator_terms(
            trainer._terms_view(),
            self.generator.state_module,
            self.discriminator.state_module,
            batch["clean"],
            batch["noisy"],
        )
        for name in ("ri", "magnitude", "time", "gan", "generator"):
            self.assertEqual(float(old[name].detach()), float(new[name].detach()), name)

    def test_accumulation_one_and_two_match(self) -> None:
        self._exercise([_noise(0)], 1, set())
        self._exercise([_noise(0), _noise(1)], 2, set())

    def test_skip_patterns_match(self) -> None:
        self._exercise([_noise(0), _noise(1), _noise(2)], 2, {2})
        self._exercise([_noise(3), _noise(4), _noise(5)], 2, {1})
        self._exercise([_noise(6), _noise(7)], 2, {1, 2})
        nan = _noise(8)
        nan["noisy"] = torch.full_like(nan["noisy"], float("nan"))
        self._exercise([nan, _noise(9), _noise(10)], 2, set())

    def test_real_pesq_update_matches(self) -> None:
        batch = {"clean": _tone(PESQ_LENGTH, 180), "noisy": _tone(PESQ_LENGTH, 90)}
        self._reload()
        torch.manual_seed(19)
        self.task.train_update([batch], 1)
        self.assertEqual(self.task.counters["optimizer_updates"], 1)
        expected_gen = {key: value.detach().cpu().clone() for key, value in self.task.generator.state_dict().items()}
        expected_disc = {
            key: value.detach().cpu().clone() for key, value in self.task.discriminator.state_dict().items()
        }
        self._reload()
        trainer = self._trainer(1)
        torch.manual_seed(19)
        trainer.consume_micro_batch(_as_batch(batch, 0), self.runtime, update_index=0)
        self.assertEqual(trainer.counters["optimizer_updates"], 1)
        self.assertEqual(_max_tree(expected_gen, trainer.generator.state_module.generator.state_dict()), 0.0)
        self.assertEqual(_max_tree(expected_disc, trainer.discriminator.state_module.discriminator.state_dict()), 0.0)

    def test_validation_keeps_failures_in_the_denominator(self) -> None:
        included = {
            "alignment_pad": False,
            "generator_loss_included": True,
            "losses": {"generator": {"status": "ok", "value": 2.0}},
        }
        missing = {
            "alignment_pad": False,
            "generator_loss_included": False,
            "losses": {"generator": {"status": "na", "value": None}},
        }
        alignment = {"alignment_pad": True, "generator_loss_included": False, "losses": {}}
        totals = local_validation_totals([included, missing, alignment])
        self.assertEqual(totals["batches"], 2.0)
        self.assertEqual(totals["failed"], 1.0)
        self.assertEqual(totals["included"], 1.0)
        self.assertEqual(totals["alignment"], 1.0)
        self.assertEqual(totals["mean"], 1.0)
        self.assertEqual(official_validation_mean([1.0, 3.0]), 2.0)
        batch = _noise(11)
        metric = _metric_script(set())
        trainer = self._trainer(1)
        with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
            torch.manual_seed(5)
            old = self.task.validation_report(batch)
            metric.reset()
            torch.manual_seed(5)
            new = MossFormerGANTask.validation_report(
                trainer._validation_proxy(self.runtime),
                {"clean": batch["clean"], "noisy": batch["noisy"]},
            )
        self.assertEqual(old["losses"]["generator"]["status"], "ok")
        self.assertEqual(old["losses"]["generator"]["value"], new["losses"]["generator"]["value"])
        self.assertEqual(old["generator_loss_included"], True)
        metric_fail = _metric_script({1})
        with patch("src.experiments.official.mossformergan.reference.metric_targets", metric_fail):
            failed = self.task.validation_report(batch)
        self.assertEqual(failed["losses"]["generator"]["status"], "na")
        self.assertIsNone(failed["losses"]["generator"]["value"])
        self.assertFalse(failed["generator_loss_included"])

    def test_plateau_restore_and_stop_match_including_resume(self) -> None:
        values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        trainer = self._trainer(1)

        def apply_both(value: float) -> None:
            summary = {"losses": {"generator": {"status": "ok", "value": value}}}
            self.task.step_schedulers(summary)
            trainer._apply_plateau(summary)

        apply_both(values[0])
        with torch.no_grad():
            next(self.task.generator.parameters()).add_(0.25)
            next(self.generator.state_module.parameters()).add_(0.25)
        for value in values[1:5]:
            apply_both(value)
        self.assertEqual(self.task._stalls, 4)
        self.assertEqual(trainer._stalls, 4)
        saved = {
            "disc": {key: value.detach().cpu().clone() for key, value in self.discriminator.state_module.state_dict().items()},
            "gen": {key: value.detach().cpu().clone() for key, value in self.generator.state_module.state_dict().items()},
            "optim_d": self.optim_d.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "task_disc": {key: value.detach().cpu().clone() for key, value in self.task.discriminator.state_dict().items()},
            "task_gen": {key: value.detach().cpu().clone() for key, value in self.task.generator.state_dict().items()},
            "task_optim_d": self.task.optim_d.state_dict(),
            "task_optim_g": self.task.optim_g.state_dict(),
            "trainer": trainer.state_dict(),
        }
        apply_both(values[5])
        self.assertEqual(self.task._stalls, 5)
        self.assertEqual(trainer._stalls, 5)
        self.assertEqual(trainer.optim_g.param_groups[0]["lr"], self.task.optim_g.param_groups[0]["lr"])
        self.assertEqual(trainer.optim_d.param_groups[0]["lr"], 1e-3)
        self.assertEqual(self.task.optim_d.param_groups[0]["lr"], 1e-3)
        self.assertAlmostEqual(trainer.optim_g.param_groups[0]["lr"], 2.5e-4)
        halted_gen = {key: value.detach().cpu().clone() for key, value in self.generator.state_module.state_dict().items()}
        self.generator.state_module.load_state_dict(saved["gen"])
        self.discriminator.state_module.load_state_dict(saved["disc"])
        self.optim_g.load_state_dict(saved["optim_g"])
        self.optim_d.load_state_dict(saved["optim_d"])
        resumed = self._trainer(1)
        resumed.load_state_dict(saved["trainer"])
        resumed._apply_plateau({"losses": {"generator": {"status": "ok", "value": values[5]}}})
        self.assertEqual(_max_tree(halted_gen, resumed.generator.state_module.state_dict()), 0.0)
        self.assertEqual(resumed.optim_g.param_groups[0]["lr"], trainer.optim_g.param_groups[0]["lr"])
        self.assertEqual(resumed._stalls, 5)
        for value in values[6:10]:
            apply_both(value)
        self.assertEqual(trainer._stalls, 9)
        self.assertFalse(trainer.stop_training)
        ninth = trainer.state_dict()
        ninth_gen = {key: value.detach().cpu().clone() for key, value in self.generator.state_module.state_dict().items()}
        ninth_optim_g = self.optim_g.state_dict()
        ninth_optim_d = self.optim_d.state_dict()
        apply_both(values[10])
        self.assertTrue(trainer.stop_training)
        self.assertTrue(self.task.stop_training)
        self.assertEqual(trainer._stalls, 10)
        self.assertEqual(trainer.optim_g.param_groups[0]["lr"], self.task.optim_g.param_groups[0]["lr"])
        stopped_lr = trainer.optim_g.param_groups[0]["lr"]
        self.generator.state_module.load_state_dict(ninth_gen)
        self.optim_g.load_state_dict(ninth_optim_g)
        self.optim_d.load_state_dict(ninth_optim_d)
        again = trainer
        again.load_state_dict(ninth)
        again._apply_plateau({"losses": {"generator": {"status": "ok", "value": values[10]}}})
        self.assertTrue(again.stop_training)
        self.assertEqual(again._stalls, 10)
        self.assertEqual(again.optim_g.param_groups[0]["lr"], stopped_lr)

    def test_rejected_wiring(self) -> None:
        shared = torch.optim.AdamW(self.generator.parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        shared._experiment_component_id = "optim_d"  # type: ignore[attr-defined]
        with self.assertRaises(BuildError):
            MossFormerGANTrainer(
                self.generator,
                self.discriminator,
                _IdleData(),
                self.optim_g,
                shared,
                max_logical_steps=1,
            )
        research = MossFormerGANTrainer(
            self.generator,
            self.discriminator,
            _IdleData(),
            self.optim_g,
            self.optim_d,
            max_logical_steps=1,
            experiment_claim="research",
        )
        research._check_claim()
        leakage = MossFormerGANTrainer(
            self.generator,
            self.discriminator,
            _IdleData(),
            self.optim_g,
            self.optim_d,
            max_logical_steps=1,
            experiment_claim="leakage_safe",
        )
        with self.assertRaises(RuntimeFailure):
            leakage._check_claim()
        bad = torch.optim.AdamW(self.optim_d.param_groups[0]["params"], lr=1e-3, weight_decay=0.0)
        bad._experiment_component_id = "optim_d"  # type: ignore[attr-defined]
        with self.assertRaises(BuildError):
            MossFormerGANTrainer(
                self.generator,
                self.discriminator,
                _IdleData(),
                self.optim_g,
                bad,
                max_logical_steps=1,
            )

    def test_profile_must_match_the_claim(self) -> None:
        cases = (
            ("official_faithful", "leakage_safe", False, True),
            ("official_faithful", "official_faithful", False, True),
            ("leakage_safe", "official_faithful", False, True),
            ("leakage_safe", "leakage_safe", True, True),
        )
        for claim, profile, uses_test, rejected in cases:
            data = _ProfileData(profile, uses_test)
            trainer = MossFormerGANTrainer(
                self.generator,
                self.discriminator,
                data,
                self.optim_g,
                self.optim_d,
                max_logical_steps=1,
                experiment_claim=claim,
            )
            with self.subTest(claim=claim, profile=profile, uses_test=uses_test):
                if rejected:
                    with self.assertRaises(RuntimeFailure):
                        trainer._check_claim()
                else:
                    trainer._check_claim()
        for claim, profile, uses_test in (
            ("official_faithful", "official_faithful", True),
            ("leakage_safe", "leakage_safe", False),
            ("official_faithful", None, None),
            ("leakage_safe", None, None),
        ):
            trainer = MossFormerGANTrainer(
                self.generator,
                self.discriminator,
                _ProfileData(profile, uses_test),
                self.optim_g,
                self.optim_d,
                max_logical_steps=1,
                experiment_claim=claim,
            )
            with self.subTest(claim=claim, profile=profile, uses_test=uses_test, accepted=True):
                trainer._check_claim()

    def test_leakage_safe_update_matches_the_official_update(self) -> None:
        batches = [_as_batch(_noise(index), index) for index in range(2)]
        metric = _metric_script(set())
        torch_state = torch.get_rng_state().clone()

        def _once(claim: str) -> dict[str, object]:
            torch.set_rng_state(torch_state.clone())
            metric.reset()
            self._reload()
            trainer = self._trainer(2, experiment_claim=claim)
            captured: dict[str, object] = {}
            original = trainer._step_pair

            def _capture(runtime: ProcessRuntime) -> dict[str, object]:
                captured["generator_grads"] = _clone_grads(self.generator.state_module)
                captured["discriminator_grads"] = _clone_grads(self.discriminator.state_module)
                return original(runtime)

            trainer._step_pair = _capture  # type: ignore[method-assign]
            result = None
            with patch("src.experiments.official.mossformergan.reference.metric_targets", metric):
                for index, batch in enumerate(batches):
                    result = trainer.consume_micro_batch(batch, self.runtime, update_index=index)
            self.assertIsNotNone(result)
            captured["log"] = result
            captured["generator"] = {
                key: value.detach().cpu().clone()
                for key, value in self.generator.state_module.state_dict().items()
            }
            captured["discriminator"] = {
                key: value.detach().cpu().clone()
                for key, value in self.discriminator.state_module.state_dict().items()
            }
            captured["optim_g"] = self.optim_g.state_dict()
            captured["optim_d"] = self.optim_d.state_dict()
            return captured

        official = _once("official_faithful")
        leakage = _once("leakage_safe")
        self.assertEqual(_max_tree(official["log"], leakage["log"]), 0.0)
        self.assertEqual(_max_tree(official["generator_grads"], leakage["generator_grads"]), 0.0)
        self.assertEqual(_max_tree(official["discriminator_grads"], leakage["discriminator_grads"]), 0.0)
        self.assertEqual(_max_tree(official["generator"], leakage["generator"]), 0.0)
        self.assertEqual(_max_tree(official["discriminator"], leakage["discriminator"]), 0.0)
        self.assertEqual(_max_tree(official["optim_g"], leakage["optim_g"]), 0.0)
        self.assertEqual(_max_tree(official["optim_d"], leakage["optim_d"]), 0.0)

    def test_amp_and_mid_pair_checkpoint_are_refused(self) -> None:
        trainer = self._trainer(1)
        with self.assertRaises(RuntimeFailure):
            trainer._refuse_amp(type("Runtime", (), {"precision": "amp_fp16"})())
        trainer._in_optimizer_pair = True
        with self.assertRaises(RuntimeFailure):
            trainer._offer(self.runtime, _CheckpointStub(), "step")
        trainer._in_optimizer_pair = False

    def test_partial_resume_and_all_fail_stop(self) -> None:
        def _constant(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor:
            del enhanced, sample_rate
            return torch.full((clean.shape[0],), 0.4)

        with tempfile.TemporaryDirectory(prefix="moss-partial-") as raw, patch(
            "src.experiments.official.mossformergan.reference.metric_targets", _constant
        ):
            root = Path(raw)
            first_dir = root / "clean"
            first_dir.mkdir()
            clean = build_experiment(load_experiment(_yaml(first_dir, partial=True, steps=1, updates=4, accumulation=2)))
            try:
                result = clean.run()
            finally:
                clean.close()
            self.assertEqual(result.stop_reason, "completed")
            self.assertEqual(result.logical_step, 1)
            failed_dir = root / "failed"
            failed_dir.mkdir()
            failed = build_experiment(load_experiment(_yaml(failed_dir, partial=True, steps=1, updates=4, accumulation=2)))
            failed.get("trainer").fail_phase = "after_generator_step"
            with self.assertRaises(RuntimeFailure):
                failed.run()
            self.assertTrue((failed_dir / "archive/step_00000000.pt").is_file())
            self.assertFalse((failed_dir / "archive/step_00000001.pt").is_file())
            resumed = build_experiment(load_experiment(_yaml(failed_dir, partial=True, steps=1, updates=4, accumulation=2)))
            try:
                resumed_result = resumed.run(resume=failed_dir / "archive/step_00000000.pt")
            finally:
                resumed.close()
            self.assertEqual(resumed_result.logical_step, 1)
            clean_state = _load(first_dir / "archive/step_00000001.pt")["states"]
            resumed_state = _load(failed_dir / "archive/step_00000001.pt")["states"]
            for key in ("generator", "discriminator", "optim_g", "optim_d"):
                self.assertEqual(_max_tree(clean_state[key], resumed_state[key], key), 0.0)
            self.assertEqual(resumed_state["trainer"]["logical_step"], 1)
            self.assertEqual(resumed_state["trainer"]["accu_count"], 0)
            self.assertEqual(resumed_state["data"]["update_index"], 2)
            broken = failed_dir / "archive/step_00000001.pt"
            broken.write_bytes(b"not-a-checkpoint")
            again = build_experiment(load_experiment(_yaml(failed_dir, partial=True, steps=1, updates=4, accumulation=2)))
            with self.assertRaises(CheckpointError):
                again.run(resume=broken)
            again.close()

        with tempfile.TemporaryDirectory(prefix="moss-fail-") as raw:
            directory = Path(raw)
            config = _yaml(directory, partial=False, steps=5, updates=2, accumulation=2, validate=False)
            experiment = build_experiment(load_experiment(config))

            def _none(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> None:
                del clean, enhanced, sample_rate
                return None

            with patch("src.experiments.official.mossformergan.reference.metric_targets", _none):
                try:
                    result = experiment.run()
                finally:
                    experiment.close()
            self.assertEqual(result.stop_reason, "completed")
            self.assertEqual(result.logical_step, 0)
            payload = _load(directory / "archive/step_00000000.pt")
            trainer_state = payload["states"]["trainer"]
            self.assertEqual(trainer_state["halt_reason"], "no_successful_micro_batch")
            self.assertEqual(trainer_state["counters"]["micro_batches_failed"], 2)
            self.assertEqual(trainer_state["counters"]["micro_batches_succeeded"], 0)
            self.assertEqual(trainer_state["counters"]["official_step"], 2)
            self.assertEqual(payload["states"]["data"]["update_index"], 0)
            self.assertEqual(payload["states"]["data"]["epoch"], 1)
            self.assertEqual(trainer_state["logical_step"], 0)

    def test_committed_yaml_declares_the_official_algorithm(self) -> None:
        config = load_experiment(ROOT / "tests/fixtures/configs/experiments/mossformergan_se_16k.yaml")
        components = config.components
        self.assertEqual(components["generator"].class_path, "src.experiments.official.mossformergan.models.MossFormerGANGenerator")
        self.assertEqual(
            components["discriminator"].class_path,
            "src.experiments.official.mossformergan.models.MossFormerGANDiscriminator",
        )
        self.assertEqual(components["trainer"].class_path, "src.experiments.official.mossformergan.trainer.MossFormerGANTrainer")
        self.assertEqual(components["optim_g"].kwargs["lr"], 0.0005)
        self.assertEqual(components["optim_d"].kwargs["lr"], 0.001)
        self.assertEqual(components["optim_g"].kwargs["weight_decay"], 0.01)
        self.assertEqual(components["trainer"].kwargs["accumulation"], 2)
        self.assertEqual(components["trainer"].kwargs["max_logical_steps"], 1)
        self.assertEqual(components["trainer"].kwargs["grad_clip"], 10.0)
        self.assertTrue(components["generator"].prepare.sync_batchnorm)
        self.assertTrue(components["generator"].prepare.find_unused_parameters)
        self.assertEqual(components["generator"].prepare.distribute, "ddp")
        self.assertEqual(components["batcher"].kwargs["group_size"], 2)
        protocol = (ROOT / "configs/data/protocols/mossformergan_se_16k.fixed_pair.yaml").read_text(encoding="utf-8")
        self.assertIn("rank_mod_fixed_v1", protocol)
        self.assertNotIn("scheduler", components)

    def test_cpu_gloo_increment_update_and_skip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="moss-gloo-") as raw:
            root = Path(raw)
            increment = root / "increment"
            increment.mkdir()
            completed = _torchrun(increment, "increment", 180)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            ranks = [
                json.loads((increment / "increment-rank0.json").read_text(encoding="utf-8")),
                json.loads((increment / "increment-rank1.json").read_text(encoding="utf-8")),
            ]
            self.assertTrue(all(row["differ"] for row in ranks))
            self.assertTrue(all(row["ranks_agree"] for row in ranks))
            self.assertEqual(ranks[0]["each"], ranks[1]["each"])

            trained = root / "trained"
            trained.mkdir()
            _yaml(
                trained,
                partial=False,
                steps=1,
                updates=1,
                accumulation=1,
                validate=True,
                distribute="ddp",
                timeout=300,
            )
            run = _torchrun(trained, "run", 600)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            trained_ranks = [
                json.loads((trained / "worker-rank0.json").read_text(encoding="utf-8")),
                json.loads((trained / "worker-rank1.json").read_text(encoding="utf-8")),
            ]
            self.assertEqual([row["logical_step"] for row in trained_ranks], [1, 1])
            self.assertEqual([row["counters"]["optimizer_updates"] for row in trained_ranks], [1, 1])
            self.assertGreaterEqual(trained_ranks[0]["average_calls"], 2)
            payload = _load(trained / "archive/step_00000001.pt")
            self.assertEqual(payload["world_size"], 2)
            self.assertTrue(payload["parameter_ranks_agree"])
            self.assertEqual(payload["states"]["trainer"]["plateau_history"][-1]["action"], "improve")

            skipped = root / "skipped"
            skipped.mkdir()
            _yaml(
                skipped,
                partial=False,
                steps=4,
                updates=2,
                accumulation=2,
                validate=False,
                distribute="ddp",
                timeout=300,
            )
            skip = _torchrun(skipped, "skip", 600)
            self.assertEqual(skip.returncode, 0, skip.stderr + skip.stdout)
            skip_ranks = [
                json.loads((skipped / "worker-rank0.json").read_text(encoding="utf-8")),
                json.loads((skipped / "worker-rank1.json").read_text(encoding="utf-8")),
            ]
            self.assertEqual([row["logical_step"] for row in skip_ranks], [0, 0])
            self.assertEqual([row["halt_reason"] for row in skip_ranks], ["no_successful_micro_batch"] * 2)
            self.assertEqual([row["counters"]["micro_batches_failed"] for row in skip_ranks], [2, 2])
            self.assertEqual([row["average_calls"] for row in skip_ranks], [0, 0])


class _ProfileData:
    """Identity-only data. It does not supply batches."""

    group_size = 1
    epoch = 0
    update_index = 0

    def __init__(self, profile: str | None, selection_uses_test: bool | None) -> None:
        self.profile = profile
        self.selection_uses_test = selection_uses_test

    def train_epoch(self, epoch: int, *, start_update: int):
        del epoch, start_update
        raise AssertionError("profile data has no epoch")

    def validation_epoch(self, epoch: int):
        del epoch
        return iter(())

    def identity(self) -> dict[str, object]:
        return {
            "algorithm": "fixed_pair.mossformergan_se_16k",
            "profile": self.profile,
            "sample_rate": 16000,
            "selection_uses_test": self.selection_uses_test,
            "teacher": "dry_clean",
        }


class _IdleData:
    group_size = 1
    epoch = 0
    update_index = 0

    def train_epoch(self, epoch: int, *, start_update: int):
        del epoch, start_update
        raise AssertionError("idle data has no epoch")

    def validation_epoch(self, epoch: int):
        del epoch
        return iter(())

    def identity(self) -> dict[str, object]:
        return {
            "algorithm": "fixed_pair.mossformergan_se_16k",
            "profile": "official_faithful",
            "sample_rate": 16000,
            "selection_uses_test": True,
            "teacher": "dry_clean",
        }


class _CheckpointStub:
    directory = Path("/tmp")

    def offer(self, event: str, *, logical_step: int, metrics: object = None, prepare: object = None) -> None:
        del event, logical_step, metrics, prepare
        raise AssertionError("checkpoint offer ran inside an optimizer pair")


if __name__ == "__main__":
    unittest.main()
