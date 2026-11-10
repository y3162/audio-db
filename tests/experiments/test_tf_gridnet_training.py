"""Phase 4-B: TF-GridNet MS-SNSD training on the experiment framework.

The objective is the existing ``mr_l1_tfd`` function. SI-SDR is not part of
the loss or the checkpoint selection. Epoch-end scheduler behavior is tested
with a fixed-length synthetic batch source. One test drives the real
``SnapshotDataModule`` chunk plan on a synthetic snapshot.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import BuildError, RuntimeFailure
from src.experiments.losses import OFFICIAL_WINDOWS, MultiResolutionL1Loss
from src.experiments.models.tf_gridnet import TFGridNetModel
from src.experiments.runtime import PreparedModel, ProcessRuntime
from src.experiments.tf_gridnet import (
    TFGridNetTrainer,
    _write_nbest,
    early_stop,
    rank_validation_loss,
    weighted_validation_loss,
)
from src.se.models.tf_gridnet.model import TFGridNet
from src.experiments.calc.average import average_state_dicts
from src.experiments.calc.optim import clip_gradients
from src.experiments.calc.spectral import multi_resolution_l1
from src.experiments.calc.tf_gridnet import TFGridNetTask
from tests.experiments.test_chunk_data import _write_snapshot

ROOT = Path(__file__).resolve().parents[2]
TINY = {
    "activation": "prelu",
    "attn_approx_qk_dim": 8,
    "attn_n_head": 1,
    "emb_dim": 4,
    "emb_hs": 1,
    "emb_ks": 1,
    "eps": 1e-5,
    "lstm_hidden_units": 4,
    "n_fft": 16,
    "n_layers": 1,
    "stride": 8,
}


class TFGridNetDeclarationTests(unittest.TestCase):
    def test_committed_yaml_declares_the_official_algorithm(self) -> None:
        config = load_experiment(ROOT / "tests/fixtures/configs/experiments/tf_gridnet_ms_snsd.yaml")
        model = config.components["model"]
        loss = config.components["loss"]
        trainer = config.components["trainer"]
        self.assertEqual(model.class_path, "src.experiments.models.tf_gridnet.TFGridNetModel")
        self.assertEqual(model.kwargs["variant"], "ms-snsd")
        self.assertEqual(model.kwargs["init"], "xavier_uniform")
        self.assertEqual(model.kwargs["architecture"]["n_fft"], 512)
        self.assertEqual(model.kwargs["architecture"]["n_layers"], 4)
        self.assertEqual(loss.class_path, "src.experiments.losses.MultiResolutionL1Loss")
        self.assertEqual(tuple(loss.kwargs["windows"]), OFFICIAL_WINDOWS)
        self.assertEqual(loss.kwargs["time_domain_weight"], 0.5)
        self.assertIs(loss.kwargs["normalize_variance"], False)
        self.assertEqual(config.components["optimizer"].class_path, "torch.optim.Adam")
        self.assertEqual(config.components["optimizer"].kwargs["lr"], 0.001)
        self.assertEqual(config.components["optimizer"].kwargs["weight_decay"], 0.0)
        self.assertEqual(tuple(config.components["optimizer"].kwargs["betas"]), (0.9, 0.999))
        scheduler = config.components["scheduler"]
        self.assertEqual(scheduler.class_path, "torch.optim.lr_scheduler.ReduceLROnPlateau")
        self.assertEqual(scheduler.kwargs["mode"], "min")
        self.assertEqual(scheduler.kwargs["factor"], 0.7)
        self.assertEqual(scheduler.kwargs["patience"], 1)
        self.assertEqual(trainer.class_path, "src.experiments.tf_gridnet.TFGridNetTrainer")
        self.assertEqual(trainer.kwargs["grad_clip"], 5.0)
        self.assertEqual(trainer.kwargs["keep_nbest"], 5)
        self.assertEqual(trainer.kwargs["early_stopping_patience"], 5)
        self.assertEqual(config.components["data"].class_path, "src.experiments.data.SnapshotDataModule")
        self.assertEqual(config.components["sampler"].class_path, "src.experiments.data.ChunkSampler")
        self.assertEqual(config.components["source"].kwargs["snapshot"], "tf_gridnet.ms_snsd.espnet_recipe@1")
        text = (ROOT / "src/experiments/tf_gridnet.py").read_text(encoding="utf-8")
        self.assertNotIn("OnDemandChunkData", text)
        self.assertNotIn("valid.si_snr.best", text)
        self.assertNotIn("valid.si_snr.ave", text)

    def test_official_widths_build_one_ms_snsd_generator(self) -> None:
        config = load_experiment(ROOT / "tests/fixtures/configs/experiments/tf_gridnet_ms_snsd.yaml")
        model = TFGridNetModel(seed=0, architecture=config.components["model"].kwargs["architecture"])
        self.assertIsInstance(model.generator, TFGridNet)
        self.assertEqual(model.n_fft, 512)
        self.assertEqual(model.stride, 256)
        self.assertEqual([parameter for parameter in model.parameters()], [parameter for parameter in model.generator.parameters()])
        exported = model.export_for_inference()
        self.assertTrue(exported)
        self.assertTrue(all(key.startswith("separator.") for key in exported))

    def test_rejects_urgent_variant_and_variance_normalization(self) -> None:
        with self.assertRaises(BuildError):
            TFGridNetModel(seed=0, variant="urgent24", architecture=TINY)
        with self.assertRaises(BuildError):
            MultiResolutionL1Loss(normalize_variance=True)


class TFGridNetConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_build_uses_one_initialized_parameter_set(self) -> None:
        experiment = build_experiment(load_experiment(_scripted_yaml(self.base / "build", steps=1, updates=1)))
        try:
            prepared = experiment.get("model")
            self.assertIsInstance(prepared, PreparedModel)
            model = prepared.state_module
            self.assertIs(prepared.module, model)
            self.assertIsInstance(model, TFGridNetModel)
            self.assertIsInstance(model.generator, TFGridNet)
            generator_ids = [id(parameter) for parameter in model.generator.parameters()]
            module_ids = [id(parameter) for parameter in model.parameters()]
            prepared_ids = [id(parameter) for parameter in prepared.parameters]
            optimizer_ids = [
                id(parameter)
                for group in experiment.get("optimizer").param_groups
                for parameter in group["params"]
            ]
            self.assertEqual(module_ids, generator_ids)
            self.assertEqual(prepared_ids, generator_ids)
            self.assertEqual(optimizer_ids, generator_ids)
            self.assertEqual(len(generator_ids), len(set(generator_ids)))
            task = _task()
            self.assertEqual(_max_abs(model.generator.state_dict(), task.generator.state_dict()), 0.0)
        finally:
            experiment.close()

    def test_forward_loss_and_update_match_the_current_task(self) -> None:
        directory = self.base / "match"
        experiment = build_experiment(load_experiment(_scripted_yaml(directory, steps=1, updates=1, max_epochs=1)))
        model = experiment.get("model").state_module
        task = _task()
        self.assertEqual(_max_abs(model.generator.state_dict(), task.generator.state_dict()), 0.0)
        view = experiment.get("data").train_epoch(0, start_update=0)
        try:
            batch = view.batch(0)[0]
        finally:
            view.close()
        staged_model = TFGridNetModel(seed=0, architecture=TINY)
        staged_model.load_state_dict(model.state_dict())
        staged_task = _task()
        staged_task.generator.load_state_dict(model.generator.state_dict())
        _assert_stage(self, staged_model, staged_task, experiment.get("loss"), batch.inputs, batch.targets)
        reference = _task()
        reference.generator.load_state_dict(model.generator.state_dict())
        reference.train_update(
            [{"clean": batch.targets, "noisy": batch.inputs}],
            1,
        )
        result = experiment.run()
        self.assertEqual(result.logical_step, 1)
        self.assertEqual(result.optimizer_updates, {"optimizer": 1})
        self.assertEqual(_max_abs(model.generator.state_dict(), reference.generator.state_dict()), 0.0)
        self.assertEqual(
            _max_abs(_adam_state(model.generator, experiment.get("optimizer")), _adam_state(reference.generator, reference.optim_g)),
            0.0,
        )
        rows = _metrics(directory)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lr"], 0.001)
        self.assertTrue(torch.isfinite(torch.tensor(rows[0]["loss"])))

    def test_epoch_end_runs_validation_then_scheduler_once(self) -> None:
        directory = self.base / "epoch"
        seen: list[tuple[str, int, float, float | None]] = []

        def record(trainer: TFGridNetTrainer, phase: str) -> None:
            best = trainer.scheduler.state_dict()["best"]
            loss = None if trainer.boundary is None else trainer.boundary.get("validation_loss")
            seen.append((phase, int(trainer.scheduler.last_epoch), float(best), None if loss is None else float(loss)))

        experiment = build_experiment(load_experiment(_scripted_yaml(directory, steps=2, updates=2, max_epochs=1)))
        with patch.object(TFGridNetTrainer, "_on_boundary_phase", record):
            result = experiment.run()
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual([item[0] for item in seen], ["pending_validation", "pending_scheduler", "pending_selection", "in_epoch"])
        self.assertEqual(seen[0][1], 0)
        self.assertTrue(seen[0][2] == float("inf"))
        self.assertIsNone(seen[0][3])
        self.assertEqual(seen[1][1], 0)
        self.assertTrue(seen[1][2] == float("inf"))
        self.assertIsNotNone(seen[1][3])
        self.assertEqual(seen[2][1], 1)
        self.assertEqual(seen[2][2], seen[2][3])
        rows = _metrics(directory)
        self.assertEqual([row["lr"] for row in rows], [0.001, 0.001])
        payload = _load(directory / "step_00000002.pt")
        trainer = payload["states"]["trainer"]
        self.assertEqual(trainer["phase"], "in_epoch")
        self.assertIsNone(trainer["boundary"])
        self.assertEqual(len(trainer["history"]), 1)
        self.assertEqual(trainer["history"][0]["epoch"], 1)
        self.assertEqual(trainer["completed_boundaries"][0]["validation_weight"], 3)
        self.assertTrue(trainer["completed_boundaries"][0]["validation_done"])
        self.assertTrue(trainer["completed_boundaries"][0]["scheduler_updated"])
        self.assertTrue(trainer["completed_boundaries"][0]["selection_done"])
        self.assertEqual(payload["states"]["scheduler"]["last_epoch"], 1)
        self.assertEqual(payload["states"]["scheduler"]["best"], trainer["history"][0]["loss"])
        self.assertFalse((directory / "valid.si_snr.best.pt").exists())
        self.assertTrue((directory / "1epoch.pt").is_file())
        self.assertTrue((directory / "valid.loss.ave_1best.pt").is_symlink())

    def test_mid_epoch_stop_does_not_validate_or_step_the_scheduler(self) -> None:
        for name, stop in (("budget", None), ("interrupt", 1)):
            directory = self.base / f"mid-{name}"
            result_holder = build_experiment(
                load_experiment(_scripted_yaml(directory, steps=1 if stop is None else 2, updates=2, stop_after=stop))
            )
            result = result_holder.run()
            self.assertEqual(result.logical_step, 1)
            payload = _load(directory / "step_00000001.pt")
            self.assertEqual(payload["label"], "step")
            self.assertIsNone(payload["validation_pending"])
            self.assertEqual(payload["states"]["trainer"]["phase"], "in_epoch")
            self.assertEqual(payload["states"]["trainer"]["history"], [])
            self.assertEqual(payload["states"]["scheduler"]["last_epoch"], 0)
            self.assertEqual(payload["states"]["scheduler"]["best"], float("inf"))
            self.assertEqual(list(directory.glob("*epoch.pt")), [])
            if stop is None:
                self.assertEqual(result.stop_reason, "completed")
                self.assertTrue((directory / "complete.json").is_file())
            else:
                self.assertEqual(result.stop_reason, "interrupted")
                self.assertFalse((directory / "complete.json").is_file())

    def test_boundary_resume_matches_a_continuous_epoch(self) -> None:
        continuous = self.base / "continuous"
        interrupted = self.base / "boundary"
        build_experiment(load_experiment(_scripted_yaml(continuous, steps=2, updates=2, max_epochs=1))).run()
        build_experiment(
            load_experiment(_scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1, stop_after=2))
        ).run()
        pending = _load(interrupted / "step_00000002.pt")
        self.assertEqual(pending["label"], "validation_pending")
        self.assertEqual(pending["validation_pending"], {"logical_step": 2})
        self.assertEqual(pending["states"]["trainer"]["phase"], "pending_validation")
        self.assertFalse(pending["states"]["trainer"]["boundary"]["validation_done"])
        self.assertFalse(pending["states"]["trainer"]["boundary"]["scheduler_updated"])
        self.assertFalse(pending["states"]["trainer"]["boundary"]["selection_done"])
        self.assertTrue(pending["states"]["trainer"]["boundary"]["optimizer_updated"])
        self.assertEqual(pending["states"]["scheduler"]["last_epoch"], 0)
        resumed = _cli(
            _scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1),
            "--resume",
            str(interrupted / "step_00000002.pt"),
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        left = _load(continuous / "step_00000002.pt")
        right = _load(interrupted / "step_00000002.pt")
        self.assertEqual(right["label"], "step")
        self.assertIsNone(right["validation_pending"])
        _assert_checkpoint(self, left, right)
        _assert_artifacts(self, continuous, interrupted)
        self.assertEqual(right["states"]["trainer"]["completed_boundaries"][0]["scheduler_updated"], True)
        self.assertEqual(len(right["states"]["trainer"]["history"]), 1)

    def test_mid_epoch_resume_matches_a_continuous_run(self) -> None:
        continuous = self.base / "mid-continuous"
        interrupted = self.base / "mid-resume"
        build_experiment(load_experiment(_scripted_yaml(continuous, steps=2, updates=2, max_epochs=1))).run()
        first = build_experiment(
            load_experiment(_scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1, stop_after=1))
        )
        self.assertEqual(first.run().stop_reason, "interrupted")
        second = build_experiment(load_experiment(_scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1)))
        self.assertEqual(second.run(resume=interrupted / "step_00000001.pt").stop_reason, "completed")
        _assert_checkpoint(self, _load(continuous / "step_00000002.pt"), _load(interrupted / "step_00000002.pt"))
        _assert_artifacts(self, continuous, interrupted)

    def test_weighted_validation_uses_sample_counts(self) -> None:
        loss, weight = weighted_validation_loss([(1.0, 2), (5.0, 1), (3.0, 2)])
        self.assertEqual(weight, 5)
        self.assertEqual(loss, 2.6)
        rank_mean = ((1.0 * 2 + 5.0) / 3 + 3.0) / 2
        self.assertNotEqual(loss, rank_mean)
        directory = self.base / "weights"
        build_experiment(load_experiment(_scripted_yaml(directory, steps=2, updates=2, max_epochs=1))).run()
        trainer = _load(directory / "step_00000002.pt")["states"]["trainer"]
        self.assertEqual(trainer["completed_boundaries"][0]["validation_weight"], 3)

    def test_nbest_rank_average_and_ties(self) -> None:
        history = [
            {"epoch": 4, "loss": 1.0},
            {"epoch": 2, "loss": 1.0},
            {"epoch": 3, "loss": 0.5},
            {"epoch": 1, "loss": 2.0},
        ]
        self.assertEqual([row["epoch"] for row in rank_validation_loss(history, 3)], [3, 2, 4])
        self.assertEqual(rank_validation_loss(history, 1)[0]["epoch"], 3)
        tied = [{"epoch": 4, "loss": 1.0}, {"epoch": 2, "loss": 1.0}]
        self.assertEqual(rank_validation_loss(tied, 1)[0]["epoch"], 2)
        self.assertTrue(early_stop([{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 2.0}], 2, 0))
        self.assertFalse(early_stop([{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 2.0}], 2, 5))
        directory = self.base / "nbest"
        directory.mkdir()
        first = {"separator.count": torch.tensor(3), "separator.weight": torch.ones(2)}
        second = {"separator.count": torch.tensor(4), "separator.weight": torch.ones(2) * 3}
        _write_nbest(directory, first, 1, [{"epoch": 1, "loss": 2.0}], 2, 1)
        self.assertTrue((directory / "valid.loss.ave_1best.pt").is_symlink())
        self.assertEqual(os.readlink(directory / "valid.loss.ave.pt"), "valid.loss.ave_1best.pt")
        self.assertFalse((directory / "valid.loss.ave_2best.pt").exists())
        _write_nbest(
            directory,
            second,
            2,
            rank_validation_loss([{"epoch": 1, "loss": 2.0}, {"epoch": 2, "loss": 1.0}], 2),
            2,
            2,
        )
        averaged = torch.load(directory / "valid.loss.ave_2best.pt", map_location="cpu", weights_only=True)
        expected = average_state_dicts([first, second])
        self.assertEqual(_max_abs(averaged, expected), 0.0)
        self.assertEqual(int(averaged["separator.count"]), 7)
        self.assertEqual(float(averaged["separator.weight"][0]), 2.0)
        self.assertEqual(os.readlink(directory / "valid.loss.ave.pt"), "valid.loss.ave_2best.pt")
        self.assertTrue((directory / "1epoch.pt").is_file())
        self.assertTrue((directory / "2epoch.pt").is_file())

    def test_snapshot_datamodule_completes_one_epoch(self) -> None:
        spec = _write_snapshot(self.base / "snapshots", derived=self.base / "derived", kind="ms", name="phase4b.ms")
        directory = self.base / "snapshot-run"
        experiment = build_experiment(load_experiment(_snapshot_yaml(directory, spec)))
        try:
            self.assertEqual(experiment.get("data").__class__.__name__, "SnapshotDataModule")
            self.assertIsInstance(experiment.get("model").state_module.generator, TFGridNet)
            result = experiment.run()
        except Exception:
            experiment.close()
            raise
        self.assertEqual(result.stop_reason, "completed")
        self.assertGreaterEqual(result.logical_step, 1)
        payload = _load(directory / f"step_{result.logical_step:08d}.pt")
        self.assertEqual(payload["states"]["trainer"]["phase"], "in_epoch")
        self.assertEqual(len(payload["states"]["trainer"]["history"]), 1)
        self.assertEqual(payload["states"]["scheduler"]["last_epoch"], 1)
        self.assertGreater(payload["states"]["trainer"]["completed_boundaries"][0]["validation_weight"], 0)

    def test_cpu_gloo_ranks_share_validation_and_updates(self) -> None:
        directory = self.base / "gloo"
        config = _scripted_yaml(directory, steps=2, updates=2, max_epochs=1, distribute="ddp")
        completed = _torchrun(config)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        payload = _load(directory / "step_00000002.pt")
        self.assertEqual(payload["world_size"], 2)
        self.assertTrue(payload["parameter_ranks_agree"])
        self.assertEqual(payload["states"]["trainer"]["optimizer_updates"], {"optimizer": 2})
        self.assertEqual(payload["states"]["trainer"]["completed_boundaries"][0]["validation_weight"], 5)
        self.assertEqual(payload["states"]["scheduler"]["last_epoch"], 1)
        rank0 = _metrics_file(directory / "samples-rank0.jsonl")
        rank1 = _metrics_file(directory / "samples-rank1.jsonl")
        self.assertEqual(len(rank0), 2)
        self.assertEqual(len(rank1), 2)
        self.assertNotEqual([row["sample_id"] for row in rank0], [row["sample_id"] for row in rank1])

    def test_ddp_prepare_keeps_parameter_identity(self) -> None:
        directory = self.base / "identity"
        directory.mkdir()
        completed = _torchrun_module("tests.experiments.tf_gridnet_identity_worker", directory)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        ranks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("rank*.json"))]
        self.assertEqual(len(ranks), 2)
        for payload in ranks:
            self.assertTrue(payload["match"])
            self.assertTrue(payload["wrapped"])
            self.assertTrue(payload["state_is_unwrapped"])
            self.assertGreater(payload["parameters"], 0)

    def test_rejects_bad_shapes_and_a_step_scheduler(self) -> None:
        model = TFGridNetModel(seed=1, architecture=TINY)
        with self.assertRaises(RuntimeFailure):
            model(torch.randn(2, 4, 8))
        with self.assertRaises(RuntimeFailure):
            model(torch.randn(1, 4))
        directory = self.base / "bad-scheduler"
        text = _scripted_text(directory, steps=1, updates=1).replace(
            "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "torch.optim.lr_scheduler.StepLR",
        ).replace("      mode: min\n      factor: 0.7\n      patience: 1\n", "      step_size: 1\n      gamma: 0.5\n")
        path = directory / "experiment.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(BuildError):
            build_experiment(load_experiment(path)).close()


def _yaml_architecture(architecture: dict[str, object]) -> str:
    text = json.dumps(architecture)
    return text.replace("1e-05", "1.0e-5")


def _task() -> TFGridNetTask:
    return TFGridNetTask(
        {
            "grad_clip": 5.0,
            "loss": "official",
            "model": {"architecture": TINY, "init": "xavier_uniform", "sampling_rate": 16000, "variant": "ms-snsd"},
            "optimizer": {"betas": [0.9, 0.999], "lr": 0.001, "name": "adam", "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "seed": 0,
        },
        torch.device("cpu"),
    )


def _assert_stage(test: unittest.TestCase, model: TFGridNetModel, task: TFGridNetTask, loss: MultiResolutionL1Loss, noisy: torch.Tensor, clean: torch.Tensor) -> None:
    model.train()
    task.train_mode()
    estimate = model(noisy)
    reference = task._enhance(task.generator, noisy)
    test.assertEqual(_max_abs(estimate, reference), 0.0)
    left = loss(estimate, clean)
    right = multi_resolution_l1(clean, reference, OFFICIAL_WINDOWS, time_domain_weight=0.5, normalize_variance=False)
    test.assertEqual(_max_abs(left, right), 0.0)
    left.backward()
    right.backward()
    test.assertEqual(_max_abs(_grads(model.generator), _grads(task.generator)), 0.0)
    test.assertTrue(any(float(value.abs().sum()) > 0 for value in _grads(model.generator).values()))
    runtime = ProcessRuntime(device="cpu", precision="fp32")
    try:
        runtime.clip(model.parameters(), 5.0)
        clip_gradients(task.generator, 5.0)
        test.assertEqual(_max_abs(_grads(model.generator), _grads(task.generator)), 0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
        applied = runtime.step(optimizer)
        runtime.end_step()
        task.optim_g.step()
        test.assertTrue(applied.applied)
        test.assertEqual(_max_abs(model.generator.state_dict(), task.generator.state_dict()), 0.0)
        test.assertEqual(
            _max_abs(_adam_state(model.generator, optimizer), _adam_state(task.generator, task.optim_g)),
            0.0,
        )
    finally:
        runtime.shutdown()


def _grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.grad.detach() for name, parameter in module.named_parameters() if parameter.grad is not None}


def _adam_state(module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, dict[str, object]]:
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    return {names[id(parameter)]: dict(state) for parameter, state in optimizer.state.items()}


def _scripted_yaml(
    directory: Path,
    *,
    steps: int,
    updates: int,
    max_epochs: int | None = None,
    stop_after: int | None = None,
    distribute: str = "none",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(
        _scripted_text(directory, steps=steps, updates=updates, max_epochs=max_epochs, stop_after=stop_after, distribute=distribute),
        encoding="utf-8",
    )
    yaml.safe_load(path.read_text(encoding="utf-8"))
    return path


def _scripted_text(
    directory: Path,
    *,
    steps: int,
    updates: int,
    max_epochs: int | None = None,
    stop_after: int | None = None,
    distribute: str = "none",
) -> str:
    epochs = "null" if max_epochs is None else str(max_epochs)
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    return f"""schema: audio-db-experiment/v1
seed: 0
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
    timeout_seconds: 60
output:
  directory: {directory}
components:
  model:
    role: module
    class: src.experiments.models.tf_gridnet.TFGridNetModel
    kwargs:
      seed: 0
      variant: ms-snsd
      init: xavier_uniform
      architecture: {_yaml_architecture(TINY)}
    prepare: {{trainable: true, distribute: {distribute}}}
  loss:
    role: loss
    class: src.experiments.losses.MultiResolutionL1Loss
    kwargs:
      windows: [256, 512, 768, 1024]
      time_domain_weight: 0.5
      normalize_variance: false
  data:
    role: data
    class: tests.experiments.tf_gridnet_data.ScriptedChunkData
    kwargs:
      updates_per_epoch: {updates}
      micro_batch: 1
      length: 32
      seed: 17
      valid_sizes: [[2, 1], [2]]
  optimizer:
    role: optimizer
    class: torch.optim.Adam
    parameters_of: model
    kwargs:
      lr: 0.001
      betas: [0.9, 0.999]
      weight_decay: 0.0
  scheduler:
    role: scheduler
    class: torch.optim.lr_scheduler.ReduceLROnPlateau
    kwargs:
      mode: min
      factor: 0.7
      patience: 1
    inject:
      optimizer: {{ref: optimizer}}
  trainer:
    role: trainer
    class: src.experiments.tf_gridnet.TFGridNetTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: 1
      grad_clip: 5.0
      keep_nbest: 5
      max_epochs: {epochs}
      checkpoint_interval: 1
{stop}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
      scheduler: {{ref: scheduler}}
"""


def _snapshot_yaml(directory: Path, spec: dict[str, object]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(
        f"""schema: audio-db-experiment/v1
seed: 0
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
output:
  directory: {directory}
components:
  model:
    role: module
    class: src.experiments.models.tf_gridnet.TFGridNetModel
    kwargs:
      seed: 0
      variant: ms-snsd
      init: xavier_uniform
      architecture: {_yaml_architecture(TINY)}
    prepare: {{trainable: true, distribute: none}}
  loss:
    role: loss
    class: src.experiments.losses.MultiResolutionL1Loss
    kwargs:
      windows: [256, 512, 768, 1024]
      time_domain_weight: 0.5
      normalize_variance: false
  source:
    role: other
    class: src.experiments.data.SnapshotSource
    kwargs:
      snapshot: {spec["selector"]}
      snapshot_root: {spec["root"]}
      split: train
      derived: {spec["derived"]}
      corpora: {spec["corpora"]}
      generated: {spec["generated"]}
  sampler:
    role: other
    class: src.experiments.data.ChunkSampler
    kwargs:
      seed: 0
      num_samples_per_epoch: 2
      chunk_length: 32
      shift_ratio: 0.5
      num_cache_chunks: 2
      shuffle: false
    inject:
      source: {{ref: source}}
  protocol:
    role: other
    class: src.experiments.data.ChunkProtocol
    kwargs: {{}}
    inject:
      source: {{ref: source}}
  batcher:
    role: other
    class: src.experiments.data.MicroBatcher
    kwargs:
      micro_batch: 1
      group_size: 1
  data:
    role: data
    class: src.experiments.data.SnapshotDataModule
    kwargs:
      prefetch: false
    inject:
      source: {{ref: source}}
      sampler: {{ref: sampler}}
      protocol: {{ref: protocol}}
      batcher: {{ref: batcher}}
  optimizer:
    role: optimizer
    class: torch.optim.Adam
    parameters_of: model
    kwargs:
      lr: 0.001
      betas: [0.9, 0.999]
      weight_decay: 0.0
  scheduler:
    role: scheduler
    class: torch.optim.lr_scheduler.ReduceLROnPlateau
    kwargs:
      mode: min
      factor: 0.7
      patience: 1
    inject:
      optimizer: {{ref: optimizer}}
  trainer:
    role: trainer
    class: src.experiments.tf_gridnet.TFGridNetTrainer
    kwargs:
      max_logical_steps: 200
      accumulation: 1
      grad_clip: 5.0
      keep_nbest: 5
      max_epochs: 1
      checkpoint_interval: 1
    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
      scheduler: {{ref: scheduler}}
""",
        encoding="utf-8",
    )
    yaml.safe_load(path.read_text(encoding="utf-8"))
    return path


def _load(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(path.name)
    return payload


def _metrics(directory: Path) -> list[dict[str, object]]:
    return _metrics_file(directory / "metrics.jsonl")


def _metrics_file(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _assert_artifacts(test: unittest.TestCase, left: Path, right: Path) -> None:
    _assert_equal(_metrics(left), _metrics(right), "metrics")
    for name in ("1epoch.pt", "valid.loss.best.pt"):
        _assert_equal(
            torch.load(left / name, map_location="cpu", weights_only=True),
            torch.load(right / name, map_location="cpu", weights_only=True),
            name,
        )
    for name in ("valid.loss.ave_1best.pt", "valid.loss.ave.pt"):
        test.assertEqual(os.readlink(left / name), os.readlink(right / name))


def _assert_checkpoint(test: unittest.TestCase, left: dict[str, object], right: dict[str, object]) -> None:
    for key in ("states", "rng_by_rank", "logical_step", "optimizer_updates", "label"):
        _assert_equal(left[key], right[key], key)


def _assert_equal(left: object, right: object, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(path)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(path)
        for key, value in left.items():
            _assert_equal(value, right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError(path)
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            _assert_equal(lhs, rhs, f"{path}[{index}]")
        return
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError(path)
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def _max_abs(left: object, right: object) -> float:
    values: list[float] = []

    def walk(lhs: object, rhs: object) -> None:
        if isinstance(lhs, torch.Tensor):
            if not isinstance(rhs, torch.Tensor):
                raise AssertionError(type(rhs))
            values.append(float((lhs.detach().cpu() - rhs.detach().cpu()).abs().max()))
            return
        if isinstance(lhs, dict):
            if not isinstance(rhs, dict) or set(lhs) != set(rhs):
                raise AssertionError(set(lhs) ^ set(rhs))
            for key, value in lhs.items():
                walk(value, rhs[key])
            return
        if lhs != rhs:
            raise AssertionError(f"{lhs!r} != {rhs!r}")

    walk(left, right)
    return max(values) if values else 0.0


def _cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _torchrun(config: Path) -> subprocess.CompletedProcess[str]:
    return _torchrun_command(
        [
            "-m",
            "src.experiments.cli",
            "--config",
            str(config),
        ]
    )


def _torchrun_module(module: str, directory: Path) -> subprocess.CompletedProcess[str]:
    return _torchrun_command(["-m", module, str(directory)])


def _torchrun_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
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
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        "--master_port",
        str(_free_port()),
        *arguments,
    ]
    return subprocess.run(command, cwd=ROOT, env=env, check=False, capture_output=True, text=True, timeout=180)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
