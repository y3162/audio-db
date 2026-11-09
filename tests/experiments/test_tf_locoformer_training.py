"""Phase 4-C: TF-Locoformer DNS training on the experiment framework.

The objective is the existing ``mr_l1_tfd`` function. The scheduler is the
existing ``WarmupReduceLROnPlateau``. Epoch-end behavior is tested with a
fixed-length synthetic batch source. One test drives the real
``SnapshotDataModule`` chunk plan on a synthetic DNS snapshot.
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
from src.experiments.models.tf_locoformer import TFLocoformerModel
from src.experiments.runtime import PreparedModel, ProcessRuntime, StepResult
from src.experiments.tf_gridnet import early_stop, rank_validation_loss
from src.experiments.tf_locoformer import (
    RESEARCH_VALIDATION_PROFILE,
    TFLocoformerTrainer,
    _write_selection,
    per_sample_si_snr,
    rank_validation_si_snr,
)
from src.se.models.tf_locoformer.inference import load as load_enhancer_checkpoint
from src.se.models.tf_locoformer.model import TFLocoformer
from src.training.core.average import average_state_dicts
from src.training.se.optim import WarmupReduceLROnPlateau, clip_gradients
from src.training.se.spectral import multi_resolution_l1, si_sdr_loss
from src.training.se.tf_locoformer import TFLocoformerTask
from tests.experiments.test_chunk_data import _write_snapshot
from tests.experiments.tf_gridnet_data import ScriptedChunkData

ROOT = Path(__file__).resolve().parents[2]
WARMUP = 4
TINY = {
    "attention_dim": 4,
    "conv1d_kernel": 4,
    "conv1d_shift": 1,
    "dropout": 0.0,
    "emb_dim": 4,
    "eps": 1e-5,
    "ffn_hidden_dim": [8, 8],
    "ffn_type": ["swiglu_conv1d", "swiglu_conv1d"],
    "flash_attention": False,
    "hop_length": 8,
    "n_fft": 16,
    "n_heads": 1,
    "n_layers": 1,
    "norm_type": "rmsgroupnorm",
    "normalize_variance": True,
    "num_groups": 4,
    "num_spk": 1,
    "pos_enc": "rope",
    "tf_order": "ft",
}


class TFLocoformerDeclarationTests(unittest.TestCase):
    def test_committed_yaml_declares_the_dns_algorithm(self) -> None:
        config = load_experiment(ROOT / "configs/experiments/tf_locoformer_dns.yaml")
        model = config.components["model"]
        loss = config.components["loss"]
        trainer = config.components["trainer"]
        self.assertEqual(model.class_path, "src.experiments.models.tf_locoformer.TFLocoformerModel")
        self.assertEqual(model.kwargs["variant"], "dns")
        self.assertEqual(model.kwargs["init"], "xavier_uniform")
        architecture = model.kwargs["architecture"]
        self.assertEqual(architecture["n_fft"], 256)
        self.assertEqual(architecture["hop_length"], 128)
        self.assertIs(architecture["normalize_variance"], True)
        self.assertEqual(architecture["n_layers"], 6)
        self.assertEqual(architecture["emb_dim"], 128)
        self.assertEqual(architecture["attention_dim"], 128)
        self.assertEqual(architecture["ffn_hidden_dim"], [384, 384])
        self.assertEqual(loss.class_path, "src.experiments.losses.MultiResolutionL1Loss")
        self.assertEqual(tuple(loss.kwargs["windows"]), OFFICIAL_WINDOWS)
        self.assertEqual(loss.kwargs["time_domain_weight"], 0.5)
        self.assertIs(loss.kwargs["normalize_variance"], False)
        optimizer = config.components["optimizer"]
        self.assertEqual(optimizer.class_path, "torch.optim.AdamW")
        self.assertEqual(optimizer.kwargs["lr"], 0.001)
        self.assertEqual(optimizer.kwargs["weight_decay"], 0.01)
        self.assertEqual(tuple(optimizer.kwargs["betas"]), (0.9, 0.999))
        scheduler = config.components["scheduler"]
        self.assertEqual(scheduler.class_path, "src.training.se.optim.WarmupReduceLROnPlateau")
        self.assertEqual(scheduler.kwargs["warmup_steps"], 4000)
        self.assertEqual(scheduler.kwargs["mode"], "min")
        self.assertEqual(scheduler.kwargs["factor"], 0.5)
        self.assertEqual(scheduler.kwargs["patience"], 3)
        self.assertEqual(scheduler.kwargs["threshold"], 1e-4)
        self.assertEqual(scheduler.kwargs["threshold_mode"], "rel")
        self.assertEqual(scheduler.kwargs["cooldown"], 0)
        self.assertEqual(scheduler.kwargs["min_lr"], 0.0)
        self.assertEqual(scheduler.kwargs["eps"], 1e-8)
        self.assertEqual(trainer.class_path, "src.experiments.tf_locoformer.TFLocoformerTrainer")
        self.assertEqual(trainer.kwargs["grad_clip"], 5.0)
        self.assertEqual(trainer.kwargs["keep_nbest"], 5)
        self.assertEqual(trainer.kwargs["early_stopping_patience"], 10)
        self.assertEqual(trainer.kwargs["max_logical_steps"], 1)
        self.assertEqual(trainer.kwargs["validation_profile"], RESEARCH_VALIDATION_PROFILE)
        self.assertEqual(
            config.components["source"].kwargs["snapshot"],
            "tf_locoformer.dns.accept_recipe.research_valid_fixed@1",
        )
        self.assertEqual(config.components["protocol"].class_path, "src.experiments.data.ChunkProtocol")
        self.assertEqual(config.components["sampler"].kwargs["chunk_length"], 64000)
        self.assertEqual(config.components["sampler"].kwargs["num_samples_per_epoch"], 1000)
        self.assertEqual(config.components["batcher"].kwargs["group_size"], 1)
        text = (ROOT / "src/experiments/tf_locoformer.py").read_text(encoding="utf-8")
        self.assertIn("step_batch", text)
        self.assertIn("step_validation", text)
        self.assertIn(RESEARCH_VALIDATION_PROFILE, text)
        self.assertNotIn("self.optimizer.step", text)
        self.assertNotIn("self.scheduler.step(", text)

    def test_official_widths_build_one_dns_generator(self) -> None:
        config = load_experiment(ROOT / "configs/experiments/tf_locoformer_dns.yaml")
        model = TFLocoformerModel(seed=0, architecture=config.components["model"].kwargs["architecture"])
        self.assertIsInstance(model.generator, TFLocoformer)
        self.assertEqual(model.n_fft, 256)
        self.assertEqual(model.hop_length, 128)
        self.assertTrue(model.normalize_variance)
        self.assertEqual(model.generator.separator.n_layers, 6)
        self.assertEqual(
            [parameter for parameter in model.parameters()],
            [parameter for parameter in model.generator.parameters()],
        )
        exported = model.export_for_inference()
        self.assertTrue(exported)
        self.assertTrue(all(key.startswith("separator.") for key in exported))

    def test_yaml_numeric_strings_and_rejected_variants(self) -> None:
        architecture = dict(TINY)
        architecture["eps"] = "1e-5"
        architecture["n_fft"] = "16"
        model = TFLocoformerModel(seed=0, architecture=architecture)
        self.assertEqual(model.n_fft, 16)
        with self.assertRaises(BuildError):
            TFLocoformerModel(seed=0, variant="urgent", architecture=TINY)
        with self.assertRaises(BuildError):
            MultiResolutionL1Loss(normalize_variance=True)


class TFLocoformerConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_build_uses_one_initialized_parameter_set(self) -> None:
        experiment = build_experiment(load_experiment(_scripted_yaml(self.base / "build", steps=1, updates=2)))
        try:
            prepared = experiment.get("model")
            self.assertIsInstance(prepared, PreparedModel)
            model = prepared.state_module
            self.assertIs(prepared.module, model)
            self.assertIsInstance(model, TFLocoformerModel)
            self.assertIsInstance(model.generator, TFLocoformer)
            self.assertTrue(model.normalize_variance)
            self.assertFalse(experiment.get("loss").normalize_variance)
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
            self.assertIsInstance(experiment.get("optimizer"), torch.optim.AdamW)
            self.assertIsInstance(experiment.get("scheduler"), WarmupReduceLROnPlateau)
            task = _task()
            self.assertEqual(_max_abs(model.generator.state_dict(), task.generator.state_dict()), 0.0)
        finally:
            experiment.close()

    def test_loss_matches_the_existing_function_and_keeps_variance_off(self) -> None:
        loss = MultiResolutionL1Loss()
        target = torch.randn(2, 64)
        estimate = target + 0.1 * torch.randn(2, 64)
        left = loss(estimate, target)
        right = multi_resolution_l1(
            target, estimate, OFFICIAL_WINDOWS, time_domain_weight=0.5, normalize_variance=False
        )
        normalized = multi_resolution_l1(
            target, estimate, OFFICIAL_WINDOWS, time_domain_weight=0.5, normalize_variance=True
        )
        self.assertEqual(_max_abs(left, right), 0.0)
        self.assertNotEqual(_max_abs(left, normalized), 0.0)
        positive = per_sample_si_snr(estimate, target)
        negative = si_sdr_loss(estimate, target)
        self.assertEqual(_max_abs(positive.mean(), -negative), 0.0)
        self.assertNotEqual(float(positive.mean()), float(negative))

    def test_forward_loss_and_update_match_the_current_task(self) -> None:
        directory = self.base / "match"
        experiment = build_experiment(load_experiment(_scripted_yaml(directory, steps=1, updates=2)))
        model = experiment.get("model").state_module
        task = _task()
        self.assertEqual(_max_abs(model.generator.state_dict(), task.generator.state_dict()), 0.0)
        view = experiment.get("data").train_epoch(0, start_update=0)
        try:
            batch = view.batch(0)[0]
        finally:
            view.close()
        staged_model = TFLocoformerModel(seed=0, architecture=TINY)
        staged_model.load_state_dict(model.state_dict())
        staged_task = _task()
        staged_task.generator.load_state_dict(model.generator.state_dict())
        _assert_stage(self, staged_model, staged_task, experiment.get("loss"), batch.inputs, batch.targets)
        reference = _task()
        reference.generator.load_state_dict(model.generator.state_dict())
        report = reference.train_update([{"clean": batch.targets, "noisy": batch.inputs}], 1)
        result = experiment.run()
        self.assertEqual(result.logical_step, 1)
        self.assertEqual(result.optimizer_updates, {"optimizer": 1})
        self.assertEqual(_max_abs(model.generator.state_dict(), reference.generator.state_dict()), 0.0)
        self.assertEqual(
            _max_abs(_optim_state(model.generator, experiment.get("optimizer")), _optim_state(reference.generator, reference.optim_g)),
            0.0,
        )
        self.assertEqual(_max_abs(_scheduler_state(experiment.get("scheduler")), _scheduler_state(reference.sched_g)), 0.0)
        rows = _metrics(directory)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lr"], 0.001)
        self.assertEqual(rows[0]["scheduled_lr"], 0.001 / WARMUP)
        self.assertEqual(rows[0]["step_num"], 1)
        self.assertEqual(rows[0]["loss"], report["mr_l1_tfd"])
        self.assertEqual(rows[0]["si_sdr"], report["si_sdr"])
        self.assertEqual(experiment.get("scheduler").step_num, 1)

    def test_second_update_uses_the_warmup_rate(self) -> None:
        directory = self.base / "second"
        experiment = build_experiment(load_experiment(_scripted_yaml(directory, steps=2, updates=3)))
        model = experiment.get("model").state_module
        view = experiment.get("data").train_epoch(0, start_update=0)
        try:
            first = view.batch(0)[0]
            second = view.batch(1)[0]
        finally:
            view.close()
        reference = _task()
        reference.generator.load_state_dict(model.generator.state_dict())
        reference.train_update([{"clean": first.targets, "noisy": first.inputs}], 1)
        self.assertEqual(reference.optim_g.param_groups[0]["lr"], 0.001 / WARMUP)
        reference.train_update([{"clean": second.targets, "noisy": second.inputs}], 1)
        experiment.run()
        self.assertEqual(_max_abs(model.generator.state_dict(), reference.generator.state_dict()), 0.0)
        self.assertEqual(
            _max_abs(_optim_state(model.generator, experiment.get("optimizer")), _optim_state(reference.generator, reference.optim_g)),
            0.0,
        )
        self.assertEqual(_max_abs(_scheduler_state(experiment.get("scheduler")), _scheduler_state(reference.sched_g)), 0.0)
        rows = _metrics(directory)
        self.assertEqual([row["lr"] for row in rows], [0.001, 0.001 / WARMUP])
        self.assertEqual([row["scheduled_lr"] for row in rows], [0.001 / WARMUP, 0.001 * 2 / WARMUP])
        self.assertEqual([row["step_num"] for row in rows], [1, 2])

    def test_warmup_boundaries_plateau_and_restored_rate(self) -> None:
        initial = 0.001
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = _plateau(optimizer, warmup_steps=4000)
        self.assertEqual(optimizer.param_groups[0]["lr"], initial)
        optimizer.step()
        scheduler.step_batch()
        self.assertEqual(scheduler.step_num, 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], initial / 4000)
        while scheduler.step_num < 3999:
            scheduler.step_batch()
        self.assertEqual(scheduler.step_num, 3999)
        self.assertEqual(optimizer.param_groups[0]["lr"], initial * 3999 / 4000)
        scheduler.step_batch()
        self.assertEqual(scheduler.step_num, 4000)
        self.assertEqual(optimizer.param_groups[0]["lr"], initial)
        scheduler.step_batch()
        self.assertEqual(scheduler.step_num, 4001)
        self.assertEqual(optimizer.param_groups[0]["lr"], initial)

        warming = _plateau(
            torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01),
            warmup_steps=4000,
        )
        warming.optimizer.step()
        warming.step_batch()
        warmed = warming.optimizer.param_groups[0]["lr"]
        warming.step_validation(1.0)
        for _ in range(3):
            warming.step_validation(1.0)
        self.assertEqual(warming.num_bad_epochs, 3)
        self.assertEqual(warming.optimizer.param_groups[0]["lr"], warmed)
        warming.step_validation(1.0)
        self.assertEqual(warming.num_bad_epochs, 0)
        self.assertEqual(warming.last_epoch, 5)
        self.assertEqual(warming.optimizer.param_groups[0]["lr"], warmed * 0.5)
        warming.step_batch()
        self.assertEqual(warming.step_num, 2)
        self.assertEqual(warming.optimizer.param_groups[0]["lr"], initial * 2 / 4000)

        improving = _plateau(
            torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01),
            warmup_steps=4000,
        )
        improving.step_validation(1.0)
        improving.step_validation(0.5)
        self.assertEqual(improving.best, 0.5)
        self.assertEqual(improving.num_bad_epochs, 0)
        improving.step_validation(0.5)
        self.assertEqual(improving.num_bad_epochs, 1)
        self.assertEqual(improving.best, 0.5)

        held = _plateau(
            torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01),
            warmup_steps=4000,
        )
        for _ in range(10):
            held.step_batch()
        saved = held.state_dict()
        saved_lr = held.optimizer.param_groups[0]["lr"]
        restarted = _plateau(
            torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01),
            warmup_steps=4000,
        )
        restarted.load_state_dict(saved)
        restarted.step_batch()
        held.step_batch()
        self.assertEqual(restarted.step_num, 11)
        self.assertEqual(held.step_num, 11)
        self.assertEqual(restarted.optimizer.param_groups[0]["lr"], initial * 11 / 4000)
        self.assertEqual(held.optimizer.param_groups[0]["lr"], initial * 11 / 4000)
        self.assertNotEqual(saved_lr, initial)

        finished = _plateau(
            torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01),
            warmup_steps=2,
        )
        finished.step_batch()
        finished.step_batch()
        finished.step_batch()
        self.assertEqual(finished.step_num, 3)
        self.assertEqual(finished.optimizer.param_groups[0]["lr"], initial)
        opt_state = finished.optimizer.state_dict()
        sched_state = finished.state_dict()
        other = _plateau(
            torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01),
            warmup_steps=2,
        )
        other.optimizer.load_state_dict(opt_state)
        other.load_state_dict(sched_state)
        other.step_batch()
        finished.step_batch()
        self.assertEqual(other.optimizer.param_groups[0]["lr"], finished.optimizer.param_groups[0]["lr"])
        self.assertEqual(other.step_num, finished.step_num)

        floored = _plateau(
            torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial, betas=(0.9, 0.999), weight_decay=0.01),
            warmup_steps=1,
            min_lr=0.0008,
            cooldown=1,
        )
        floored.step_batch()
        floored.step_validation(1.0)
        for _ in range(4):
            floored.step_validation(2.0)
        self.assertEqual(floored.optimizer.param_groups[0]["lr"], 0.0008)
        self.assertEqual(floored.cooldown_counter, 1)
        floored.step_validation(3.0)
        self.assertEqual(floored.cooldown_counter, 0)
        self.assertEqual(floored.num_bad_epochs, 0)

    def test_epoch_end_runs_validation_then_plateau_once(self) -> None:
        directory = self.base / "epoch"
        seen: list[tuple[str, int, int]] = []

        def record(trainer: TFLocoformerTrainer, phase: str) -> None:
            seen.append((phase, int(trainer.scheduler.step_num), int(trainer.scheduler.last_epoch)))

        experiment = build_experiment(load_experiment(_scripted_yaml(directory, steps=2, updates=2, max_epochs=1)))
        with patch.object(TFLocoformerTrainer, "_on_boundary_phase", record):
            result = experiment.run()
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual([item[0] for item in seen], ["pending_validation", "pending_scheduler", "pending_selection", "in_epoch"])
        self.assertEqual(seen[0][1:], (2, 0))
        self.assertEqual(seen[1][1:], (2, 0))
        self.assertEqual(seen[2][1:], (2, 1))
        self.assertEqual(seen[3][1:], (2, 1))
        payload = _load(directory / "step_00000002.pt")
        trainer = payload["states"]["trainer"]
        self.assertEqual(trainer["phase"], "in_epoch")
        self.assertIsNone(trainer["boundary"])
        self.assertEqual(len(trainer["history"]), 1)
        self.assertEqual(trainer["history"][0]["epoch"], 1)
        self.assertEqual(trainer["completed_boundaries"][0]["validation_weight"], 3)
        self.assertTrue(trainer["completed_boundaries"][0]["warmup_updated"])
        self.assertTrue(trainer["completed_boundaries"][0]["validation_done"])
        self.assertTrue(trainer["completed_boundaries"][0]["scheduler_updated"])
        self.assertTrue(trainer["completed_boundaries"][0]["selection_done"])
        self.assertEqual(payload["states"]["scheduler"]["step_num"], 2)
        self.assertEqual(payload["states"]["scheduler"]["last_epoch"], 1)
        self.assertEqual(payload["states"]["scheduler"]["best"], trainer["history"][0]["loss"])
        self.assertNotEqual(trainer["history"][0]["si_snr"], _metrics(directory)[-1]["si_sdr"])
        loss_meta = json.loads((directory / "valid.loss.best.pt.json").read_text(encoding="utf-8"))
        si_meta = json.loads((directory / "valid.si_snr.best.pt.json").read_text(encoding="utf-8"))
        self.assertEqual(loss_meta["criterion"], "valid.loss.min")
        self.assertEqual(si_meta["criterion"], "valid.si_snr.max")
        self.assertEqual(si_meta["sign"], "si_sdr_db")
        self.assertEqual(loss_meta["validation_profile"], "scripted")
        self.assertEqual(si_meta["value"], trainer["history"][0]["si_snr"])
        expected = _expected_validation(experiment.get("model").state_module)
        self.assertEqual(trainer["history"][0]["loss"], expected["loss"])
        self.assertEqual(trainer["history"][0]["si_snr"], expected["si_snr"])
        self.assertNotEqual(expected["loss"], expected["rank_mean_loss"])

    def test_mid_epoch_stop_does_not_validate_or_step_plateau(self) -> None:
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
            self.assertEqual(payload["states"]["scheduler"]["step_num"], 1)
            self.assertEqual(payload["states"]["scheduler"]["best"], float("inf"))
            self.assertEqual(list(directory.glob("*epoch.pt")), [])
            if stop is None:
                self.assertEqual(result.stop_reason, "completed")
                self.assertTrue((directory / "complete.json").is_file())
            else:
                self.assertEqual(result.stop_reason, "interrupted")
                self.assertFalse((directory / "complete.json").is_file())

    def test_skipped_optimizer_step_does_not_advance_warmup(self) -> None:
        directory = self.base / "skip"
        experiment = build_experiment(load_experiment(_scripted_yaml(directory, steps=1, updates=1, max_epochs=1)))

        def skip(self, optimizer: torch.optim.Optimizer) -> StepResult:
            del self, optimizer
            return StepResult(applied=False)

        with patch.object(ProcessRuntime, "step", skip):
            result = experiment.run()
        self.assertEqual(result.logical_step, 1)
        self.assertEqual(result.optimizer_updates, {"optimizer": 0})
        payload = _load(directory / "step_00000001.pt")
        self.assertEqual(payload["states"]["scheduler"]["step_num"], 0)
        self.assertEqual(payload["states"]["scheduler"]["last_epoch"], 1)
        self.assertEqual(payload["states"]["trainer"]["completed_boundaries"][0]["warmup_updated"], False)
        self.assertTrue(payload["states"]["trainer"]["completed_boundaries"][0]["scheduler_updated"])
        rows = _metrics(directory)
        self.assertEqual(rows[0]["lr"], 0.001)
        self.assertEqual(rows[0]["scheduled_lr"], 0.001)
        self.assertFalse(rows[0]["applied"])
        self.assertEqual(rows[0]["step_num"], 0)

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
        self.assertTrue(pending["states"]["trainer"]["boundary"]["warmup_updated"])
        self.assertEqual(pending["states"]["scheduler"]["step_num"], 2)
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
        self.assertEqual(right["states"]["scheduler"]["step_num"], 2)
        self.assertEqual(right["states"]["scheduler"]["last_epoch"], 1)
        self.assertEqual(right["states"]["scheduler"]["num_bad_epochs"], 0)

    def test_selection_failure_resumes_from_the_pending_checkpoint(self) -> None:
        continuous = self.base / "select-continuous"
        interrupted = self.base / "select-fail"
        build_experiment(load_experiment(_scripted_yaml(continuous, steps=2, updates=2, max_epochs=1))).run()

        def fail_once(self, *args: object, **kwargs: object) -> None:
            del self, args, kwargs
            raise RuntimeError("selection write failed")

        experiment = build_experiment(load_experiment(_scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1)))
        with patch.object(TFLocoformerTrainer, "_select", fail_once):
            with self.assertRaises(RuntimeError):
                experiment.run()
        pending = _load(interrupted / "step_00000002.pt")
        self.assertEqual(pending["label"], "validation_pending")
        self.assertEqual(pending["states"]["trainer"]["history"], [])
        resumed = _cli(
            _scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1),
            "--resume",
            str(interrupted / "step_00000002.pt"),
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        _assert_checkpoint(self, _load(continuous / "step_00000002.pt"), _load(interrupted / "step_00000002.pt"))
        _assert_artifacts(self, continuous, interrupted)

    def test_mid_epoch_resume_matches_a_continuous_run(self) -> None:
        continuous = self.base / "mid-continuous"
        interrupted = self.base / "mid-resume"
        build_experiment(load_experiment(_scripted_yaml(continuous, steps=2, updates=2, max_epochs=1))).run()
        first = build_experiment(
            load_experiment(_scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1, stop_after=1))
        )
        self.assertEqual(first.run().stop_reason, "interrupted")
        pending = _load(interrupted / "step_00000001.pt")
        self.assertEqual(pending["states"]["scheduler"]["step_num"], 1)
        self.assertEqual(pending["states"]["optimizer"]["param_groups"][0]["lr"], 0.001 / WARMUP)
        second = build_experiment(load_experiment(_scripted_yaml(interrupted, steps=2, updates=2, max_epochs=1)))
        self.assertEqual(second.run(resume=interrupted / "step_00000001.pt").stop_reason, "completed")
        _assert_checkpoint(self, _load(continuous / "step_00000002.pt"), _load(interrupted / "step_00000002.pt"))
        _assert_artifacts(self, continuous, interrupted)
        rows = _metrics(interrupted)
        self.assertEqual(rows[1]["lr"], 0.001 / WARMUP)

    def test_weighted_validation_uses_sample_counts(self) -> None:
        directory = self.base / "weights"
        experiment = build_experiment(load_experiment(_scripted_yaml(directory, steps=2, updates=2, max_epochs=1)))
        experiment.run()
        trainer = _load(directory / "step_00000002.pt")["states"]["trainer"]
        self.assertEqual(trainer["completed_boundaries"][0]["validation_weight"], 3)
        expected = _expected_validation(experiment.get("model").state_module)
        self.assertEqual(trainer["history"][0]["loss"], expected["loss"])
        self.assertEqual(trainer["history"][0]["si_snr"], expected["si_snr"])
        self.assertEqual(expected["weight"], 3)

    def test_nbest_rank_average_and_ties(self) -> None:
        history = [
            {"epoch": 4, "loss": 1.0, "si_snr": 3.0},
            {"epoch": 2, "loss": 1.0, "si_snr": 9.0},
            {"epoch": 3, "loss": 0.5, "si_snr": 9.0},
            {"epoch": 1, "loss": 2.0, "si_snr": 4.0},
        ]
        self.assertEqual([row["epoch"] for row in rank_validation_loss(history, 3)], [3, 2, 4])
        self.assertEqual([row["epoch"] for row in rank_validation_si_snr(history, 3)], [2, 3, 1])
        self.assertEqual(rank_validation_si_snr(history, 1)[0]["epoch"], 2)
        tied = [{"epoch": 4, "loss": 1.0, "si_snr": 5.0}, {"epoch": 2, "loss": 1.0, "si_snr": 5.0}]
        self.assertEqual(rank_validation_loss(tied, 1)[0]["epoch"], 2)
        self.assertEqual(rank_validation_si_snr(tied, 1)[0]["epoch"], 2)
        self.assertTrue(early_stop([{"epoch": 1, "loss": 1.0, "si_snr": 0.0}, {"epoch": 2, "loss": 2.0, "si_snr": 100.0}], 2, 0))
        self.assertFalse(early_stop([{"epoch": 1, "loss": 1.0, "si_snr": 0.0}, {"epoch": 2, "loss": 2.0, "si_snr": 100.0}], 2, 5))
        directory = self.base / "nbest"
        directory.mkdir()
        first = {"separator.count": torch.tensor(3), "separator.weight": torch.ones(2)}
        second = {"separator.count": torch.tensor(4), "separator.weight": torch.ones(2) * 3}
        third = {"separator.count": torch.tensor(8), "separator.weight": torch.ones(2) * 9}
        rows = [{"epoch": 1, "loss": 2.0, "si_snr": 5.0}]
        _write_selection(directory, first, 1, rank_validation_loss(rows, 2), rank_validation_si_snr(rows, 2), 2, 1, "scripted")
        self.assertTrue((directory / "valid.loss.ave_1best.pt").is_symlink())
        self.assertEqual(os.readlink(directory / "valid.loss.ave.pt"), "valid.loss.ave_1best.pt")
        self.assertFalse((directory / "valid.loss.ave_2best.pt").exists())
        self.assertFalse((directory / "valid.si_snr.ave.pt").exists())
        rows.append({"epoch": 2, "loss": 2.0, "si_snr": 5.0})
        _write_selection(directory, second, 2, rank_validation_loss(rows, 2), rank_validation_si_snr(rows, 2), 2, 2, "scripted")
        self.assertEqual(_max_abs(torch.load(directory / "valid.loss.best.pt", map_location="cpu", weights_only=True), first), 0.0)
        self.assertEqual(_max_abs(torch.load(directory / "valid.si_snr.best.pt", map_location="cpu", weights_only=True), first), 0.0)
        rows.append({"epoch": 3, "loss": 0.1, "si_snr": 9.0})
        _write_selection(directory, third, 3, rank_validation_loss(rows, 2), rank_validation_si_snr(rows, 2), 2, 3, "scripted")
        averaged = torch.load(directory / "valid.loss.ave_2best.pt", map_location="cpu", weights_only=True)
        expected = average_state_dicts(
            [
                torch.load(directory / "3epoch.pt", map_location="cpu", weights_only=True),
                torch.load(directory / "1epoch.pt", map_location="cpu", weights_only=True),
            ]
        )
        self.assertEqual(_max_abs(averaged, expected), 0.0)
        self.assertEqual(int(averaged["separator.count"]), 11)
        self.assertEqual(float(averaged["separator.weight"][0]), 5.0)
        self.assertEqual(os.readlink(directory / "valid.loss.ave.pt"), "valid.loss.ave_2best.pt")
        self.assertTrue((directory / "1epoch.pt").is_file())
        self.assertFalse((directory / "2epoch.pt").exists())
        self.assertTrue((directory / "3epoch.pt").is_file())
        self.assertEqual(_max_abs(torch.load(directory / "valid.si_snr.best.pt", map_location="cpu", weights_only=True), third), 0.0)
        meta = json.loads((directory / "valid.si_snr.best.pt.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["criterion"], "valid.si_snr.max")
        self.assertEqual(meta["validation_profile"], "scripted")
        self.assertEqual(meta["value"], 9.0)

    def test_export_matches_the_inference_loader(self) -> None:
        model = TFLocoformerModel(seed=2, architecture=TINY)
        exported = model.export_for_inference()
        prefixed = self.base / "export-prefixed.pt"
        plain = self.base / "export-plain.pt"
        torch.save(exported, prefixed)
        torch.save({key.removeprefix("separator."): value for key, value in exported.items()}, plain)
        prefixed_enhancer = load_enhancer_checkpoint(prefixed, "cpu", TINY, "dns")
        plain_enhancer = load_enhancer_checkpoint(plain, "cpu", TINY, "dns")
        reference = {key.removeprefix("separator."): value for key, value in exported.items()}
        self.assertEqual(_max_abs(prefixed_enhancer.model.separator.state_dict(), reference), 0.0)
        self.assertEqual(_max_abs(plain_enhancer.model.separator.state_dict(), reference), 0.0)
        self.assertTrue(all(isinstance(key, str) and key.startswith("separator.") for key in exported))

    def test_snapshot_datamodule_completes_one_epoch(self) -> None:
        spec = _write_snapshot(self.base / "snapshots", derived=self.base / "derived", kind="dns", name="phase4c.dns")
        directory = self.base / "snapshot-run"
        experiment = build_experiment(load_experiment(_snapshot_yaml(directory, spec)))
        try:
            self.assertEqual(experiment.get("data").__class__.__name__, "SnapshotDataModule")
            self.assertEqual(experiment.get("protocol").__class__.__name__, "ChunkProtocol")
            self.assertIsInstance(experiment.get("model").state_module.generator, TFLocoformer)
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
        self.assertEqual(payload["states"]["scheduler"]["step_num"], result.logical_step)
        self.assertGreater(payload["states"]["trainer"]["completed_boundaries"][0]["validation_weight"], 0)
        self.assertIn("si_snr", payload["states"]["trainer"]["history"][0])

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
        self.assertEqual(payload["states"]["scheduler"]["step_num"], 2)
        self.assertEqual(payload["states"]["scheduler"]["best"], payload["states"]["trainer"]["history"][0]["loss"])
        rank0 = _metrics_file(directory / "samples-rank0.jsonl")
        rank1 = _metrics_file(directory / "samples-rank1.jsonl")
        self.assertEqual(len(rank0), 2)
        self.assertEqual(len(rank1), 2)
        self.assertNotEqual([row["sample_id"] for row in rank0], [row["sample_id"] for row in rank1])

    def test_ddp_prepare_keeps_parameter_identity(self) -> None:
        directory = self.base / "identity"
        directory.mkdir()
        completed = _torchrun_module("tests.experiments.tf_locoformer_identity_worker", directory)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        ranks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("rank*.json"))]
        self.assertEqual(len(ranks), 2)
        for payload in ranks:
            self.assertTrue(payload["match"])
            self.assertTrue(payload["wrapped"])
            self.assertTrue(payload["state_is_unwrapped"])
            self.assertGreater(payload["parameters"], 0)

    def test_rejects_bad_shapes_silence_and_a_step_scheduler(self) -> None:
        model = TFLocoformerModel(seed=1, architecture=TINY)
        with self.assertRaises(RuntimeFailure):
            model(torch.randn(2, 4, 8))
        with self.assertRaises(RuntimeFailure):
            model(torch.randn(1, 8))
        with self.assertRaises(RuntimeFailure):
            model(torch.zeros(1, 32))
        with self.assertRaises(RuntimeFailure):
            model(torch.full((1, 32), 0.2))
        silent = TFLocoformerModel(seed=1, architecture={**TINY, "normalize_variance": False})
        restored = silent(torch.zeros(1, 32))
        self.assertEqual(tuple(restored.shape), (1, 32))
        directory = self.base / "bad-scheduler"
        text = _scripted_text(directory, steps=1, updates=1).replace(
            "src.training.se.optim.WarmupReduceLROnPlateau",
            "torch.optim.lr_scheduler.StepLR",
        )
        text = text.replace(
            "      warmup_steps: 4\n      mode: min\n      factor: 0.5\n      patience: 3\n"
            "      threshold: 1.0e-4\n      threshold_mode: rel\n      cooldown: 0\n"
            "      min_lr: 0.0\n      eps: 1.0e-8\n",
            "      step_size: 1\n      gamma: 0.5\n",
        )
        path = directory / "experiment.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(BuildError):
            build_experiment(load_experiment(path)).close()
        mismatched = build_experiment(load_experiment(_scripted_yaml(self.base / "group", steps=1, updates=1)))
        mismatched.get("data").group_size = 2
        with self.assertRaises(RuntimeFailure):
            mismatched.run()


def _plateau(optimizer: torch.optim.Optimizer, **overrides: object) -> WarmupReduceLROnPlateau:
    values = {
        "cooldown": 0,
        "eps": 1e-8,
        "factor": 0.5,
        "min_lr": 0.0,
        "mode": "min",
        "patience": 3,
        "threshold": 1e-4,
        "threshold_mode": "rel",
        "warmup_steps": 4000,
    }
    values.update(overrides)
    return WarmupReduceLROnPlateau(optimizer, **values)  # type: ignore[arg-type]


def _yaml_architecture(architecture: dict[str, object]) -> str:
    text = json.dumps(architecture)
    return text.replace("1e-05", "1.0e-5")


def _task(warmup: int = WARMUP) -> TFLocoformerTask:
    return TFLocoformerTask(
        {
            "grad_clip": 5.0,
            "loss": "official",
            "model": {"architecture": TINY, "init": "xavier_uniform", "variant": "dns"},
            "optimizer": {"betas": [0.9, 0.999], "lr": 0.001, "name": "adamw", "weight_decay": 0.01},
            "scheduler": {
                "cooldown": 0,
                "eps": 1e-8,
                "factor": 0.5,
                "min_lr": 0.0,
                "mode": "min",
                "name": "warmup_reduce_on_plateau",
                "patience": 3,
                "threshold": 1e-4,
                "threshold_mode": "rel",
                "warmup_steps": warmup,
            },
            "seed": 0,
        },
        torch.device("cpu"),
    )


def _assert_stage(
    test: unittest.TestCase,
    model: TFLocoformerModel,
    task: TFLocoformerTask,
    loss: MultiResolutionL1Loss,
    noisy: torch.Tensor,
    clean: torch.Tensor,
) -> None:
    model.train()
    task.train_mode()
    estimate = model(noisy)
    reference = task.generator(noisy)
    test.assertEqual(_max_abs(estimate, reference), 0.0)
    left = loss(estimate, clean)
    right = multi_resolution_l1(clean, reference, OFFICIAL_WINDOWS, time_domain_weight=0.5, normalize_variance=False)
    test.assertEqual(_max_abs(left, right), 0.0)
    test.assertGreater(float(left.abs()), 0.0)
    left.backward()
    right.backward()
    test.assertEqual(_max_abs(_grads(model.generator), _grads(task.generator)), 0.0)
    test.assertTrue(any(float(value.abs().sum()) > 0 for value in _grads(model.generator).values()))
    positive = per_sample_si_snr(estimate.detach(), clean).mean()
    negative = si_sdr_loss(estimate.detach(), clean)
    test.assertEqual(_max_abs(positive, -negative), 0.0)
    runtime = ProcessRuntime(device="cpu", precision="fp32")
    try:
        runtime.clip(model.parameters(), 5.0)
        clip_gradients(task.generator, 5.0)
        test.assertEqual(_max_abs(_grads(model.generator), _grads(task.generator)), 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = _plateau(optimizer, warmup_steps=WARMUP)
        task_scheduler = _plateau(task.optim_g, warmup_steps=WARMUP)
        test.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        test.assertEqual(task.optim_g.param_groups[0]["lr"], 0.001)
        applied = runtime.step(optimizer)
        runtime.end_step()
        task.optim_g.step()
        test.assertTrue(applied.applied)
        test.assertEqual(_max_abs(model.generator.state_dict(), task.generator.state_dict()), 0.0)
        test.assertEqual(
            _max_abs(_optim_state(model.generator, optimizer), _optim_state(task.generator, task.optim_g)),
            0.0,
        )
        scheduler.step_batch()
        task_scheduler.step_batch()
        test.assertEqual(scheduler.step_num, 1)
        test.assertEqual(optimizer.param_groups[0]["lr"], 0.001 / WARMUP)
        test.assertEqual(task.optim_g.param_groups[0]["lr"], 0.001 / WARMUP)
        test.assertEqual(_max_abs(scheduler.state_dict(), task_scheduler.state_dict()), 0.0)
    finally:
        runtime.shutdown()


def _grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.grad.detach() for name, parameter in module.named_parameters() if parameter.grad is not None}


def _scheduler_state(scheduler: WarmupReduceLROnPlateau) -> dict[str, object]:
    state = dict(scheduler.state_dict())
    state.pop("_experiment_component_id", None)
    return state


def _optim_state(module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, dict[str, object]]:
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    return {names[id(parameter)]: dict(state) for parameter, state in optimizer.state.items()}


def _expected_validation(model: TFLocoformerModel) -> dict[str, float]:
    data = ScriptedChunkData(updates_per_epoch=2, micro_batch=1, length=32, seed=17, valid_sizes=((2, 1), (2,)))
    data.world_size = 1
    data.rank = 0
    loss_fn = MultiResolutionL1Loss()
    loss_sum = 0.0
    si_sum = 0.0
    count = 0
    rank_losses: list[float] = []
    was = model.training
    model.eval()
    with torch.no_grad():
        for batch in data.validation_epoch(0):
            prediction = model(batch.inputs)
            value = float(loss_fn(prediction, batch.targets))
            scores = per_sample_si_snr(prediction, batch.targets)
            weight = int(batch.inputs.shape[0])
            loss_sum += value * weight
            si_sum += float(scores.sum())
            count += weight
            rank_losses.append(value)
    if was:
        model.train()
    return {
        "loss": loss_sum / count,
        "rank_mean_loss": sum(rank_losses) / len(rank_losses),
        "si_snr": si_sum / count,
        "weight": float(count),
    }


def _scripted_yaml(
    directory: Path,
    *,
    steps: int,
    updates: int,
    max_epochs: int | None = None,
    stop_after: int | None = None,
    distribute: str = "none",
    warmup: int = WARMUP,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(
        _scripted_text(
            directory,
            steps=steps,
            updates=updates,
            max_epochs=max_epochs,
            stop_after=stop_after,
            distribute=distribute,
            warmup=warmup,
        ),
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
    warmup: int = WARMUP,
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
    class: src.experiments.models.tf_locoformer.TFLocoformerModel
    kwargs:
      seed: 0
      variant: dns
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
    class: torch.optim.AdamW
    parameters_of: model
    kwargs:
      lr: 0.001
      betas: [0.9, 0.999]
      weight_decay: 0.01
  scheduler:
    role: scheduler
    class: src.training.se.optim.WarmupReduceLROnPlateau
    kwargs:
      warmup_steps: {warmup}
      mode: min
      factor: 0.5
      patience: 3
      threshold: 1.0e-4
      threshold_mode: rel
      cooldown: 0
      min_lr: 0.0
      eps: 1.0e-8
    inject:
      optimizer: {{ref: optimizer}}
  trainer:
    role: trainer
    class: src.experiments.tf_locoformer.TFLocoformerTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: 1
      grad_clip: 5.0
      keep_nbest: 5
      max_epochs: {epochs}
      checkpoint_interval: 1
      validation_profile: scripted
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
    class: src.experiments.models.tf_locoformer.TFLocoformerModel
    kwargs:
      seed: 0
      variant: dns
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
    class: torch.optim.AdamW
    parameters_of: model
    kwargs:
      lr: 0.001
      betas: [0.9, 0.999]
      weight_decay: 0.01
  scheduler:
    role: scheduler
    class: src.training.se.optim.WarmupReduceLROnPlateau
    kwargs:
      warmup_steps: {WARMUP}
      mode: min
      factor: 0.5
      patience: 3
      threshold: 1.0e-4
      threshold_mode: rel
      cooldown: 0
      min_lr: 0.0
      eps: 1.0e-8
    inject:
      optimizer: {{ref: optimizer}}
  trainer:
    role: trainer
    class: src.experiments.tf_locoformer.TFLocoformerTrainer
    kwargs:
      max_logical_steps: 200
      accumulation: 1
      grad_clip: 5.0
      keep_nbest: 5
      max_epochs: 1
      checkpoint_interval: 1
      validation_profile: scripted
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
    for name in ("1epoch.pt", "valid.loss.best.pt", "valid.si_snr.best.pt"):
        _assert_equal(
            torch.load(left / name, map_location="cpu", weights_only=True),
            torch.load(right / name, map_location="cpu", weights_only=True),
            name,
        )
    for name in ("valid.loss.best.pt.json", "valid.si_snr.best.pt.json"):
        _assert_equal(
            json.loads((left / name).read_text(encoding="utf-8")),
            json.loads((right / name).read_text(encoding="utf-8")),
            name,
        )
    for name in ("valid.loss.ave_1best.pt", "valid.loss.ave.pt"):
        test.assertEqual(os.readlink(left / name), os.readlink(right / name))


def _assert_checkpoint(test: unittest.TestCase, left: dict[str, object], right: dict[str, object]) -> None:
    del test
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
            if lhs.dtype != rhs.dtype or tuple(lhs.shape) != tuple(rhs.shape):
                raise AssertionError((tuple(lhs.shape), tuple(rhs.shape), lhs.dtype, rhs.dtype))
            if lhs.numel() == 0:
                return
            values.append(float((lhs.detach().cpu() - rhs.detach().cpu()).abs().max()))
            return
        if isinstance(lhs, dict):
            if not isinstance(rhs, dict) or set(lhs) != set(rhs):
                raise AssertionError(set(lhs) ^ set(rhs))
            for key, value in lhs.items():
                walk(value, rhs[key])
            return
        if isinstance(lhs, (list, tuple)):
            if not isinstance(rhs, (list, tuple)) or len(lhs) != len(rhs):
                raise AssertionError(path_label(lhs, rhs))
            for left_item, right_item in zip(lhs, rhs):
                walk(left_item, right_item)
            return
        if lhs != rhs:
            raise AssertionError(f"{lhs!r} != {rhs!r}")

    walk(left, right)
    return max(values) if values else 0.0


def path_label(left: object, right: object) -> str:
    return f"{left!r} != {right!r}"


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
    return _torchrun_command(["-m", "src.experiments.cli", "--config", str(config)])


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
