"""Phase 4-D: MP-SENet official GAN training on the experiment framework.

CPU / FP32 comparisons use the existing ``MPSENetTask`` update. A PESQ stub
and the installed PESQ library are separate tests. Short 1,600-sample clips
make ``metric_targets`` return None; the 16,000-sample test uses real PESQ.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.data import Batch
from src.experiments.errors import BuildError, CheckpointError, RuntimeFailure
from src.experiments.official.mp_senet.waveform import MPSENetWaveformAdapter
from src.experiments.official.mp_senet.models import MPSENetGANGenerator, MPSENetMetricDiscriminator
from src.experiments.official.mp_senet.trainer import (
    BEST_CHECKPOINT_START_EPOCH,
    LEAKAGE_SAFE_VALIDATION_INTERVAL,
    OFFICIAL_CHECKPOINT_INTERVAL,
    OFFICIAL_GAMMA,
    OFFICIAL_LEARNING_RATE,
    OFFICIAL_VALIDATION_INTERVAL,
    OFFICIAL_WEIGHT_DECAY,
)
from src.experiments.runtime import PreparePolicy, ProcessRuntime
from src.experiments.data.ids import canonical_text, digest
from src.experiments.data.snapshot import (
    FORMAT,
    SCHEMA_VERSION,
    arrays_from_assets,
    arrays_from_cases,
    arrays_from_items,
    arrays_from_members,
    file_hashes,
    load_snapshot,
    snapshot_hash,
)
from src.experiments.calc.rng import set_determinism
from src.experiments.calc.losses import metric_targets
from src.experiments.official.mp_senet.reference import MPSENetTask

ROOT = Path(__file__).resolve().parents[2]
ARCH = {
    "beta": 2.0,
    "compress_factor": 0.3,
    "dense_channel": 4,
    "hop_size": 100,
    "n_fft": 400,
    "num_tsconformers": 1,
    "sampling_rate": 16000,
    "win_size": 400,
}
LENGTH = 1600


class MPSENetGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_determinism(True, cuda=False)
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)

    @classmethod
    def tearDownClass(cls) -> None:
        set_determinism(False, cuda=False)
        cls._temporary.cleanup()

    def test_committed_yaml_declares_the_official_algorithm(self) -> None:
        path = ROOT / "tests/fixtures/configs/experiments/mp_senet_gan.yaml"
        text = path.read_text(encoding="utf-8")
        config = load_experiment(path)
        self.assertEqual(config.schema, "audio-db-experiment/v1")
        self.assertIn("MPSENetGANTrainer", text)
        self.assertIn("MPSENetGANGenerator", text)
        self.assertIn("MPSENetMetricDiscriminator", text)
        self.assertIn("experiment_claim: official_faithful", text)
        self.assertIn("mp_senet.vb28.ondemand.official@1", text)
        self.assertIn("leakage_safe", text)
        self.assertNotIn("cv_synthetic_small", text)
        trainer = config.components["trainer"]
        self.assertEqual(trainer.kwargs["validation_interval"], OFFICIAL_VALIDATION_INTERVAL)
        self.assertEqual(trainer.kwargs["checkpoint_interval"], OFFICIAL_CHECKPOINT_INTERVAL)
        self.assertEqual(trainer.kwargs["best_checkpoint_start_epoch"], BEST_CHECKPOINT_START_EPOCH)
        self.assertEqual(trainer.kwargs["max_logical_steps"], 1)
        self.assertEqual(config.components["optim_g"].kwargs["lr"], OFFICIAL_LEARNING_RATE)
        self.assertEqual(config.components["optim_d"].kwargs["weight_decay"], OFFICIAL_WEIGHT_DECAY)
        self.assertEqual(config.components["scheduler_g"].kwargs["gamma"], OFFICIAL_GAMMA)
        self.assertEqual(config.components["scheduler_d"].kwargs["gamma"], OFFICIAL_GAMMA)
        self.assertEqual(config.components["generator"].kwargs["n_fft"], 400)
        self.assertEqual(config.components["generator"].kwargs["num_tsconformers"], 4)
        self.assertIn(str(LEAKAGE_SAFE_VALIDATION_INTERVAL), text)
        self.assertTrue(issubclass(MPSENetWaveformAdapter, torch.nn.Module))

    def test_builder_initialization_matches_the_task(self) -> None:
        directory = self.base / "init"
        experiment = build_experiment(load_experiment(_yaml(directory, steps=1, updates=2)))
        try:
            task = _task()
            generator = experiment.get("generator").state_module.generator
            discriminator = experiment.get("metric_discriminator").state_module.discriminator
            self.assertEqual(_max_diff(task.generator.state_dict(), generator.state_dict()), 0.0)
            self.assertEqual(_max_diff(task.discriminator.state_dict(), discriminator.state_dict()), 0.0)
            owners = set(experiment._graph.states.ids())
            self.assertTrue(
                {
                    "data",
                    "generator",
                    "metric_discriminator",
                    "optim_d",
                    "optim_g",
                    "runtime",
                    "scheduler_d",
                    "scheduler_g",
                    "trainer",
                }
                <= owners
            )
            generator_ids = {id(parameter) for parameter in experiment.get("generator").parameters}
            discriminator_ids = {id(parameter) for parameter in experiment.get("metric_discriminator").parameters}
            optim_g_ids = {id(parameter) for group in experiment.get("optim_g").param_groups for parameter in group["params"]}
            optim_d_ids = {id(parameter) for group in experiment.get("optim_d").param_groups for parameter in group["params"]}
            self.assertEqual(generator_ids, optim_g_ids)
            self.assertEqual(discriminator_ids, optim_d_ids)
            self.assertFalse(generator_ids & discriminator_ids)
        finally:
            experiment.close()

    def test_shared_parameters_and_bad_claim_are_rejected(self) -> None:
        directory = self.base / "shared"
        text = _yaml_text(directory, steps=1, updates=2).replace(
            "parameters_of: metric_discriminator",
            "parameters_of: generator",
        )
        path = directory / "experiment.yaml"
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(BuildError):
            build_experiment(load_experiment(path)).close()
        wrong = self.base / "claim"
        wrong_text = _yaml_text(wrong, steps=1, updates=2).replace(
            "experiment_claim: official_faithful",
            "experiment_claim: leakage_safe",
        )
        wrong_path = wrong / "experiment.yaml"
        wrong.mkdir(parents=True, exist_ok=True)
        wrong_path.write_text(wrong_text, encoding="utf-8")
        experiment = build_experiment(load_experiment(wrong_path))
        with self.assertRaises(RuntimeFailure):
            experiment.run()

    def test_one_stubbed_update_matches_the_task(self) -> None:
        report = _compare_update(accumulation=1, length=LENGTH, stub=True)
        self.assertEqual(report["init_generator"], 0.0)
        self.assertEqual(report["init_discriminator"], 0.0)
        for name in (
            "losses",
            "discriminator_grad",
            "generator_grad",
            "discriminator_parameters",
            "generator_parameters",
            "optim_d",
            "optim_g",
        ):
            self.assertEqual(report[name], 0.0, name)
        self.assertTrue(report["graph_alive"])

    def test_accumulation_two_matches_and_keeps_the_graph(self) -> None:
        report = _compare_update(accumulation=2, length=LENGTH, stub=True)
        self.assertEqual(report["losses"], 0.0)
        self.assertEqual(report["generator_parameters"], 0.0)
        self.assertEqual(report["discriminator_parameters"], 0.0)
        self.assertEqual(report["optim_g"], 0.0)
        self.assertEqual(report["optim_d"], 0.0)
        self.assertGreater(report["rss_kb"], 0)

    def test_real_pesq_update_matches_the_task(self) -> None:
        report = _compare_update(accumulation=1, length=16000, stub=False, sine=True)
        self.assertIsNotNone(report["pesq_target"])
        self.assertEqual(report["losses"], 0.0)
        self.assertEqual(report["generator_grad"], 0.0)
        self.assertEqual(report["discriminator_grad"], 0.0)
        self.assertEqual(report["generator_parameters"], 0.0)
        self.assertEqual(report["discriminator_parameters"], 0.0)
        self.assertEqual(report["optim_g"], 0.0)
        self.assertEqual(report["optim_d"], 0.0)

    def test_pesq_failure_still_updates_both_optimizers(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        try:
            trainer, _gen, disc = _manual_trainer(runtime, accumulation=1)
            before_g = {name: tensor.detach().clone() for name, tensor in trainer.generator.state_module.generator.named_parameters()}
            before_d = {name: tensor.detach().clone() for name, tensor in disc.named_parameters()}
            batch = _random_batch(length=LENGTH, seed=5)
            steps = {"g": 0, "d": 0}
            trainer.optim_g.step = _count_step(trainer.optim_g.step, steps, "g")
            trainer.optim_d.step = _count_step(trainer.optim_d.step, steps, "d")

            def fail_one(clean, enhanced, sample_rate):
                del enhanced, sample_rate
                if clean.shape[0] != 1:
                    return None
                return None

            with patch("src.experiments.calc.losses.metric_targets", fail_one):
                metrics = trainer.official_step([batch], runtime)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["generator"])))
            self.assertEqual(steps, {"g": 1, "d": 1})
            self.assertGreater(_parameter_shift(before_g, trainer.generator.state_module.generator), 0.0)
            self.assertGreater(_parameter_shift(before_d, disc), 0.0)
        finally:
            runtime.shutdown()

    def test_missing_pesq_import_returns_none_and_validation_is_unavailable(self) -> None:
        time = torch.arange(16000, dtype=torch.float32) / 16000
        clean = (0.2 * torch.sin(2 * torch.pi * 200 * time)).unsqueeze(0)
        estimate = clean + 0.01
        self.assertIsNotNone(metric_targets(clean, estimate, 16000))
        import pesq as pesq_lib

        with patch.object(pesq_lib, "pesq", side_effect=ImportError("pesq is not installed")):
            self.assertIsNone(metric_targets(clean, estimate, 16000))
        directory = self.base / "unavailable"
        experiment = build_experiment(
            load_experiment(_yaml(directory, steps=1, updates=2, validation_interval=1, best_start=0))
        )
        try:
            trainer = experiment.get("trainer")
            with patch("src.evaluation.metrics.pesq.library_status", return_value="pesq is not installed"):
                summary = trainer._validate(experiment._graph.runtime, 0, "trainer")
            self.assertEqual(summary["pesq"]["status"], "unavailable")
            self.assertIsNone(summary["pesq"]["value"])
            decision = trainer._consider_best(0, summary)
            self.assertFalse(decision["selected"])
            self.assertIsNone(trainer.best_pesq)
            self.assertNotEqual(summary["pesq"]["value"], 0)
        finally:
            experiment.close()

    def test_amp_and_mid_update_checkpoint_are_refused(self) -> None:
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        try:
            trainer, _gen, _disc = _manual_trainer(runtime, accumulation=1)
            runtime.precision = "amp_fp16"
            with self.assertRaises(RuntimeFailure) as caught:
                trainer.official_step([_random_batch(LENGTH, 1)], runtime)
            self.assertIn("amp_fp16", str(caught.exception))
            runtime.precision = "fp32"
            trainer._in_official_step = True
            with self.assertRaises(RuntimeFailure):
                trainer._checkpoint(experiment_checkpoint := _FakeCheckpoint(), logical_step=1, label="step")
            self.assertEqual(experiment_checkpoint.saves, [])
        finally:
            runtime.shutdown()

    def test_scheduler_is_independent_of_validation(self) -> None:
        mid = self.base / "sched-mid"
        experiment = build_experiment(
            load_experiment(_yaml(mid, steps=1, updates=4, validation_interval=1, best_start=40))
        )
        phases: list[str] = []
        experiment.get("trainer")._on_boundary_phase = lambda phase: phases.append(phase)
        try:
            result = experiment.run()
        except Exception:
            experiment.close()
            raise
        trainer = experiment.get("trainer")
        self.assertEqual(result.logical_step, 1)
        self.assertEqual(trainer.scheduler_steps, 0)
        self.assertEqual(experiment.get("optim_g").param_groups[0]["lr"], OFFICIAL_LEARNING_RATE)
        self.assertEqual(experiment.get("optim_d").param_groups[0]["lr"], OFFICIAL_LEARNING_RATE)
        self.assertEqual(phases, ["pending_boundary", "in_epoch"])
        payload = _load(mid / "step_00000001.pt")
        self.assertEqual(payload["label"], "step")
        self.assertIsNone(payload["validation_pending"])

    def test_epoch_end_validates_before_the_scheduler(self) -> None:
        directory = self.base / "sched-end"
        experiment = build_experiment(
            load_experiment(_yaml(directory, steps=2, updates=2, validation_interval=2, best_start=0))
        )
        phases: list[str] = []
        experiment.get("trainer")._on_boundary_phase = lambda phase: phases.append(str(phase))
        try:
            experiment.run()
        except Exception:
            experiment.close()
            raise
        trainer = experiment.get("trainer")
        self.assertEqual(trainer.scheduler_steps, 1)
        self.assertAlmostEqual(experiment.get("optim_g").param_groups[0]["lr"], OFFICIAL_LEARNING_RATE * OFFICIAL_GAMMA)
        self.assertAlmostEqual(experiment.get("optim_d").param_groups[0]["lr"], OFFICIAL_LEARNING_RATE * OFFICIAL_GAMMA)
        self.assertEqual(phases, ["pending_boundary", "pending_scheduler", "in_epoch"])
        self.assertEqual(len(trainer.history), 1)
        self.assertEqual(trainer.history[0]["pesq_status"], "ok")
        self.assertEqual(trainer.history[0]["pesq"], -1.0)
        self.assertTrue(trainer.history[0]["selected"])
        self.assertTrue((directory / "best_generator.pt").is_file())
        stored = torch.load(directory / "best_generator.pt", map_location="cpu", weights_only=False)
        self.assertEqual(_max_diff(stored["generator"], experiment.get("generator").state_module.generator.state_dict()), 0.0)

    def test_best_selection_keeps_ties_and_ignores_unavailable(self) -> None:
        directory = self.base / "best-rules"
        experiment = build_experiment(load_experiment(_yaml(directory, steps=1, updates=2, best_start=40)))
        try:
            trainer = experiment.get("trainer")
            early = trainer._consider_best(39, {"pesq": {"status": "ok", "value": 2.0}})
            self.assertFalse(early["selected"])
            self.assertIsNone(trainer.best_pesq)
            trainer.best_checkpoint_start_epoch = 0
            missing = trainer._consider_best(0, {"pesq": {"status": "unavailable", "value": None}})
            self.assertFalse(missing["selected"])
            self.assertIsNone(trainer.best_pesq)
            first = trainer._consider_best(1, {"pesq": {"status": "ok", "value": 1.5}})
            tie = trainer._consider_best(2, {"pesq": {"status": "ok", "value": 1.5}})
            lower = trainer._consider_best(3, {"pesq": {"status": "ok", "value": 1.4}})
            higher = trainer._consider_best(4, {"pesq": {"status": "ok", "value": 1.6}})
            self.assertTrue(first["selected"])
            self.assertFalse(tie["selected"])
            self.assertEqual(tie["reason"], "not_higher")
            self.assertFalse(lower["selected"])
            self.assertTrue(higher["selected"])
            self.assertEqual(trainer.best_pesq, 1.6)
            self.assertEqual(trainer.best_epoch, 4)
        finally:
            experiment.close()

    def test_mid_epoch_and_boundary_resume_match(self) -> None:
        continuous = self.base / "continuous"
        interrupted = self.base / "mid"
        build_experiment(load_experiment(_yaml(continuous, steps=2, updates=4, validation_interval=100))).run()
        first = build_experiment(load_experiment(_yaml(interrupted, steps=2, updates=4, validation_interval=100, stop_after=1)))
        self.assertEqual(first.run().stop_reason, "interrupted")
        second = build_experiment(load_experiment(_yaml(interrupted, steps=2, updates=4, validation_interval=100)))
        self.assertEqual(second.run(resume=interrupted / "step_00000001.pt").stop_reason, "completed")
        _assert_same_checkpoint(continuous / "step_00000002.pt", interrupted / "step_00000002.pt")
        self.assertEqual(_metrics(continuous), _metrics(interrupted))

        boundary = self.base / "boundary"
        full = self.base / "boundary-full"
        build_experiment(load_experiment(_yaml(full, steps=2, updates=2, validation_interval=2, best_start=0))).run()
        stopped = build_experiment(
            load_experiment(_yaml(boundary, steps=2, updates=2, validation_interval=2, best_start=0, stop_after=2))
        )
        self.assertEqual(stopped.run().stop_reason, "interrupted")
        pending = _load(boundary / "step_00000002.pt")
        self.assertEqual(pending["label"], "validation_pending")
        self.assertFalse(pending["states"]["trainer"]["boundary"]["validation_done"])
        self.assertFalse(pending["states"]["trainer"]["boundary"]["scheduler_done"])
        resumed = _cli(
            _yaml(boundary, steps=2, updates=2, validation_interval=2, best_start=0),
            "--resume",
            str(boundary / "step_00000002.pt"),
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        _assert_same_checkpoint(full / "step_00000002.pt", boundary / "step_00000002.pt")
        self.assertEqual(_load(boundary / "step_00000002.pt")["label"], "step")
        self.assertEqual(_metrics(full), _metrics(boundary))

    def test_corrupt_checkpoint_is_refused(self) -> None:
        directory = self.base / "corrupt"
        experiment = build_experiment(
            load_experiment(_yaml(directory, steps=2, updates=2, validation_interval=100, stop_after=1))
        )
        self.assertEqual(experiment.run().stop_reason, "interrupted")
        broken = directory / "step_00000001.pt"
        broken.write_bytes(b"not-a-checkpoint")
        again = build_experiment(load_experiment(_yaml(directory, steps=2, updates=2, validation_interval=100)))
        with self.assertRaises(CheckpointError):
            again.run(resume=broken)

    def test_snapshot_pipeline_matches_one_task_update(self) -> None:
        spec = _write_snapshot(self.base / "snap-root", derived=self.base / "snap-derived", name="phase4d.safe", profile="leakage_safe")
        directory = self.base / "snap-run"
        config = _snapshot_yaml(directory, spec, claim="leakage_safe", profile="leakage_safe")
        experiment = build_experiment(load_experiment(config))
        try:
            data = experiment.get("data")
            view = data.train_epoch(0, start_update=0)
            try:
                batch = view.batch(0)[0]
            finally:
                view.close()
            task = _task()
            self.assertEqual(
                _max_diff(task.generator.state_dict(), experiment.get("generator").state_module.generator.state_dict()),
                0.0,
            )
            torch.manual_seed(19)
            old = task._official([{"clean": batch.targets.clone(), "noisy": batch.inputs.clone()}], 1)
            torch.manual_seed(19)
            trainer = experiment.get("trainer")
            new = trainer.official_step(
                [Batch(inputs=batch.inputs.clone(), targets=batch.targets.clone(), sample_id="s", realization_id="s")],
                experiment._graph.runtime,
            )
            for key in old:
                self.assertEqual(old[key], new[key], key)
            self.assertEqual(
                _max_diff(task.generator.state_dict(), experiment.get("generator").state_module.generator.state_dict()),
                0.0,
            )
        finally:
            experiment.close()

    def test_cpu_gloo_agrees_and_resumes(self) -> None:
        directory = self.base / "gloo"
        config = _yaml(directory, steps=1, updates=2, validation_interval=1, best_start=0, distribute="ddp")
        completed = _torchrun_module(config, "run")
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        ranks = [_json(directory / "worker-rank0.json"), _json(directory / "worker-rank1.json")]
        self.assertEqual([row["average_calls"] for row in ranks], [1, 1])
        self.assertEqual([row["synced_during_generator_backward"] for row in ranks], [False, False])
        self.assertEqual([row["optimizer_updates"]["optim_g"] for row in ranks], [1, 1])
        self.assertEqual([row["optimizer_updates"]["optim_d"] for row in ranks], [1, 1])
        payload = _load(directory / "step_00000001.pt")
        self.assertEqual(payload["world_size"], 2)
        self.assertTrue(payload["parameter_ranks_agree"])
        self.assertEqual(payload["states"]["trainer"]["logical_step"], 1)

        resume_dir = self.base / "gloo-resume"
        first_config = _yaml(resume_dir, steps=2, updates=2, validation_interval=100, distribute="ddp", stop_after=1)
        first = _torchrun_cli(first_config)
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        second = _torchrun_cli(
            _yaml(resume_dir, steps=2, updates=2, validation_interval=100, distribute="ddp"),
            "--resume",
            str(resume_dir / "step_00000001.pt"),
        )
        self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
        restored = _load(resume_dir / "step_00000002.pt")
        self.assertEqual(restored["world_size"], 2)
        self.assertTrue(restored["parameter_ranks_agree"])
        self.assertEqual(restored["states"]["trainer"]["logical_step"], 2)

    def test_nan_rank_and_split_pesq_do_not_hang(self) -> None:
        nan_dir = self.base / "nan"
        config = _yaml(nan_dir, steps=1, updates=2, validation_interval=100, distribute="ddp", timeout=30)
        completed = _torchrun_module(config, "nan")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("rejected its micro-batch", completed.stderr + completed.stdout)
        split_dir = self.base / "pesq-split"
        split = _torchrun_module(
            _yaml(split_dir, steps=1, updates=2, validation_interval=100, distribute="ddp"),
            "pesq_split",
        )
        self.assertEqual(split.returncode, 0, split.stderr + split.stdout)
        payload = _load(split_dir / "step_00000001.pt")
        self.assertTrue(payload["parameter_ranks_agree"])
        self.assertEqual(payload["states"]["trainer"]["optimizer_updates"]["optim_g"], 1)
        self.assertEqual(payload["states"]["trainer"]["optimizer_updates"]["optim_d"], 1)


class _FakeCheckpoint:
    def __init__(self) -> None:
        self.saves: list[str] = []
        self.directory = Path(".")

    def save(self, directory: Path, *, logical_step: int, label: str) -> None:
        del directory, logical_step
        self.saves.append(label)


def _task() -> MPSENetTask:
    torch.manual_seed(0)
    return MPSENetTask(
        {
            "loss": "official",
            "model": dict(ARCH),
            "optimizer": {"betas": [0.8, 0.99], "lr": OFFICIAL_LEARNING_RATE, "weight_decay": OFFICIAL_WEIGHT_DECAY},
            "scheduler": {"gamma": OFFICIAL_GAMMA},
            "seed": 7,
        },
        torch.device("cpu"),
    )


def _manual_trainer(runtime: ProcessRuntime, accumulation: int) -> tuple[object, MPSENetGANGenerator, torch.nn.Module]:
    torch.manual_seed(0)
    generator = MPSENetGANGenerator(seed=7, **ARCH)
    discriminator = MPSENetMetricDiscriminator(seed=7, **ARCH)
    prepared_g = runtime.prepare_model(generator, PreparePolicy(trainable=True, distribute="none"))
    prepared_d = runtime.prepare_model(discriminator, PreparePolicy(trainable=True, distribute="none"))
    optim_g = torch.optim.AdamW(prepared_g.parameters, lr=OFFICIAL_LEARNING_RATE, betas=(0.8, 0.99), weight_decay=OFFICIAL_WEIGHT_DECAY)
    optim_d = torch.optim.AdamW(prepared_d.parameters, lr=OFFICIAL_LEARNING_RATE, betas=(0.8, 0.99), weight_decay=OFFICIAL_WEIGHT_DECAY)
    optim_g._experiment_component_id = "optim_g"
    optim_d._experiment_component_id = "optim_d"
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=OFFICIAL_GAMMA)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=OFFICIAL_GAMMA)

    class Data:
        group_size = accumulation
        epoch = 0
        update_index = 0

        def identity(self) -> dict[str, object]:
            return {"algorithm": "fixed_pair.mp_senet", "profile": "official_faithful", "selection_uses_test": True}

        def train_epoch(self, epoch: int, *, start_update: int):
            del epoch, start_update
            raise RuntimeError

        def validation_epoch(self, epoch: int):
            del epoch
            return iter(())

    from src.experiments.official.mp_senet.trainer import MPSENetGANTrainer

    trainer = MPSENetGANTrainer(
        prepared_g,
        prepared_d,
        Data(),
        optim_g,
        optim_d,
        scheduler_g,
        scheduler_d,
        max_logical_steps=1,
        experiment_claim="official_faithful",
        accumulation=accumulation,
        validation_interval=0,
        checkpoint_interval=1,
        best_checkpoint_start_epoch=40,
    )
    return trainer, generator, discriminator.discriminator


def _compare_update(*, accumulation: int, length: int, stub: bool, sine: bool = False) -> dict[str, float | bool | None]:
    import resource

    def targets(clean, enhanced, sample_rate):
        del enhanced, sample_rate
        return torch.full((clean.shape[0],), 0.25)

    batches = [_tone_batch(length, index) if sine else _random_batch(length, index) for index in range(accumulation)]
    torch.manual_seed(0)
    task = _task_without_reseed()
    init_g = _max_diff(task.generator.state_dict(), task.generator.state_dict())
    torch.manual_seed(0)
    probe_g = MPSENetGANGenerator(seed=7, **ARCH)
    probe_d = MPSENetMetricDiscriminator(seed=7, **ARCH)
    init_generator = _max_diff(task.generator.state_dict(), probe_g.generator.state_dict())
    init_discriminator = _max_diff(task.discriminator.state_dict(), probe_d.discriminator.state_dict())
    old_box: dict[str, list[torch.Tensor | None]] = {}
    _capture(task.optim_d, old_box, "d")
    _capture(task.optim_g, old_box, "g")
    _seed_streams(21)
    patcher = patch("src.experiments.official.mp_senet.reference.metric_targets", targets) if stub else nullpatch()
    with patcher:
        old = task._official([{"clean": batch.targets.clone(), "noisy": batch.inputs.clone()} for batch in batches], accumulation)
    runtime = ProcessRuntime(device="cpu", precision="fp32")
    try:
        trainer, generator, discriminator = _manual_trainer(runtime, accumulation)
        # _manual_trainer rebuilt from the same seed. Compare against the task's initial weights
        # by rebuilding the task comparison modules above; trainer modules match probe if both used seed 7
        # after manual_seed(0). _manual_trainer calls manual_seed(0) itself.
        self_init_g = _max_diff(probe_g.generator.state_dict(), generator.generator.state_dict())
        self_init_d = _max_diff(probe_d.discriminator.state_dict(), discriminator.state_dict())
        if self_init_g != 0.0 or self_init_d != 0.0:
            raise AssertionError(f"rebuild drifted g={self_init_g} d={self_init_d}")
        new_box: dict[str, list[torch.Tensor | None]] = {}
        _capture(trainer.optim_d, new_box, "d")
        _capture(trainer.optim_g, new_box, "g")
        alive = {"value": False}

        def hook(phase: str, payload: object = None) -> None:
            if phase == "after_discriminator_step" and isinstance(payload, list):
                audio = payload[0][1]["audio"]
                alive["value"] = audio.grad_fn is not None and bool(payload[0][1]["mag_hat"].requires_grad)

        trainer._on_update_phase = hook
        _seed_streams(21)
        new_patch = patch("src.experiments.calc.losses.metric_targets", targets) if stub else nullpatch()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with new_patch:
            new = trainer.official_step(
                [Batch(inputs=batch.inputs.clone(), targets=batch.targets.clone(), sample_id=batch.sample_id, realization_id=batch.realization_id) for batch in batches],
                runtime,
            )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        del before
        pesq_target = None if stub else metric_targets(batches[0].targets, batches[0].targets, 16000)
        # The real target is scored on clean versus the enhanced audio inside the update.
        # Report whether that call returned a tensor by repeating it on the stored waveforms
        # only for the sine case, where the library accepts the reference.
        if not stub:
            pesq_target = metric_targets(batches[0].targets, batches[0].inputs, 16000)
        return {
            "discriminator_grad": _grad_max(old_box["d"], new_box["d"]),
            "discriminator_parameters": _max_diff(task.discriminator.state_dict(), discriminator.state_dict()),
            "generator_grad": _grad_max(old_box["g"], new_box["g"]),
            "generator_parameters": _max_diff(task.generator.state_dict(), generator.generator.state_dict()),
            "graph_alive": alive["value"],
            "init_discriminator": init_discriminator,
            "init_generator": init_generator,
            "losses": max(abs(old[key] - new[key]) for key in old),
            "optim_d": _optim_max(task.optim_d, trainer.optim_d),
            "optim_g": _optim_max(task.optim_g, trainer.optim_g),
            "pesq_target": None if pesq_target is None else float(pesq_target.reshape(-1)[0]),
            "rss_kb": float(rss),
            "unused": init_g,
        }
    finally:
        runtime.shutdown()


class nullpatch:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _seed_streams(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _task_without_reseed() -> MPSENetTask:
    return MPSENetTask(
        {
            "loss": "official",
            "model": dict(ARCH),
            "optimizer": {"betas": [0.8, 0.99], "lr": OFFICIAL_LEARNING_RATE, "weight_decay": OFFICIAL_WEIGHT_DECAY},
            "scheduler": {"gamma": OFFICIAL_GAMMA},
            "seed": 7,
        },
        torch.device("cpu"),
    )


def _random_batch(length: int, seed: int) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    noisy = (torch.randn(1, length, generator=generator) * 0.05).contiguous()
    clean = (torch.randn(1, length, generator=generator) * 0.05).contiguous()
    return Batch(inputs=noisy, targets=clean, sample_id=f"b{seed}", realization_id=f"b{seed}")


def _tone_batch(length: int, seed: int) -> Batch:
    time = torch.arange(length, dtype=torch.float32) / 16000
    clean = (0.2 * torch.sin(2 * torch.pi * (180 + seed) * time)).unsqueeze(0).contiguous()
    generator = torch.Generator().manual_seed(seed + 9)
    noisy = (clean + 0.01 * torch.randn(1, length, generator=generator)).contiguous()
    return Batch(inputs=noisy, targets=clean, sample_id=f"t{seed}", realization_id=f"t{seed}")


def _capture(optim: torch.optim.Optimizer, box: dict[str, list[torch.Tensor | None]], key: str) -> None:
    original = optim.step

    def wrapped(*args, **kwargs):
        grads: list[torch.Tensor | None] = []
        for group in optim.param_groups:
            for parameter in group["params"]:
                grads.append(None if parameter.grad is None else parameter.grad.detach().clone())
        box[key] = grads
        return original(*args, **kwargs)

    optim.step = wrapped


def _count_step(original, box: dict[str, int], key: str):
    def wrapped(*args, **kwargs):
        box[key] += 1
        return original(*args, **kwargs)

    return wrapped


def _grad_max(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> float:
    if len(left) != len(right):
        return 1.0
    worst = 0.0
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            return 1.0
        worst = max(worst, float((lhs - rhs).abs().max()))
    return worst


def _optim_max(left: torch.optim.Optimizer, right: torch.optim.Optimizer) -> float:
    left_params = [parameter for group in left.param_groups for parameter in group["params"]]
    right_params = [parameter for group in right.param_groups for parameter in group["params"]]
    worst = 0.0
    for lhs, rhs in zip(left_params, right_params):
        left_state = left.state[lhs]
        right_state = right.state[rhs]
        if set(left_state) != set(right_state):
            return 1.0
        for key in left_state:
            if torch.is_tensor(left_state[key]):
                worst = max(worst, float((left_state[key] - right_state[key]).abs().max()))
            elif left_state[key] != right_state[key]:
                return 1.0
    return worst


def _max_diff(left: dict, right: dict) -> float:
    if set(left) != set(right):
        return 1.0
    worst = 0.0
    for key in left:
        if not torch.is_tensor(left[key]):
            continue
        if left[key].shape != right[key].shape:
            return 1.0
        worst = max(worst, float((left[key].detach().float() - right[key].detach().float()).abs().max()))
    return worst


def _parameter_shift(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    worst = 0.0
    for name, tensor in module.named_parameters():
        worst = max(worst, float((before[name] - tensor.detach()).abs().max()))
    return worst


def _yaml(
    directory: Path,
    *,
    steps: int,
    updates: int,
    validation_interval: int = 100,
    best_start: int = 40,
    distribute: str = "none",
    stop_after: int | None = None,
    accumulation: int = 1,
    profile: str = "official_faithful",
    selection_uses_test: bool = True,
    timeout: int = 60,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(
        _yaml_text(
            directory,
            steps=steps,
            updates=updates,
            validation_interval=validation_interval,
            best_start=best_start,
            distribute=distribute,
            stop_after=stop_after,
            accumulation=accumulation,
            profile=profile,
            selection_uses_test=selection_uses_test,
            timeout=timeout,
        ),
        encoding="utf-8",
    )
    return path


def _yaml_text(
    directory: Path,
    *,
    steps: int,
    updates: int,
    validation_interval: int = 100,
    best_start: int = 40,
    distribute: str = "none",
    stop_after: int | None = None,
    accumulation: int = 1,
    profile: str = "official_faithful",
    selection_uses_test: bool = True,
    timeout: int = 60,
) -> str:
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    claim = profile
    arch = ", ".join(f"{key}: {value}" for key, value in ARCH.items())
    return f"""schema: audio-db-experiment/v1
seed: 7
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
    timeout_seconds: {timeout}
output:
  directory: {directory}
components:
  generator:
    role: module
    class: src.experiments.official.mp_senet.models.MPSENetGANGenerator
    kwargs: {{seed: 7, {arch}}}
    prepare: {{trainable: true, distribute: {distribute}}}
  metric_discriminator:
    role: module
    class: src.experiments.official.mp_senet.models.MPSENetMetricDiscriminator
    kwargs: {{seed: 7, {arch}}}
    prepare: {{trainable: true, distribute: {distribute}}}
  data:
    role: data
    class: tests.experiments.mp_senet_gan_data.WavePairData
    kwargs:
      updates_per_epoch: {updates}
      micro_batch: 1
      length: {LENGTH}
      seed: 17
      profile: {profile}
      selection_uses_test: {str(selection_uses_test).lower()}
      group_size: {accumulation}
  optim_g:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: generator
    kwargs:
      lr: 0.0005
      betas: [0.8, 0.99]
      weight_decay: 0.01
  optim_d:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: metric_discriminator
    kwargs:
      lr: 0.0005
      betas: [0.8, 0.99]
      weight_decay: 0.01
  scheduler_g:
    role: scheduler
    class: torch.optim.lr_scheduler.ExponentialLR
    kwargs:
      gamma: 0.99
    inject:
      optimizer: {{ref: optim_g}}
  scheduler_d:
    role: scheduler
    class: torch.optim.lr_scheduler.ExponentialLR
    kwargs:
      gamma: 0.99
    inject:
      optimizer: {{ref: optim_d}}
  trainer:
    role: trainer
    class: src.experiments.official.mp_senet.trainer.MPSENetGANTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: {accumulation}
      experiment_claim: {claim}
      validation_interval: {validation_interval}
      checkpoint_interval: 1
      best_checkpoint_start_epoch: {best_start}
{stop}    inject:
      generator: {{ref: generator}}
      discriminator: {{ref: metric_discriminator}}
      data: {{ref: data}}
      optim_g: {{ref: optim_g}}
      optim_d: {{ref: optim_d}}
      scheduler_g: {{ref: scheduler_g}}
      scheduler_d: {{ref: scheduler_d}}
"""


def _snapshot_yaml(directory: Path, spec: dict[str, object], *, claim: str, profile: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    arch = ", ".join(f"{key}: {value}" for key, value in ARCH.items())
    path.write_text(
        f"""schema: audio-db-experiment/v1
seed: 7
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
  generator:
    role: module
    class: src.experiments.official.mp_senet.models.MPSENetGANGenerator
    kwargs: {{seed: 7, {arch}}}
    prepare: {{trainable: true, distribute: none}}
  metric_discriminator:
    role: module
    class: src.experiments.official.mp_senet.models.MPSENetMetricDiscriminator
    kwargs: {{seed: 7, {arch}}}
    prepare: {{trainable: true, distribute: none}}
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
  protocol:
    role: other
    class: src.experiments.data.FixedPairProtocol
    kwargs: {{}}
    inject:
      source: {{ref: source}}
  sampler:
    role: other
    class: src.experiments.data.OrderSampler
    kwargs:
      seed: 17
    inject:
      source: {{ref: source}}
      protocol: {{ref: protocol}}
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
      require_profile: {profile}
    inject:
      source: {{ref: source}}
      sampler: {{ref: sampler}}
      protocol: {{ref: protocol}}
      batcher: {{ref: batcher}}
  optim_g:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: generator
    kwargs:
      lr: 0.0005
      betas: [0.8, 0.99]
      weight_decay: 0.01
  optim_d:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: metric_discriminator
    kwargs:
      lr: 0.0005
      betas: [0.8, 0.99]
      weight_decay: 0.01
  scheduler_g:
    role: scheduler
    class: torch.optim.lr_scheduler.ExponentialLR
    kwargs:
      gamma: 0.99
    inject:
      optimizer: {{ref: optim_g}}
  scheduler_d:
    role: scheduler
    class: torch.optim.lr_scheduler.ExponentialLR
    kwargs:
      gamma: 0.99
    inject:
      optimizer: {{ref: optim_d}}
  trainer:
    role: trainer
    class: src.experiments.official.mp_senet.trainer.MPSENetGANTrainer
    kwargs:
      max_logical_steps: 1
      accumulation: 1
      experiment_claim: {claim}
      validation_interval: 100
      checkpoint_interval: 1
      best_checkpoint_start_epoch: 40
    inject:
      generator: {{ref: generator}}
      discriminator: {{ref: metric_discriminator}}
      data: {{ref: data}}
      optim_g: {{ref: optim_g}}
      optim_d: {{ref: optim_d}}
      scheduler_g: {{ref: scheduler_g}}
      scheduler_d: {{ref: scheduler_d}}
""",
        encoding="utf-8",
    )
    return path


def _write_snapshot(root: Path, *, derived: Path, name: str, profile: str) -> dict[str, object]:
    corpora = derived.parent / "corpora"
    generated = derived.parent / "generated"
    corpora.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    assets = []
    cases = []
    total = 4
    for index in range(total):
        noisy = np.linspace(-0.2, 0.5, LENGTH, dtype=np.float32) * ((index % 5) + 1) / 5
        clean = np.linspace(0.4, -0.3, LENGTH, dtype=np.float32) * ((index % 3) + 1) / 5
        noisy_rel = f"toy/{name}/n{index}.wav"
        clean_rel = f"toy/{name}/c{index}.wav"
        noisy_path = derived / noisy_rel
        clean_path = derived / clean_rel
        _write_wav(noisy_path, noisy)
        _write_wav(clean_path, clean)
        for rel, path in ((noisy_rel, noisy_path), (clean_rel, clean_path)):
            assets.append(
                {
                    "asset_id": f"derived/{rel}",
                    "channels": 1,
                    "frames": LENGTH,
                    "integrity": "sha256",
                    "rel_path": rel,
                    "sample_rate": 16000,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        cases.append(
            {
                "build_id": None,
                "case_id": f"{name}-case-{index:02d}",
                "case_role": "recorded",
                "clean_asset_id": f"derived/{clean_rel}",
                "doc_kind": "case",
                "frames": LENGTH,
                "gen_index": None,
                "noise_asset_id": None,
                "noisy_asset_id": f"derived/{noisy_rel}",
                "protocol_id": "protocol",
                "sample_rate": 16000,
                "speaker_id": None,
                "teacher_kind": "dry_clean",
                "utterance_key": None,
            }
        )
    train_ids = [f"{name}-case-{index:02d}" for index in range(2)]
    valid_ids = [f"{name}-case-02"]
    test_ids = [f"{name}-case-03"]
    items = []
    bindings = []
    for slot, ids in (("train.cases", train_ids), ("valid.cases", valid_ids), ("test.cases", test_ids)):
        case_set_id = digest(ids)
        bindings.append({"case_set_id": case_set_id, "material_set_id": None, "slot": slot})
        for position, case_id in enumerate(ids):
            items.append({"case_id": case_id, "case_set_id": case_set_id, "position": position})
    uses_test = profile == "official_faithful"
    claim = "official_reproduction" if uses_test else "leakage_safe_comparison"
    rule = {
        "delivery": "on_demand",
        "epoch_unit": "utterance",
        "global_batch": 1,
        "normalize": "noisy_rms_unit_v1",
        "order": ["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "validation": {"batch": 1, "crop": "full_length"},
        "window": LENGTH,
    }
    dataset_id = digest({"algorithm": "fixed_pair.mp_senet", "name": name, "phase": "4d", "profile": profile})
    dataset = {
        "algorithm": "fixed_pair.mp_senet",
        "algorithm_version": 1,
        "bindings": bindings,
        "claim_label": claim,
        "dataset_id": dataset_id,
        "declared_status": "ready",
        "kind": "fixed_pair",
        "name": name,
        "profile": profile,
        "protocol_id": "protocol",
        "rule_json": rule,
        "selection_uses_test": uses_test,
        "version": 1,
    }
    parts = {
        "assets": assets,
        "case_set_items": items,
        "cases": cases,
        "dataset": dataset,
        "docs": [],
        "generated": [],
        "material_set_members": [],
    }
    folder = root / dataset_id
    folder.mkdir(parents=True)
    digest_value = snapshot_hash(parts)
    manifest = {
        "files": file_hashes(parts),
        "provenance": {"format": FORMAT},
        "schema_version": SCHEMA_VERSION,
        "snapshot_hash": digest_value,
    }
    (folder / "dataset.json").write_text(canonical_text(parts["dataset"]) + "\n", encoding="utf-8")
    (folder / "generated.json").write_text(canonical_text(parts["generated"]) + "\n", encoding="utf-8")
    (folder / "docs.jsonl").write_text("", encoding="utf-8")
    (folder / "manifest.json").write_text(canonical_text(manifest) + "\n", encoding="utf-8")
    np.savez(folder / "assets.npz", **arrays_from_assets(parts["assets"]))
    np.savez(folder / "cases.npz", **arrays_from_cases(parts["cases"]))
    np.savez(folder / "case_set_items.npz", **arrays_from_items(parts["case_set_items"]))
    np.savez(folder / "material_set_members.npz", **arrays_from_members(parts["material_set_members"]))
    loaded = load_snapshot(folder)
    if loaded.snapshot_hash != digest_value:
        raise RuntimeError("synthetic snapshot hash did not round-trip")
    return {
        "corpora": corpora,
        "derived": derived,
        "generated": generated,
        "root": root,
        "selector": f"{name}@1",
    }


def _write_wav(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, 16000, subtype="PCM_16")


def _load(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(path.name)
    return payload


def _metrics(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_same_checkpoint(left: Path, right: Path) -> None:
    first = _load(left)
    second = _load(right)
    for key in ("states", "rng_by_rank", "logical_step", "optimizer_updates", "label"):
        _assert_equal(first[key], second[key], key)


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


def _cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _torchrun_cli(config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _torchrun_command(["-m", "src.experiments.cli", "--config", str(config), *extra])


def _torchrun_module(config: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return _torchrun_command(["-m", "tests.experiments.mp_senet_gan_worker", str(config.parent), mode])


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
