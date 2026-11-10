"""Phase 4-H SEMamba++ GAN connection tests.

CPU comparisons use the reference selective scan and a narrow generator.
The official-width declaration is ``configs/experiments/semambapp_gsr.yaml``.
It is loaded here and not trained: one 3-second official-width step is a
separate measurement. CUDA fused and NCCL live in ``test_cuda_amp`` so a
hidden GPU does not add a skip to this module.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.experiments.api import build_experiment, load_experiment
from src.experiments.data.fixed_pair import MicroBatcher, OrderSampler, SnapshotDataModule, SnapshotSource
from src.experiments.data.online import SemambaProtocol
from src.experiments.data.sample import Batch
from src.experiments.errors import BuildError, CheckpointError, RuntimeFailure
from src.experiments.models.semambapp import SEMambaPPGenerator, scan_execution
from src.experiments.runtime import ProcessRuntime
from src.experiments.calc.semambapp import SEMambaTask

_ROOT = Path(__file__).resolve().parents[2]
_ARCH = {
    "d_conv": 2,
    "d_state": 4,
    "expand": 2,
    "hid_feature": 8,
    "input_channel": 2,
    "n_fft": 400,
    "norm_epsilon": 1e-5,
    "num_tfmamba": 1,
    "output_channel": 1,
    "scan_backend": "reference",
}


class LoopData:
    """Short fixed waveforms for the GAN loop. Rank changes the draw."""

    def __init__(
        self,
        seed: int,
        length: int,
        count: int,
        group_size: int = 1,
        poison_rank: int = -1,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("length must be a positive integer")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be a positive integer")
        rank = int(os.environ.get("RANK", "0"))
        generator = torch.Generator().manual_seed(seed + rank * 1009)
        self.waves = [
            (
                torch.randn(1, length, generator=generator),
                torch.randn(1, length, generator=generator),
            )
            for _ in range(count)
        ]
        self.group_size = group_size
        self.poison_rank = poison_rank
        self.rank = rank
        self.epoch = 0
        self.update_index = 0
        self._closed_views = 0

    def identity(self) -> dict[str, object]:
        return {
            "algorithm": "online.semambapp_gsr",
            "claim_label": "protocol_conformant_substitute_materials",
            "group_size": self.group_size,
        }

    @property
    def plan_hash(self) -> str:
        updates = len(self.waves) - (len(self.waves) % self.group_size)
        return f"loop-{updates}-{self.group_size}"

    def train_epoch(self, epoch: int, start_update: int = 0) -> "_LoopView":
        del epoch
        updates = len(self.waves) - (len(self.waves) % self.group_size)
        if updates < 1:
            raise RuntimeError("epoch does not contain a full update group")
        if start_update < 0 or start_update > updates:
            raise RuntimeError(f"start_update {start_update} is outside 0..{updates}")
        return _LoopView(self, updates, self.plan_hash)

    def validation_epoch(self, epoch: int):
        del epoch
        if self.rank != 0:
            return iter(())
        return iter(self._batch(index, "valid") for index in range(min(1, len(self.waves))))

    def state_dict(self) -> dict[str, int | str]:
        return {
            "epoch": int(self.epoch),
            "plan_hash": self.plan_hash,
            "update_index": int(self.update_index),
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.update_index = int(state["update_index"])

    def _batch(self, index: int, split: str) -> Batch:
        noisy, clean = self.waves[index]
        if self.poison_rank == self.rank:
            noisy = torch.full_like(noisy, float("nan"))
        return Batch(
            inputs=noisy.clone(),
            targets=clean.clone(),
            sample_id=f"{split}-{index}",
            realization_id=f"real-{index}",
        )


class _LoopView:
    def __init__(self, module: LoopData, updates: int, plan_hash: str) -> None:
        self._module = module
        self.updates = updates
        self.plan_hash = plan_hash

    def batch(self, update_index: int) -> list[Batch]:
        if update_index < 0 or update_index >= self.updates:
            raise RuntimeError(f"update index {update_index} is outside the view")
        return [self._module._batch(update_index, "train")]

    def close(self) -> None:
        self._module._closed_views += 1


class SEMambaPPAlgorithmTests(unittest.TestCase):
    def test_factory_source_has_no_model_name_branch(self) -> None:
        factory = (_ROOT / "src" / "experiments" / "factory.py").read_text(encoding="utf-8")
        builder = (_ROOT / "src" / "experiments" / "builder.py").read_text(encoding="utf-8")
        for name in ("semambapp", "SEMamba", "MultiScaleSubbandCQT"):
            self.assertNotIn(name, factory)
            self.assertNotIn(name, builder)

    def test_initial_parameters_match_the_task_in_builder_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = build_experiment(load_experiment(_yaml(Path(directory), steps=1, count=1)))
            try:
                task = _task()
                self.assertEqual(_state_gap(task.generator, experiment.get("generator").state_module.generator), 0.0)
                self.assertEqual(
                    _state_gap(task.discriminator_mrd, experiment.get("mrd").state_module.discriminator),
                    0.0,
                )
                self.assertEqual(
                    _state_gap(task.discriminator_band, experiment.get("cqt").state_module.discriminator),
                    0.0,
                )
                generator_ids = {id(parameter) for parameter in experiment.get("optim_g").param_groups[0]["params"]}
                discriminator_ids = {
                    id(parameter)
                    for group in experiment.get("optim_d").param_groups
                    for parameter in group["params"]
                }
                self.assertFalse(generator_ids & discriminator_ids)
                self.assertEqual(len(experiment.get("optim_d").param_groups), 2)
            finally:
                experiment.close()

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(BuildError):
            SEMambaPPGenerator(seed=1, hid_feature=8, num_tfmamba=1, d_state=4, d_conv=2, expand=2, scan_backend="fast")

    def test_cpu_fused_setting_executes_reference(self) -> None:
        module = SEMambaPPGenerator(
            seed=1,
            hid_feature=8,
            num_tfmamba=1,
            d_state=4,
            d_conv=2,
            expand=2,
            scan_backend="fused",
        )
        report = scan_execution(module.generator, torch.device("cpu"))
        self.assertEqual(report["configured"], "fused")
        self.assertEqual(report["executed"], "reference")

    def test_accumulation_one_matches_the_task(self) -> None:
        _assert_matches_task(self, accumulation=1, count=1)

    def test_accumulation_two_reforwards_and_matches_the_task(self) -> None:
        _assert_matches_task(self, accumulation=2, count=2)

    def test_freeze_is_restored_when_generator_loss_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = build_experiment(load_experiment(_yaml(Path(directory), steps=1, count=1)))
            try:
                trainer = experiment.get("trainer")
                batch = _random_batch()

                def boom(phase: str, payload: object = None) -> None:
                    del payload
                    if phase == "before_generator_backward":
                        raise RuntimeError("boom")

                trainer._on_update_phase = boom
                with self.assertRaises(RuntimeError):
                    trainer.official_step([batch], experiment._graph.runtime)
                self.assertFalse(trainer.frozen_during_generator)
                for module in (experiment.get("mrd").state_module, experiment.get("cqt").state_module):
                    for parameter in module.parameters():
                        self.assertTrue(parameter.requires_grad)
            finally:
                experiment.close()

    def test_scheduler_waits_for_the_epoch_and_skips_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mid = root / "mid"
            build_experiment(load_experiment(_yaml(mid, steps=1, count=4, validation_interval=0))).run()
            saved = torch.load(mid / "step_00000001.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["states"]["trainer"]["scheduler_steps"], 0)
            self.assertEqual(saved["states"]["optim_g"]["param_groups"][0]["lr"], 0.0002)
            self.assertEqual(saved["states"]["optim_d"]["param_groups"][0]["lr"], 0.0002)
            self.assertFalse((mid / "best_generator.pt").exists())

            done = root / "done"
            build_experiment(
                load_experiment(_yaml(done, steps=2, count=2, validation_interval=2, best_start=0))
            ).run()
            finished = torch.load(done / "step_00000002.pt", map_location="cpu", weights_only=False)
            trainer_state = finished["states"]["trainer"]
            self.assertEqual(trainer_state["scheduler_steps"], 1)
            self.assertEqual(trainer_state["phase"], "in_epoch")
            self.assertEqual(len(trainer_state["history"]), 1)
            self.assertFalse(trainer_state["history"][0]["selected"])
            self.assertEqual(trainer_state["history"][0]["reason"], "semambapp_does_not_select_best_generator")
            summary = trainer_state["history"][0]["summary"]
            self.assertEqual(summary["losses"]["generator"]["status"], "ok")
            self.assertEqual(summary["losses"]["discriminator"]["reason"], "validation reports the waveform L1, not the GAN")
            self.assertEqual(summary["si_sdr"]["status"], "ok")
            self.assertIn(summary["pesq"]["status"], {"ok", "na", "unavailable", "error"})
            self.assertIn(summary["stoi"]["status"], {"ok", "na", "unavailable", "error"})
            self.assertEqual(summary["cases"][0]["lengths"], {"input": 2048, "output": 2048, "reference": 2048})
            self.assertAlmostEqual(finished["states"]["optim_g"]["param_groups"][0]["lr"], 0.0002 * 0.99)
            self.assertAlmostEqual(finished["states"]["optim_d"]["param_groups"][0]["lr"], 0.0002 * 0.99)
            self.assertFalse((done / "best_generator.pt").exists())
            self.assertEqual(finished["states"]["data"]["epoch"], 1)
            self.assertEqual(finished["states"]["data"]["update_index"], 0)

    def test_resume_matches_a_continuous_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = root / "continuous"
            interrupted = root / "mid"
            build_experiment(load_experiment(_yaml(continuous, steps=2, count=4, validation_interval=0))).run()
            first = build_experiment(load_experiment(_yaml(interrupted, steps=2, count=4, validation_interval=0, stop_after=1)))
            self.assertEqual(first.run().stop_reason, "interrupted")
            second = build_experiment(load_experiment(_yaml(interrupted, steps=2, count=4, validation_interval=0)))
            self.assertEqual(second.run(resume=interrupted / "step_00000001.pt").stop_reason, "completed")
            _assert_same_checkpoint(continuous / "step_00000002.pt", interrupted / "step_00000002.pt")
            self.assertEqual(_metric_lines(continuous), _metric_lines(interrupted))

    def test_epoch_boundary_resume_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            boundary = root / "boundary"
            build_experiment(load_experiment(_yaml(full, steps=2, count=2, validation_interval=2, best_start=0))).run()
            stopped = build_experiment(
                load_experiment(_yaml(boundary, steps=2, count=2, validation_interval=2, best_start=0, stop_after=2))
            )
            self.assertEqual(stopped.run().stop_reason, "interrupted")
            pending = torch.load(boundary / "step_00000002.pt", map_location="cpu", weights_only=False)
            self.assertEqual(pending["label"], "validation_pending")
            self.assertFalse(pending["states"]["trainer"]["boundary"]["validation_done"])
            self.assertFalse(pending["states"]["trainer"]["boundary"]["scheduler_done"])
            resumed = _cli(
                _yaml(boundary, steps=2, count=2, validation_interval=2, best_start=0),
                "--resume",
                str(boundary / "step_00000002.pt"),
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr[-2000:] + resumed.stdout[-500:])
            _assert_same_checkpoint(full / "step_00000002.pt", boundary / "step_00000002.pt")
            self.assertEqual(_metric_lines(full), _metric_lines(boundary))

    def test_corrupt_checkpoint_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = build_experiment(load_experiment(_yaml(root, steps=2, count=2, validation_interval=0, stop_after=1)))
            self.assertEqual(experiment.run().stop_reason, "interrupted")
            broken = root / "step_00000001.pt"
            broken.write_bytes(b"not-a-checkpoint")
            again = build_experiment(load_experiment(_yaml(root, steps=2, count=2, validation_interval=0)))
            with self.assertRaises(CheckpointError):
                again.run(resume=broken)

    def test_dataset_identity_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_experiment(load_experiment(_yaml(root, steps=2, count=1, validation_interval=0, stop_after=1))).run()
            path = root / "step_00000001.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["dataset_identity"]["data"]["claim_label"] = "other-claim"
            torch.save(payload, path)
            again = build_experiment(load_experiment(_yaml(root, steps=2, count=1, validation_interval=0)))
            with self.assertRaises(CheckpointError) as caught:
                again.run(resume=path)
            self.assertIn("dataset identity", str(caught.exception))

    def test_amp_and_group_size_and_official_claim_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = build_experiment(load_experiment(_yaml(root / "amp", steps=1, count=1)))
            try:
                class _Amp:
                    precision = "amp_fp16"

                with self.assertRaises(RuntimeFailure):
                    experiment.get("trainer")._refuse_amp(_Amp())
            finally:
                experiment.close()
            mismatch = build_experiment(load_experiment(_yaml(root / "group", steps=1, count=2, group_size=2, accumulation=1)))
            with self.assertRaises(RuntimeFailure):
                mismatch.run()
            with self.assertRaises(BuildError):
                build_experiment(load_experiment(_yaml(root / "official", steps=1, count=1, claim="official_reproduction")))

    def test_cpu_gloo_accumulation_agrees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for accumulation, count in ((1, 2), (2, 2)):
                target = root / f"gloo-{accumulation}"
                completed = _torchrun(
                    _yaml(target, steps=1, count=count, group_size=accumulation, accumulation=accumulation, validation_interval=0)
                )
                self.assertEqual(completed.returncode, 0, completed.stderr[-2500:])
                payload = torch.load(target / "step_00000001.pt", map_location="cpu", weights_only=False)
                self.assertTrue(payload["parameter_ranks_agree"])
                self.assertEqual(payload["world_size"], 2)
                trainer_state = payload["states"]["trainer"]
                row = trainer_state["step_log"][0]
                if accumulation == 1:
                    self.assertEqual(row["generator_forwards"], 1)
                    self.assertEqual(row["generator_previews"], 0)
                    self.assertEqual(row["generator_reforwards"], 0)
                else:
                    self.assertEqual(row["generator_forwards"], 0)
                    self.assertEqual(row["generator_previews"], 2)
                    self.assertEqual(row["generator_reforwards"], 2)

    def test_nonfinite_rank_does_not_hang(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "poison"
            completed = _torchrun(
                _yaml(target, steps=1, count=1, poison_rank=1),
                timeout=180,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("rejected its micro-batch", completed.stderr + completed.stdout)

    def test_smoke_snapshot_contract(self) -> None:
        source = SnapshotSource("semambapp.gsr.smoke@1", _ROOT / "data/db/snapshots", "train")
        protocol = SemambaProtocol(source)
        sampler = OrderSampler(source, protocol, seed=1234)
        batcher = MicroBatcher(micro_batch=1, group_size=3)
        data = SnapshotDataModule(source, sampler, protocol, batcher, prefetch=False)
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        try:
            data.setup(runtime)
            identity = data.identity()
            self.assertEqual(identity["algorithm"], "online.semambapp_gsr")
            self.assertEqual(identity["claim_label"], "protocol_conformant_substitute_materials")
            self.assertNotEqual(identity["claim_label"], "official_reproduction")
            count = protocol.material_count("train")
            view = data.train_epoch(0, start_update=0)
            try:
                self.assertEqual(view.updates, count - (count % 3))
                self.assertEqual(view.updates % 3, 0)
                self.assertLess(view.updates, count)
                plan_hash = view.plan_hash
                again = data.train_epoch(0, start_update=0)
                try:
                    self.assertEqual(again.plan_hash, plan_hash)
                finally:
                    again.close()
                first = data.materialize(0, 0, view._groups)
                second = data.materialize(0, 0, view._groups)
            finally:
                view.close()
            self.assertTrue(torch.equal(first.inputs, second.inputs))
            self.assertTrue(torch.equal(first.targets, second.targets))
            self.assertEqual(first.sample_id, second.sample_id)
            self.assertEqual(first.realization_id, second.realization_id)
            self.assertTrue(first.sample_id)
            self.assertTrue(first.realization_id)
            self.assertEqual(tuple(first.inputs.shape), (1, 48000))
            self.assertEqual(tuple(first.targets.shape), (1, 48000))
            self.assertEqual(first.teacher_kind, "dry_clean")
            self.assertEqual(data.update_index, 0)
            valid = SnapshotSource("semambapp.gsr.smoke@1", _ROOT / "data/db/snapshots", "valid")
            valid_protocol = SemambaProtocol(valid)
            valid_sampler = OrderSampler(valid, valid_protocol, seed=1234)
            valid_data = SnapshotDataModule(
                valid,
                valid_sampler,
                valid_protocol,
                MicroBatcher(micro_batch=1, group_size=1),
                prefetch=False,
            )
            valid_data.setup(runtime)
            valid_batches = list(valid_data.validation_epoch(0))
            self.assertGreater(len(valid_batches), 0)
            self.assertEqual(tuple(valid_batches[0].inputs.shape[-1:]), (48000,))
            valid_data.close()
        finally:
            data.close()
            runtime.shutdown()

    def test_one_smoke_microbatch_updates(self) -> None:
        source = SnapshotSource("semambapp.gsr.smoke@1", _ROOT / "data/db/snapshots", "train")
        protocol = SemambaProtocol(source)
        sampler = OrderSampler(source, protocol, seed=1234)
        data = SnapshotDataModule(
            source,
            sampler,
            protocol,
            MicroBatcher(micro_batch=1, group_size=1),
            prefetch=False,
        )
        runtime = ProcessRuntime(device="cpu", precision="fp32")
        data.setup(runtime)
        view = data.train_epoch(0, start_update=0)
        try:
            batch = view.batch(0)[0]
        finally:
            view.close()
        with tempfile.TemporaryDirectory() as directory:
            experiment = build_experiment(load_experiment(_yaml(Path(directory), steps=1, count=1, length=2048)))
            try:
                weight = experiment.get("generator").state_module.generator.state_dict()["TSMamba.0.tlinears.0.weight"]
                before = weight.detach().clone()
                metrics = experiment.get("trainer").official_step([batch], experiment._graph.runtime)
                after = weight.detach().clone()
            finally:
                experiment.close()
        self.assertEqual(metrics["scan_executed"], "reference")
        self.assertEqual(metrics["generator_forwards"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(float(metrics["generator"]))))
        self.assertTrue(torch.isfinite(torch.tensor(float(metrics["discriminator"]))))
        self.assertEqual(tuple(batch.inputs.shape), (1, 48000))
        self.assertGreater(float((before - after).abs().max()), 0.0)
        self.assertEqual(data.update_index, 0)
        data.close()
        runtime.shutdown()

    def test_declaration_uses_the_substitute_snapshot(self) -> None:
        config = load_experiment(_ROOT / "configs" / "experiments" / "semambapp_gsr.yaml")
        text = (_ROOT / "configs" / "experiments" / "semambapp_gsr.yaml").read_text(encoding="utf-8")
        self.assertIn("semambapp.gsr.smoke@1", text)
        self.assertNotIn("semambapp.gsr.official@1", text.split("stays blocked")[-1])
        self.assertEqual(config.components["trainer"].kwargs["max_logical_steps"], 1)
        self.assertEqual(config.components["trainer"].kwargs["claim_label"], "protocol_conformant_substitute_materials")
        self.assertIsNone(config.components["trainer"].kwargs["grad_clip"])
        self.assertEqual(config.components["generator"].kwargs["scan_backend"], "reference")
        self.assertEqual(config.components["generator"].kwargs["hid_feature"], 48)
        self.assertEqual(config.components["optim_d"].parameters_of[0].component, "mrd")
        self.assertEqual(config.components["optim_d"].parameters_of[1].component, "cqt")


def _assert_matches_task(test: unittest.TestCase, *, accumulation: int, count: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        experiment = build_experiment(
            load_experiment(
                _yaml(
                    Path(directory),
                    steps=1,
                    count=count,
                    group_size=accumulation,
                    accumulation=accumulation,
                )
            )
        )
        try:
            task = _task()
            test.assertEqual(_state_gap(task.generator, experiment.get("generator").state_module.generator), 0.0)
            generator = torch.Generator().manual_seed(19)
            waves = [
                (torch.randn(1, 2048, generator=generator), torch.randn(1, 2048, generator=generator))
                for _ in range(accumulation)
            ]
            task_batches = [{"clean": clean.clone(), "noisy": noisy.clone()} for noisy, clean in waves]
            task.capture_update = True
            task.capture_gradients = True
            old = task.train_update(task_batches, accumulation)
            trainer = experiment.get("trainer")
            trainer.capture_gradients = True
            batches = [
                Batch(inputs=noisy.clone(), targets=clean.clone(), sample_id=f"s{index}", realization_id=f"r{index}")
                for index, (noisy, clean) in enumerate(waves)
            ]
            new = trainer.official_step(batches, experiment._graph.runtime)
            for key in old:
                if isinstance(old[key], float):
                    test.assertEqual(old[key], new[key], key)
            if accumulation == 1:
                test.assertEqual(trainer.generator_graph_forwards, 1)
                test.assertEqual(trainer.generator_preview_forwards, 0)
                test.assertEqual(trainer.generator_reforward_forwards, 0)
                test.assertIn("generator_backward_retained", trainer.sync_trace)
            else:
                test.assertEqual(trainer.generator_graph_forwards, 0)
                test.assertEqual(trainer.generator_preview_forwards, accumulation)
                test.assertEqual(trainer.generator_reforward_forwards, accumulation)
                test.assertEqual(trainer.sync_trace.count("generator_preview_nograd"), accumulation)
            test.assertEqual(_state_gap(task.generator, experiment.get("generator").state_module.generator), 0.0)
            test.assertEqual(_state_gap(task.discriminator_mrd, experiment.get("mrd").state_module.discriminator), 0.0)
            test.assertEqual(_state_gap(task.discriminator_band, experiment.get("cqt").state_module.discriminator), 0.0)
            test.assertEqual(
                _adam_gap(task.optim_g, task.generator.parameters(), experiment.get("optim_g"), experiment.get("generator").state_module.generator.parameters()),
                0.0,
            )
            test.assertEqual(
                _adam_gap(
                    task.optim_d,
                    list(task.discriminator_mrd.parameters()) + list(task.discriminator_band.parameters()),
                    experiment.get("optim_d"),
                    list(experiment.get("mrd").state_module.discriminator.parameters())
                    + list(experiment.get("cqt").state_module.discriminator.parameters()),
                ),
                0.0,
            )
            test.assertEqual(set(task.last_capture["generator_grad"]), set(trainer.last_generator_grads))
            for key, value in task.last_capture["generator_grad"].items():
                test.assertEqual(_tensor_gap(value, trainer.last_generator_grads[key]), 0.0, key)
            test.assertEqual(set(task.last_capture["discriminator_grad"]), set(trainer.last_discriminator_grads))
            for key, value in task.last_capture["discriminator_grad"].items():
                test.assertEqual(_tensor_gap(value, trainer.last_discriminator_grads[key]), 0.0, key)
            test.assertEqual(new["scan_configured"], "reference")
            test.assertEqual(new["scan_executed"], "reference")
        finally:
            experiment.close()


def _task() -> SEMambaTask:
    return SEMambaTask(
        {
            "loss": "official",
            "model": {
                "architecture": dict(_ARCH),
                "name": "semambapp",
                "sampling_rate": 16000,
                "stft": {"hop": 100, "win": 400},
                "variant": "semambapp",
            },
            "optimizer": {
                "betas": [0.8, 0.99],
                "lr": 0.0002,
                "name": "adamw",
                "weight_decay": 0.01,
            },
            "scheduler": {"gamma": 0.99, "name": "exponential"},
            "seed": 1234,
        },
        torch.device("cpu"),
    )


def _random_batch() -> Batch:
    generator = torch.Generator().manual_seed(3)
    return Batch(
        inputs=torch.randn(1, 2048, generator=generator),
        targets=torch.randn(1, 2048, generator=generator),
        sample_id="s",
        realization_id="r",
    )


def _yaml(
    directory: Path,
    *,
    steps: int,
    count: int,
    group_size: int = 1,
    accumulation: int = 1,
    validation_interval: int = 0,
    stop_after: int | None = None,
    best_start: int = 40,
    claim: str = "protocol_conformant_substitute_materials",
    poison_rank: int = -1,
    scan_backend: str = "reference",
    device: str = "cpu",
    length: int = 2048,
    deterministic: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    path = directory / "experiment.yaml"
    path.write_text(
        f"""schema: audio-db-experiment/v1
seed: 1234
deterministic: {str(deterministic).lower()}
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: {device}
    precision: fp32
output:
  directory: {directory}
components:
  generator:
    role: module
    class: src.experiments.models.semambapp.SEMambaPPGenerator
    kwargs:
      seed: 1234
      hid_feature: 8
      num_tfmamba: 1
      d_state: 4
      d_conv: 2
      expand: 2
      norm_epsilon: 1.0e-5
      input_channel: 2
      output_channel: 1
      n_fft: 400
      scan_backend: {scan_backend}
      hop: 100
      win: 400
      sampling_rate: 16000
    prepare:
      trainable: true
      distribute: ddp
  mrd:
    role: module
    class: src.experiments.models.semambapp.SEMambaPPMRD
    kwargs:
      seed: 1234
      hid_feature: 8
      num_tfmamba: 1
      d_state: 4
      d_conv: 2
      expand: 2
      norm_epsilon: 1.0e-5
      input_channel: 2
      output_channel: 1
      n_fft: 400
      scan_backend: {scan_backend}
      hop: 100
      win: 400
      sampling_rate: 16000
    prepare:
      trainable: true
      distribute: ddp
  cqt:
    role: module
    class: src.experiments.models.semambapp.SEMambaPPCQTDiscriminator
    kwargs:
      seed: 1234
      hid_feature: 8
      num_tfmamba: 1
      d_state: 4
      d_conv: 2
      expand: 2
      norm_epsilon: 1.0e-5
      input_channel: 2
      output_channel: 1
      n_fft: 400
      scan_backend: {scan_backend}
      hop: 100
      win: 400
      sampling_rate: 16000
    prepare:
      trainable: true
      distribute: ddp
  data:
    role: data
    class: tests.experiments.test_semambapp.LoopData
    kwargs:
      seed: 1234
      length: {length}
      count: {count}
      group_size: {group_size}
      poison_rank: {poison_rank}
  optim_g:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: generator
    kwargs:
      lr: 0.0002
      betas: [0.8, 0.99]
      weight_decay: 0.01
  optim_d:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of:
      - mrd
      - cqt
    kwargs:
      lr: 0.0002
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
    class: src.experiments.semambapp.SEMambaPPTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: {accumulation}
      validation_interval: {validation_interval}
      checkpoint_interval: 1
      best_checkpoint_start_epoch: {best_start}
      claim_label: {claim}
{stop}    inject:
      generator: {{ref: generator}}
      mrd: {{ref: mrd}}
      cqt: {{ref: cqt}}
      data: {{ref: data}}
      optim_g: {{ref: optim_g}}
      optim_d: {{ref: optim_d}}
      scheduler_g: {{ref: scheduler_g}}
      scheduler_d: {{ref: scheduler_d}}
""",
        encoding="utf-8",
    )
    return path


def _state_gap(left: torch.nn.Module, right: torch.nn.Module) -> float:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if set(left_state) != set(right_state):
        return 1.0
    worst = 0.0
    for key in left_state:
        worst = max(worst, _tensor_gap(left_state[key], right_state[key]))
    return worst


def _tensor_gap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        return 1.0
    if not left.dtype.is_floating_point:
        return 0.0 if torch.equal(left.cpu(), right.cpu()) else 1.0
    return float((left.detach().cpu() - right.detach().cpu()).abs().max())


def _adam_gap(left_optim, left_params, right_optim, right_params) -> float:
    worst = 0.0
    for left, right in zip(left_params, right_params, strict=True):
        left_state = left_optim.state[left]
        right_state = right_optim.state[right]
        for key in ("exp_avg", "exp_avg_sq"):
            worst = max(worst, _tensor_gap(left_state[key], right_state[key]))
        if int(left_state["step"]) != int(right_state["step"]):
            return 1.0
    return worst


def _assert_same_checkpoint(left: Path, right: Path) -> None:
    lhs = torch.load(left, map_location="cpu", weights_only=False)
    rhs = torch.load(right, map_location="cpu", weights_only=False)
    for key in ("states", "rng_by_rank", "logical_step", "optimizer_updates", "label"):
        _walk(lhs[key], rhs[key])


def _walk(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or _tensor_gap(left, right) != 0.0:
            raise AssertionError("tensor mismatch")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(sorted(set(left) ^ set(right))[:8])
        for key in left:
            _walk(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError("list mismatch")
        for item, other in zip(left, right, strict=True):
            _walk(item, other)
        return
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError("array mismatch")
        return
    if left != right:
        raise AssertionError(f"{left!r} != {right!r}")


def _metric_lines(directory: Path) -> list[str]:
    return (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()


def _cli(config: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *args],
        cwd=_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _torchrun(config: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
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
            "src.experiments.cli",
            "--config",
            str(config),
        ],
        cwd=_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
