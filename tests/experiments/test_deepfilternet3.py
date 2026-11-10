"""Phase 4-G DeepFilterNet3 connection tests.

Numerical comparisons use the same kernel on both sides. A locally rebuilt
``libdf_rfft.so`` is preferred when it is present. The tracked binary is not
replaced. Fallback success is not reported as a native-kernel match against
libdf; it is a match between the old task and the new trainer on one path.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_RFFT = _ROOT / "build" / "df-rfft" / "release" / "libdf_rfft.so"
if _LOCAL_RFFT.is_file():
    os.environ.setdefault("DF_RFFT_LIBRARY", str(_LOCAL_RFFT))

import torch

from src.experiments.api import Experiment
from src.experiments.config import load_experiment
from src.experiments.data.fixed_pair import MicroBatcher, SnapshotDataModule, SnapshotSource
from src.experiments.data.online import Dfn3IndexSampler, Dfn3Protocol
from src.experiments.deepfilternet3 import (
    component_losses,
    note_early_stop,
    schedule_index,
)
from src.experiments.models.deepfilternet3 import DeepFilterNet3Model
from src.se.models.deepfilternet3.dsp import DFStream, _rfft_kernel, rfft_library_path
from src.experiments.calc.batch_schedule import scheduled_batch_size
from src.experiments.calc.deepfilternet import DeepFilterNetTask, deepfilter_cosine_schedules

_ARCH = {
    "conv_ch": 16,
    "nb_erb": 8,
    "nb_df": 8,
    "emb_hidden_dim": 16,
    "emb_num_layers": 2,
    "df_hidden_dim": 16,
    "df_num_layers": 1,
    "enc_lin_groups": 8,
    "lin_groups": 8,
    "df_order": 5,
}
_SHIPPED = [(0, 16), (2, 24), (5, 32), (10, 64), (20, 128), (40, 9999)]


def _task_config() -> dict:
    return {
        "seed": 7,
        "loss": "official",
        "grad_clip": 1.0,
        "model": {"name": "deepfilternet3", "variant": "deepfilternet3", "architecture": dict(_ARCH)},
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-12,
            "amsgrad": True,
        },
        "scheduler": {
            "name": "deepfilter_cosine",
            "steps_per_epoch": 4,
            "epochs": 120,
            "lr_min": 1e-6,
            "lr_warmup": 1e-4,
            "warmup_epochs": 3,
            "weight_decay_end": 0.01,
        },
    }


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


class DeepFilterNet3LibraryTests(unittest.TestCase):
    def test_native_library_status_is_explicit(self) -> None:
        kernel = _rfft_kernel()
        if kernel is None:
            self.assertFalse(rfft_library_path().is_file() and os.access(rfft_library_path(), os.R_OK))
            return
        self.assertEqual(kernel.df_rfft_abi(), 1)
        self.assertTrue(rfft_library_path().is_file())


class DeepFilterNet3AlgorithmTests(unittest.TestCase):
    def test_factory_source_has_no_model_name_branch(self) -> None:
        factory = (_ROOT / "src" / "experiments" / "factory.py").read_text(encoding="utf-8")
        builder = (_ROOT / "src" / "experiments" / "builder.py").read_text(encoding="utf-8")
        for name in ("deepfilternet3", "DeepFilterNet3", "DfNet"):
            self.assertNotIn(name, factory)
            self.assertNotIn(name, builder)

    def test_forward_dsp_loss_clip_and_adamw_match_the_task(self) -> None:
        torch.manual_seed(0)
        clean = torch.randn(1, 4800)
        noisy = clean + 0.05 * torch.randn(1, 4800)
        task = DeepFilterNetTask(_task_config(), torch.device("cpu"))
        model = DeepFilterNet3Model(seed=7, architecture=_ARCH)
        model.generator.load_state_dict(task.generator.state_dict())
        self.assertTrue(any("df_fc_a" in name for name in model.export_for_inference()))
        params = model.generator.params
        stream = DFStream(
            params.sr, params.fft_size, params.hop_size, params.nb_erb, params.nb_df,
            params.min_nb_freqs, params.norm_tau, torch.device("cpu"),
        )
        fresh = DFStream(
            params.sr, params.fft_size, params.hop_size, params.nb_erb, params.nb_df,
            params.min_nb_freqs, params.norm_tau, torch.device("cpu"),
        )
        padded = torch.nn.functional.pad(noisy, (0, params.fft_size))
        stream.reset()
        left = stream.features(stream.analyze(padded))
        fresh.reset()
        right = fresh.features(fresh.analyze(padded))
        for lhs, rhs in zip(left, right):
            self.assertEqual(_max_diff(lhs, rhs), 0.0)
        parts = component_losses(stream, model, clean, noisy)
        reference = task._loss(task.generator, clean.clone(), noisy.clone())
        self.assertEqual(_max_diff(parts["total"], reference), 0.0)
        self.assertEqual(_max_diff(parts["spectral"] + parts["local_snr"], parts["total"]), 0.0)
        enhanced, mask, lsnr, coefs = model.generator(*task._analyze_features(noisy)[:3])
        self.assertEqual(enhanced.shape[0], 1)
        self.assertEqual(mask.shape[-1], params.nb_erb)
        self.assertEqual(lsnr.shape[0], 1)
        self.assertGreater(coefs.numel(), 0)
        self.assertTrue(torch.isfinite(enhanced).all())
        clone = DeepFilterNet3Model(seed=7, architecture=_ARCH)
        clone.generator.load_state_dict(task.generator.state_dict())
        optimizer = torch.optim.AdamW(
            clone.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-12, amsgrad=True,
        )
        for group in optimizer.param_groups:
            group["lr"] = task._df_lrs[0]
            group["weight_decay"] = task._df_wds[0]
        optimizer.zero_grad(set_to_none=True)
        loss = component_losses(stream, clone, clean, noisy)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clone.parameters(), 1.0)
        task.train_update([{"clean": clean, "noisy": noisy}], 1)
        optimizer.step()
        paired = zip(clone.parameters(), task.generator.parameters())
        self.assertEqual(max(_max_diff(left, right) for left, right in paired), 0.0)
        self.assertEqual(optimizer.param_groups[0]["lr"], task.optim_g.param_groups[0]["lr"])
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], task.optim_g.param_groups[0]["weight_decay"])
        state_diff = 0.0
        left_state = [value for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)]
        right_state = [value for state in task.optim_g.state.values() for value in state.values() if torch.is_tensor(value)]
        self.assertEqual(len(left_state), len(right_state))
        for lhs, rhs in zip(left_state, right_state):
            state_diff = max(state_diff, _max_diff(lhs, rhs))
        self.assertEqual(state_diff, 0.0)

    def test_cosine_index_is_not_a_cumulative_update_count(self) -> None:
        config = _task_config()
        rates, decays = deepfilter_cosine_schedules(config)
        self.assertEqual(rates[0], 1e-4)
        self.assertAlmostEqual(rates[3 * 4 - 1], 0.001)
        self.assertEqual(len(rates), 4 * 120)
        self.assertEqual(len(decays), len(rates))
        self.assertEqual(schedule_index(1, 0, 2), 2)
        self.assertNotEqual(schedule_index(1, 0, 2), 4)
        rebuilt = dict(config)
        rebuilt["scheduler"] = dict(config["scheduler"])
        rebuilt["scheduler"]["steps_per_epoch"] = 2
        retargeted, _retargeted_decay = deepfilter_cosine_schedules(rebuilt)
        self.assertEqual(retargeted[schedule_index(1, 0, 2)], retargeted[2])
        self.assertNotEqual(len(retargeted), len(rates))

    def test_batch_thresholds_match_the_shipped_cap(self) -> None:
        self.assertEqual(scheduled_batch_size(0, 64, _SHIPPED), 16)
        self.assertEqual(scheduled_batch_size(2, 64, _SHIPPED), 24)
        self.assertEqual(scheduled_batch_size(5, 64, _SHIPPED), 32)
        self.assertEqual(scheduled_batch_size(10, 64, _SHIPPED), 64)
        self.assertEqual(scheduled_batch_size(20, 64, _SHIPPED), 64)
        self.assertEqual(scheduled_batch_size(40, 64, _SHIPPED), 64)

    def test_early_stop_resets_on_a_strict_improvement(self) -> None:
        metric, count, stop = note_early_stop(None, 0, 1.0, 2)
        self.assertEqual((metric, count, stop), (1.0, 0, False))
        metric, count, stop = note_early_stop(metric, count, 1.0, 2)
        self.assertEqual((count, stop), (1, False))
        metric, count, stop = note_early_stop(metric, count, 0.5, 2)
        self.assertEqual((metric, count, stop), (0.5, 0, False))
        metric, count, stop = note_early_stop(metric, count, 0.5, 2)
        metric, count, stop = note_early_stop(metric, count, 0.5, 2)
        self.assertTrue(stop)
        self.assertEqual(count, 2)

    def test_smoke_plan_changes_width_without_reusing_the_old_groups(self) -> None:
        source = SnapshotSource(snapshot="dfn3.online.smoke@1", snapshot_root="data/db/snapshots", split="train")
        sampler = Dfn3IndexSampler(source, seed=7, global_batch=None)
        protocol = Dfn3Protocol(source)
        data = SnapshotDataModule(source, sampler, protocol, MicroBatcher(1, 1, True), prefetch=False)
        data.setup(_Runtime())
        try:
            self.assertEqual(data.identity()["dataset_name"], "dfn3.online.smoke")
            self.assertEqual(data.identity()["claim_label"], "protocol_conformant_substitute_materials")
            first = data.train_epoch(0, start_update=0)
            try:
                first_hash = first.plan_hash
                first_groups = first._groups
                sample = first_groups[0][0]
                rendered = protocol.render(sample)
                batch = first.batch(0)[0]
                self.assertEqual(batch.inputs.shape[-1], 144000)
                self.assertEqual(int(batch.inputs.shape[-1]), rendered[0].shape[0])
                self.assertEqual(float(torch.from_numpy(rendered[0]).sub(batch.inputs[0]).abs().max()), 0.0)
                self.assertEqual(float(torch.from_numpy(rendered[1]).sub(batch.targets[0]).abs().max()), 0.0)
            finally:
                first.close()
            saved = data.identity()
            data.rebind_micro_batch(2)
            self.assertEqual(data.batcher.micro_batch, 2)
            second = data.train_epoch(0, start_update=0)
            try:
                self.assertNotEqual(second.plan_hash, first_hash)
                self.assertNotEqual(second._groups, first_groups)
                self.assertLess(second.updates, first.updates)
            finally:
                second.close()
            data.align_scheduled_batch(saved)
            restored = data.train_epoch(0, start_update=0)
            try:
                self.assertEqual(restored.plan_hash, first_hash)
                self.assertEqual(restored._groups, first_groups)
            finally:
                restored.close()
        finally:
            data.close()


class DeepFilterNet3ExperimentTests(unittest.TestCase):
    def test_parameter_identity_and_resume_on_the_smoke_snapshot(self) -> None:
        directory = Path("/tmp/phase4g/resume")
        if directory.exists():
            for child in directory.iterdir():
                child.unlink()
        else:
            directory.mkdir(parents=True)
        path = directory / "experiment.yaml"
        path.write_text(_tiny_yaml(directory, stop_after=1), encoding="utf-8")
        first = Experiment(load_experiment(path))
        result = first.run()
        self.assertEqual(result.logical_step, 1)
        experiment = first
        del experiment
        model = None
        built = Experiment(load_experiment(path))
        built.build()
        module = built.get("model").state_module
        optimizer = built.get("optimizer")
        self.assertEqual(
            {id(parameter) for parameter in optimizer.param_groups[0]["params"]},
            {id(parameter) for parameter in module.parameters()},
        )
        built.close()
        checkpoint = directory / "step_00000001.pt"
        self.assertTrue(checkpoint.is_file())
        path.write_text(_tiny_yaml(directory, stop_after=2), encoding="utf-8")
        second = Experiment(load_experiment(path))
        resumed = second.run(resume=checkpoint)
        self.assertEqual(resumed.logical_step, 2)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(payload["format"], "audio-db-experiment-v1")
        self.assertIsInstance(payload["provenance_sha256"], str)
        self.assertEqual(
            hashlib.sha256((directory / "data_provenance.json").read_bytes()).hexdigest(),
            payload["provenance_sha256"],
        )
        metrics = (directory / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(metrics), 2)

    def test_epoch_boundary_rebinds_before_the_next_plan(self) -> None:
        directory = Path("/tmp/phase4g/boundary")
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            child.unlink()
        config = directory / "experiment.yaml"
        config.write_text(_boundary_yaml(directory), encoding="utf-8")
        result = Experiment(load_experiment(config)).run()
        self.assertEqual(result.logical_step, 9)
        payload = torch.load(directory / "step_00000009.pt", map_location="cpu", weights_only=False)
        identity = payload["dataset_identity"]["data"]
        self.assertEqual(identity["micro_batch"], 2)
        self.assertEqual(payload["states"]["data"]["epoch"], 1)
        self.assertEqual(payload["states"]["data"]["update_index"], 1)
        self.assertEqual(payload["states"]["trainer"]["schedule_steps"], 4)
        lines = (directory / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 9)
        self.assertTrue((directory / "valid.loss.best.pt").is_file())

    def test_cpu_gloo_one_step_agrees(self) -> None:
        directory = Path("/tmp/phase4g/gloo")
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            child.unlink()
        config = directory / "experiment.yaml"
        config.write_text(_tiny_yaml(directory, stop_after=None, world_batch=2), encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
        completed = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", "2", "--master_port", str(_free_port()),
                "-m", "src.experiments.cli", "--config", str(config),
            ],
            cwd=_ROOT, env=env, check=False, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        payload = torch.load(directory / "step_00000001.pt", map_location="cpu", weights_only=False)
        self.assertTrue(payload["parameter_ranks_agree"])
        self.assertEqual(payload["world_size"], 2)


def _tiny_yaml(directory: Path, stop_after: int | None, world_batch: int = 1) -> str:
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    return f"""schema: audio-db-experiment/v1
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
  model:
    role: module
    class: src.experiments.models.deepfilternet3.DeepFilterNet3Model
    kwargs:
      seed: 7
      variant: deepfilternet3
      architecture:
        conv_ch: 16
        nb_erb: 8
        nb_df: 8
        emb_hidden_dim: 16
        emb_num_layers: 2
        df_hidden_dim: 16
        df_num_layers: 1
        enc_lin_groups: 8
        lin_groups: 8
        df_order: 5
    prepare:
      trainable: true
      distribute: ddp
      find_unused_parameters: true
  source:
    role: other
    class: src.experiments.data.SnapshotSource
    kwargs:
      snapshot: dfn3.online.smoke@1
      snapshot_root: data/db/snapshots
      split: train
  sampler:
    role: other
    class: src.experiments.data.Dfn3IndexSampler
    kwargs:
      seed: 7
      global_batch: {world_batch}
    inject:
      source: {{ref: source}}
  protocol:
    role: other
    class: src.experiments.data.Dfn3Protocol
    kwargs: {{}}
    inject:
      source: {{ref: source}}
  batcher:
    role: other
    class: src.experiments.data.MicroBatcher
    kwargs:
      micro_batch: 1
      group_size: 1
      keep_partial: true
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
      weight_decay: 1.0e-12
      amsgrad: true
  trainer:
    role: trainer
    class: src.experiments.deepfilternet3.DeepFilterNet3Trainer
    kwargs:
      max_logical_steps: 2
      accumulation: 1
      grad_clip: 1.0
      max_epochs: 120
      early_stopping_patience: 25
      steps_per_epoch: 4
      epochs: 120
      lr_min: 1.0e-6
      lr_warmup: 1.0e-4
      warmup_epochs: 3
      weight_decay_end: 0.01
      batch_size_limit: 64
      checkpoint_interval: 1
{stop}    inject:
      model: {{ref: model}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""


def _boundary_yaml(directory: Path) -> str:
    text = _tiny_yaml(directory, stop_after=None)
    text = text.replace("      max_logical_steps: 2\n", "      max_logical_steps: 9\n")
    text = text.replace(
        "      batch_size_limit: 64\n",
        "      batch_size_limit: 64\n"
        "      batch_schedule:\n"
        "        - [0, 1]\n"
        "        - [1, 2]\n",
    )
    return text.replace("      global_batch: 1\n", "")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Runtime:
    world_size = 1
    rank = 0
    device = torch.device("cpu")


if __name__ == "__main__":
    unittest.main()
