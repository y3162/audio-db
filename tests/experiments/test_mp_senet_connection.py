"""Phase 4-A: MP-SENet generator connected through SnapshotDataModule.

The objective is waveform L1. It is not the official GAN loss.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import BuildError, CheckpointError, RuntimeFailure, StateError
from src.experiments.models.mp_senet import MPSENetWaveformAdapter
from src.experiments.runtime import PreparedModel
from src.reproduction.ids import canonical_text, digest
from src.reproduction.snapshot_reader import (
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
from src.se.models.mp_senet.model import MPNet
from src.training.se.mp_senet import MPSENetTask
from tests.training.support import tiny_config

ROOT = Path(__file__).resolve().parents[2]
WAVEFORM_LENGTH = 64
MODEL = {
    "dense_channel": 4,
    "num_tsconformers": 1,
    "n_fft": 32,
    "hop_size": 8,
    "win_size": 32,
    "compress_factor": 0.3,
    "beta": 2.0,
    "sampling_rate": 16000,
}


class MPSENetConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)
        cls.spec = _write_snapshot(cls.base / "snapshots", derived=cls.base / "derived", name="phase4a.mp")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_committed_yaml_declares_the_connection(self) -> None:
        config = load_experiment(ROOT / "configs/experiments/mp_senet_l1.yaml")
        model = config.components["model"]
        self.assertEqual(model.class_path, "src.experiments.models.mp_senet.MPSENetWaveformAdapter")
        self.assertEqual(model.kwargs["sampling_rate"], 16000)
        self.assertEqual(config.components["loss"].class_path, "torch.nn.L1Loss")
        self.assertEqual(config.components["optimizer"].class_path, "torch.optim.AdamW")
        self.assertEqual(config.components["data"].class_path, "src.experiments.data.SnapshotDataModule")
        self.assertEqual(config.components["trainer"].class_path, "src.experiments.trainers.StandardTrainer")
        self.assertEqual(config.components["source"].kwargs["snapshot"], "mp_senet.vb28.ondemand.safe@1")
        self.assertEqual(config.components["source"].kwargs["snapshot_root"], "data/db/snapshots")

    def test_build_uses_mpnet_parameters_once(self) -> None:
        experiment = build_experiment(load_experiment(_write_yaml(self.base / "build", steps=1)))
        try:
            prepared = experiment.get("model")
            self.assertIsInstance(prepared, PreparedModel)
            adapter = prepared.state_module
            self.assertIsInstance(adapter, MPSENetWaveformAdapter)
            self.assertIs(prepared.module, adapter)
            self.assertIsInstance(adapter.generator, MPNet)
            generator_ids = [id(parameter) for parameter in adapter.generator.parameters()]
            adapter_ids = [id(parameter) for parameter in adapter.parameters()]
            optimizer_ids = [
                id(parameter)
                for group in experiment.get("optimizer").param_groups
                for parameter in group["params"]
            ]
            prepared_ids = [id(parameter) for parameter in prepared.parameters]
            self.assertEqual(len(generator_ids), len(set(generator_ids)))
            self.assertEqual(adapter_ids, generator_ids)
            self.assertEqual(optimizer_ids, generator_ids)
            self.assertEqual(prepared_ids, generator_ids)
            self.assertEqual(adapter.n_fft, 32)
            self.assertEqual(adapter.hop_size, 8)
            self.assertEqual(adapter.win_size, 32)
            self.assertEqual(adapter.compress_factor, 0.3)
            self.assertEqual(adapter.sampling_rate, 16000)
        finally:
            experiment.close()

    def test_snapshot_batch_forward_keeps_the_graph(self) -> None:
        experiment = build_experiment(load_experiment(_write_yaml(self.base / "forward", steps=1)))
        try:
            adapter = experiment.get("model").state_module
            data = experiment.get("data")
            self.assertEqual(data.source.dataset()["rule_json"]["sample_rate"], adapter.sampling_rate)
            view = data.train_epoch(0, start_update=0)
            try:
                batch = view.batch(0)[0]
            finally:
                view.close()
            self.assertEqual(tuple(batch.inputs.shape), (1, WAVEFORM_LENGTH))
            self.assertEqual(tuple(batch.targets.shape), tuple(batch.inputs.shape))
            self.assertEqual(batch.inputs.dtype, torch.float32)
            self.assertEqual(batch.inputs.device.type, "cpu")

            def rejected(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("inference entry was called")

            with (
                patch("src.se.load_enhancer", rejected),
                patch("src.se.api.load_enhancer", rejected),
                patch("src.se.api._MPEnhancer._enhance_one", rejected),
            ):
                enhanced = adapter(batch.inputs)
            self.assertEqual(tuple(enhanced.shape), tuple(batch.inputs.shape))
            self.assertEqual(enhanced.dtype, torch.float32)
            self.assertEqual(enhanced.device.type, "cpu")
            self.assertTrue(bool(torch.isfinite(enhanced).all()))
            self.assertIsNotNone(enhanced.grad_fn)
        finally:
            experiment.close()

    def test_l1_backward_reaches_every_generator_parameter(self) -> None:
        adapter = _adapter()
        adapter.train()
        noisy = torch.randn(1, WAVEFORM_LENGTH)
        clean = torch.randn(1, WAVEFORM_LENGTH)
        enhanced = adapter(noisy)
        loss = F.l1_loss(enhanced, clean)
        self.assertTrue(bool(torch.isfinite(loss).all()))
        loss.backward()
        missing = []
        for name, parameter in adapter.generator.named_parameters():
            gradient = parameter.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum()) == 0:
                missing.append(name)
        self.assertEqual(missing, [])

    def test_adamw_changes_parameters_with_nonzero_gradients(self) -> None:
        directory = self.base / "update"
        experiment = build_experiment(load_experiment(_write_yaml(directory, steps=1)))
        adapter = experiment.get("model").state_module
        before = {name: parameter.detach().clone() for name, parameter in adapter.generator.named_parameters()}
        result = experiment.run()
        self.assertEqual(result.logical_step, 1)
        self.assertEqual(result.optimizer_updates, {"optimizer": 1})
        optimizer = experiment.get("optimizer")
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        changed = []
        for name, parameter in adapter.generator.named_parameters():
            gradient = parameter.grad
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.0)
            self.assertFalse(torch.equal(before[name], parameter.detach()))
            self.assertTrue(bool(torch.isfinite(parameter).all()))
            changed.append(name)
            state = optimizer.state[parameter]
            self.assertIn("exp_avg", state)
            self.assertTrue(bool(torch.isfinite(state["exp_avg"]).all()))
        self.assertEqual(len(changed), len(before))
        rows = _metrics(directory)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["logical_step"], 1)
        self.assertTrue(np.isfinite(rows[0]["loss"]))
        self.assertEqual(rows[0]["optimizer_updates"], {"optimizer": 1})

    def test_forward_matches_the_current_generate_path(self) -> None:
        torch.manual_seed(19)
        adapter = _adapter()
        noisy = torch.randn(2, WAVEFORM_LENGTH)
        clean = torch.randn(2, WAVEFORM_LENGTH)
        task = MPSENetTask(tiny_config(self.base / "task-ckpt"), torch.device("cpu"))
        task.generator.load_state_dict(adapter.generator.state_dict())
        self.assertEqual(task.spec["n_fft"], adapter.n_fft)
        self.assertEqual(task.spec["hop_size"], adapter.hop_size)
        self.assertEqual(task.spec["win_size"], adapter.win_size)
        self.assertEqual(task.spec["compress_factor"], adapter.compress_factor)
        with torch.no_grad():
            enhanced = adapter(noisy)
            reference = task._generate(clean, noisy, generator=task.generator)["audio"]
        self.assertEqual(tuple(noisy.shape), (2, WAVEFORM_LENGTH))
        self.assertEqual(tuple(enhanced.shape), tuple(reference.shape))
        self.assertEqual(enhanced.dtype, torch.float32)
        self.assertEqual(enhanced.device.type, "cpu")
        waveform_error = float((enhanced - reference).abs().max())
        loss_error = float((F.l1_loss(enhanced, clean) - F.l1_loss(reference, clean)).abs())
        self.assertEqual(waveform_error, 0.0)
        self.assertEqual(loss_error, 0.0)
        adapter.train()
        task.generator.train()
        left_optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=0.0)
        right_optimizer = torch.optim.AdamW(task.generator.parameters(), lr=0.001, weight_decay=0.0)
        left_optimizer.zero_grad(set_to_none=True)
        right_optimizer.zero_grad(set_to_none=True)
        F.l1_loss(adapter(noisy), clean).backward()
        F.l1_loss(task._generate(clean, noisy, generator=task.generator)["audio"], clean).backward()
        left_optimizer.step()
        right_optimizer.step()
        for left_parameter, right_parameter in zip(
            adapter.generator.parameters(), task.generator.parameters(), strict=True
        ):
            self.assertEqual(float((left_parameter.detach() - right_parameter.detach()).abs().max()), 0.0)

    def test_checkpoint_resume_matches_a_continuous_run(self) -> None:
        continuous_dir = self.base / "continuous"
        interrupted_dir = self.base / "interrupted"
        continuous = build_experiment(load_experiment(_write_yaml(continuous_dir, steps=2)))
        continuous_result = continuous.run()
        self.assertEqual(continuous_result.logical_step, 2)
        self.assertEqual(continuous_result.optimizer_updates, {"optimizer": 2})

        interrupted = build_experiment(
            load_experiment(_write_yaml(interrupted_dir, steps=2, stop_after=1))
        )
        interrupted_result = interrupted.run()
        self.assertEqual(interrupted_result.stop_reason, "interrupted")
        self.assertEqual(interrupted_result.logical_step, 1)
        saved = interrupted_dir / "step_00000001.pt"
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        self.assertEqual(payload["format"], "audio-db-experiment-v1")
        self.assertTrue(any(key.startswith("generator.") for key in payload["states"]["model"]))
        self.assertIn("state", payload["states"]["optimizer"])
        self.assertEqual(payload["states"]["trainer"]["logical_step"], 1)
        self.assertIsInstance(payload["states"]["data"]["plan_hash"], str)
        self.assertTrue(payload["states"]["data"]["plan_hash"])
        self.assertIn("snapshot_hash", payload["states"]["data"]["identity"])
        self.assertEqual(payload["dataset_identity"]["data"]["snapshot_hash"], payload["states"]["data"]["identity"]["snapshot_hash"])
        self.assertEqual(len(payload["rng_by_rank"]), 1)
        self.assertFalse((interrupted_dir / "complete.json").is_file())

        tampered = interrupted_dir / "tampered_identity.pt"
        identity_payload = torch.load(saved, map_location="cpu", weights_only=False)
        identity_payload["dataset_identity"]["data"]["snapshot_hash"] = "0" * 64
        torch.save(identity_payload, tampered)
        rejected = build_experiment(load_experiment(_write_yaml(interrupted_dir, steps=2)))
        with self.assertRaises(CheckpointError):
            rejected.run(resume=tampered)

        shape_payload = torch.load(saved, map_location="cpu", weights_only=False)
        key = next(item for item in shape_payload["states"]["model"] if item.startswith("generator."))
        shape_payload["states"]["model"][key] = torch.zeros(3)
        shape_path = interrupted_dir / "tampered_state.pt"
        torch.save(shape_payload, shape_path)
        rejected_state = build_experiment(load_experiment(_write_yaml(interrupted_dir, steps=2)))
        with self.assertRaises(CheckpointError):
            rejected_state.run(resume=shape_path)

        resumed = build_experiment(load_experiment(_write_yaml(interrupted_dir, steps=2)))
        resumed_result = resumed.run(resume=saved)
        self.assertEqual(resumed_result.logical_step, 2)
        self.assertEqual(resumed_result.optimizer_updates, {"optimizer": 2})
        left = torch.load(continuous_dir / "step_00000002.pt", map_location="cpu", weights_only=False)
        right = torch.load(interrupted_dir / "step_00000002.pt", map_location="cpu", weights_only=False)
        _assert_same_tensors(left["states"]["model"], right["states"]["model"])
        _assert_same_tensors(left["states"]["optimizer"], right["states"]["optimizer"])
        self.assertEqual(left["states"]["trainer"], right["states"]["trainer"])
        self.assertEqual(left["states"]["data"]["plan_hash"], right["states"]["data"]["plan_hash"])
        self.assertEqual(left["dataset_identity"], right["dataset_identity"])
        self.assertEqual(_metrics(continuous_dir), _metrics(interrupted_dir))

    def test_rejects_invalid_shapes_settings_and_parameter_targets(self) -> None:
        adapter = _adapter()
        with self.assertRaises(RuntimeFailure):
            adapter(torch.randn(WAVEFORM_LENGTH))
        with self.assertRaises(RuntimeFailure):
            adapter(torch.randn(1, 1, WAVEFORM_LENGTH))
        with self.assertRaises(RuntimeFailure):
            adapter(torch.randn(1, 60))
        with self.assertRaises(BuildError):
            MPSENetWaveformAdapter(**{**MODEL, "dense_channel": 2})
        with self.assertRaises(BuildError):
            MPSENetWaveformAdapter(**{**MODEL, "win_size": 64})
        with self.assertRaises(BuildError):
            MPSENetWaveformAdapter(**{**MODEL, "sampling_rate": 8000})
        with self.assertRaises(BuildError):
            MPSENetWaveformAdapter(**{**MODEL, "n_fft": 0})

        mismatch = build_experiment(load_experiment(_write_yaml(self.base / "identity", steps=1)))
        try:
            data = mismatch.get("data")
            state = data.state_dict()
            state["identity"]["snapshot_hash"] = "f" * 64
            with self.assertRaises(StateError):
                data.load_state_dict(state)
        finally:
            mismatch.close()

        with self.assertRaises(BuildError):
            build_experiment(load_experiment(_write_yaml(self.base / "bad-opt", steps=1, parameters_of="loss")))
        with self.assertRaises(BuildError):
            build_experiment(
                load_experiment(_write_yaml(self.base / "frozen-opt", steps=1, trainable=False))
            )


def _adapter() -> MPSENetWaveformAdapter:
    return MPSENetWaveformAdapter(**MODEL)


def _write_yaml(
    directory: Path,
    *,
    steps: int,
    stop_after: int | None = None,
    parameters_of: str = "model",
    trainable: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    text = f"""schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
output:
  directory: "{directory}"
components:
  model:
    role: module
    class: src.experiments.models.mp_senet.MPSENetWaveformAdapter
    kwargs:
      dense_channel: 4
      num_tsconformers: 1
      n_fft: 32
      hop_size: 8
      win_size: 32
      compress_factor: 0.3
      beta: 2.0
      sampling_rate: 16000
    prepare: {{trainable: {str(trainable).lower()}, distribute: none}}
  loss:
    role: loss
    class: torch.nn.L1Loss
  source:
    role: other
    class: src.experiments.data.SnapshotSource
    kwargs:
      snapshot: {MPSENetConnectionTests.spec["selector"]}
      snapshot_root: {MPSENetConnectionTests.spec["root"]}
      split: train
      derived: {MPSENetConnectionTests.spec["derived"]}
      corpora: {MPSENetConnectionTests.spec["corpora"]}
      generated: {MPSENetConnectionTests.spec["generated"]}
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
      require_profile: leakage_safe
    inject:
      source: {{ref: source}}
      sampler: {{ref: sampler}}
      protocol: {{ref: protocol}}
      batcher: {{ref: batcher}}
  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: {parameters_of}
    kwargs:
      lr: 0.001
      weight_decay: 0.0
  trainer:
    role: trainer
    class: src.experiments.trainers.StandardTrainer
    kwargs:
      max_logical_steps: {steps}
      accumulation: 1
{stop}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
"""
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(text)
    return path


def _metrics(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json_row(line))
    return rows


def json_row(line: str) -> dict[str, object]:
    import json

    value = json.loads(line)
    if not isinstance(value, dict):
        raise AssertionError("metric row is not an object")
    return value


def _assert_same_tensors(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch {tuple(left.shape)} vs {tuple(right.shape)}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"keys {sorted(set(left) ^ set(right))}")
        for key in left:
            _assert_same_tensors(left[key], right[key])
        return
    if left != right:
        raise AssertionError(f"{left!r} != {right!r}")


def _write_snapshot(root: Path, *, derived: Path, name: str) -> dict[str, object]:
    corpora = derived.parent / "corpora"
    generated = derived.parent / "generated"
    corpora.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    assets = []
    cases = []
    total = 4
    for index in range(total):
        noisy = np.linspace(-0.2, 0.5, WAVEFORM_LENGTH, dtype=np.float32) * ((index % 5) + 1)
        clean = np.linspace(0.4, -0.3, WAVEFORM_LENGTH, dtype=np.float32) * ((index % 3) + 1)
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
                    "rel_path": rel,
                    "sample_rate": 16000,
                    "frames": WAVEFORM_LENGTH,
                    "channels": 1,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "integrity": "sha256",
                }
            )
        cases.append(
            {
                "build_id": None,
                "case_id": f"{name}-case-{index:02d}",
                "case_role": "recorded",
                "clean_asset_id": f"derived/{clean_rel}",
                "doc_kind": "case",
                "frames": WAVEFORM_LENGTH,
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
    rule = {
        "delivery": "on_demand",
        "epoch_unit": "utterance",
        "global_batch": 1,
        "normalize": "noisy_rms_unit_v1",
        "order": ["fixed_single_shuffle_v1", "torch_distributed_sampler_v1"],
        "sample_rate": 16000,
        "teacher": "dry_clean",
        "validation": {"batch": 1, "crop": "full_length"},
        "window": WAVEFORM_LENGTH,
    }
    dataset_id = digest({"algorithm": "fixed_pair.mp_senet", "name": name, "phase": "4a"})
    dataset = {
        "algorithm": "fixed_pair.mp_senet",
        "algorithm_version": 1,
        "bindings": bindings,
        "claim_label": "leakage_safe_comparison",
        "dataset_id": dataset_id,
        "declared_status": "ready",
        "kind": "fixed_pair",
        "name": name,
        "profile": "leakage_safe",
        "protocol_id": "protocol",
        "rule_json": rule,
        "selection_uses_test": False,
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
    directory = root / dataset_id
    directory.mkdir(parents=True)
    digest_value = snapshot_hash(parts)
    manifest = {
        "files": file_hashes(parts),
        "provenance": {"format": FORMAT},
        "schema_version": SCHEMA_VERSION,
        "snapshot_hash": digest_value,
    }
    (directory / "dataset.json").write_text(canonical_text(parts["dataset"]) + "\n", encoding="utf-8")
    (directory / "generated.json").write_text(canonical_text(parts["generated"]) + "\n", encoding="utf-8")
    (directory / "docs.jsonl").write_text("", encoding="utf-8")
    (directory / "manifest.json").write_text(canonical_text(manifest) + "\n", encoding="utf-8")
    np.savez(directory / "assets.npz", **arrays_from_assets(parts["assets"]))
    np.savez(directory / "cases.npz", **arrays_from_cases(parts["cases"]))
    np.savez(directory / "case_set_items.npz", **arrays_from_items(parts["case_set_items"]))
    np.savez(directory / "material_set_members.npz", **arrays_from_members(parts["material_set_members"]))
    loaded = load_snapshot(directory)
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


if __name__ == "__main__":
    unittest.main()
