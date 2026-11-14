"""Phase 4-F: boundaries shared by the four migrated trainers.

The per-model numerical tests stay in their own modules. This module checks
that those trainers still enter through one factory, refuse each other's
checkpoints, and export inference weights the existing loaders accept.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.evaluation.provenance import judge_checkpoint
from src.experiments.api import Experiment
from src.experiments.checkpoint import FORMAT
from src.experiments.config import load_experiment
from src.experiments.data import ChunkSampler, MicroBatcher
from src.experiments.errors import CheckpointError, DataError, ExperimentError
from src.experiments.official.mossformergan.models import MossFormerGANGenerator
from src.experiments.official.mp_senet.models import MPSENetGANGenerator
from src.experiments.official.tf_gridnet.model import TFGridNetModel
from src.experiments.official.tf_locoformer.model import TFLocoformerModel
from src.se import load_enhancer
from src.se.models.mossformergan.inference import _enhance_chunk
from src.experiments.calc.rng import set_determinism
from tests.experiments.test_mp_senet_gan import ARCH, _yaml as mp_yaml
from tests.experiments.test_mossformergan import _yaml as moss_yaml
from tests.experiments.test_tf_gridnet_training import TINY as GRID_TINY
from tests.experiments.test_tf_gridnet_training import _scripted_yaml as grid_yaml
from tests.experiments.test_tf_locoformer_training import TINY as LOCO_TINY
from tests.experiments.test_tf_locoformer_training import _scripted_yaml as loco_yaml

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = {
    "mp_senet": ROOT / "tests/fixtures/configs/experiments/mp_senet_gan.yaml",
    "mossformergan": ROOT / "tests/fixtures/configs/experiments/mossformergan_se_16k.yaml",
    "tf_gridnet": ROOT / "tests/fixtures/configs/experiments/tf_gridnet_ms_snsd.yaml",
    "tf_locoformer": ROOT / "tests/fixtures/configs/experiments/tf_locoformer_dns.yaml",
}


def _pesq_stub(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor:
    del enhanced, sample_rate
    return torch.full((clean.shape[0],), 0.25)


def _lines(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def _assert_bound_provenance(directory: Path, payload: dict, name: str) -> None:
    digest = payload["provenance_sha256"]
    assert isinstance(digest, str) and len(digest) == 64, name
    raw = (directory / "data_provenance.json").read_bytes()
    sidecar = (directory / "archive/step_00000002.data_provenance.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    assert sidecar == raw, name
    judgment = judge_checkpoint(directory / "archive/step_00000002.pt", [])
    assert judgment["independent"] is not True, name
    assert judgment["confirmed_independent"] is False, name


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} is not a checkpoint mapping")
    return payload


def _max_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise AssertionError(f"state keys differ: {sorted(set(left) ^ set(right))[:8]}")
    worst = 0.0
    for key, value in left.items():
        other = right[key]
        if value.dtype != other.dtype or value.shape != other.shape:
            raise AssertionError(f"{key}: {value.dtype} {tuple(value.shape)} vs {other.dtype} {tuple(other.shape)}")
        if not value.dtype.is_floating_point:
            if not torch.equal(value, other):
                raise AssertionError(f"{key} integer state differs")
            continue
        worst = max(worst, float((value - other).abs().max()))
    return worst


class Phase4FDeclarationTests(unittest.TestCase):
    def test_factory_and_builder_do_not_branch_on_model_names(self) -> None:
        factory = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in ("src/experiments/core/factory.py", "src/experiments/factory.py")
        )
        builder = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in ("src/experiments/core/builder.py", "src/experiments/builder.py")
        )
        for name in ("mp_senet", "mossformer", "tf_gridnet", "tf_locoformer", "MPSENet", "TFGridNet", "TFLocoformer"):
            self.assertNotIn(name, factory)
            self.assertNotIn(name, builder)

    def test_official_yaml_matrix_uses_one_schema(self) -> None:
        configs = {name: load_experiment(path) for name, path in OFFICIAL.items()}
        for name, config in configs.items():
            self.assertEqual(config.schema, "audio-db-experiment/v1", name)
            self.assertEqual(config.entry, "trainer", name)
            self.assertTrue(config.deterministic, name)
            self.assertEqual(config.runtime_class, "src.experiments.runtime.ProcessRuntime", name)
            self.assertEqual(config.runtime_kwargs["device"], "cpu", name)
            self.assertEqual(config.runtime_kwargs["precision"], "fp32", name)
            self.assertEqual(config.components["trainer"].kwargs["max_logical_steps"], 1, name)
            self.assertNotIn("update_order", config.components["trainer"].kwargs, name)
            self.assertEqual(config.components["data"].class_path, "src.experiments.data.SnapshotDataModule", name)
        mp = configs["mp_senet"]
        self.assertEqual(mp.components["optim_g"].parameters_of[0].component, "generator")
        self.assertEqual(mp.components["optim_d"].parameters_of[0].component, "metric_discriminator")
        self.assertEqual(mp.components["scheduler_g"].class_path, "torch.optim.lr_scheduler.ExponentialLR")
        self.assertEqual(mp.components["scheduler_d"].class_path, "torch.optim.lr_scheduler.ExponentialLR")
        self.assertEqual(mp.components["generator"].prepare.distribute, "none")
        self.assertEqual(mp.components["data"].kwargs["require_profile"], "official_faithful")
        moss = configs["mossformergan"]
        self.assertNotIn("scheduler", moss.components)
        self.assertEqual(moss.components["optim_g"].parameters_of[0].component, "generator")
        self.assertEqual(moss.components["optim_d"].parameters_of[0].component, "discriminator")
        self.assertEqual(moss.components["optim_g"].kwargs["lr"], 0.0005)
        self.assertEqual(moss.components["optim_d"].kwargs["lr"], 0.001)
        self.assertEqual(moss.components["trainer"].kwargs["accumulation"], 2)
        self.assertTrue(moss.components["generator"].prepare.sync_batchnorm)
        self.assertEqual(moss.components["generator"].prepare.distribute, "ddp")
        grid = configs["tf_gridnet"]
        self.assertEqual(grid.components["optimizer"].class_path, "torch.optim.Adam")
        self.assertEqual(grid.components["optimizer"].parameters_of[0].component, "model")
        self.assertEqual(grid.components["scheduler"].class_path, "torch.optim.lr_scheduler.ReduceLROnPlateau")
        self.assertEqual(grid.components["model"].prepare.distribute, "ddp")
        self.assertEqual(grid.components["sampler"].kwargs["global_batch"], 4)
        self.assertEqual(grid.components["batcher"].kwargs["micro_batch"], 2)
        loco = configs["tf_locoformer"]
        self.assertEqual(loco.components["optimizer"].class_path, "torch.optim.AdamW")
        self.assertEqual(loco.components["scheduler"].class_path, "src.experiments.calc.optim.WarmupReduceLROnPlateau")
        self.assertEqual(loco.components["trainer"].kwargs["validation_profile"], "tf_locoformer.dns.accept_recipe.research_valid_fixed@1")
        self.assertEqual(loco.components["model"].prepare.distribute, "ddp")
        self.assertEqual(loco.components["sampler"].kwargs["global_batch"], 4)
        self.assertEqual(loco.components["batcher"].kwargs["micro_batch"], 1)

    def test_committed_chunk_batches_reject_world_size_one(self) -> None:
        grid = load_experiment(OFFICIAL["tf_gridnet"])
        loco = load_experiment(OFFICIAL["tf_locoformer"])
        grid_sampler = ChunkSampler(object(), global_batch=grid.components["sampler"].kwargs["global_batch"])
        loco_sampler = ChunkSampler(object(), global_batch=loco.components["sampler"].kwargs["global_batch"])
        grid_batch = MicroBatcher(
            micro_batch=grid.components["batcher"].kwargs["micro_batch"],
            group_size=grid.components["batcher"].kwargs["group_size"],
        )
        loco_batch = MicroBatcher(
            micro_batch=loco.components["batcher"].kwargs["micro_batch"],
            group_size=loco.components["batcher"].kwargs["group_size"],
        )
        with self.assertRaises(DataError):
            grid_sampler.bind_batch(grid_batch, 1)
        grid_sampler.bind_batch(grid_batch, 2)
        self.assertEqual(grid_sampler._bound_world, 2)
        with self.assertRaises(DataError):
            loco_sampler.bind_batch(loco_batch, 1)
        loco_sampler.bind_batch(loco_batch, 4)
        self.assertEqual(loco_sampler._bound_world, 4)


class Phase4FCheckpointAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_determinism(True, cuda=False)
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)

    @classmethod
    def tearDownClass(cls) -> None:
        set_determinism(False, cuda=False)
        cls._temporary.cleanup()

    def test_interrupted_resume_keeps_one_metric_line_per_logical_step(self) -> None:
        runs = {
            "mp_senet": self._pair(self.base / "mp", self._mp),
            "mossformergan": self._pair(self.base / "moss", self._moss, pesq=True),
            "tf_gridnet": self._pair(self.base / "grid", self._grid),
            "tf_locoformer": self._pair(self.base / "loco", self._loco),
        }
        for name, directory in runs.items():
            payload = _load(directory / "archive/step_00000002.pt")
            self.assertEqual(payload["format"], FORMAT, name)
            self.assertEqual(payload["schema_version"], 1, name)
            _assert_bound_provenance(directory, payload, name)
            self.assertIsInstance(payload["code_identity"]["git_commit"], str, name)
            self.assertIsInstance(payload["code_identity"]["dirty"], bool, name)
            self.assertIn("data", payload["dataset_identity"], name)
            self.assertIsInstance(payload["states"]["data"]["plan_hash"], str, name)
            self.assertGreaterEqual(len(payload["states"]["data"]["plan_hash"]), 8, name)
            metrics = _lines(directory / "metrics.jsonl")
            self.assertEqual([row["logical_step"] for row in metrics], [1, 2], name)
            self.assertEqual(json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"], "COMPLETED", name)
            self.assertTrue((directory / "complete.json").is_file(), name)
        self._refuse_completed_resume(runs["tf_gridnet"])
        self._refuse_foreign_checkpoint(runs)
        self._refuse_tampered_grid_checkpoint(runs["tf_gridnet"])

    def test_inference_exports_round_trip_without_a_discriminator(self) -> None:
        self._export_mp_senet()
        self._export_mossformer()
        self._export_gridnet()
        self._export_locoformer()

    def _pair(self, directory: Path, write, *, pesq: bool = False) -> Path:
        first = write(directory, stop_after=1)
        self._execute(first, pesq=pesq)
        stopped = _lines(directory / "metrics.jsonl")
        self.assertEqual([row["logical_step"] for row in stopped], [1])
        self.assertEqual(json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"], "INTERRUPTED")
        self.assertFalse((directory / "complete.json").exists())
        second = write(directory, stop_after=None)
        checkpoint = directory / "archive/step_00000001.pt"
        before = checkpoint.read_bytes()
        self._execute(second, resume=checkpoint, pesq=pesq)
        self.assertEqual(checkpoint.read_bytes(), before)
        return directory

    def _execute(self, config: Path, *, resume: Path | None = None, pesq: bool = False) -> None:
        if pesq:
            with patch("src.experiments.official.mossformergan.reference.metric_targets", _pesq_stub):
                Experiment(load_experiment(config)).run(resume=resume)
            return
        Experiment(load_experiment(config)).run(resume=resume)

    def _mp(self, directory: Path, *, stop_after: int | None) -> Path:
        return mp_yaml(directory, steps=2, updates=4, validation_interval=100, stop_after=stop_after)

    def _moss(self, directory: Path, *, stop_after: int | None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = moss_yaml(directory, steps=2, updates=4, accumulation=1, partial=False, validate=False)
        text = path.read_text(encoding="utf-8")
        if stop_after is not None:
            text = text.replace(
                "      experiment_claim: research\n",
                "      experiment_claim: research\n      stop_after_logical_steps: 1\n",
                1,
            )
        path.write_text(text, encoding="utf-8")
        return path

    def _grid(self, directory: Path, *, stop_after: int | None) -> Path:
        return grid_yaml(directory, steps=2, updates=4, stop_after=stop_after)

    def _loco(self, directory: Path, *, stop_after: int | None) -> Path:
        return loco_yaml(directory, steps=2, updates=4, stop_after=stop_after)

    def _refuse_completed_resume(self, directory: Path) -> None:
        metrics = (directory / "metrics.jsonl").read_bytes()
        checkpoint = (directory / "archive/step_00000002.pt").read_bytes()
        with self.assertRaises(CheckpointError) as caught:
            Experiment(load_experiment(directory / "experiment.yaml")).run(resume=directory / "archive/step_00000002.pt")
        self.assertIn("complete.json", str(caught.exception))
        self.assertEqual((directory / "metrics.jsonl").read_bytes(), metrics)
        self.assertEqual((directory / "archive/step_00000002.pt").read_bytes(), checkpoint)

    def _refuse_foreign_checkpoint(self, runs: dict[str, Path]) -> None:
        order = ("mp_senet", "tf_gridnet", "tf_locoformer", "mossformergan")
        for source_name, target_name in zip(order, order[1:] + order[:1], strict=True):
            source = runs[source_name]
            source_bytes = (source / "archive/step_00000001.pt").read_bytes()
            target = self.base / f"foreign-{source_name}-into-{target_name}"
            config = self._rewrite(target_name, target)
            payload = _load(source / "archive/step_00000001.pt")
            payload["output_directory"] = str(target.resolve())
            foreign = target / "archive/step_00000001.pt"
            foreign.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, foreign)
            before = foreign.read_bytes()
            with self.assertRaises(ExperimentError):
                Experiment(load_experiment(config)).run(resume=foreign)
            self.assertEqual(foreign.read_bytes(), before)
            self.assertEqual((source / "archive/step_00000001.pt").read_bytes(), source_bytes)

    def _refuse_tampered_grid_checkpoint(self, directory: Path) -> None:
        original = (directory / "archive/step_00000001.pt").read_bytes()
        cases = {
            "plan": lambda payload: payload["states"]["data"].__setitem__("plan_hash", "f" * 64),
            "dataset": lambda payload: payload["dataset_identity"]["data"].__setitem__("seed", -1),
            "world": lambda payload: payload.__setitem__("world_size", 2),
        }
        for name, mutate in cases.items():
            target = self.base / f"grid-tamper-{name}"
            config = self._grid(target, stop_after=None)
            payload = _load(directory / "archive/step_00000001.pt")
            payload["output_directory"] = str(target.resolve())
            mutate(payload)
            foreign = target / "archive/step_00000001.pt"
            foreign.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, foreign)
            with self.assertRaises(ExperimentError):
                Experiment(load_experiment(config)).run(resume=foreign)
        corrupt = self.base / "grid-corrupt"
        config = self._grid(corrupt, stop_after=None)
        broken = corrupt / "archive/step_00000001.pt"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"not-a-checkpoint")
        with self.assertRaises(ExperimentError):
            Experiment(load_experiment(config)).run(resume=broken)
        self.assertEqual((directory / "archive/step_00000001.pt").read_bytes(), original)

    def _rewrite(self, name: str, directory: Path) -> Path:
        writers = {
            "mp_senet": self._mp,
            "mossformergan": self._moss,
            "tf_gridnet": self._grid,
            "tf_locoformer": self._loco,
        }
        return writers[name](directory, stop_after=None)

    def _export_mp_senet(self) -> None:
        generator = MPSENetGANGenerator(seed=7, **ARCH)
        exported = generator.export_for_inference()
        self.assertEqual(set(exported), {"generator"})
        path = self.base / "mp-export.pt"
        torch.save(exported, path)
        enhancer = load_enhancer("mp_senet", path, "cpu", ARCH)
        self.assertEqual(_max_diff(generator.generator.state_dict(), enhancer.model.state_dict()), 0.0)
        magnitude = torch.rand(1, ARCH["n_fft"] // 2 + 1, 4)
        phase = torch.rand(1, ARCH["n_fft"] // 2 + 1, 4)
        generator.eval()
        enhancer.model.eval()
        with torch.inference_mode():
            before = generator.generator(magnitude, phase)
            after = enhancer.model(magnitude, phase)
        self.assertEqual(_max_diff(_as_dict(before), _as_dict(after)), 0.0)
        waveform = torch.randn(800)
        enhanced = enhancer.enhance(waveform, 16000)
        self.assertEqual(tuple(enhanced.shape), (800,))
        self.assertTrue(bool(torch.isfinite(enhanced).all()))

    def _export_mossformer(self) -> None:
        generator = MossFormerGANGenerator(seed=7, num_channel=64)
        exported = generator.export_for_inference()
        self.assertEqual(set(exported), {"model"})
        path = self.base / "moss-export.pt"
        torch.save(exported, path)
        enhancer = load_enhancer(
            "mossformergan",
            path,
            "cpu",
            {"n_fft": 400, "win_len": 400, "hop": 100, "num_channel": 64},
            variant="se_16k",
        )
        self.assertEqual(_max_diff(generator.generator.state_dict(), enhancer.model.state_dict()), 0.0)
        waveform = torch.randn(2400)
        generator.eval()
        enhancer.model.eval()
        with torch.inference_mode():
            before = _enhance_chunk(generator.generator, waveform, 400, 400, 100)
            after = _enhance_chunk(enhancer.model, waveform, 400, 400, 100)
        self.assertEqual(float((before - after).abs().max()), 0.0)
        enhanced = enhancer.enhance(waveform, 16000)
        self.assertEqual(tuple(enhanced.shape), (2400,))
        self.assertTrue(bool(torch.isfinite(enhanced).all()))

    def _export_gridnet(self) -> None:
        model = TFGridNetModel(seed=0, architecture=GRID_TINY)
        exported = model.export_for_inference()
        self.assertTrue(exported)
        self.assertTrue(all(key.startswith("separator.") for key in exported))
        path = self.base / "grid-export.pt"
        torch.save(exported, path)
        enhancer = load_enhancer("tf_gridnet", path, "cpu", GRID_TINY, variant="ms-snsd")
        waveform = torch.randn(1, 64)
        model.eval()
        with torch.inference_mode():
            before = model(waveform)
            after = enhancer.enhance(waveform[0], 16000)
        self.assertEqual(tuple(after.shape), (64,))
        self.assertEqual(float((before[0] - after).abs().max()), 0.0)

    def _export_locoformer(self) -> None:
        model = TFLocoformerModel(seed=0, architecture=LOCO_TINY)
        exported = model.export_for_inference()
        self.assertTrue(exported)
        self.assertTrue(all(key.startswith("separator.") for key in exported))
        path = self.base / "loco-export.pt"
        torch.save(exported, path)
        enhancer = load_enhancer("tf_locoformer", path, "cpu", LOCO_TINY, variant="dns")
        waveform = torch.randn(1, 64)
        model.eval()
        with torch.inference_mode():
            before = model(waveform)
            after = enhancer.enhance(waveform[0], 16000)
        self.assertEqual(tuple(after.shape), (64,))
        self.assertEqual(float((before[0] - after).abs().max()), 0.0)


def _as_dict(value: object) -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {"output": value}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if isinstance(item, torch.Tensor)}
    if isinstance(value, (tuple, list)):
        return {str(index): item for index, item in enumerate(value) if isinstance(item, torch.Tensor)}
    raise AssertionError(f"unsupported forward output {type(value).__name__}")

