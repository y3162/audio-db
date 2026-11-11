"""Phase 4-I: six-model boundaries that the per-model suites do not share.

Numerical equality stays in the Phase 4-A through 4-H modules. This module
checks the six committed YAML declarations, DeepFilterNet3 and SEMamba++
inference export, and checkpoint refusal across the six trainers. A saved
checkpoint carries a provenance sidecar. A run that never writes a checkpoint
does not.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import torch

from src.evaluation.provenance import judge_checkpoint
from src.experiments.api import Experiment
from src.experiments.checkpoint import FORMAT
from src.experiments.config import load_experiment
from src.experiments.errors import ExperimentError
from src.experiments.official.deepfilternet3.model import DeepFilterNet3Model
from src.experiments.official.semambapp.models import SEMambaPPGenerator, scan_execution
from src.se import load_enhancer
from src.experiments.calc.rng import set_determinism
from src.experiments.data.snapshot import published_snapshot_directories
from tests.experiments.test_deepfilternet3 import _ARCH as DFN_ARCH
from tests.experiments.test_deepfilternet3 import _tiny_yaml
from tests.experiments.test_mossformergan import _yaml as moss_yaml
from tests.experiments.test_mp_senet_gan import _yaml as mp_yaml
from tests.experiments.test_semambapp import _yaml as semamba_yaml
from tests.experiments.test_tf_gridnet_training import _scripted_yaml as grid_yaml
from tests.experiments.test_tf_locoformer_training import _scripted_yaml as loco_yaml

ROOT = Path(__file__).resolve().parents[2]
TRACKED_RFFT = ROOT / "src/se/models/deepfilternet3/libdf_rfft.so"
LOCAL_RFFT = ROOT / "build/df-rfft/release/libdf_rfft.so"
SNAPSHOTS = ROOT / "data/db/snapshots"
OFFICIAL = {
    "mp_senet": ROOT / "tests/fixtures/configs/experiments/mp_senet_gan.yaml",
    "mossformergan": ROOT / "tests/fixtures/configs/experiments/mossformergan_se_16k.yaml",
    "tf_gridnet": ROOT / "tests/fixtures/configs/experiments/tf_gridnet_ms_snsd.yaml",
    "tf_locoformer": ROOT / "tests/fixtures/configs/experiments/tf_locoformer_dns.yaml",
    "deepfilternet3": ROOT / "tests/fixtures/configs/experiments/deepfilternet3.yaml",
    "semambapp": ROOT / "tests/fixtures/configs/experiments/semambapp_gsr.yaml",
}
READY = {
    "mp_senet.vb28.ondemand.official": "official_reproduction",
    "mossformergan.vb28.ondemand.official": "official_reproduction",
    "tf_gridnet.ms_snsd.espnet_recipe": "espnet_public_se_recipe_reproduction",
    "tf_locoformer.dns.accept_recipe.research_valid_fixed": "research_checkpoint_selection",
    "dfn3.online.smoke": "protocol_conformant_substitute_materials",
    "semambapp.gsr.smoke": "protocol_conformant_substitute_materials",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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
        if value.shape != other.shape or value.dtype != other.dtype:
            raise AssertionError(key)
        if not value.dtype.is_floating_point:
            if not torch.equal(value, other):
                raise AssertionError(key)
            continue
        worst = max(worst, float((value.detach().cpu() - other.detach().cpu()).abs().max()))
    return worst


def _datasets() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for child in published_snapshot_directories(SNAPSHOTS):
        path = child / "dataset.json"
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        name = document.get("name")
        if isinstance(name, str):
            found[name] = document
    return found


def _probe_rfft(library: str) -> dict[str, object]:
    code = (
        "import json, os\n"
        "os.environ['DF_RFFT_LIBRARY'] = %r\n"
        "from src.se.models.deepfilternet3.dsp import _rfft_kernel, rfft_library_path\n"
        "try:\n"
        "    kernel = _rfft_kernel()\n"
        "    abi = None if kernel is None else int(kernel.df_rfft_abi())\n"
        "    print(json.dumps({'ok': True, 'none': kernel is None, 'abi': abi, 'path': str(rfft_library_path())}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok': False, 'error': type(exc).__name__, 'message': str(exc)}))\n"
    ) % library
    env = os.environ.copy()
    env["DF_RFFT_LIBRARY"] = library
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr[-1500:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


class Phase4IDeclarationTests(unittest.TestCase):
    def test_factory_and_builder_do_not_branch_on_the_six_models(self) -> None:
        factory = (ROOT / "src/experiments/factory.py").read_text(encoding="utf-8")
        builder = (ROOT / "src/experiments/builder.py").read_text(encoding="utf-8")
        for name in (
            "mp_senet",
            "mossformer",
            "tf_gridnet",
            "tf_locoformer",
            "deepfilternet",
            "semamba",
            "DfNet",
            "MPSENet",
            "TFGridNet",
            "TFLocoformer",
        ):
            self.assertNotIn(name, factory)
            self.assertNotIn(name, builder)

    def test_six_yaml_files_share_the_experiment_schema(self) -> None:
        configs = {name: load_experiment(path) for name, path in OFFICIAL.items()}
        for name, config in configs.items():
            self.assertEqual(config.schema, "audio-db-experiment/v1", name)
            self.assertEqual(config.entry, "trainer", name)
            self.assertEqual(config.runtime_class, "src.experiments.runtime.ProcessRuntime", name)
            self.assertEqual(config.runtime_kwargs["device"], "cpu", name)
            self.assertEqual(config.runtime_kwargs["precision"], "fp32", name)
            self.assertNotIn("world_size", config.runtime_kwargs, name)
            trainer = config.components["trainer"]
            self.assertEqual(trainer.kwargs["max_logical_steps"], 1, name)
            self.assertGreaterEqual(int(trainer.kwargs["checkpoint_interval"]), 1, name)
            self.assertEqual(config.components["data"].class_path, "src.experiments.data.SnapshotDataModule", name)
        mp = configs["mp_senet"]
        self.assertEqual(mp.components["optim_g"].parameters_of[0].component, "generator")
        self.assertEqual(mp.components["optim_d"].parameters_of[0].component, "metric_discriminator")
        self.assertEqual(mp.components["generator"].prepare.distribute, "none")
        self.assertEqual(mp.components["trainer"].kwargs["checkpoint_interval"], 5000)
        moss = configs["mossformergan"]
        self.assertNotIn("scheduler", moss.components)
        self.assertEqual(moss.components["optim_g"].parameters_of[0].component, "generator")
        self.assertEqual(moss.components["optim_d"].parameters_of[0].component, "discriminator")
        self.assertTrue(moss.components["generator"].prepare.sync_batchnorm)
        self.assertEqual(moss.components["trainer"].kwargs["accumulation"], 2)
        self.assertEqual(moss.components["trainer"].kwargs["checkpoint_interval"], 5786)
        grid = configs["tf_gridnet"]
        self.assertEqual(grid.components["optimizer"].class_path, "torch.optim.Adam")
        self.assertEqual(grid.components["scheduler"].class_path, "torch.optim.lr_scheduler.ReduceLROnPlateau")
        self.assertEqual(grid.components["model"].prepare.distribute, "ddp")
        self.assertEqual(grid.components["sampler"].kwargs["global_batch"], 4)
        self.assertEqual(grid.components["batcher"].kwargs["micro_batch"], 2)
        loco = configs["tf_locoformer"]
        self.assertEqual(loco.components["optimizer"].class_path, "torch.optim.AdamW")
        self.assertEqual(loco.components["scheduler"].class_path, "src.experiments.calc.optim.WarmupReduceLROnPlateau")
        self.assertEqual(
            loco.components["trainer"].kwargs["validation_profile"],
            "tf_locoformer.dns.accept_recipe.research_valid_fixed@1",
        )
        self.assertEqual(loco.components["sampler"].kwargs["global_batch"], 4)
        self.assertEqual(loco.components["batcher"].kwargs["micro_batch"], 1)
        dfn = configs["deepfilternet3"]
        self.assertTrue(dfn.components["model"].prepare.find_unused_parameters)
        self.assertFalse(dfn.components["model"].prepare.sync_batchnorm)
        self.assertEqual(dfn.components["optimizer"].class_path, "torch.optim.AdamW")
        self.assertEqual(dfn.components["source"].kwargs["snapshot"], "dfn3.online.smoke@1")
        self.assertEqual(dfn.components["trainer"].kwargs["checkpoint_interval"], 1)
        self.assertEqual(dfn.components["trainer"].kwargs["batch_schedule"][0], [0, 16])
        semamba = configs["semambapp"]
        self.assertEqual(semamba.components["optim_g"].parameters_of[0].component, "generator")
        self.assertEqual(
            [item.component for item in semamba.components["optim_d"].parameters_of],
            ["mrd", "cqt"],
        )
        self.assertEqual(semamba.components["generator"].kwargs["scan_backend"], "reference")
        self.assertEqual(semamba.components["source"].kwargs["snapshot"], "semambapp.gsr.smoke@1")
        self.assertEqual(semamba.components["trainer"].kwargs["claim_label"], "protocol_conformant_substitute_materials")
        self.assertNotIn("best_generator", semamba.components)

    def test_ready_snapshots_and_blocked_official_sets_stay_distinct(self) -> None:
        documents = _datasets()
        for name, claim in READY.items():
            self.assertIn(name, documents, name)
            self.assertEqual(documents[name]["claim_label"], claim, name)
            self.assertEqual(documents[name]["declared_status"], "ready", name)
            self.assertEqual(documents[name]["profile"], "official_faithful", name)
        for blocked in ("dfn3.online.official", "semambapp.gsr.official"):
            self.assertNotIn(blocked, documents)
        p8 = duckdb.connect(str(ROOT / "data/db/reproduction-p8.duckdb"), read_only=True)
        p9 = duckdb.connect(str(ROOT / "data/db/reproduction-p9.duckdb"), read_only=True)
        try:
            dfn = p8.execute(
                "select declared_status, claim_label from repro.datasets where name = 'dfn3.online.official'"
            ).fetchone()
            semamba = p9.execute(
                "select declared_status, claim_label from repro.datasets where name = 'semambapp.gsr.official'"
            ).fetchone()
        finally:
            p8.close()
            p9.close()
        self.assertEqual(dfn, ("blocked", "official_reproduction"))
        self.assertEqual(semamba, ("blocked", "official_reproduction"))

    def test_three_rfft_paths_stay_distinct(self) -> None:
        self.assertTrue(TRACKED_RFFT.is_file())
        self.assertTrue(LOCAL_RFFT.is_file())
        tracked = _sha256(TRACKED_RFFT)
        local = _sha256(LOCAL_RFFT)
        self.assertNotEqual(tracked, local)
        missing = _probe_rfft("/tmp/audio-db-phase4i-missing-libdf_rfft.so")
        tracked_probe = _probe_rfft(str(TRACKED_RFFT))
        local_probe = _probe_rfft(str(LOCAL_RFFT))
        self.assertEqual(missing, {"ok": True, "none": True, "abi": None, "path": "/tmp/audio-db-phase4i-missing-libdf_rfft.so"})
        self.assertFalse(tracked_probe["ok"])
        self.assertEqual(tracked_probe["error"], "RuntimeError")
        self.assertIn("failed to load", str(tracked_probe["message"]))
        self.assertEqual(local_probe["ok"], True)
        self.assertEqual(local_probe["abi"], 1)
        self.assertFalse(local_probe["none"])
        self.assertEqual(_sha256(TRACKED_RFFT), tracked)
        self.assertEqual(_sha256(LOCAL_RFFT), local)


class Phase4IExportTests(unittest.TestCase):
    def test_deepfilternet3_export_loads_strictly_without_a_discriminator(self) -> None:
        model = DeepFilterNet3Model(seed=7, architecture=DFN_ARCH)
        exported = model.export_for_inference()
        self.assertTrue(any("df_fc_a" in name for name in exported))
        self.assertFalse(any("discriminator" in name for name in exported))
        path = Path(tempfile.mkdtemp(prefix="phase4i-dfn-")) / "model.pt"
        torch.save(exported, path)
        enhancer = load_enhancer("deepfilternet3", path, "cpu", dict(DFN_ARCH))
        self.assertEqual(_max_diff(model.generator.state_dict(), enhancer.model.state_dict()), 0.0)
        waveform = torch.randn(4800)
        enhanced = enhancer.enhance(waveform, 48000)
        self.assertEqual(tuple(enhanced.shape), (4800,))
        self.assertTrue(bool(torch.isfinite(enhanced).all()))

    def test_semambapp_reference_and_fused_weights_load_into_each_other(self) -> None:
        kwargs = dict(
            seed=1234,
            hid_feature=8,
            num_tfmamba=1,
            d_state=4,
            d_conv=2,
            expand=2,
            norm_epsilon=1e-5,
            input_channel=2,
            output_channel=1,
            n_fft=400,
            hop=100,
            win=400,
            sampling_rate=16000,
        )
        reference = SEMambaPPGenerator(**kwargs, scan_backend="reference")
        fused = SEMambaPPGenerator(**kwargs, scan_backend="fused")
        self.assertEqual(scan_execution(reference, torch.device("cpu"))["executed"], "reference")
        fused_scan = scan_execution(fused, torch.device("cpu"))
        self.assertEqual(fused_scan["configured"], "fused")
        self.assertEqual(fused_scan["executed"], "reference")
        self.assertEqual(_max_diff(reference.generator.state_dict(), fused.generator.state_dict()), 0.0)
        exported = reference.export_for_inference()
        self.assertEqual(set(exported), {"generator"})
        fused.generator.load_state_dict(exported["generator"], strict=True)
        reference.generator.load_state_dict(fused.export_for_inference()["generator"], strict=True)
        path = Path(tempfile.mkdtemp(prefix="phase4i-semamba-")) / "generator.pt"
        torch.save(exported, path)
        config = {
            "hid_feature": 8,
            "num_tfmamba": 1,
            "d_state": 4,
            "d_conv": 2,
            "expand": 2,
            "norm_epsilon": 1e-5,
            "input_channel": 2,
            "output_channel": 1,
            "n_fft": 400,
        }
        enhancer = load_enhancer("semambapp", path, "cpu", {**config, "scan_backend": "reference"})
        fused_enhancer = load_enhancer("semambapp", path, "cpu", {**config, "scan_backend": "fused"})
        self.assertEqual(_max_diff(reference.generator.state_dict(), enhancer.model.state_dict()), 0.0)
        self.assertEqual(_max_diff(reference.generator.state_dict(), fused_enhancer.model.state_dict()), 0.0)
        self.assertEqual(scan_execution(fused_enhancer.model, torch.device("cpu"))["executed"], "reference")
        waveform = torch.randn(800)
        reference.eval()
        enhanced = enhancer.enhance(waveform, 16000)
        self.assertEqual(tuple(enhanced.shape), (800,))
        self.assertTrue(bool(torch.isfinite(enhanced).all()))


class Phase4ICheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_determinism(True, cuda=False)
        cls._temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls._temporary.name)
        cls.dfn = cls.base / "dfn"
        cls.semamba = cls.base / "semamba"
        cls.wide_interval = cls.base / "dfn-wide-interval"
        cls.mp_wide_interval = cls.base / "mp-wide-interval"
        _run(_dfn_config(cls.dfn, steps=1, interval=1))
        _run(_semamba_config(cls.semamba))
        _run(_dfn_config(cls.wide_interval, steps=1, interval=5000))
        _run(_mp_wide(cls.mp_wide_interval))

    @classmethod
    def tearDownClass(cls) -> None:
        set_determinism(False, cuda=False)
        cls._temporary.cleanup()

    def test_a_wide_checkpoint_interval_completes_without_a_step_file(self) -> None:
        for directory in (self.wide_interval, self.mp_wide_interval):
            self.assertEqual(json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"], "COMPLETED")
            self.assertTrue((directory / "complete.json").is_file())
            self.assertFalse(list(directory.glob("step_*.pt")))
            self.assertFalse((directory / "data_provenance.json").exists())
            metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual([row["logical_step"] for row in metrics], [1])
            saved = (directory / "metrics.jsonl").read_bytes()
            with self.assertRaises(ExperimentError) as caught:
                Experiment(load_experiment(directory / "experiment.yaml")).run(resume=directory / "missing.pt")
            self.assertIn("complete.json", str(caught.exception))
            self.assertEqual((directory / "metrics.jsonl").read_bytes(), saved)

    def test_saved_checkpoints_keep_the_experiment_contract(self) -> None:
        for directory in (self.dfn, self.semamba):
            payload = _load(directory / "step_00000001.pt")
            self.assertEqual(payload["format"], FORMAT)
            digest = payload["provenance_sha256"]
            self.assertIsInstance(digest, str)
            self.assertEqual(len(digest), 64)
            raw = (directory / "data_provenance.json").read_bytes()
            sidecar = (directory / "step_00000001.data_provenance.json").read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            self.assertEqual(sidecar, raw)
            judgment = judge_checkpoint(directory / "step_00000001.pt", [])
            self.assertNotEqual(judgment["independent"], True)
            self.assertFalse(judgment["confirmed_independent"])

    def test_trainers_refuse_each_others_checkpoints(self) -> None:
        donors = {
            "deepfilternet3": self.dfn / "step_00000001.pt",
            "semambapp": self.semamba / "step_00000001.pt",
        }
        targets = {
            "mp_senet": lambda directory: mp_yaml(directory, steps=1, updates=1, validation_interval=100),
            "mossformergan": _small_moss,
            "tf_gridnet": lambda directory: grid_yaml(directory, steps=1, updates=2),
            "tf_locoformer": lambda directory: loco_yaml(directory, steps=1, updates=2),
            "deepfilternet3": lambda directory: _dfn_config(directory, steps=1, interval=1),
            "semambapp": lambda directory: _semamba_config(directory),
        }
        for target_name, write in targets.items():
            for donor_name, donor in donors.items():
                if target_name == donor_name:
                    continue
                original = donor.read_bytes()
                directory = self.base / f"foreign-{donor_name}-into-{target_name}"
                config = write(directory)
                payload = _load(donor)
                payload["output_directory"] = str(directory.resolve())
                foreign = directory / donor.name
                torch.save(payload, foreign)
                before = foreign.read_bytes()
                with self.assertRaises(ExperimentError):
                    Experiment(load_experiment(config)).run(resume=foreign)
                self.assertEqual(foreign.read_bytes(), before)
                self.assertEqual(donor.read_bytes(), original)

    def test_dfn3_resume_refuses_plan_identity_world_and_corrupt_bytes(self) -> None:
        original = (self.dfn / "step_00000001.pt").read_bytes()
        metrics = (self.dfn / "metrics.jsonl").read_bytes()
        complete = self.dfn / "complete.json"
        held = complete.read_bytes()
        complete.unlink()
        try:
            cases = {
                "plan": lambda payload: payload["states"]["data"].__setitem__("plan_hash", "f" * 64),
                "dataset": lambda payload: payload["dataset_identity"]["data"].__setitem__("seed", -1),
                "world": lambda payload: payload.__setitem__("world_size", 2),
                "components": lambda payload: payload.__setitem__("component_ids", ["not-a-model"]),
            }
            for name, mutate in cases.items():
                payload = _load(self.dfn / "step_00000001.pt")
                mutate(payload)
                foreign = self.dfn / f"tamper-{name}.pt"
                torch.save(payload, foreign)
                before = foreign.read_bytes()
                with self.assertRaises(ExperimentError):
                    Experiment(load_experiment(self.dfn / "experiment.yaml")).run(resume=foreign)
                self.assertEqual(foreign.read_bytes(), before)
            broken = self.dfn / "tamper-corrupt.pt"
            broken.write_bytes(b"not-a-checkpoint")
            with self.assertRaises(ExperimentError):
                Experiment(load_experiment(self.dfn / "experiment.yaml")).run(resume=broken)
        finally:
            complete.write_bytes(held)
        self.assertEqual((self.dfn / "step_00000001.pt").read_bytes(), original)
        self.assertEqual((self.dfn / "metrics.jsonl").read_bytes(), metrics)


def _run(config: Path) -> None:
    Experiment(load_experiment(config)).run()


def _dfn_config(directory: Path, *, steps: int, interval: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = _tiny_yaml(directory, stop_after=None)
    text = text.replace("      max_logical_steps: 2\n", f"      max_logical_steps: {steps}\n")
    text = text.replace("      checkpoint_interval: 1\n", f"      checkpoint_interval: {interval}\n")
    path = directory / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _semamba_config(directory: Path) -> Path:
    return semamba_yaml(directory, steps=1, count=1, accumulation=1, validation_interval=0)


def _mp_wide(directory: Path) -> Path:
    path = mp_yaml(directory, steps=1, updates=4, validation_interval=100)
    text = path.read_text(encoding="utf-8").replace("      checkpoint_interval: 1\n", "      checkpoint_interval: 5000\n", 1)
    path.write_text(text, encoding="utf-8")
    return path


def _small_moss(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = moss_yaml(directory, steps=1, updates=1, accumulation=1, partial=False, validate=False)
    text = path.read_text(encoding="utf-8").replace("num_channel: 64", "num_channel: 8")
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
