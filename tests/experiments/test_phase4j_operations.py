"""Phase 4-J: native library selection, provenance, and stop boundaries.

Numerical equality stays in the Phase 4-A through 4-I modules. CUDA cases live
in ``test_cuda_amp`` so hiding CUDA does not add a skip here.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.evaluation.provenance import canonical_bytes, judge_checkpoint
from src.experiments.api import Experiment
from src.experiments.checkpoint import CheckpointError, _assert_replicated_states
from src.experiments.config import load_experiment
from src.experiments.official.deepfilternet3.trainer import DeepFilterNet3Trainer
from src.experiments.official.mossformergan.trainer import MossFormerGANTrainer
from src.experiments.official.mp_senet.trainer import MPSENetGANTrainer
from src.experiments.provenance import judge_experiment_checkpoint
from src.experiments.official.semambapp.trainer import SEMambaPPTrainer
from src.experiments.signals import clear_interrupt, ranks_should_stop, request_interrupt
from src.experiments.official.tf_gridnet.trainer import TFGridNetTrainer
from src.experiments.official.tf_locoformer.trainer import TFLocoformerTrainer
from src.se.models.deepfilternet3.dsp import _rfft_kernel, rfft_runtime_record
from tests.experiments.test_ddp import _torchrun, _write_yaml

ROOT = Path(__file__).resolve().parents[2]
LOCAL_RFFT = ROOT / "build" / "df-rfft" / "release" / "libdf_rfft.so"
TRACKED_RFFT = ROOT / "src" / "se" / "models" / "deepfilternet3" / "libdf_rfft.so"


class NativeLibraryTests(unittest.TestCase):
    def test_clean_process_selects_a_loadable_library(self) -> None:
        self.assertTrue(LOCAL_RFFT.is_file())
        env = os.environ.copy()
        env.pop("DF_RFFT_LIBRARY", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, os\n"
                "os.environ.pop('DF_RFFT_LIBRARY', None)\n"
                "from src.se.models.deepfilternet3.dsp import rfft_runtime_record\n"
                "print(json.dumps(rfft_runtime_record(), sort_keys=True))\n",
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-1500:])
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(record["kernel"], "native")
        self.assertEqual(record["abi"], 1)
        self.assertIn(record["source"], {"tracked", "local_build"})
        self.assertEqual(record["sha256"], hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest())
        if record["source"] == "local_build":
            self.assertEqual(Path(record["path"]).resolve(), LOCAL_RFFT.resolve())
            self.assertNotEqual(record["sha256"], hashlib.sha256(TRACKED_RFFT.read_bytes()).hexdigest())

    def test_unloadable_file_is_not_a_silent_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.so"
            path.write_bytes(b"not-an-elf")
            previous = os.environ.get("DF_RFFT_LIBRARY")
            os.environ["DF_RFFT_LIBRARY"] = str(path)
            try:
                with self.assertRaises(RuntimeError) as caught:
                    _rfft_kernel()
            finally:
                if previous is None:
                    os.environ.pop("DF_RFFT_LIBRARY", None)
                else:
                    os.environ["DF_RFFT_LIBRARY"] = previous
            self.assertIn("failed to load", str(caught.exception))

    def test_missing_library_still_falls_back(self) -> None:
        previous = os.environ.get("DF_RFFT_LIBRARY")
        os.environ["DF_RFFT_LIBRARY"] = "/tmp/audio-db-phase4j-missing-libdf_rfft.so"
        try:
            self.assertIsNone(_rfft_kernel())
            record = rfft_runtime_record()
        finally:
            if previous is None:
                os.environ.pop("DF_RFFT_LIBRARY", None)
            else:
                os.environ["DF_RFFT_LIBRARY"] = previous
        self.assertEqual(record["kernel"], "fallback")
        self.assertEqual(record["source"], "environment")


class ProvenanceContractTests(unittest.TestCase):
    def test_new_checkpoint_binds_a_hash_and_legacy_none_stays_unproven(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = _write_yaml(
                directory,
                steps=2,
                batch_size=2,
                examples=8,
                features=4,
                accumulation=1,
                optimizer="sgd",
                distribute="none",
                stop_after=1,
            )
            first = Experiment(load_experiment(config))
            result = first.run()
            self.assertEqual(result.stop_reason, "interrupted")
            output = directory / "run"
            step1 = output / "step_00000001.pt"
            payload = torch.load(step1, map_location="cpu", weights_only=False)
            digest = payload["provenance_sha256"]
            self.assertIsInstance(digest, str)
            raw_bytes = (output / "data_provenance.json").read_bytes()
            self.assertEqual(hashlib.sha256(raw_bytes).hexdigest(), digest)
            self.assertEqual((output / "step_00000001.data_provenance.json").read_bytes(), raw_bytes)
            judgment = judge_checkpoint(step1, [])
            self.assertNotEqual(judgment["independent"], True)
            self.assertFalse(judgment["confirmed_independent"])
            legacy = dict(payload)
            legacy["provenance_sha256"] = None
            torch.save(legacy, step1)
            (output / "data_provenance.json").unlink()
            (output / "step_00000001.data_provenance.json").unlink()
            resumed_config = _write_yaml(
                directory,
                steps=2,
                batch_size=2,
                examples=8,
                features=4,
                accumulation=1,
                optimizer="sgd",
                distribute="none",
            )
            second = Experiment(load_experiment(resumed_config))
            resumed = second.run(resume=step1)
            self.assertEqual(resumed.logical_step, 2)
            kept = torch.load(step1, map_location="cpu", weights_only=False)
            self.assertIsNone(kept["provenance_sha256"])
            legacy_judgment = judge_experiment_checkpoint(step1, [])
            self.assertEqual(legacy_judgment["independent"], "UNKNOWN")
            self.assertTrue(legacy_judgment["legacy_checkpoint"])
            self.assertTrue((output / "data_provenance.json").is_file())
            step2 = torch.load(output / "step_00000002.pt", map_location="cpu", weights_only=False)
            self.assertIsInstance(step2["provenance_sha256"], str)
            metrics = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual([row["logical_step"] for row in metrics], [1, 2])

    def test_sidecar_tamper_and_dataset_swap_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = _write_yaml(
                directory,
                steps=1,
                batch_size=2,
                examples=8,
                features=4,
                accumulation=1,
                optimizer="sgd",
                distribute="none",
            )
            Experiment(load_experiment(config)).run()
            output = directory / "run"
            (output / "complete.json").unlink()
            step = output / "step_00000001.pt"
            original = step.read_bytes()
            sidecar = output / "step_00000001.data_provenance.json"
            saved = sidecar.read_bytes()
            sidecar.write_bytes(saved + b" ")
            with self.assertRaises(CheckpointError):
                Experiment(load_experiment(config)).run(resume=step)
            self.assertEqual(step.read_bytes(), original)
            sidecar.write_bytes(saved)
            document = json.loads(saved.decode("utf-8"))
            document["dataset_identity"]["data"]["examples"] = 999
            swapped = canonical_bytes(document)
            payload = torch.load(step, map_location="cpu", weights_only=False)
            payload["provenance_sha256"] = hashlib.sha256(swapped).hexdigest()
            foreign = output / "step_00000009.pt"
            torch.save(payload, foreign)
            (output / "step_00000009.data_provenance.json").write_bytes(swapped)
            (output / "data_provenance.json").write_bytes(swapped)
            try:
                with self.assertRaises(CheckpointError) as caught:
                    Experiment(load_experiment(config)).run(resume=foreign)
                self.assertIn("provenance", str(caught.exception))
            finally:
                (output / "data_provenance.json").write_bytes(saved)
            self.assertEqual(step.read_bytes(), original)

    def test_comparison_failure_keeps_the_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = _write_yaml(
                directory,
                steps=1,
                batch_size=2,
                examples=8,
                features=4,
                accumulation=1,
                optimizer="sgd",
                distribute="none",
            )
            experiment = Experiment(load_experiment(config))
            experiment.run()
            output = directory / "run"
            previous = (output / "step_00000001.pt").read_bytes()
            again = Experiment(load_experiment(config))
            again.build()
            with patch(
                "src.experiments.checkpoint._assert_replicated_states",
                side_effect=CheckpointError("ranks disagree at state.model"),
            ):
                with self.assertRaises(CheckpointError):
                    again._checkpoint.save(output, logical_step=2, label="step")
            self.assertFalse((output / "step_00000002.pt").is_file())
            self.assertEqual((output / "step_00000001.pt").read_bytes(), previous)
            again.close()

    def test_replicated_state_difference_is_reported(self) -> None:
        class _Owner:
            def __init__(self, blob: dict) -> None:
                self._blob = blob

            def state_dict(self) -> dict:
                return self._blob

        class _States:
            def __init__(self, blob: dict) -> None:
                self._owner = _Owner(blob)

            def ids(self) -> tuple[str, ...]:
                return ("model",)

            def owner(self, state_id: str) -> _Owner:
                del state_id
                return self._owner

        class _Runtime:
            world_size = 2

            def __init__(self, blobs: list[dict]) -> None:
                self._blobs = blobs

            def gather_objects(self, local: dict) -> list[dict]:
                del local
                return self._blobs

        left = {"weight": torch.zeros(4)}
        right = {"weight": torch.ones(4)}
        started = time.perf_counter()
        with self.assertRaises(CheckpointError) as caught:
            _assert_replicated_states(_States(left), _Runtime([left, right]))
        elapsed = time.perf_counter() - started
        self.assertIn("ranks disagree", str(caught.exception))
        matched = _Runtime([left, left])
        _assert_replicated_states(_States(left), matched)
        nbytes = sum(tensor.nbytes for tensor in left.values())
        Path("/tmp/phase4j-rank-compare.txt").write_text(
            f"tensors 1 bytes {nbytes} mismatch_seconds {elapsed:.6f}\n",
            encoding="utf-8",
        )

    def test_rank_state_tamper_refuses_the_save(self) -> None:
        for tamper, optimizer, scheduler in (
            ("optimizer", "adamw", False),
            ("scheduler", "adamw", True),
            ("trainer", "sgd", False),
        ):
            with self.subTest(tamper=tamper):
                directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
                config = _write_yaml(
                    directory,
                    steps=2,
                    batch_size=4,
                    examples=16,
                    features=4,
                    accumulation=1,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model="dropout",
                    trainer="tests.experiments.support.TamperTrainer",
                    trainer_kwargs=f"      tamper: {tamper}\n",
                )
                completed = _torchrun(config, timeout=120)
                self.assertNotEqual(completed.returncode, 0, completed.stderr + completed.stdout)
                self.assertIn("ranks disagree", completed.stderr + completed.stdout)
                output = directory / "run"
                self.assertFalse((output / "step_00000002.pt").is_file())
                previous = output / "step_00000001.pt"
                digest = (output / "previous.sha256").read_text(encoding="utf-8").strip()
                self.assertEqual(hashlib.sha256(previous.read_bytes()).hexdigest(), digest)


class StopBoundaryTests(unittest.TestCase):
    def test_trainers_observe_the_interrupt_flag(self) -> None:
        trainers = (
            MPSENetGANTrainer,
            MossFormerGANTrainer,
            TFGridNetTrainer,
            TFLocoformerTrainer,
            DeepFilterNet3Trainer,
            SEMambaPPTrainer,
        )
        request_interrupt()
        try:
            for cls in trainers:
                probe = object.__new__(cls)
                probe.stop_after_logical_steps = None
                probe.logical_step = 0
                self.assertTrue(probe._stop_requested(None), cls.__name__)
            self.assertTrue(ranks_should_stop(None))
        finally:
            clear_interrupt()
        probe = object.__new__(MPSENetGANTrainer)
        probe.stop_after_logical_steps = None
        probe.logical_step = 0
        self.assertFalse(probe._stop_requested(None))

    def test_sigterm_stops_after_a_finished_step_and_resume_keeps_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            output = directory / "run"
            config = directory / "experiment.yaml"
            config.write_text(
                textwrap.dedent(
                    f"""\
                    schema: audio-db-experiment/v1
                    seed: 1
                    deterministic: true
                    entry: trainer
                    runtime:
                      class: src.experiments.runtime.ProcessRuntime
                      kwargs:
                        device: cpu
                        precision: fp32
                        timeout_seconds: 30
                    output:
                      directory: {output}
                    components:
                      trainer:
                        role: trainer
                        class: tests.experiments.phase4j_support.InterruptProbe
                        kwargs:
                          max_logical_steps: 30
                          checkpoint_interval: 1
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            process = subprocess.Popen(
                [sys.executable, "-m", "src.experiments.cli", "--config", str(config)],
                cwd=ROOT,
                env=env,
                start_new_session=True,
            )
            deadline = time.monotonic() + 20
            step = output / "step_00000001.pt"
            while time.monotonic() < deadline and not step.is_file():
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertTrue(step.is_file(), "interrupt probe did not publish a step")
            process.send_signal(signal.SIGTERM)
            code = process.wait(timeout=30)
            self.assertEqual(code, 0)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "INTERRUPTED")
            self.assertFalse((output / "complete.json").is_file())
            show = subprocess.run(
                [sys.executable, str(ROOT / "scripts/experiments/show_status.py"), str(output)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(show.returncode, 0, show.stderr)
            self.assertIn("INTERRUPTED", show.stdout)
            finished = subprocess.run(
                [sys.executable, "-m", "src.experiments.cli", "--config", str(config), "--resume", str(step)],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(finished.returncode, 0, finished.stderr[-2000:])
            final = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "COMPLETED")
            metrics = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            steps = [row["logical_step"] for row in metrics]
            self.assertEqual(steps, list(range(1, 31)))
            self.assertEqual(len(steps), len(set(steps)))


class JudgeMaterialTests(unittest.TestCase):
    def test_presence_of_a_file_is_not_independence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint = directory / "step_00000001.pt"
            checkpoint.write_bytes(b"unused")
            missing = judge_checkpoint(checkpoint, [{"set_name": "eval", "sha256": "a" * 64, "source_sha256": "b" * 64, "lineage": "source"}])
            self.assertEqual(missing["independent"], "UNKNOWN")
            incomplete = {
                "claim_label": "protocol_conformant_substitute_materials",
                "format": "audio-db-data-provenance-v1",
                "materials": [],
                "pipeline": "experiment",
                "schema_version": 1,
                "selection_uses_test": False,
            }
            _write_provenance(directory, checkpoint, incomplete)
            empty = judge_checkpoint(checkpoint, [])
            self.assertEqual(empty["independent"], "UNKNOWN")
            self.assertFalse(empty["confirmed_independent"])
            train = "a" * 64
            selection = "d" * 64
            other = "b" * 64
            third = "c" * 64
            complete = {
                "claim_label": "official_recipe_scaled_down",
                "format": "audio-db-data-provenance-v1",
                "materials": [
                    {
                        "asset_id": "clean",
                        "lineage": "source",
                        "role": "train",
                        "sha256": train,
                        "source_sha256": None,
                    },
                    {
                        "asset_id": "valid",
                        "lineage": "source",
                        "role": "selection",
                        "sha256": selection,
                        "source_sha256": None,
                    },
                ],
                "pipeline": "reproduction",
                "schema_version": 1,
                "selection_uses_test": False,
            }
            _write_provenance(directory, checkpoint, complete)
            overlapped = judge_checkpoint(
                checkpoint,
                [{"set_name": "eval", "sha256": train, "source_sha256": None, "lineage": "source"}],
            )
            self.assertIs(overlapped["independent"], False)
            disjoint = judge_checkpoint(
                checkpoint,
                [{"set_name": "eval", "sha256": other, "source_sha256": third, "lineage": "source"}],
            )
            self.assertIs(disjoint["independent"], True)
            self.assertTrue(disjoint["confirmed_independent"])
            hole = dict(complete)
            hole["materials"] = [
                {"asset_id": "clean", "lineage": "unknown", "role": "train", "sha256": None, "source_sha256": None}
            ]
            _write_provenance(directory, checkpoint, hole)
            unknown = judge_checkpoint(
                checkpoint,
                [{"set_name": "eval", "sha256": other, "source_sha256": third, "lineage": "source"}],
            )
            self.assertEqual(unknown["independent"], "UNKNOWN")


def _write_provenance(directory: Path, checkpoint: Path, document: dict) -> None:
    raw = canonical_bytes(document)
    (directory / "data_provenance.json").write_bytes(raw)
    checkpoint.with_suffix(".data_provenance.json").write_bytes(raw)


if __name__ == "__main__":
    unittest.main()
