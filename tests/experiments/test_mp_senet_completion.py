"""MP-SENet completion gates: checkpoint safe point, then structure.

The CUDA single-rank path used to keep a step-checkpoint request in
``_mp_io`` and offer it from ``drain_previous`` while the next official
update still had ``_in_official_step`` set. That mixes step N's progress
with step N+1's gradients. The offer must happen after step N's optimizers
and before step N+1's forward. Draining during the next update may flush
only the previous log line.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.api import build_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import RuntimeFailure
from src.experiments.execution.journal import StepJournal
from src.experiments.official.mp_senet.metrics import aggregate_reports
from src.experiments.runtime import ProcessRuntime
from tests.experiments.test_mp_senet_gan import _manual_trainer, _yaml_text
from tests.numerics.reference.mp_senet_task import MPSENetTask


class _Device:
    type = "cuda"


class _CudaView:
    """Reports a CUDA world of one rank without initializing a device."""

    device = _Device()
    world_size = 1
    rank = 0

    def gather_objects(self, value: object) -> list[object]:
        return [value]


class _Lifecycle:
    def __init__(self, directory: Path, trainer: object) -> None:
        self.output_directory = directory
        self.config_identity = None
        self.events: list[dict[str, object]] = []
        self._trainer = trainer

    def save_if_needed(self, event: str, logical_step: int | None = None, metrics: object = None, prepare: object = None, progress: object = None) -> None:
        del progress
        inside = bool(getattr(self._trainer, "_in_official_step", False))
        self.events.append({"event": event, "logical_step": logical_step, "inside": inside, "metrics": metrics})
        if event == "step" and inside:
            raise RuntimeFailure("refusing to checkpoint between the discriminator step and the generator step")
        if prepare is not None:
            prepare(self)


def _local() -> dict[str, float | str]:
    return {
        "complex": 0.1,
        "consistency": 0.2,
        "discriminator": 0.3,
        "generator": 0.4,
        "lr_d": 0.0005,
        "lr_g": 0.0005,
        "magnitude": 0.5,
        "metric": 0.6,
        "phase": 0.7,
        "sample_id": "b0",
        "time": 0.8,
    }


class CudaCheckpointSafePointTests(unittest.TestCase):
    def test_step_checkpoint_is_requested_before_the_next_update(self) -> None:
        """CUDA accumulation 1, non-boundary step: save at the safe point.

        A following drain stands in for the next update's PESQ overlap. It
        must not offer another checkpoint, and it must not raise the
        mid-pair guard.
        """

        runtime = ProcessRuntime(device="cpu", precision="fp32")
        try:
            trainer, _generator, _discriminator = _manual_trainer(runtime, accumulation=1)
            directory = Path(tempfile.mkdtemp())
            trainer.journal = StepJournal(directory, runtime)
            trainer.journal.begin(restored=False, logical_step=0)
            trainer.lifecycle = _Lifecycle(directory, trainer)
            trainer.progress.logical_step = 1
            continued = trainer._record_step(
                _CudaView(),
                _local(),
                epoch=0,
                epoch_done=False,
                will_validate=False,
            )
            self.assertTrue(continued)
            self.assertEqual(trainer.progress.step_log[-1]["logical_step"], 1)
            self.assertTrue(trainer.lifecycle.events, "step checkpoint was not requested at the safe point")
            self.assertEqual(trainer.lifecycle.events[0]["event"], "step")
            self.assertEqual(trainer.lifecycle.events[0]["logical_step"], 1)
            self.assertFalse(trainer.lifecycle.events[0]["inside"])
            metrics = (directory / "metrics.jsonl").read_text(encoding="utf-8")
            self.assertIn('"logical_step": 1', metrics)
            trainer._in_official_step = True
            trainer._drain_mp_io()
            self.assertEqual(len(trainer.lifecycle.events), 1)
            trainer._in_official_step = True
            with self.assertRaises(RuntimeFailure) as caught:
                trainer._offer("step")
            self.assertIn("refusing to checkpoint", str(caught.exception))
        finally:
            runtime.shutdown()

    @unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA required")
    def test_cuda_fit_saves_three_completed_steps(self) -> None:
        directory = Path(tempfile.mkdtemp())
        _write_cuda_yaml(directory, steps=3, updates=8, validation_interval=100)
        experiment = build_experiment(load_experiment(directory / "experiment.yaml"))
        try:
            result = experiment.run()
        finally:
            experiment.close()
        self.assertEqual(result.logical_step, 3)
        rows = _metrics(directory)
        self.assertEqual([row["logical_step"] for row in rows], [1, 2, 3])
        for step in (1, 2, 3):
            payload = torch.load(directory / f"archive/step_{step:08d}.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["logical_step"], step)
            trainer_state = payload["states"]["trainer"]
            self.assertEqual(trainer_state["logical_step"], step)
            self.assertEqual(trainer_state["optimizer_updates"]["optim_g"], step)
            self.assertEqual(trainer_state["optimizer_updates"]["optim_d"], step)
            self.assertEqual(len(trainer_state["step_log"]), step)
            self.assertEqual(payload["states"]["data"]["update_index"], step)
        first = torch.load(directory / "archive/step_00000001.pt", map_location="cpu", weights_only=False)
        second = torch.load(directory / "archive/step_00000002.pt", map_location="cpu", weights_only=False)
        self.assertGreater(_weight_gap(first, second), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA required")
    def test_cuda_stop_resume_reaches_the_same_step_as_a_continuous_run(self) -> None:
        continuous = Path(tempfile.mkdtemp())
        interrupted = Path(tempfile.mkdtemp())
        _write_cuda_yaml(continuous, steps=4, updates=8, validation_interval=100)
        _write_cuda_yaml(interrupted, steps=4, updates=8, validation_interval=100, stop_after=2)
        first = build_experiment(load_experiment(continuous / "experiment.yaml"))
        try:
            first.run()
        finally:
            first.close()
        stopped = build_experiment(load_experiment(interrupted / "experiment.yaml"))
        try:
            stopped_result = stopped.run()
        finally:
            stopped.close()
        self.assertEqual(stopped_result.stop_reason, "interrupted")
        self.assertEqual(stopped_result.logical_step, 2)
        _write_cuda_yaml(interrupted, steps=4, updates=8, validation_interval=100)
        resume = build_experiment(load_experiment(interrupted / "experiment.yaml"))
        try:
            resumed = resume.run(resume=interrupted / "archive/step_00000002.pt")
        finally:
            resume.close()
        self.assertEqual(resumed.logical_step, 4)
        left = torch.load(continuous / "archive/step_00000004.pt", map_location="cpu", weights_only=False)
        right = torch.load(interrupted / "archive/step_00000004.pt", map_location="cpu", weights_only=False)
        self.assertEqual(left["logical_step"], 4)
        self.assertEqual(right["logical_step"], 4)
        self.assertEqual(left["states"]["trainer"]["logical_step"], right["states"]["trainer"]["logical_step"])
        self.assertEqual(left["states"]["trainer"]["optimizer_updates"], right["states"]["trainer"]["optimizer_updates"])
        self.assertEqual(len(left["states"]["trainer"]["step_log"]), 4)
        self.assertEqual(len(right["states"]["trainer"]["step_log"]), 4)
        self.assertEqual([row["logical_step"] for row in _metrics(continuous)], [1, 2, 3, 4])
        self.assertEqual([row["logical_step"] for row in _metrics(interrupted)], [1, 2, 3, 4])
        stopped_step = torch.load(interrupted / "archive/step_00000002.pt", map_location="cpu", weights_only=False)
        self.assertGreater(_weight_gap(stopped_step, right), 0.0)
        # AdaptiveMaxPool has no deterministic CUDA backward, so this YAML turns
        # deterministic algorithms off. Bitwise equality with a second process
        # is not the contract here. CPU resume covers that comparison.

    @unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA required")
    def test_unsaved_cuda_step_flushes_its_log_inside_the_next_update(self) -> None:
        directory = Path(tempfile.mkdtemp())
        text = _yaml_text(directory, steps=3, updates=8, validation_interval=100).replace("device: cpu", "device: cuda")
        text = text.replace("deterministic: true", "deterministic: false")
        text = text.replace("interval: 1", "interval: 2")
        (directory / "experiment.yaml").write_text(text, encoding="utf-8")
        seen: list[tuple[object, ...]] = []
        from src.experiments.official.mp_senet.trainer import MPSENetGANTrainer

        original_drain = MPSENetGANTrainer._drain_mp_io
        original_offer = MPSENetGANTrainer._offer

        def drain(self: MPSENetGANTrainer) -> None:
            seen.append(("drain", self._in_official_step, self._recorder.pending_logical_step))
            original_drain(self)

        def offer(self: MPSENetGANTrainer, event: str, metrics: dict[str, float] | None = None) -> None:
            seen.append(("offer", event, self._in_official_step, self.logical_step))
            original_offer(self, event, metrics)

        MPSENetGANTrainer._drain_mp_io = drain
        MPSENetGANTrainer._offer = offer
        try:
            experiment = build_experiment(load_experiment(directory / "experiment.yaml"))
            try:
                experiment.run()
            finally:
                experiment.close()
        finally:
            MPSENetGANTrainer._drain_mp_io = original_drain
            MPSENetGANTrainer._offer = original_offer
        offers = [item for item in seen if item[0] == "offer" and item[1] == "step"]
        self.assertTrue(offers)
        self.assertTrue(all(item[2] is False for item in offers))
        drains = [item for item in seen if item[0] == "drain"]
        self.assertIn(("drain", True, 1), drains)
        self.assertFalse((directory / "archive/step_00000001.pt").exists())
        self.assertTrue((directory / "archive/step_00000002.pt").is_file())
        rows = _metrics(directory)
        self.assertEqual([row["logical_step"] for row in rows], [1, 2, 3])

    def test_cpu_epoch_end_checkpoint_records_validation_before_the_scheduler(self) -> None:
        self._assert_epoch_end_checkpoint(device="cpu", deterministic=True)

    @unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA required")
    def test_cuda_epoch_end_validates_then_steps_the_scheduler(self) -> None:
        self._assert_epoch_end_checkpoint(device="cuda", deterministic=False)

    def _assert_epoch_end_checkpoint(self, *, device: str, deterministic: bool) -> None:
        directory = Path(tempfile.mkdtemp())
        _write_cuda_yaml(
            directory,
            steps=2,
            updates=2,
            validation_interval=2,
            best_start=0,
            device=device,
            deterministic=deterministic,
        )
        experiment = build_experiment(load_experiment(directory / "experiment.yaml"))
        try:
            result = experiment.run()
        finally:
            experiment.close()
        self.assertEqual(result.logical_step, 2)
        payload = torch.load(directory / "archive/step_00000002.pt", map_location="cpu", weights_only=False)
        trainer_state = payload["states"]["trainer"]
        self.assertEqual(trainer_state["phase"], "in_epoch")
        self.assertEqual(trainer_state["scheduler_steps"], 1)
        self.assertIsNone(trainer_state["boundary"])
        self.assertEqual(len(trainer_state["completed_boundaries"]), 1)
        self.assertTrue(trainer_state["completed_boundaries"][0]["validate"])
        self.assertTrue(trainer_state["completed_boundaries"][0]["scheduler"])
        self.assertEqual(trainer_state["history"][0]["logical_step"], 2)
        self.assertTrue((directory / "archive/step_00000001.pt").is_file())
        self.assertEqual([row["logical_step"] for row in _metrics(directory)], [1, 2])

    @unittest.skipUnless(torch.cuda.is_available(), "NOT_RUN: CUDA required")
    def test_cuda_stub_and_real_pesq_both_finish_an_overlapped_step(self) -> None:
        from src.experiments.calc import losses as loss_lib

        stub_calls: list[str] = []
        real = loss_lib.metric_targets

        def stub(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor:
            stub_calls.append("stub")
            return torch.zeros(clean.shape[0], dtype=torch.float32)

        loss_lib.metric_targets = stub
        try:
            stub_dir = Path(tempfile.mkdtemp())
            _write_cuda_yaml(stub_dir, steps=2, updates=4, validation_interval=100)
            experiment = build_experiment(load_experiment(stub_dir / "experiment.yaml"))
            try:
                result = experiment.run()
            finally:
                experiment.close()
        finally:
            loss_lib.metric_targets = real
        self.assertEqual(result.logical_step, 2)
        self.assertGreaterEqual(len(stub_calls), 2)
        self.assertTrue((stub_dir / "archive/step_00000001.pt").is_file())
        self.assertTrue((stub_dir / "archive/step_00000002.pt").is_file())

        from src.experiments.runtime import ProcessRuntime
        from tests.experiments.test_mp_senet_gan import _manual_trainer, _tone_batch

        seen: list[object] = []

        def wrapped(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> torch.Tensor | None:
            value = real(clean, enhanced, sample_rate)
            seen.append(value)
            return value

        from src.experiments.execution.journal import StepJournal

        runtime = ProcessRuntime(device="cuda", precision="fp32")
        loss_lib.metric_targets = wrapped
        try:
            trainer, _generator, _discriminator = _manual_trainer(runtime, accumulation=1)
            journal_dir = Path(tempfile.mkdtemp())
            trainer.journal = StepJournal(journal_dir, runtime)
            local = trainer.official_step([_tone_batch(16000, 1)], runtime)
        finally:
            loss_lib.metric_targets = real
            runtime.shutdown()
        self.assertFalse(trainer._in_official_step)
        self.assertTrue(seen)
        self.assertTrue(any(torch.is_tensor(item) for item in seen), seen)
        self.assertIn("generator", local)
        self.assertIn("discriminator", local)


class ValidationSeparationTests(unittest.TestCase):
    def test_official_validation_skips_the_reference_task_and_keeps_rng(self) -> None:
        directory = Path(tempfile.mkdtemp())
        _write_cuda_yaml(directory, steps=1, updates=2, validation_interval=1, device="cpu", deterministic=True)
        experiment = build_experiment(load_experiment(directory / "experiment.yaml"))
        try:
            trainer = experiment.get("trainer")
            runtime = experiment._graph.runtime
            before = torch.random.get_rng_state().clone()
            cursor = (int(trainer.data.epoch), int(trainer.data.update_index))
            summary = trainer._validator.evaluate(runtime, 0, "trainer")
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual((int(trainer.data.epoch), int(trainer.data.update_index)), cursor)
            shadow = trainer._validator._shadow
            self.assertIsNotNone(shadow)
            self.assertFalse(hasattr(shadow, "optim_g"))
            self.assertFalse(hasattr(shadow, "optim_d"))
            self.assertNotIn("MPSENetTask", type(shadow).__name__)
            task = MPSENetTask(
                {
                    "loss": "official",
                    "model": dict(trainer.generator.unwrapped().architecture),
                    "optimizer": {"betas": [0.8, 0.99], "lr": 0.0005, "weight_decay": 0.01},
                    "scheduler": {"gamma": 0.99},
                    "seed": int(trainer.generator.unwrapped().seed),
                },
                torch.device("cpu"),
            )
            task.generator.load_state_dict(trainer.generator.unwrapped().generator.state_dict(), strict=True)
            task.discriminator.load_state_dict(
                trainer.discriminator.unwrapped().discriminator.state_dict(), strict=True
            )
            task.eval_mode()
            batch = next(iter(trainer.data.validation_epoch(0)))
            report = task.validation_report({"clean": batch.targets, "noisy": batch.inputs})
            self.assertEqual(aggregate_reports([report])["pesq"], summary["pesq"])
            self.assertEqual(aggregate_reports([report])["losses"], summary["losses"])
        finally:
            experiment.close()

    def test_src_does_not_import_tests_or_the_reference_task(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for path in (root / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                for module in modules:
                    self.assertFalse(module == "tests" or module.startswith("tests."), f"{path} imports {module}")
        package = root / "src/experiments/official/mp_senet"
        for path in package.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("MPSENetTask", text, path.name)
            self.assertNotIn("CheckpointManager", text, path.name)
        for name in ("training_step.py", "update_accumulated.py", "update_cuda_overlap.py", "validation.py", "recording.py", "boundary.py"):
            text = (package / name).read_text(encoding="utf-8")
            self.assertNotIn("save_if_needed", text, name)
            self.assertNotIn("MPSENetGANTrainer", text, name)
            self.assertNotIn("ResearchSession", text, name)


def _write_cuda_yaml(
    directory: Path,
    *,
    steps: int,
    updates: int,
    validation_interval: int,
    stop_after: int | None = None,
    device: str = "cuda",
    deterministic: bool = False,
    best_start: int = 40,
    accumulation: int = 1,
    distribute: str = "none",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = _yaml_text(
        directory,
        steps=steps,
        updates=updates,
        validation_interval=validation_interval,
        stop_after=stop_after,
        best_start=best_start,
        accumulation=accumulation,
        distribute=distribute,
    )
    text = text.replace("device: cpu", f"device: {device}")
    if not deterministic:
        text = text.replace("deterministic: true", "deterministic: false")
    (directory / "experiment.yaml").write_text(text, encoding="utf-8")


def _metrics(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _weight_gap(left: dict[str, object], right: dict[str, object]) -> float:
    worst = 0.0
    left_states = left["states"]
    right_states = right["states"]
    assert isinstance(left_states, dict) and isinstance(right_states, dict)
    for name in ("generator", "metric_discriminator"):
        left_module = left_states[name]
        right_module = right_states[name]
        assert isinstance(left_module, dict) and isinstance(right_module, dict)
        for key, value in left_module.items():
            other = right_module[key]
            if torch.is_tensor(value) and torch.is_tensor(other):
                worst = max(worst, float((value.detach().float() - other.detach().float()).abs().max()))
    return worst


def _nccl_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


class NCCLMPSENetTests(unittest.TestCase):
    @unittest.skipUnless(_nccl_available(), "NOT_RUN: two CUDA devices required")
    def test_two_ranks_agree_on_one_step(self) -> None:
        directory = Path(tempfile.mkdtemp())
        text = _yaml_text(directory, steps=1, updates=2, validation_interval=100, distribute="ddp")
        text = text.replace("device: cpu", "device: cuda").replace("deterministic: true", "deterministic: false")
        (directory / "experiment.yaml").write_text(text, encoding="utf-8")
        completed = _torchrun(directory)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        ranks = [json.loads((directory / f"worker-rank{rank}.json").read_text()) for rank in (0, 1)]
        self.assertEqual([row["optimizer_updates"]["optim_g"] for row in ranks], [1, 1])
        self.assertEqual([row["optimizer_updates"]["optim_d"] for row in ranks], [1, 1])
        payload = torch.load(directory / "archive/step_00000001.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["world_size"], 2)
        self.assertTrue(payload["parameter_ranks_agree"])
        self.assertEqual([row["average_calls"] for row in ranks], [1, 1])

    @unittest.skipUnless(_nccl_available(), "NOT_RUN: two CUDA devices required")
    def test_two_ranks_average_gradients_when_accumulation_is_two(self) -> None:
        directory = Path(tempfile.mkdtemp())
        _write_cuda_yaml(
            directory,
            steps=1,
            updates=2,
            validation_interval=100,
            accumulation=2,
            distribute="ddp",
        )
        completed = _torchrun(directory)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        ranks = [json.loads((directory / f"worker-rank{rank}.json").read_text()) for rank in (0, 1)]
        self.assertEqual([row["logical_step"] for row in ranks], [1, 1])
        self.assertEqual([row["optimizer_updates"]["optim_g"] for row in ranks], [1, 1])
        self.assertEqual([row["optimizer_updates"]["optim_d"] for row in ranks], [1, 1])
        self.assertEqual([row["average_calls"] for row in ranks], [1, 1])
        self.assertEqual([row["synced_during_generator_backward"] for row in ranks], [False, False])
        payload = torch.load(directory / "archive/step_00000001.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["world_size"], 2)
        self.assertTrue(payload["parameter_ranks_agree"])


def _torchrun(directory: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
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
            "-m",
            "tests.experiments.mp_senet_gan_worker",
            str(directory),
            "run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


if __name__ == "__main__":
    unittest.main()
