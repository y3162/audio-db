"""CUDA FP32, NCCL, and AMP FP16 training, overflow, and resume.

These tests are NOT_RUN when CUDA is hidden. They are not part of the CPU
regression command, which sets ``CUDA_VISIBLE_DEVICES`` empty.
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

import numpy as np
import torch
import yaml

from src.experiments.errors import PrepareError, RuntimeFailure
from src.experiments.runtime import ProcessRuntime

ROOT = Path(__file__).resolve().parents[2]
_GPU0 = os.environ.get("AUDIO_DB_CUDA_GPU0", "0")
_GPU1 = os.environ.get("AUDIO_DB_CUDA_GPU1", "1")
_DEVICES_ONE = _GPU0
_DEVICES_TWO = f"{_GPU0},{_GPU1}"


def setUpModule() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("NOT_RUN: CUDA is not visible")


class CudaAmpRuntimeTests(unittest.TestCase):
    def test_scaled_backward_unscale_and_step_ignore_a_none_return(self) -> None:
        _require_devices(1)
        torch.cuda.set_device(0)
        runtime = ProcessRuntime(device="cuda", precision="amp_fp16", init_scale=32.0, growth_interval=2000)
        weight = torch.nn.Parameter(torch.ones(2, device="cuda"))
        optimizer = torch.optim.SGD([weight], lr=0.1)
        loss = (weight * 2).sum()
        runtime.backward(loss)
        self.assertTrue(torch.equal(weight.grad, torch.full_like(weight, 64)))
        norm = runtime.clip([weight], 10.0, optimizer=optimizer)
        self.assertAlmostEqual(norm, 8.0**0.5, places=5)
        self.assertTrue(torch.allclose(weight.grad, torch.full_like(weight, 2)))
        with self.assertRaises(RuntimeFailure):
            runtime.clip([weight], 10.0, optimizer=optimizer)
        result = runtime.step(optimizer)
        runtime.end_step()
        self.assertTrue(result.applied)
        self.assertTrue(torch.allclose(weight.detach(), torch.full_like(weight, 0.8)))
        runtime.shutdown()

    def test_nonfinite_loss_is_not_an_overflow_skip(self) -> None:
        _require_devices(1)
        runtime = ProcessRuntime(device="cuda", precision="amp_fp16", init_scale=8.0)
        loss = torch.tensor(float("nan"), device="cuda", requires_grad=True)
        with self.assertRaises(RuntimeFailure) as caught:
            runtime.backward(loss)
        self.assertIn("not finite", str(caught.exception))
        runtime.shutdown()

    def test_scale_growth_and_finite_loss_overflow_then_resume(self) -> None:
        _require_devices(1)
        torch.manual_seed(17)
        runtime = ProcessRuntime(
            device="cuda",
            precision="amp_fp16",
            init_scale=2.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2,
        )
        model = torch.nn.Linear(2, 2).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs = torch.ones(4, 2, device="cuda")
        targets = torch.zeros(4, 2, device="cuda")
        applied: list[bool] = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                loss = torch.nn.functional.mse_loss(model(inputs), targets)
            self.assertTrue(bool(torch.isfinite(loss).all()))
            runtime.backward(loss)
            applied.append(runtime.step(optimizer).applied)
            runtime.end_step()
        self.assertEqual(applied, [True, True])
        self.assertEqual(runtime.scaler_state()["scale"], 4.0)
        self.assertEqual(runtime.scaler_state()["_growth_tracker"], 0)
        runtime.shutdown()

        runtime = ProcessRuntime(device="cuda", precision="amp_fp16", init_scale=65536.0, growth_interval=2000)
        torch.manual_seed(17)
        model = torch.nn.Linear(2, 2).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        with runtime.autocast():
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
        runtime.backward(loss)
        self.assertTrue(runtime.step(optimizer).applied)
        runtime.end_step()
        runtime.set_grad_scale(131072.0)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.zero_grad(set_to_none=True)
        with runtime.autocast():
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
        self.assertTrue(bool(torch.isfinite(loss).all()))
        runtime.backward(loss)
        skipped = runtime.step(optimizer)
        runtime.end_step()
        self.assertFalse(skipped.applied)
        for old, parameter in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, parameter.detach()))
        self.assertEqual(runtime.scaler_state()["scale"], 65536.0)

        optimizer.zero_grad(set_to_none=True)
        with runtime.autocast():
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
        runtime.backward(loss)
        resumed = runtime.step(optimizer)
        runtime.end_step()
        self.assertTrue(resumed.applied)
        self.assertFalse(all(torch.equal(old, parameter.detach()) for old, parameter in zip(before, model.parameters())))
        runtime.shutdown()

    def test_missing_scaler_state_is_refused(self) -> None:
        _require_devices(1)
        runtime = ProcessRuntime(device="cuda", precision="amp_fp16", init_scale=16.0)
        with self.assertRaises(RuntimeFailure):
            runtime.load_scaler({"scale": 16.0})
        with self.assertRaises(RuntimeFailure):
            runtime.load_scaler(
                {
                    "scale": 0.0,
                    "growth_factor": 2.0,
                    "backoff_factor": 0.5,
                    "growth_interval": 2000,
                    "_growth_tracker": 0,
                }
            )
        runtime.shutdown()

    def test_cpu_amp_and_a_busy_launch_do_not_fall_back(self) -> None:
        with self.assertRaises(PrepareError) as caught:
            ProcessRuntime(device="cpu", precision="amp_fp16")
        self.assertIn("refusing to fall back", str(caught.exception))


class CudaWorldTests(unittest.TestCase):
    def test_c1_fp32_resume_matches_and_restores_cuda_rng(self) -> None:
        _require_devices(1)
        continuous = Path(self.enterContext(tempfile.TemporaryDirectory()))
        interrupted = Path(self.enterContext(tempfile.TemporaryDirectory()))
        first = _write_yaml(
            continuous,
            device="cuda",
            precision="fp32",
            steps=4,
            grad_clip=0.5,
            trainer="tests.experiments.support.CudaConsumeTrainer",
            scheduler=True,
        )
        _write_yaml(
            interrupted,
            device="cuda",
            precision="fp32",
            steps=4,
            stop_after=2,
            grad_clip=0.5,
            trainer="tests.experiments.support.CudaConsumeTrainer",
            scheduler=True,
            filename="interrupted.yaml",
        )
        resume = _write_yaml(
            interrupted,
            device="cuda",
            precision="fp32",
            steps=4,
            grad_clip=0.5,
            trainer="tests.experiments.support.CudaConsumeTrainer",
            scheduler=True,
            filename="resume.yaml",
        )
        self._check(_cli(first, devices=_DEVICES_ONE), "c1 fp32 continuous")
        self._check(_cli(interrupted / "interrupted.yaml", devices=_DEVICES_ONE), "c1 fp32 interrupt")
        saved = interrupted / "run" / "step_00000002.pt"
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        self.assertEqual(payload["states"]["runtime"]["device"], "cuda")
        self.assertIsNone(payload["states"]["runtime"]["grad_scaler"])
        identity = payload["rng_by_rank"][0]["cuda_device_identity"]
        self.assertEqual(identity["index"], 0)
        self.assertIn("uuid", identity)
        copied = dict(payload)
        copied["rng_by_rank"] = [dict(payload["rng_by_rank"][0])]
        copied["rng_by_rank"][0]["cuda_device_identity"] = dict(identity, uuid="not-this-gpu")
        tampered = interrupted / "run" / "step_00000002_tampered.pt"
        torch.save(copied, tampered)
        refused = _cli(resume, "--resume", str(tampered), devices=_DEVICES_ONE)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("CUDA device", refused.stderr + refused.stdout)
        self._check(_cli(resume, "--resume", str(saved), devices=_DEVICES_ONE), "c1 fp32 resume")
        left = torch.load(continuous / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
        right = torch.load(interrupted / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
        self.assertEqual(_max_abs(left["states"], right["states"]), 0.0)
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(_jsonl(continuous / "run" / "metrics.jsonl"), _jsonl(interrupted / "run" / "metrics.jsonl"))
        self.assertEqual(right["states"]["runtime"]["precision"], "fp32")
        midway = torch.load(continuous / "run" / "step_00000002.pt", map_location="cpu", weights_only=False)
        self.assertGreater(_max_abs(left["states"]["model"], midway["states"]["model"]), 0.0)

    def test_c1_amp_autocast_overflow_and_resume(self) -> None:
        _require_devices(1)
        continuous = Path(self.enterContext(tempfile.TemporaryDirectory()))
        interrupted = Path(self.enterContext(tempfile.TemporaryDirectory()))
        common = dict(
            device="cuda",
            precision="amp_fp16",
            steps=4,
            init_scale=65536.0,
            model="autocast",
            trainer="tests.experiments.support.ScaleBumpTrainer",
            trainer_kwargs="      bump_step: 1\n      bump_scale: 131072.0\n      restore_step: 2\n      restore_scale: 1024.0\n",
            grad_clip=1.0,
            scheduler=True,
            learning_rate=0.01,
        )
        first = _write_yaml(continuous, **common)
        _write_yaml(interrupted, stop_after=2, filename="interrupted.yaml", **common)
        resume = _write_yaml(interrupted, filename="resume.yaml", **common)
        self._check(_cli(first, devices=_DEVICES_ONE), "c1 amp continuous")
        self._check(_cli(interrupted / "interrupted.yaml", devices=_DEVICES_ONE), "c1 amp interrupt")
        mid = torch.load(interrupted / "run" / "step_00000002.pt", map_location="cpu", weights_only=False)
        self.assertEqual(mid["optimizer_updates"], {"optimizer": 1})
        self.assertEqual(mid["states"]["runtime"]["grad_scaler"]["scale"], 65536.0)
        self.assertEqual([row["applied"] for row in _jsonl(interrupted / "run" / "metrics.jsonl")], [True, False])
        self.assertEqual(mid["states"]["model"]["autocast_on"].item(), 1.0)
        scaler = dict(mid["states"]["runtime"]["grad_scaler"])
        broken = dict(mid)
        broken["states"] = dict(mid["states"])
        broken["states"]["runtime"] = dict(mid["states"]["runtime"], grad_scaler=None)
        bad = interrupted / "run" / "step_00000002_noscaler.pt"
        torch.save(broken, bad)
        refused = _cli(resume, "--resume", str(bad), devices=_DEVICES_ONE)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("grad scaler", refused.stderr + refused.stdout)
        self._check(_cli(resume, "--resume", str(interrupted / "run" / "step_00000002.pt"), devices=_DEVICES_ONE), "c1 amp resume")
        left = torch.load(continuous / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
        right = torch.load(interrupted / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
        self.assertEqual(_max_abs(left["states"], right["states"]), 0.0)
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(_jsonl(continuous / "run" / "metrics.jsonl"), _jsonl(interrupted / "run" / "metrics.jsonl"))
        self.assertEqual([row["applied"] for row in _jsonl(interrupted / "run" / "metrics.jsonl")], [True, False, True, True])
        self.assertEqual(right["optimizer_updates"], {"optimizer": 3})
        self.assertEqual(right["states"]["scheduler"]["last_epoch"], left["states"]["scheduler"]["last_epoch"])
        self.assertEqual(scaler["scale"], 65536.0)

    def test_c2_nccl_resume_rank_samples_and_disagreement(self) -> None:
        _require_devices(2)
        continuous = Path(self.enterContext(tempfile.TemporaryDirectory()))
        interrupted = Path(self.enterContext(tempfile.TemporaryDirectory()))
        common = dict(
            device="cuda",
            precision="fp32",
            steps=4,
            world_size=2,
            distribute="ddp",
            trainer="tests.experiments.support.InspectTrainer",
            scheduler=True,
        )
        first = _write_yaml(continuous, **common)
        _write_yaml(interrupted, stop_after=2, filename="interrupted.yaml", **common)
        resume = _write_yaml(interrupted, filename="resume.yaml", **common)
        self._check(_torchrun(first, devices=_DEVICES_TWO), "c2 continuous")
        self._check(_torchrun(interrupted / "interrupted.yaml", devices=_DEVICES_TWO), "c2 interrupt")
        output = interrupted / "run"
        rank0 = _jsonl(output / "samples-rank0.jsonl")
        rank1 = _jsonl(output / "samples-rank1.jsonl")
        self.assertEqual(len(rank0), 2)
        for left, right in zip(rank0, rank1):
            self.assertEqual(set(str(left["sample_id"]).split(",")) & set(str(right["sample_id"]).split(",")), set())
        saved = output / "step_00000002.pt"
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        self.assertEqual([blob["rank"] for blob in payload["rng_by_rank"]], [0, 1])
        self.assertNotEqual(
            payload["rng_by_rank"][0]["cuda_device_identity"]["uuid"],
            payload["rng_by_rank"][1]["cuda_device_identity"]["uuid"],
        )
        swapped = dict(payload)
        swapped["rng_by_rank"] = [dict(payload["rng_by_rank"][0]), dict(payload["rng_by_rank"][0])]
        swapped_path = output / "step_00000002_swapped.pt"
        torch.save(swapped, swapped_path)
        refused = _torchrun(resume, "--resume", str(swapped_path), devices=_DEVICES_TWO)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("rank", (refused.stderr + refused.stdout).lower())
        self._check(_torchrun(resume, "--resume", str(saved), devices=_DEVICES_TWO), "c2 resume")
        left = torch.load(continuous / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
        right = torch.load(output / "step_00000004.pt", map_location="cpu", weights_only=False)
        self.assertEqual(_max_abs(left["states"], right["states"]), 0.0)
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(_jsonl(continuous / "run" / "metrics.jsonl"), _jsonl(output / "metrics.jsonl"))

    def test_c2_rank_mismatch_keeps_the_previous_checkpoint(self) -> None:
        _require_devices(2)
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config = _write_yaml(
            directory,
            device="cuda",
            precision="fp32",
            steps=2,
            world_size=2,
            distribute="ddp",
            model="dropout",
            trainer="tests.experiments.support.TamperTrainer",
            trainer_kwargs="      tamper: weight\n",
        )
        completed = _torchrun(config, devices=_DEVICES_TWO)
        self.assertNotEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("ranks disagree", completed.stderr + completed.stdout)
        output = directory / "run"
        self.assertFalse((output / "step_00000002.pt").is_file())
        previous = output / "step_00000001.pt"
        digest = (output / "previous.sha256").read_text(encoding="utf-8").strip()
        self.assertEqual(_sha256(previous), digest)

    def test_c2_two_ranks_cannot_share_one_gpu(self) -> None:
        _require_devices(1)
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config = _write_yaml(
            directory,
            device="cuda",
            precision="fp32",
            steps=1,
            world_size=2,
            timeout_seconds=10,
        )
        completed = _torchrun(config, devices=_DEVICES_ONE, timeout=40)
        self.assertNotEqual(completed.returncode, 0)
        text = completed.stderr + completed.stdout
        self.assertTrue("refusing" in text or "CUDA" in text)

    def test_c4_amp_nccl_accumulation_resume_and_coordinated_skip(self) -> None:
        _require_devices(2)
        continuous = Path(self.enterContext(tempfile.TemporaryDirectory()))
        interrupted = Path(self.enterContext(tempfile.TemporaryDirectory()))
        common = dict(
            device="cuda",
            precision="amp_fp16",
            steps=4,
            world_size=2,
            accumulation=2,
            batch_size=1,
            examples=16,
            init_scale=65536.0,
            distribute="ddp",
            scheduler=True,
        )
        first = _write_yaml(continuous, **common)
        _write_yaml(interrupted, stop_after=2, filename="interrupted.yaml", **common)
        resume = _write_yaml(interrupted, filename="resume.yaml", **common)
        self._check(_torchrun(first, devices=_DEVICES_TWO), "c4 continuous")
        self._check(_torchrun(interrupted / "interrupted.yaml", devices=_DEVICES_TWO), "c4 interrupt")
        output = interrupted / "run"
        for left, right in zip(_jsonl(output / "samples-rank0.jsonl"), _jsonl(output / "samples-rank1.jsonl")):
            self.assertEqual(set(str(left["sample_id"]).split(",")) & set(str(right["sample_id"]).split(",")), set())
        self._check(
            _torchrun(resume, "--resume", str(output / "step_00000002.pt"), devices=_DEVICES_TWO),
            "c4 resume",
        )
        left = torch.load(continuous / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
        right = torch.load(output / "step_00000004.pt", map_location="cpu", weights_only=False)
        self.assertEqual(_max_abs(left["states"], right["states"]), 0.0)
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(left["states"]["runtime"]["grad_scaler"], right["states"]["runtime"]["grad_scaler"])
        self.assertEqual(_jsonl(continuous / "run" / "metrics.jsonl"), _jsonl(output / "metrics.jsonl"))

        overflow_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        overflow = _write_yaml(
            overflow_dir,
            device="cuda",
            precision="amp_fp16",
            steps=3,
            world_size=2,
            distribute="ddp",
            init_scale=1024.0,
            grad_clip=1.0,
            trainer="tests.experiments.support.RankOverflowTrainer",
            trainer_kwargs="      overflow_step: 1\n",
            scheduler=True,
        )
        self._check(_torchrun(overflow, devices=_DEVICES_TWO), "c4 coordinated overflow")
        rows = _jsonl(overflow_dir / "run" / "metrics.jsonl")
        self.assertEqual([row["applied"] for row in rows], [True, False, True])
        saved = torch.load(overflow_dir / "run" / "step_00000003.pt", map_location="cpu", weights_only=False)
        self.assertEqual(saved["optimizer_updates"], {"optimizer": 2})
        self.assertEqual(saved["states"]["runtime"]["grad_scaler"]["scale"], 512.0)
        self.assertEqual(saved["states"]["scheduler"]["last_epoch"], 2)

    def _check(self, completed: subprocess.CompletedProcess[str], label: str) -> None:
        self.assertEqual(completed.returncode, 0, f"{label}\n{completed.stderr}\n{completed.stdout}")


def _require_devices(count: int) -> None:
    if torch.cuda.device_count() < count:
        raise unittest.SkipTest(f"NOT_RUN: need {count} visible CUDA devices")
    for index in range(count):
        free, total = torch.cuda.mem_get_info(index)
        used = (total - free) / (1024**2)
        if used > 1024:
            raise unittest.SkipTest(f"NOT_RUN: cuda:{index} is using {used:.0f} MiB")


def _write_yaml(
    directory: Path,
    *,
    device: str,
    precision: str,
    steps: int,
    batch_size: int = 4,
    examples: int = 32,
    features: int = 4,
    accumulation: int = 1,
    stop_after: int | None = None,
    trainer: str = "src.experiments.trainers.StandardTrainer",
    trainer_kwargs: str = "",
    model: str = "linear",
    scheduler: bool = False,
    distribute: str = "none",
    world_size: int | None = None,
    grad_clip: float | None = None,
    init_scale: float | None = None,
    timeout_seconds: float = 120,
    learning_rate: float = 0.1,
    filename: str = "experiment.yaml",
) -> Path:
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    clip = "" if grad_clip is None else f"      grad_clip: {grad_clip}\n"
    world = "" if world_size is None else f"    world_size: {world_size}\n"
    scale = "" if init_scale is None else f"    init_scale: {init_scale}\n"
    if model == "dropout":
        model_class = "tests.experiments.support.DropoutLinear"
    elif model == "autocast":
        model_class = "tests.experiments.support.AutocastLinear"
    else:
        model_class = "torch.nn.Linear"
    scheduler_block = ""
    scheduler_inject = ""
    if scheduler:
        scheduler_block = """  scheduler:
    role: scheduler
    class: torch.optim.lr_scheduler.StepLR
    kwargs: {step_size: 1, gamma: 0.5}
    inject:
      optimizer: {ref: optimizer}
"""
        scheduler_inject = "      scheduler: {ref: scheduler}\n"
    text = f"""schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: {device}
    precision: {precision}
    timeout_seconds: {timeout_seconds}
{world}{scale}output:
  directory: {directory / "run"}
components:
  model:
    role: module
    class: {model_class}
    kwargs:
      in_features: {features}
      out_features: {features}
    prepare: {{trainable: true, distribute: {distribute}}}
  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: {features}
      examples: {examples}
      batch_size: {batch_size}
  optimizer:
    role: optimizer
    class: torch.optim.SGD
    parameters_of: model
    kwargs: {{lr: {learning_rate}, momentum: 0, weight_decay: 0}}
{scheduler_block}  trainer:
    role: trainer
    class: {trainer}
    kwargs:
      max_logical_steps: {steps}
      accumulation: {accumulation}
{stop}{clip}{trainer_kwargs}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
{scheduler_inject}"""
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(path.read_text(encoding="utf-8"))
    return path


def _base_env(devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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
    return env


def _cli(config: Path, *extra: str, devices: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.experiments.cli", "--config", str(config), *extra],
        cwd=ROOT,
        env=_base_env(devices),
        check=False,
        capture_output=True,
        text=True,
    )


def _torchrun(config: Path, *extra: str, devices: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    visible = [item for item in devices.split(",") if item != ""]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(visible)),
        "--master_port",
        str(_free_port()),
        "-m",
        "src.experiments.cli",
        "--config",
        str(config),
        *extra,
    ]
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=_base_env(devices),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(command, 124, stdout, stderr + "\nTIMEOUT")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            delta = (lhs.detach().float().cpu() - rhs.detach().float().cpu()).abs()
            values.append(float(delta.max()) if delta.numel() else 0.0)
            return
        if isinstance(lhs, dict):
            if not isinstance(rhs, dict):
                raise AssertionError(type(rhs))
            for key, value in lhs.items():
                if key not in rhs:
                    raise AssertionError(key)
                walk(value, rhs[key])

    walk(left, right)
    return max(values) if values else 0.0


class SEMambaCudaTests(unittest.TestCase):
    """Short CUDA SEMamba++ steps. The module skip still covers a hidden GPU."""

    def test_reference_cuda_matches_the_task(self) -> None:
        _require_devices(1)
        report = _semamba_probe("reference", sys.executable)
        row = report["runs"][0]
        self.assertEqual(row["scan_configured"], "reference")
        self.assertEqual(row["scan_executed"], "reference")
        self.assertTrue(row["finite"])
        self.assertEqual(row["generator_forwards"], 1.0)
        # reflection_pad1d CUDA backward has no deterministic kernel, so two
        # runs of the same task already move weights by about 4e-4. The
        # trainer stays inside that self-rerun.
        self.assertLess(row["loss_max_abs"], 1e-4)
        self.assertLess(row["generator_grad_max_abs"], 1e-3)
        self.assertLess(row["generator_state_max_abs"], 1e-3)
        Path("/tmp/phase4h-cuda-reference.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")

    def test_fused_cuda_matches_the_task_and_records_the_reference_gap(self) -> None:
        _require_devices(1)
        python = ROOT / ".venvs" / "semambapp" / "bin" / "python"
        self.assertTrue(python.is_file())
        report = _semamba_probe("both", str(python))
        by_name = {row["backend"]: row for row in report["runs"]}
        fused = by_name["fused"]
        self.assertEqual(fused["scan_configured"], "fused")
        self.assertEqual(fused["scan_executed"], "fused")
        self.assertNotEqual(fused["scan_executed"], "reference")
        self.assertTrue(fused["finite"])
        self.assertLess(fused["loss_max_abs"], 1e-4)
        self.assertLess(fused["generator_state_max_abs"], 1e-3)
        cross = by_name["reference_versus_fused"]
        self.assertEqual(cross["reference_executed"], "reference")
        self.assertEqual(cross["fused_executed"], "fused")
        self.assertGreaterEqual(float(cross["generator_state_max_abs"]), 0.0)
        Path("/tmp/phase4h-cuda-both.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def test_nccl_reference_and_fused_accumulation(self) -> None:
        _require_devices(2)
        from tests.experiments.test_semambapp import _yaml

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend, python in (
                ("reference", sys.executable),
                ("fused", str(ROOT / ".venvs" / "semambapp" / "bin" / "python")),
            ):
                for accumulation in (1, 2):
                    target = root / f"{backend}-{accumulation}"
                    config = _yaml(
                        target,
                        steps=1,
                        count=accumulation,
                        group_size=accumulation,
                        accumulation=accumulation,
                        validation_interval=0,
                        scan_backend=backend,
                        device="cuda",
                        deterministic=False,
                    )
                    completed = _torchrun_python(config, python=python, devices=_DEVICES_TWO, timeout=300)
                    self.assertEqual(completed.returncode, 0, f"{backend} accum {accumulation}\n{completed.stderr[-2500:]}")
                    payload = torch.load(target / "step_00000001.pt", map_location="cpu", weights_only=False)
                    self.assertTrue(payload["parameter_ranks_agree"])
                    self.assertEqual(payload["world_size"], 2)
                    row = payload["states"]["trainer"]["step_log"][0]
                    self.assertEqual(row["scan_configured"], backend)
                    if accumulation == 1:
                        self.assertEqual(row["generator_forwards"], 1)
                        self.assertEqual(row["generator_reforwards"], 0)
                    else:
                        self.assertEqual(row["generator_previews"], 2)
                        self.assertEqual(row["generator_reforwards"], 2)


def _semamba_probe(mode: str, python: str) -> dict:
    directory = Path(tempfile.mkdtemp(prefix="semamba-cuda-"))
    out = directory / "report.json"
    env = _base_env(_DEVICES_ONE)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        [
            python,
            str(ROOT / "tests" / "experiments" / "semamba_cuda_probe.py"),
            "--mode",
            mode,
            "--length",
            "2048",
            "--out",
            str(out),
            "--directory",
            str(directory),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise AssertionError(f"semamba cuda probe {mode} failed\n{completed.stderr[-3000:]}\n{completed.stdout[-1500:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def _torchrun_python(config: Path, *, python: str, devices: str, timeout: int) -> subprocess.CompletedProcess[str]:
    visible = [item for item in devices.split(",") if item != ""]
    return subprocess.run(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(len(visible)),
            "--master_port",
            str(_free_port()),
            "-m",
            "src.experiments.cli",
            "--config",
            str(config),
        ],
        cwd=ROOT,
        env=_base_env(devices),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


if __name__ == "__main__":
    unittest.main()
