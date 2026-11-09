"""CPU Gloo DDP, distributed checkpoint, and resume."""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from src.experiments.checkpoint import DirectoryLock
from src.experiments.data import TensorPairData
from src.training.core.rng import set_determinism

ROOT = Path(__file__).resolve().parents[2]


class DistributedDataTests(unittest.TestCase):
    def test_batches_are_disjoint_and_cover_the_global_batch(self) -> None:
        data = TensorPairData(features=4, examples=16, batch_size=2, seed=3)
        data.world_size = 2
        data.plan_hash = data._plan_hash()
        self.assertEqual(data.global_batch_size, 4)
        seen: list[str] = []
        for update in range(4):
            per_rank = []
            for rank in (0, 1):
                data.rank = rank
                batch = data.train_epoch(0, start_update=0).batch(update)[0]
                per_rank.append(batch.sample_id.split(","))
            self.assertEqual(len(set(per_rank[0]) & set(per_rank[1])), 0)
            seen.extend(per_rank[0])
            seen.extend(per_rank[1])
        self.assertEqual(seen, [f"ex-{index}" for index in range(16)])
        self.assertEqual(data.update_index, 0)


class DistributedTrainingTests(unittest.TestCase):
    def test_two_ranks_train_and_match_an_averaged_gradient(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _write_yaml(
            directory,
            steps=2,
            batch_size=2,
            examples=16,
            features=4,
            accumulation=1,
            optimizer="sgd",
            trainer="tests.experiments.support.InspectTrainer",
        )
        completed = _torchrun(config)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        output = directory / "run"
        self.assertEqual(_status(output), "COMPLETED")
        metrics = _jsonl(output / "metrics.jsonl")
        self.assertEqual([row["logical_step"] for row in metrics], [1, 2])
        rank0 = _jsonl(output / "samples-rank0.jsonl")
        rank1 = _jsonl(output / "samples-rank1.jsonl")
        self.assertEqual([row["sample_id"] for row in rank0], [row["sample_id"] for row in metrics])
        self.assertEqual(len(rank1), 2)
        for left, right in zip(rank0, rank1):
            self.assertEqual(set(str(left["sample_id"]).split(",")) & set(str(right["sample_id"]).split(",")), set())
        payload = torch.load(output / "step_00000002.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["world_size"], 2)
        self.assertEqual(payload["rank_count_in_rng"], 2)
        self.assertEqual(payload["parameter_ranks_agree"], True)
        self.assertEqual([blob["rank"] for blob in payload["rng_by_rank"]], [0, 1])
        self.assertEqual(payload["optimizer_updates"], {"optimizer": 2})
        expected = _sgd_reference(steps=2, batch_size=2, examples=16, features=4, accumulation=1)
        self.assertEqual(_max_abs(payload["states"]["model"], expected), 0.0)
        names = sorted(path.name for path in output.iterdir())
        self.assertNotIn("samples-rank2.jsonl", names)

    def test_equivalent_global_batches_match_their_reductions(self) -> None:
        wide = Path(self.enterContext(self._directory()))
        narrow = Path(self.enterContext(self._directory()))
        wide_config = _write_yaml(
            wide,
            steps=1,
            batch_size=2,
            examples=8,
            features=4,
            accumulation=1,
            optimizer="sgd",
        )
        narrow_config = _write_yaml(
            narrow,
            steps=1,
            batch_size=1,
            examples=8,
            features=4,
            accumulation=2,
            optimizer="sgd",
        )
        wide_run = _torchrun(wide_config)
        narrow_run = _torchrun(narrow_config)
        self.assertEqual(wide_run.returncode, 0, wide_run.stderr + wide_run.stdout)
        self.assertEqual(narrow_run.returncode, 0, narrow_run.stderr + narrow_run.stdout)
        wide_state = torch.load(wide / "run" / "step_00000001.pt", map_location="cpu", weights_only=False)
        narrow_state = torch.load(narrow / "run" / "step_00000001.pt", map_location="cpu", weights_only=False)
        wide_expected = _sgd_reference(steps=1, batch_size=2, examples=8, features=4, accumulation=1)
        narrow_expected = _sgd_reference(steps=1, batch_size=1, examples=8, features=4, accumulation=2)
        self.assertEqual(_max_abs(wide_state["states"]["model"], wide_expected), 0.0)
        self.assertEqual(_max_abs(narrow_state["states"]["model"], narrow_expected), 0.0)
        cross_delta = _max_abs(wide_state["states"]["model"], narrow_state["states"]["model"])
        self.assertIsInstance(cross_delta, float)
        # Measured on this CPU: 7.450580596923828e-09. Each run matches its own
        # reduction with maximum absolute error 0. The cross-run gap is the
        # addition order inside the mean, so it is not a passing tolerance.

    def test_interrupted_resume_matches_a_continuous_two_rank_run(self) -> None:
        continuous_dir = Path(self.enterContext(self._directory()))
        interrupted_dir = Path(self.enterContext(self._directory()))
        continuous_config = _write_yaml(
            continuous_dir,
            steps=4,
            batch_size=4,
            examples=32,
            features=8,
            accumulation=1,
            optimizer="adamw",
            model="dropout",
            scheduler=True,
        )
        interrupted_config = _write_yaml(
            interrupted_dir,
            steps=4,
            stop_after=2,
            batch_size=4,
            examples=32,
            features=8,
            accumulation=1,
            optimizer="adamw",
            model="dropout",
            scheduler=True,
        )
        resume_config = _write_yaml(
            interrupted_dir,
            steps=4,
            batch_size=4,
            examples=32,
            features=8,
            accumulation=1,
            optimizer="adamw",
            model="dropout",
            scheduler=True,
            filename="resume.yaml",
        )
        continuous = _torchrun(continuous_config)
        self.assertEqual(continuous.returncode, 0, continuous.stderr + continuous.stdout)
        interrupted = _torchrun(interrupted_config)
        self.assertEqual(interrupted.returncode, 0, interrupted.stderr + interrupted.stdout)
        output = interrupted_dir / "run"
        self.assertEqual(_status(output), "INTERRUPTED")
        self.assertFalse((output / "complete.json").is_file())
        saved = output / "step_00000002.pt"
        mismatched = _cli(resume_config, "--resume", str(saved))
        self.assertEqual(mismatched.returncode, 1, mismatched.stdout)
        self.assertIn("world size", mismatched.stderr)
        self.assertEqual(_status(output), "INTERRUPTED")
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        copied = dict(payload)
        rng = list(copied["rng_by_rank"])
        rng[1] = rng[0]
        copied["rng_by_rank"] = rng
        copied_path = output / "copied_rng.pt"
        torch.save(copied, copied_path)
        _copy_provenance_sidecar(saved, copied_path)
        copied_run = _torchrun(resume_config, "--resume", str(copied_path))
        self.assertNotEqual(copied_run.returncode, 0)
        self.assertIn("belongs to rank", copied_run.stderr + copied_run.stdout)
        short = dict(payload)
        short["rng_by_rank"] = [payload["rng_by_rank"][0]]
        short["rank_count_in_rng"] = 1
        short_path = output / "short_rng.pt"
        torch.save(short, short_path)
        _copy_provenance_sidecar(saved, short_path)
        short_run = _torchrun(resume_config, "--resume", str(short_path))
        self.assertNotEqual(short_run.returncode, 0)
        self.assertIn("rng_by_rank", short_run.stderr + short_run.stdout)
        for name in ("metrics.jsonl", "samples-rank0.jsonl", "samples-rank1.jsonl"):
            with (output / name).open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"logical_step": 3, "loss": 123.0, "sample_id": "not-consumed"},
                        sort_keys=True,
                    )
                    + "\n"
                )
        resumed = _torchrun(resume_config, "--resume", str(saved))
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        self.assertEqual(_status(output), "COMPLETED")
        left = torch.load(continuous_dir / "run" / "step_00000004.pt", map_location="cpu", weights_only=False)
        right = torch.load(output / "step_00000004.pt", map_location="cpu", weights_only=False)
        self.assertEqual(_max_abs(left["states"], right["states"]), 0.0)
        self.assertEqual(left["logical_step"], 4)
        self.assertEqual(right["optimizer_updates"], left["optimizer_updates"])
        _assert_equal(left["rng_by_rank"], right["rng_by_rank"])
        self.assertEqual(_jsonl(continuous_dir / "run" / "metrics.jsonl"), _jsonl(output / "metrics.jsonl"))
        self.assertNotIn("not-consumed", [row["sample_id"] for row in _jsonl(output / "metrics.jsonl")])
        for name in ("samples-rank0.jsonl", "samples-rank1.jsonl"):
            self.assertEqual(_jsonl(continuous_dir / "run" / name), _jsonl(output / name))
            self.assertNotIn("not-consumed", [row["sample_id"] for row in _jsonl(output / name)])

    def test_rank_mismatch_refuses_the_save_and_keeps_the_previous_file(self) -> None:
        for tamper in ("weight", "buffer"):
            with self.subTest(tamper=tamper):
                directory = Path(self.enterContext(self._directory()))
                config = _write_yaml(
                    directory,
                    steps=2,
                    batch_size=4,
                    examples=32,
                    features=8,
                    accumulation=1,
                    optimizer="adamw",
                    model="dropout",
                    trainer="tests.experiments.support.TamperTrainer",
                    trainer_kwargs=f"      tamper: {tamper}\n",
                )
                completed = _torchrun(config)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ranks disagree", completed.stderr + completed.stdout)
                output = directory / "run"
                self.assertFalse((output / "step_00000002.pt").is_file())
                self.assertFalse((output / ".step_00000002.pt.tmp").is_file())
                previous = output / "step_00000001.pt"
                digest = (output / "previous.sha256").read_text(encoding="utf-8").strip()
                self.assertEqual(_sha256(previous), digest)
                payload = torch.load(previous, map_location="cpu", weights_only=False)
                self.assertEqual(payload["logical_step"], 1)

    def test_skip_disagreement_fails_without_hanging(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _write_yaml(
            directory,
            steps=1,
            batch_size=2,
            examples=8,
            features=4,
            accumulation=1,
            optimizer="sgd",
            trainer="tests.experiments.support.DisagreeTrainer",
            include_model=False,
        )
        completed = _torchrun(config, timeout=30)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ranks disagree", completed.stderr + completed.stdout)

    def test_a_second_launch_cannot_lock_the_directory(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        config = _write_yaml(
            directory,
            steps=1,
            batch_size=2,
            examples=8,
            features=4,
            accumulation=1,
            optimizer="sgd",
        )
        output = directory / "run"
        output.mkdir()
        held = DirectoryLock(output)
        try:
            completed = _torchrun(config, timeout=30)
        finally:
            held.release()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("already running", completed.stderr + completed.stdout)
        self.assertFalse((output / "status.json").is_file())

    def test_trainable_without_ddp_and_a_world_size_mismatch_are_refused(self) -> None:
        directory = Path(self.enterContext(self._directory()))
        plain = _write_yaml(
            directory,
            steps=1,
            batch_size=2,
            examples=8,
            features=4,
            accumulation=1,
            optimizer="sgd",
            distribute="none",
        )
        refused = _torchrun(plain, timeout=30)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("ddp", refused.stderr + refused.stdout)
        mismatched = _write_yaml(
            directory,
            steps=1,
            batch_size=2,
            examples=8,
            features=4,
            accumulation=1,
            optimizer="sgd",
            world_size=1,
            filename="mismatch.yaml",
        )
        launched = _torchrun(mismatched, timeout=30)
        self.assertNotEqual(launched.returncode, 0)
        self.assertIn("does not match WORLD_SIZE", launched.stderr + launched.stdout)

    def _directory(self):
        import tempfile

        return tempfile.TemporaryDirectory()


def _sgd_reference(*, steps: int, batch_size: int, examples: int, features: int, accumulation: int) -> dict[str, torch.Tensor]:
    """One-process replay of the Gloo average used by the DDP runs.

    Each rank sums its micro-batch gradients, then the two ranks are added and
    divided by the world size. That is the reduction DDP performs on the
    synchronizing backward.
    """

    set_determinism(True, cuda=False)
    random.seed(17)
    np.random.seed(17 % (2**32))
    torch.manual_seed(17)
    model = nn.Linear(features, features)
    data = TensorPairData(features=features, examples=examples, batch_size=batch_size, seed=0)
    data.world_size = 2
    data.plan_hash = data._plan_hash()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0)
    index = 0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        per_rank: list[list[torch.Tensor]] = []
        for rank in (0, 1):
            data.rank = rank
            clone = nn.Linear(features, features)
            clone.load_state_dict(model.state_dict())
            for offset in range(accumulation):
                batch = data.train_epoch(0, start_update=0).batch(index + offset)[0]
                loss = loss_fn(clone(batch.inputs), batch.targets) / accumulation
                loss.backward()
            per_rank.append([parameter.grad.detach().clone() for parameter in clone.parameters()])
        with torch.no_grad():
            for parameter, first, second in zip(model.parameters(), per_rank[0], per_rank[1]):
                parameter.grad = (first + second) / 2
        optimizer.step()
        index += accumulation
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _write_yaml(
    directory: Path,
    *,
    steps: int,
    batch_size: int,
    examples: int,
    features: int,
    accumulation: int,
    optimizer: str,
    stop_after: int | None = None,
    trainer: str = "src.experiments.trainers.StandardTrainer",
    trainer_kwargs: str = "",
    model: str = "linear",
    scheduler: bool = False,
    distribute: str = "ddp",
    include_model: bool = True,
    world_size: int | None = None,
    filename: str = "experiment.yaml",
) -> Path:
    stop = "" if stop_after is None else f"      stop_after_logical_steps: {stop_after}\n"
    world = "" if world_size is None else f"    world_size: {world_size}\n"
    if model == "dropout":
        model_block = f"""  model:
    role: module
    class: tests.experiments.support.DropoutLinear
    kwargs:
      in_features: {features}
      out_features: {features}
    prepare: {{trainable: true, distribute: {distribute}}}
"""
    else:
        model_block = f"""  model:
    role: module
    class: torch.nn.Linear
    kwargs:
      in_features: {features}
      out_features: {features}
    prepare: {{trainable: true, distribute: {distribute}}}
"""
    if optimizer == "sgd":
        optimizer_block = """  optimizer:
    role: optimizer
    class: torch.optim.SGD
    parameters_of: model
    kwargs: {lr: 0.1, momentum: 0, weight_decay: 0}
"""
    else:
        optimizer_block = """  optimizer:
    role: optimizer
    class: torch.optim.AdamW
    parameters_of: model
    kwargs: {lr: 0.001}
"""
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
    if include_model:
        body = f"""{model_block}  loss:
    role: loss
    class: torch.nn.MSELoss
  data:
    role: data
    class: src.experiments.data.TensorPairData
    kwargs:
      features: {features}
      examples: {examples}
      batch_size: {batch_size}
{optimizer_block}{scheduler_block}  trainer:
    role: trainer
    class: {trainer}
    kwargs:
      max_logical_steps: {steps}
      accumulation: {accumulation}
{stop}{trainer_kwargs}    inject:
      model: {{ref: model}}
      loss: {{ref: loss}}
      data: {{ref: data}}
      optimizer: {{ref: optimizer}}
{scheduler_inject}"""
    else:
        body = f"""  trainer:
    role: trainer
    class: {trainer}
    kwargs:
      max_logical_steps: {steps}
"""
    text = f"""schema: audio-db-experiment/v1
seed: 17
deterministic: true
entry: trainer
runtime:
  class: src.experiments.runtime.ProcessRuntime
  kwargs:
    device: cpu
    precision: fp32
    timeout_seconds: 20
{world}output:
  directory: {directory / "run"}
components:
{body}"""
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(path.read_text(encoding="utf-8"))
    return path


def _copy_provenance_sidecar(source: Path, dest: Path) -> None:
    sidecar = source.with_suffix(".data_provenance.json")
    if sidecar.is_file():
        dest.with_suffix(".data_provenance.json").write_bytes(sidecar.read_bytes())


def _torchrun(config: Path, *extra: str, timeout: int = 90) -> subprocess.CompletedProcess[str]:
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
            env=env,
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
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _status(directory: Path) -> str:
    return str(json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"])


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
            values.append(float((lhs.detach().cpu() - rhs.detach().cpu()).abs().max()))
            return
        if isinstance(lhs, dict):
            if not isinstance(rhs, dict):
                raise AssertionError(type(rhs))
            for key, value in lhs.items():
                walk(value, rhs[key])

    walk(left, right)
    return max(values) if values else 0.0


if __name__ == "__main__":
    unittest.main()
